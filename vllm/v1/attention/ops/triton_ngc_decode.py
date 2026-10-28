# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K decode and page scoring with a shared partial/reduction pipeline."""

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.ngc_scoring import NGCScratchBuffers


@triton.jit(do_not_specialize=["num_tiles", "scratch_width"])
def _ngc_decode_score_partials(
    query,
    cache,
    ring,
    descriptors,
    key_positions,
    table,
    due,
    decode_partials,
    segment_max,
    segment_sum,
    page_mass,
    page_tags,
    query_stride,
    cache_page_stride,
    cache_head_stride,
    cache_token_stride,
    ring_query_stride,
    ring_head_stride,
    key_position_stride,
    table_stride,
    num_tiles,
    scratch_width,
    scale,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    RING_WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    SCORE_PAGE: tl.constexpr,
    D: tl.constexpr,
    Q: tl.constexpr,
    S: tl.constexpr,
    SEGMENTS: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    segment = program % SEGMENTS
    GQA: tl.constexpr = HEADS // KV_HEADS
    GROUPS: tl.constexpr = triton.cdiv(GQA, GROUP)
    group_id = program // SEGMENTS
    req = group_id // (KV_HEADS * GROUPS)
    kv_head = group_id // GROUPS % KV_HEADS
    first_head = group_id % GROUPS * GROUP
    rows = tl.arange(0, ROWS)
    local_head = first_head + rows % GROUP
    head = kv_head * GQA + local_head
    qs = rows // GROUP
    valid_row = (local_head < GQA) & (qs <= WINDOW)
    pid = req * HEADS + head
    descriptor = descriptors + req * 8
    seq_len = tl.load(descriptor)
    ring_slot = tl.load(descriptor + 1)
    count = tl.load(descriptor + 2)
    kept = tl.load(descriptor + 3)
    score_due = tl.load(due + req)
    table_row = tl.load(descriptor + 7)
    ds = tl.arange(0, D)
    ss = tl.arange(0, S)
    score_query = valid_row & (qs > 0) & (qs <= count) & score_due
    q_pos = seq_len - count + qs - 1
    ring_row = ring_slot * RING_WINDOW + q_pos % RING_WINDOW
    recent = tl.load(
        ring
        + ring_row[:, None] * ring_query_stride
        + head[:, None] * ring_head_stride
        + ds[None, :],
        score_query[:, None] & (ds[None, :] < DIM),
        0,
    )
    current = tl.load(
        query + req * query_stride + head[:, None] * DIM + ds[None, :],
        valid_row[:, None] & (qs[:, None] == 0) & (ds[None, :] < DIM),
        0,
    )
    q = tl.where(qs[:, None] == 0, current, recent)
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    ell = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    R: tl.constexpr = S // SCORE_PAGE
    part_base = page_mass + (pid * Q + qs) * scratch_width
    tag_base = page_tags + (pid * Q + qs) * num_tiles
    tiles_per_segment = tl.cdiv(kept, SEGMENTS * S)
    first_tile = segment * tiles_per_segment
    end_tile = tl.minimum(first_tile + tiles_per_segment, tl.cdiv(kept, S))
    for tile in range(first_tile, end_tile):
        slots = tile * S + ss
        valid = slots < kept
        pages = tl.load(table + table_row * table_stride + slots // PAGE, valid, 0)
        addresses = (
            pages.to(tl.int64)[:, None] * cache_page_stride
            + kv_head * cache_head_stride
            + (slots % PAGE)[:, None] * cache_token_stride
            + ds[None, :]
        )
        k = tl.load(cache + addresses, valid[:, None] & (ds[None, :] < DIM), 0).to(
            q.dtype
        )
        logits = tl.dot(q, tl.trans(k)) * scale
        original_positions = tl.load(
            key_positions + req * key_position_stride + slots, valid, 0
        )
        allowed = (
            valid[None, :]
            & valid_row[:, None]
            & (
                (qs[:, None] == 0)
                | (
                    score_query[:, None]
                    & (original_positions[None, :] < q_pos[:, None])
                )
            )
        )
        logits = tl.where(allowed, logits, float("-inf"))
        next_m = tl.maximum(m, tl.max(logits, 1))
        safe_m = tl.where(next_m == float("-inf"), 0.0, next_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(logits - safe_m[:, None])
        ell = ell * alpha + tl.sum(p, 1)
        m = next_m
        v = tl.load(
            cache + addresses + DIM, valid[:, None] & (ds[None, :] < DIM), 0
        ).to(q.dtype)
        # Only the current query needs PV; retain tensor-core accumulation.
        decode_p = tl.where(qs[:, None] == 0, p, 0.0)
        acc = acc * alpha[:, None]
        acc += tl.dot(decode_p.to(v.dtype), v)
        if score_due:
            block_sums = tl.sum(tl.reshape(p, (ROWS, R, SCORE_PAGE)), 2)
            cols = tile * R + tl.arange(0, R)
            tl.store(
                part_base[:, None] + cols[None, :],
                block_sums,
                score_query[:, None],
            )
            tl.store(tag_base + tile, m, score_query)
    state_offsets = (pid * Q + qs) * SEGMENTS + segment
    # Empty segments explicitly publish neutral states for the reduction.
    tl.store(segment_max + state_offsets, m, valid_row)
    tl.store(segment_sum + state_offsets, ell, valid_row)
    tl.store(
        decode_partials + (pid[:, None] * SEGMENTS + segment) * D + ds[None, :],
        acc,
        valid_row[:, None] & (qs[:, None] == 0),
    )


@triton.jit(do_not_specialize=["num_pages", "num_tiles", "scratch_width", "jobs"])
def _ngc_decode_score_reduce(
    descriptors,
    due,
    decode_partials,
    segment_max,
    segment_sum,
    page_mass,
    page_tags,
    output,
    scores,
    output_stride,
    num_pages,
    num_tiles,
    scratch_width,
    jobs,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    Q: tl.constexpr,
    D: tl.constexpr,
    SEGMENTS: tl.constexpr,
    SCORE_PAGE: tl.constexpr,
    S: tl.constexpr,
    HQ: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    req = program // jobs
    job = program % jobs
    segments = tl.arange(0, SEGMENTS)
    if job < HEADS:
        pid = req * HEADS + job
        offsets = pid * Q * SEGMENTS + segments
        decode_maxima = tl.load(segment_max + offsets)
        decode_sums = tl.load(segment_sum + offsets)
        decode_final_m = tl.max(decode_maxima, 0)
        decode_final_m = tl.where(decode_final_m == float("-inf"), 0.0, decode_final_m)
        factors = tl.exp(decode_maxima - decode_final_m)
        decode_denominator = tl.sum(decode_sums * factors, 0)
        ds = tl.arange(0, D)
        partial = tl.load(
            decode_partials + (pid * SEGMENTS + segments[:, None]) * D + ds[None, :]
        )
        decode_result = tl.sum(partial * factors[:, None], 0)
        decode_result = decode_result / tl.where(
            decode_denominator > 0, decode_denominator, 1.0
        )
        tl.store(output + req * output_stride + job * DIM + ds, decode_result, ds < DIM)
    else:
        cols = (job - HEADS) * BLOCK_P + tl.arange(0, BLOCK_P)
        kept = tl.load(descriptors + req * 8 + 3)
        count = tl.load(descriptors + req * 8 + 2)
        score_due = tl.load(due + req)
        if score_due and (job - HEADS) * BLOCK_P < tl.cdiv(kept, SCORE_PAGE):
            hq = tl.arange(0, HQ)
            heads = hq // WINDOW
            queries = hq % WINDOW + 1
            valid_hq = (heads < HEADS) & (queries <= count)
            state = ((req * HEADS + heads) * Q + queries) * SEGMENTS
            score_maxima = tl.load(
                segment_max + state[:, None] + segments[None, :],
                valid_hq[:, None],
                float("-inf"),
            )
            score_sums = tl.load(
                segment_sum + state[:, None] + segments[None, :],
                valid_hq[:, None],
                0.0,
            )
            score_final_m = tl.max(score_maxima, 1)
            score_final_m = tl.where(score_final_m == float("-inf"), 0.0, score_final_m)
            score_denominator = tl.sum(
                score_sums * tl.exp(score_maxima - score_final_m[:, None]), 1
            )
            score_denominator = tl.where(score_denominator > 0, score_denominator, 1.0)
            valid = valid_hq[:, None] & (cols[None, :] < tl.cdiv(kept, SCORE_PAGE))
            row = (req * HEADS + heads) * Q + queries
            mass = tl.load(
                page_mass + row[:, None] * scratch_width + cols[None, :], valid, 0.0
            )
            tags = tl.load(
                page_tags
                + row[:, None] * num_tiles
                + (cols // (S // SCORE_PAGE))[None, :],
                valid,
                float("-inf"),
            )
            normalized = (
                mass
                * tl.exp(tags - score_final_m[:, None])
                / score_denominator[:, None]
            )
            score_result = tl.sum(tl.where(valid, normalized, 0.0), 0)
        else:
            score_result = tl.full((BLOCK_P,), 0.0, tl.float32)
        # Each page has one writer, including zero scores for non-due/tail rows.
        tl.store(scores + req * num_pages + cols, score_result, cols < num_pages)


def ngc_decode_and_score(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    ring: torch.Tensor,
    descriptors: torch.Tensor,
    key_positions: torch.Tensor,
    block_table: torch.Tensor,
    due: torch.Tensor,
    output: torch.Tensor,
    *,
    window: int,
    ring_window: int,
    page_size: int,
    max_kept: int,
    softmax_scale: float,
    num_segments: int = 16,
    tile_size: int = 16,
    scratch: "NGCScratchBuffers | None" = None,
    score_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute decode and scores in two launches sharing the KV-reading pass.

    Queries/output are [requests, heads, dim], and cache is [pages, kv_heads,
    kernel_page_size, 2*dim]. Descriptors use the packed NGC input layout.
    The first launch writes segment states and page masses. The second reduces
    both outputs without rereading KV. Only one query per request is supported.
    Scratch may be reused for sequential calls on one CUDA stream; score_buffer
    must remain separate for each layer until its eviction decisions are read.
    FP16/BF16 cached keys and values are cast to the query dtype inside the kernel.
    """
    if query.ndim != 3 or kv_cache.ndim != 4 or output.shape != query.shape:
        raise ValueError(
            "Fused NGC decode requires 3D queries/output and a 4D KV cache"
        )
    batch, heads, dim = query.shape
    kernel_page = kv_cache.shape[2]
    if (
        kv_cache.shape[-1] != 2 * dim
        or query.stride(-1) != 1
        or query.stride(1) != dim
        or output.stride(-1) != 1
        or output.stride(1) != dim
        or kv_cache.stride(-1) != 1
    ):
        raise ValueError(
            "Fused NGC decode requires contiguous heads and logical K/V storage"
        )
    if (
        query.dtype not in (torch.float16, torch.bfloat16)
        or kv_cache.dtype not in (torch.float16, torch.bfloat16)
        or ring.dtype != query.dtype
        or output.dtype != query.dtype
    ):
        raise ValueError(
            "Fused NGC decode requires FP16/BF16 queries and KV, "
            "with ring and output matching the query dtype"
        )
    if (
        page_size <= 0
        or page_size & (page_size - 1)
        or page_size < kernel_page
        or page_size % kernel_page
    ):
        raise ValueError("Scoring pages must contain whole kernel pages")
    if heads % kv_cache.shape[1] or not 0 < window <= ring_window:
        raise ValueError("Invalid GQA head ratio or recent-query window")
    if num_segments <= 0 or num_segments & (num_segments - 1):
        raise ValueError("Segment count must be a positive power of two")
    if tile_size < 16 or tile_size & (tile_size - 1):
        raise ValueError("Tile size must be a power of two of at least 16")
    if batch <= 0 or max_kept <= 0:
        raise ValueError("Fused NGC decode requires a nonempty batch and KV context")
    block_slots = max(tile_size, page_size)
    block_q = max(16, triton.next_power_of_2(window + 1))
    block_d = max(16, triton.next_power_of_2(dim))
    pages = triton.cdiv(max_kept, page_size)
    tiles = triton.cdiv(max_kept, block_slots)
    scratch_width = tiles * (block_slots // page_size)
    if scratch is None:
        from vllm.v1.worker.ngc_scoring import NGCScratchBuffers

        scratch = NGCScratchBuffers()
    if score_buffer is None:
        scores = torch.empty((batch, pages), device=query.device, dtype=torch.float32)
    else:
        if (
            score_buffer.shape != (batch, pages)
            or score_buffer.dtype != torch.float32
            or score_buffer.device != query.device
            or not score_buffer.is_contiguous()
        ):
            raise ValueError(
                "Score buffer must be contiguous FP32 [batch, pages] on query device"
            )
        scores = score_buffer
    partials = scratch.get(
        "page_mass", (batch, heads, block_q, scratch_width), query.device
    )
    tags = scratch.get("page_tags", (batch, heads, block_q, tiles), query.device)
    decode_partials = scratch.get(
        "decode", (batch, heads, num_segments, block_d), query.device
    )
    state_shape = (batch, heads, block_q, num_segments)
    segment_max = scratch.get("segment_max", state_shape, query.device)
    segment_sum = scratch.get("segment_sum", state_shape, query.device)
    gqa = heads // kv_cache.shape[1]
    group = min(gqa, 8)
    groups = triton.cdiv(gqa, group)
    _ngc_decode_score_partials[(batch * kv_cache.shape[1] * groups * num_segments,)](
        query,
        kv_cache,
        ring,
        descriptors,
        key_positions,
        block_table,
        due,
        decode_partials,
        segment_max,
        segment_sum,
        partials,
        tags,
        query.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        ring.stride(0),
        ring.stride(1),
        key_positions.stride(0),
        block_table.stride(0),
        tiles,
        scratch_width,
        softmax_scale,
        HEADS=heads,
        KV_HEADS=kv_cache.shape[1],
        DIM=dim,
        RING_WINDOW=ring_window,
        PAGE=kernel_page,
        SCORE_PAGE=page_size,
        D=block_d,
        Q=block_q,
        S=block_slots,
        SEGMENTS=num_segments,
        GROUP=group,
        ROWS=max(16, triton.next_power_of_2(group * (window + 1))),
        WINDOW=window,
    )
    jobs = heads + triton.cdiv(pages, 16)
    _ngc_decode_score_reduce[(batch * jobs,)](
        descriptors,
        due,
        decode_partials,
        segment_max,
        segment_sum,
        partials,
        tags,
        output,
        scores,
        output.stride(0),
        pages,
        tiles,
        scratch_width,
        jobs,
        HEADS=heads,
        DIM=dim,
        WINDOW=window,
        Q=block_q,
        D=block_d,
        SEGMENTS=num_segments,
        SCORE_PAGE=page_size,
        S=block_slots,
        HQ=triton.next_power_of_2(heads * window),
        BLOCK_P=16,
    )
    return scores
