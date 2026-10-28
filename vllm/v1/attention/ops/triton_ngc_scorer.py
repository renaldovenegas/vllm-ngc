# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-block attention mass of recent queries over retained cached keys.

Each program scores all recent queries for one head, reusing each K tile.
An online softmax writes unnormalized block sums and running-max tags to
scratch, then rescales them using the final maximum and denominator.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=["kept_len", "num_blocks", "num_q", "scratch_w", "num_tiles"]
)
def ngc_scorer_kernel(
    # inputs
    q_ptr,  # (num_q, num_layers, num_qo_heads, head_dim)
    k_ptr,  # (num_layers, num_pages, num_kv_heads, page_size, head_dim)
    page_table_ptr,  # (num_row_pages,) int32 — PAGE ids of the compacted row, in order
    query_positions_ptr,
    key_positions_ptr,
    # scratch, (num_layers, num_qo_heads, BLOCK_Q, ·) float32; may be uninitialized
    partials_ptr,  # last dim scratch_w — unnormalized per-block mass
    tags_ptr,  # last dim num_tiles — running max when the tile was written
    # outputs
    out_ptr,  # (num_layers, num_blocks) float32 — per-block mass sums, atomic_add'd
    #          into by every head of the layer; host must zero it first
    # sizes
    kept_len,  # slots with valid K in the row; loop bound and softmax mask
    num_blocks,  # ceil(num_slots / block_size) — out's per-layer stride
    num_q,  # actual recent query count; padding rows are masked
    scratch_w,  # partials' row stride = num_tiles * R; >= every block index written
    num_tiles,  # tags' row stride = cdiv(kept_len, BLOCK_SLOTS)
    # strides, in elements (head_dim is stride-1 for both q and k; host asserts)
    stride_q_q,  # q: next recent query
    stride_q_layer,  # q: next layer
    stride_q_head,  # q: next qo head
    stride_k_layer,  # k: next layer's K plane
    stride_k_page,  # independent of token stride, including cross-layer gaps
    stride_k_slot,  # k: next token within a page
    stride_k_head,  # k: next kv head
    softmax_scale,
    # Tile sizes depend on query count and head/page dimensions, not kept_len.
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_GROUP: tl.constexpr,  # num_qo_heads // num_kv_heads
    BLOCK_Q: tl.constexpr,  # max(16, next_power_of_2(num_q)) — 16 is tl.dot's minimum
    BLOCK_SLOTS: tl.constexpr,  # K tile; multiple of BLOCK_SIZE
    BLOCK_SIZE: tl.constexpr,  # scoring block; power-of-two multiple of PAGE_SIZE
    BLOCK_N: tl.constexpr,  # scoring blocks per fixup iteration
):
    R: tl.constexpr = BLOCK_SLOTS // BLOCK_SIZE  # NGC blocks per tile

    # The vLLM wrapper launches one layer at a time.
    pid_layer = tl.program_id(0).to(tl.int64)
    pid_head = tl.program_id(1).to(tl.int64)
    pid = pid_layer * tl.num_programs(1) + pid_head

    offs_d = tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_SLOTS)
    offs_q = tl.arange(0, BLOCK_Q)

    kv_head = pid_head // GQA_GROUP
    k_base = k_ptr + pid_layer * stride_k_layer + kv_head * stride_k_head

    # This program's private (BLOCK_Q, ·) scratch slabs.
    part_base = partials_ptr + pid * BLOCK_Q * scratch_w
    tag_base = tags_ptr + pid * BLOCK_Q * num_tiles

    # Load all num_q query vectors for this (layer, head) before the K loop.
    # BLOCK_Q >= num_q (next power of 2); queries past num_q are zeroed.
    q_base = q_ptr + pid_layer * stride_q_layer + pid_head * stride_q_head
    q_mask = offs_q[:, None] < num_q  # (BLOCK_Q, 1) — masks whole rows
    query_positions = tl.load(
        query_positions_ptr + offs_q, mask=offs_q < num_q, other=0
    )
    q_group = tl.load(
        q_base + offs_q[:, None] * stride_q_q + offs_d[None, :],
        mask=q_mask & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )  # (BLOCK_Q, HEAD_DIM)

    # online softmax
    m = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)  # running row max
    ell = tl.zeros((BLOCK_Q,), dtype=tl.float32)  # running row denom

    # Process K in tiles — each tile loaded once, reused against all queries
    for start in range(0, kept_len, BLOCK_SLOTS):
        slots = start + offs_s
        mask = slots < kept_len

        page = tl.load(page_table_ptr + slots // PAGE_SIZE, mask=mask, other=0)
        k_addr = (
            k_base
            + page.to(tl.int64)[:, None] * stride_k_page
            + (slots % PAGE_SIZE).to(tl.int64)[:, None] * stride_k_slot
            + offs_d[None, :]
        )
        k_tile = tl.load(
            k_addr,
            mask=mask[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(q_group.dtype)

        logits = (
            tl.dot(q_group, tl.trans(k_tile)) * softmax_scale
        )  # (BLOCK_Q, BLOCK_SLOTS)
        key_positions = tl.load(key_positions_ptr + slots, mask=mask, other=0)
        allowed = (
            q_mask & mask[None, :] & (key_positions[None, :] < query_positions[:, None])
        )
        logits = tl.where(allowed, logits, float("-inf"))

        # Softmax update
        m_new = tl.maximum(m, tl.max(logits, axis=1))
        # An entirely masked prefix has no finite maximum yet.
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - safe_m)
        exp_tile = tl.exp(logits - safe_m[:, None])
        ell = ell * alpha + tl.sum(exp_tile, axis=1)
        m = m_new

        # To maximally exploit that scoring is over contiguous blocks:
        # reshape to view per scoring block
        exp_blocks = tl.reshape(exp_tile, (BLOCK_Q, R, BLOCK_SIZE))
        # sum of scores per Q tensor and scoring block, shape (BLOCK_Q, R)
        block_sums = tl.sum(exp_blocks, axis=2)

        # Spill this tile's R block sums to scratch. They are final values —
        # only their scale (m) is provisional, so one tag per tile covers all R.
        # scratch_w is sized off the loop trip count, so no column mask is needed.
        tile = start // BLOCK_SLOTS
        cols_r = tile * R + tl.arange(0, R)  # (R,) global scoring-block ids
        tl.store(
            part_base + offs_q[:, None] * scratch_w + cols_r[None, :],
            block_sums,
            mask=q_mask,
        )
        tl.store(tag_base + offs_q * num_tiles + tile, m, mask=offs_q < num_q)

    tl.debug_barrier()

    # rescale to the final max, normalize by the final denom, and
    # reduce over queries.
    n_eff = tl.minimum(tl.cdiv(kept_len, BLOCK_SLOTS) * R, num_blocks)
    for n0 in range(0, n_eff, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < n_eff
        # Drop tail columns and padding query rows.
        ok = q_mask & col_mask[None, :]

        p = tl.load(
            part_base + offs_q[:, None] * scratch_w + cols[None, :], mask=ok, other=0.0
        )
        # One tag per K tile, shared by that tile's R scoring blocks.
        t = tl.load(
            tag_base + offs_q[:, None] * num_tiles + (cols // R)[None, :],
            mask=ok,
            other=float("-inf"),
        )
        final_m = tl.where(ell > 0, m, 0.0)
        denom = tl.where(ell > 0, ell, 1.0)
        v = p * tl.exp(t - final_m[:, None]) / denom[:, None]

        # Every qo head of this layer accumulates into the same row, hence atomic.
        tl.atomic_add(
            out_ptr + pid_layer * num_blocks + cols,
            tl.sum(tl.where(ok, v, 0.0), axis=0),
            mask=col_mask,
        )


def ngc_page_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    kept_len: int,
    softmax_scale: float,
    scoring_block_size: int | None = None,
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    """Score retained pages for one request and one attention layer.

    Each query attends only to retained keys with strictly smaller original
    positions, excluding its own token. Scores sum probability mass over queries,
    query heads, and tokens in each scoring block; they are not averaged.

    Args:
        query: Recent post-RoPE queries, shaped [num_q, num_q_heads, head_dim].
        key_cache: Unquantized post-RoPE keys in logical [pages, kv_heads,
            page_size, head_dim] order. For the standard vLLM cache, pass
            ``kv_cache[..., :head_dim]``. Physical strides are preserved.
            FP16/BF16 keys are cast to the query dtype inside the kernel.
        block_table: One contiguous int32/int64 request row of physical page
            IDs, in retained logical order. Referenced IDs must be valid and
            owned by this request; there must be no holes or null blocks in
            the retained prefix. Unused trailing entries are ignored.
        kept_len: Number of retained tokens. All pages except the last must
            be full. This is a host integer, not a GPU sequence-length tensor.
        softmax_scale: Scale applied to QK logits.
        scoring_block_size: Tokens per output score; defaults to page_size.
            Must be a power-of-two multiple of page_size.
        query_positions: Contiguous int32/int64 original token positions,
            shaped [num_q], on the query device.
        key_positions: Contiguous int32/int64 original token positions,
            shaped [kept_len], in the same retained logical token order as
            block_table. These are not physical cache slots or compacted indices.
            Positions need not be sorted. Must be on the query device.

    Returns:
        Float32 scores of shape [ceil(kept_len / scoring_block_size)], in
        retained logical order. Each query with at least one eligible key
        contributes total mass num_q_heads. Queries without eligible keys
        contribute zero.

    Raises:
        ValueError: Shapes, layout, or sizes violate the input contract.
        TypeError: Query, cache, or block-table dtypes are unsupported.

    """
    if query.ndim != 3 or key_cache.ndim != 4 or block_table.ndim != 1:
        raise ValueError("Expected 3D queries, a 4D key cache, and a 1D block table")
    if not (query.device == key_cache.device == block_table.device):
        raise ValueError("Queries, key cache, and block table must share a device")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("NGC scoring supports float16 and bfloat16 queries")
    if key_cache.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("NGC scoring supports float16 and bfloat16 keys")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("Block table must contain int32 or int64 page IDs")
    if query.stride(-1) != 1 or key_cache.stride(-1) != 1:
        raise ValueError("Query and key head dimensions must be contiguous")
    if not block_table.is_contiguous():
        raise ValueError("Block table row must be contiguous")
    num_q, num_heads, head_dim = query.shape
    for name, positions, size in (
        ("query_positions", query_positions, num_q),
        ("key_positions", key_positions, kept_len),
    ):
        if positions.ndim != 1 or positions.numel() != size:
            raise ValueError(f"{name} must have shape [{size}]")
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must contain int32 or int64 positions")
        if positions.device != query.device or not positions.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on the query device")
    _, num_kv_heads, page_size, key_dim = key_cache.shape
    if (
        num_heads == 0
        or num_kv_heads == 0
        or num_heads % num_kv_heads
        or head_dim == 0
        or head_dim != key_dim
    ):
        raise ValueError(
            "Head dimensions must match and query heads must group KV heads"
        )
    if page_size == 0 or page_size & (page_size - 1):
        raise ValueError("Page size must be a positive power of two")
    if kept_len < 0 or kept_len > block_table.numel() * page_size:
        raise ValueError("Retained length exceeds the block table or is negative")
    block_size = page_size if scoring_block_size is None else scoring_block_size
    if block_size <= 0 or block_size & (block_size - 1) or block_size % page_size:
        raise ValueError(
            "Scoring block size must be a power-of-two multiple of page size"
        )
    if num_heads > 65535:
        raise ValueError("Number of query heads exceeds the Triton grid limit")

    num_blocks = triton.cdiv(kept_len, block_size)
    out = torch.zeros(num_blocks, device=query.device, dtype=torch.float32)
    if kept_len == 0 or num_q == 0:
        return out

    block_q = max(16, triton.next_power_of_2(num_q))
    block_slots = max(128, block_size)
    num_tiles = triton.cdiv(kept_len, block_slots)
    scratch_w = num_tiles * (block_slots // block_size)
    partials = torch.empty(
        (num_heads, block_q, scratch_w), device=query.device, dtype=torch.float32
    )
    tags = torch.empty(
        (num_heads, block_q, num_tiles), device=query.device, dtype=torch.float32
    )
    ngc_scorer_kernel[(1, num_heads)](
        query,
        key_cache,
        block_table,
        query_positions,
        key_positions,
        partials,
        tags,
        out,
        kept_len,
        num_blocks,
        num_q,
        scratch_w,
        num_tiles,
        query.stride(0),
        0,
        query.stride(1),
        0,
        key_cache.stride(0),
        key_cache.stride(2),
        key_cache.stride(1),
        softmax_scale,
        HEAD_DIM=head_dim,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        PAGE_SIZE=page_size,
        GQA_GROUP=num_heads // num_kv_heads,
        BLOCK_Q=block_q,
        BLOCK_SLOTS=block_slots,
        BLOCK_SIZE=block_size,
        BLOCK_N=32,
    )
    return out
