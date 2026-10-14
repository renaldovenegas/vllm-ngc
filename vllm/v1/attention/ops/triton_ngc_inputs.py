# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked query capture and batched input preparation for NGC scoring."""

from vllm.triton_utils import tl, triton


@triton.jit
def capture_ngc_queries(
    query,
    rows,
    ring,
    query_stride,
    scratch_row,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    row = tl.load(rows + token)
    if row != scratch_row:
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(query + token * query_stride + offsets, offsets < WIDTH, 0)
        tl.store(ring + row.to(tl.int64) * WIDTH + offsets, values, offsets < WIDTH)


@triton.jit(do_not_specialize=["key_capacity", "page_capacity", "tiles_per_request"])
def prepare_ngc_score_inputs(
    ring,
    packed,
    descriptors,
    block_table,
    queries,
    query_positions,
    key_positions,
    block_ids,
    table_stride,
    key_capacity,
    page_capacity,
    tiles_per_request,
    WINDOW: tl.constexpr,
    RING_WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KERNEL_PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    batch = program // tiles_per_request
    offsets = program % tiles_per_request * BLOCK + tl.arange(0, BLOCK)
    descriptor = descriptors + batch * 8
    seq_len = tl.load(descriptor)
    slot = tl.load(descriptor + 1)
    count = tl.load(descriptor + 2)
    kept = tl.load(descriptor + 3)
    evicted = tl.load(descriptor + 4)
    known_pages = tl.load(descriptor + 5)
    page_offset = tl.load(descriptor + 6)
    req_index = tl.load(descriptor + 7)

    query_index = offsets // WIDTH
    positions = seq_len - count + query_index
    source_rows = slot * RING_WINDOW + positions % RING_WINDOW
    values = tl.load(
        ring + source_rows * WIDTH + offsets % WIDTH,
        query_index < count,
        0,
    )
    tl.store(
        queries + batch * WINDOW * WIDTH + offsets,
        values,
        offsets < WINDOW * WIDTH,
    )
    tl.store(
        query_positions + batch * WINDOW + offsets,
        seq_len - count + offsets,
        offsets < count,
    )

    page = offsets // PAGE_SIZE
    known_position = tl.load(
        packed + page_offset + page,
        (offsets < kept) & (page < known_pages),
        0,
    )
    original = tl.where(
        page < known_pages, known_position + offsets % PAGE_SIZE, offsets + evicted
    )
    tl.store(key_positions + batch * key_capacity + offsets, original, offsets < kept)

    num_pages = tl.cdiv(kept, PAGE_SIZE)
    ratio: tl.constexpr = PAGE_SIZE // KERNEL_PAGE_SIZE
    physical_page = tl.load(
        block_table + req_index * table_stride + offsets * ratio,
        offsets < num_pages,
        0,
    )
    tl.store(
        block_ids + batch * page_capacity + offsets,
        physical_page // ratio,
        offsets < num_pages,
    )
