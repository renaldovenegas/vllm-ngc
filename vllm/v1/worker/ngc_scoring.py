# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-slot query rings and pending page scores for NGC eviction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import floor

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.ngc_eviction import NGCEvictionProposal
from vllm.v1.utils import CpuGpuBuffer


def share_ngc_retained_lengths(
    retained: CpuGpuBuffer, num_reqs: int
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Share batch-local length views only when every request's length matches.

    Keep the independent backing rows populated for ordinary CUDA graphs, whose
    captured pointers must remain valid even when layer lengths later diverge.
    """
    shared: dict[bytes, tuple[torch.Tensor, torch.Tensor, int]] = {}
    result = []
    for column, row in enumerate(retained.np[:, :num_reqs]):
        key = row.tobytes()
        views = shared.get(key)
        if views is None:
            views = (
                retained.gpu[column, :num_reqs],
                retained.cpu[column, :num_reqs],
                int(row.max()) if num_reqs else 0,
            )
            shared[key] = views
        result.append(views)
    return result


@dataclass
class NGCCadenceCheck:
    """Per-forward request state shared by fused preparation and score handoff."""

    req_index: int
    req_id: str
    seq_len: int
    due_boundaries: dict[int, int]


@dataclass
class NGCPendingScore:
    """Borrowed score storage, consumed before the next model forward."""

    request_id: str
    layer_name: str
    group_id: int
    cadence_boundary: int
    computed_tokens: int
    retained_tokens: int
    page_size: int
    num_victims: int
    scores: torch.Tensor
    block_ids: torch.Tensor


@dataclass
class NGCPageScores:
    scores: torch.Tensor
    block_ids: torch.Tensor


@dataclass
class NGCEvictionDecision:
    """Worker-local proposal; no block has been freed or remapped."""

    group_id: int
    cadence_boundary: int
    computed_tokens: int
    retained_tokens: int
    page_size: int
    page_scores: NGCPageScores
    victim_indices: torch.Tensor
    victim_block_ids: torch.Tensor

    def to_proposal(
        self, request_id: str, layer_name: str, table_version: int
    ) -> NGCEvictionProposal:
        """Copy results to CPU, waiting for scoring and its cache reads to finish."""
        return NGCEvictionProposal(
            request_id=request_id,
            layer_name=layer_name,
            group_id=self.group_id,
            table_version=table_version,
            cadence_boundary=self.cadence_boundary,
            computed_tokens=self.computed_tokens,
            retained_tokens=self.retained_tokens,
            page_size=self.page_size,
            block_ids=tuple(self.page_scores.block_ids.tolist()),
            scores=tuple(self.page_scores.scores.tolist()),
            victim_indices=tuple(self.victim_indices.tolist()),
            victim_block_ids=tuple(self.victim_block_ids.tolist()),
        )


def select_ngc_victims(
    scores: torch.Tensor,
    retained_tokens: int,
    page_size: int,
    eviction_rate: float,
    min_retained_tokens: int,
    *,
    check_finite: bool = True,
) -> torch.Tensor:
    """Rank full pages, protecting the newest page and a minimum token budget.

    Equal scores prefer earlier logical pages. Even a full newest page is
    protected. Partial pages are never selected, so each victim frees page_size
    tokens. Rounding down can produce an empty decision for small caches.
    """
    if page_size <= 0 or retained_tokens < 0 or min_retained_tokens <= 0:
        raise ValueError("Invalid NGC page size or retained-token budget")
    if not 0 <= eviction_rate < 1:
        raise ValueError("NGC eviction_rate must be in [0, 1)")
    num_pages = (retained_tokens + page_size - 1) // page_size
    if scores.ndim != 1 or scores.numel() != num_pages:
        raise ValueError("Expected one NGC score per allocator page")
    if check_finite and not bool(torch.isfinite(scores).all()):
        raise ValueError("NGC page scores must be finite")
    eligible_pages = max(num_pages - 1, 0)
    budget = max((retained_tokens - min_retained_tokens) // page_size, 0)
    num_victims = min(floor(eviction_rate * eligible_pages), budget)
    if num_victims == 0:
        return torch.empty(0, dtype=torch.int64, device=scores.device)
    return torch.argsort(scores[:eligible_pages], stable=True)[:num_victims]


@dataclass
class NGCQueryHistory:
    """Scores and the pending decision for one request and layer."""

    page_scores: NGCPageScores | None = None
    decision: NGCEvictionDecision | None = None


@dataclass
class NGCScoreInput:
    req_id: str
    layer_name: str
    req_index: int
    retained_tokens: int
    evicted_tokens: int
    page_size: int
    kernel_page_size: int
    page_positions: tuple[int, ...]
    block_table: torch.Tensor


@dataclass
class NGCPreparedScoreInput:
    queries: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    block_ids: torch.Tensor


@dataclass
class NGCScoreWorkspace:
    queries: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    block_ids: torch.Tensor


class NGCScratchBuffers:
    """Grow-only FP32 storage; shaped views keep kernel indexing contiguous.

    Scratch can be shared by successive layers on the same CUDA stream. Outputs
    that survive a layer must use separate names or a separate buffer owner.
    """

    def __init__(self) -> None:
        self.buffers: dict[str, torch.Tensor] = {}

    def get(
        self, name: str, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        size = 1
        for dimension in shape:
            size *= dimension
        buffer = self.buffers.get(name)
        if buffer is None or buffer.numel() < size or buffer.device != device:
            capacity = 1 << (max(size, 1) - 1).bit_length()
            buffer = torch.empty(capacity, dtype=torch.float32, device=device)
            self.buffers[name] = buffer
        return buffer[:size].view(shape)


@dataclass
class NGCFusedDecodeState:
    inputs: list[NGCScoreInput]
    descriptors: torch.Tensor
    workspace: NGCScoreWorkspace
    due: torch.Tensor
    scores: torch.Tensor | None = None
    score_buffer: torch.Tensor | None = None


class NGCScoringBuffers:
    """Per-layer query rings keyed by request slot, independent of batch order.

    Each layer keeps the last ``window`` post-RoPE queries of every request slot
    at row ``slot * window + position % window``. ``prepare_capture`` computes a
    destination row per kept token on the CPU; ``capture`` uses a masked copy
    between persistent buffers, so CUDA graphs can record it. The scratch-row
    sentinel suppresses loads and stores for queries outside the capture window.

    Query positions are original sequence positions. Scheduling a request again
    invalidates its pending scores. Callers must also discard pending scores if
    they change a block table before applying an eviction decision.
    """

    def __init__(self, *, include_debug_scores: bool = False):
        self.include_debug_scores = include_debug_scores
        self.histories: dict[tuple[str, str], NGCQueryHistory] = {}
        self._scored_boundaries: dict[tuple[str, str], int] = {}
        self._checked_boundaries: dict[tuple[str, int], int] = {}
        self._handoff: CpuGpuBuffer | None = None
        self._pending_scores: dict[tuple[str, str], NGCPendingScore] = {}
        self._score_input_layout: tuple | None = None
        self._rings: dict[str, torch.Tensor] = {}
        self._windows: dict[str, int] = {}
        self._window = 0
        self._scratch_row = 0
        self._capture_rows: CpuGpuBuffer | None = None
        self._capture_groups: dict[tuple[int, int | None], CpuGpuBuffer] = {}
        self._capture_layer_groups: dict[str, tuple[int, int | None]] = {}
        self._capture_active: dict[tuple[int, int | None], bool] = {}
        self._capture_high = 0
        self._slots: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._seq_lens: dict[str, int] = {}
        self._score_inputs: CpuGpuBuffer | None = None
        self._score_workspaces: dict[str, NGCScoreWorkspace] = {}
        self._prepared_groups: dict[str, tuple[torch.Tensor, NGCScoreWorkspace]] = {}
        self.fused_inputs: dict[tuple[str, str], NGCPreparedScoreInput] = {}
        self._fused_decode: dict[str, NGCFusedDecodeState] = {}
        self._fused_scratch = NGCScratchBuffers()
        self._fused_scores = NGCScratchBuffers()
        self._fused_due: dict[str, torch.Tensor] = {}
        self._fused_due_batch: CpuGpuBuffer | None = None

    def allocate(
        self,
        layers: dict[str, tuple[int, int, int]],
        *,
        num_slots: int,
        max_num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = False,
        cadences: dict[str, int] | None = None,
    ) -> None:
        """Reserve rings for ``{layer: (window, num_heads, head_size)}``."""
        if not layers or num_slots <= 0 or max_num_tokens <= 0:
            raise ValueError("NGC query rings need layers, slots, and tokens")
        if any(window <= 0 for window, _, _ in layers.values()):
            raise ValueError("NGC query windows must be positive")
        if cadences is not None and (
            cadences.keys() != layers.keys() or any(c <= 0 for c in cadences.values())
        ):
            raise ValueError("NGC capture needs one positive cadence per layer")
        self._window = max(window for window, _, _ in layers.values())
        self._scratch_row = num_slots * self._window
        self._windows = {name: window for name, (window, _, _) in layers.items()}
        self._rings = {
            name: torch.zeros(
                (self._scratch_row + 1, heads, head_size), dtype=dtype, device=device
            )
            for name, (_, heads, head_size) in layers.items()
        }
        self._capture_groups = {}
        self._capture_layer_groups = {}
        self._capture_active = {}
        for name, (window, _, _) in layers.items():
            group = (window, cadences[name] if cadences is not None else None)
            self._capture_layer_groups[name] = group
            if group not in self._capture_groups:
                rows = CpuGpuBuffer(
                    max_num_tokens,
                    dtype=torch.int64,
                    device=device,
                    pin_memory=pin_memory,
                )
                rows.np.fill(self._scratch_row)
                rows.gpu.fill_(self._scratch_row)
                self._capture_groups[group] = rows
                self._capture_active[group] = False
        self._capture_rows = next(iter(self._capture_groups.values()))
        self._capture_high = 0
        self._free_slots = list(range(num_slots - 1, -1, -1))
        self._slots = {}
        self._seq_lens = {}
        self._score_inputs = None
        self._score_input_layout = None
        self._pending_scores.clear()
        self._fused_due_batch = None
        self._score_workspaces = {}
        self._prepared_groups.clear()
        self.reset_fused_decode()
        self._fused_scratch = NGCScratchBuffers()
        self._fused_scores = NGCScratchBuffers()
        self._fused_due.clear()

    def _slot(self, req_id: str) -> int:
        slot = self._slots.get(req_id)
        if slot is None:
            if not self._free_slots:
                raise RuntimeError("More NGC requests than reserved query slots")
            slot = self._free_slots.pop()
            self._slots[req_id] = slot
        return slot

    def prepare_capture(
        self,
        req_ids: Sequence[str],
        req_indices: np.ndarray,
        query_pos: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        positions: np.ndarray,
        seq_lens: Sequence[int],
    ) -> None:
        """Keep the queries needed at the current or next scoring boundary."""
        assert self._capture_rows is not None
        num_tokens = len(positions)
        high = max(num_tokens, self._capture_high)
        destinations = None
        for group, buffer in self._capture_groups.items():
            window, cadence = group
            keep = query_pos >= num_scheduled_tokens[req_indices] - window
            if cadence is not None:
                computed = np.asarray(seq_lens, dtype=np.int64)
                checked = np.fromiter(
                    (
                        self._checked_boundaries.get((req_id, cadence), 0)
                        for req_id in req_ids
                    ),
                    np.int64,
                    len(req_ids),
                )
                due = computed // cadence * cadence > checked
                token_count = positions + 1
                next_boundary = np.maximum(
                    cadence, (token_count + cadence - 1) // cadence * cadence
                )
                keep &= due[req_indices] | (next_boundary - token_count < window)
            active = bool(keep.any())
            if active and destinations is None:
                slots = np.fromiter(
                    (self._slot(req_id) for req_id in req_ids), np.int64, len(req_ids)
                )
                destinations = (
                    slots[req_indices] * self._window + positions % self._window
                )
            if active or self._capture_active[group]:
                rows = buffer.np
                rows[:num_tokens] = self._scratch_row
                if active:
                    assert destinations is not None
                    rows[:num_tokens][keep] = destinations[keep]
                rows[num_tokens:high] = self._scratch_row
                buffer.copy_to_gpu(high)
            self._capture_active[group] = active
        self._capture_high = num_tokens
        scheduled = set(req_ids)
        for req_id, seq_len in zip(req_ids, seq_lens):
            self._seq_lens[req_id] = int(seq_len)
        for key in [key for key in self.histories if key[0] in scheduled]:
            del self.histories[key]
        for key in [key for key in self._pending_scores if key[0] in scheduled]:
            del self._pending_scores[key]

    def reset_capture(self) -> None:
        """Send every row to scratch, for dummy runs and graph capture."""
        self.reset_fused_decode()
        if self._capture_rows is None:
            return
        for group, buffer in self._capture_groups.items():
            buffer.np[: self._capture_high] = self._scratch_row
            buffer.copy_to_gpu(self._capture_high)
            self._capture_active[group] = False
        self._capture_high = 0

    def capture(self, layer_name: str, query: torch.Tensor) -> None:
        """Record one layer's queries; safe inside a CUDA graph."""
        group = self._capture_layer_groups[layer_name]
        if (
            not self._capture_active[group]
            and is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL
        ):
            return
        ring = self._rings[layer_name]
        rows = self._capture_groups[group]
        num_tokens = query.shape[0]
        if query.device.type == "cpu":
            indices = rows.gpu[:num_tokens]
            keep = indices != self._scratch_row
            ring.index_copy_(
                0, indices[keep], query.reshape(num_tokens, *ring.shape[1:])[keep]
            )
        elif num_tokens:
            from vllm.v1.attention.ops.triton_ngc_inputs import capture_ngc_queries

            width = ring.shape[1] * ring.shape[2]
            query = query.reshape(num_tokens, width)
            capture_ngc_queries[(num_tokens, (width + 255) // 256)](
                query,
                rows.gpu,
                ring,
                query.stride(0),
                self._scratch_row,
                WIDTH=width,
                BLOCK=256,
            )

    def prepare_score_inputs(
        self,
        inputs: Sequence[NGCScoreInput],
        *,
        gather_queries: bool = True,
        materialize_keys: set[tuple[str, str]] | None = None,
    ) -> dict[tuple[str, str], NGCPreparedScoreInput]:
        """Prepare page metadata and optionally gather historical queries.

        With ``gather_queries=False``, returned query views are not populated;
        fused decode reads the rings after forward captures the current query.
        ``materialize_keys`` limits returned per-request views, while batched
        metadata is still prepared for every input required by decode.
        """
        self._prepared_groups = {}
        if not inputs:
            return {}
        from vllm.v1.attention.ops.triton_ngc_inputs import prepare_ngc_score_inputs

        grouped: dict[str, list[NGCScoreInput]] = {}
        for item in inputs:
            grouped.setdefault(item.layer_name, []).append(item)
        ordered = [item for group in grouped.values() for item in group]
        descriptor_size = 8 * len(ordered)
        size = descriptor_size + sum(len(item.page_positions) for item in ordered)
        device = self._rings[ordered[0].layer_name].device
        if self._score_inputs is None or self._score_inputs.gpu.numel() < size:
            self._score_inputs = CpuGpuBuffer(
                1 << (max(size, 1) - 1).bit_length(),
                dtype=torch.int64,
                device=device,
                pin_memory=device.type == "cuda",
            )
            self._score_input_layout = None
        layout = tuple(
            (item.req_id, item.layer_name, item.req_index, item.page_positions)
            for item in ordered
        )
        positions_changed = layout != self._score_input_layout
        packed = self._score_inputs.np
        descriptors = packed[:descriptor_size].reshape(-1, 8)
        offset = descriptor_size
        for i, item in enumerate(ordered):
            seq_len = self._seq_lens[item.req_id]
            count = min(self._windows[item.layer_name], seq_len)
            known = len(item.page_positions)
            descriptors[i] = (
                seq_len,
                self._slots[item.req_id],
                count,
                item.retained_tokens,
                item.evicted_tokens,
                known,
                offset,
                item.req_index,
            )
            if positions_changed:
                packed[offset : offset + known] = item.page_positions
            offset += known
        self._score_inputs.copy_to_gpu(size if positions_changed else descriptor_size)
        self._score_input_layout = layout

        prepared = {}
        start = 0
        for layer_name, items in grouped.items():
            ring = self._rings[layer_name]
            window = self._windows[layer_name]
            batch = len(items)
            page_size = items[0].page_size
            max_kept = max(item.retained_tokens for item in items)
            workspace = self._score_workspaces.get(layer_name)
            if (
                workspace is None
                or workspace.queries.shape[0] < batch
                or workspace.key_positions.shape[1] < max_kept
            ):
                batch_capacity = 1 << (batch - 1).bit_length()
                key_capacity = 1 << (max(max_kept, 1) - 1).bit_length()
                if workspace is not None:
                    batch_capacity = max(batch_capacity, workspace.queries.shape[0])
                    key_capacity = max(key_capacity, workspace.key_positions.shape[1])
                workspace = NGCScoreWorkspace(
                    queries=torch.empty(
                        (batch_capacity, window, *ring.shape[1:]),
                        device=device,
                        dtype=ring.dtype,
                    ),
                    query_positions=torch.empty(
                        (batch_capacity, window),
                        device=device,
                        dtype=torch.int64,
                    ),
                    key_positions=torch.empty(
                        (batch_capacity, key_capacity),
                        device=device,
                        dtype=torch.int64,
                    ),
                    block_ids=torch.empty(
                        (batch_capacity, (key_capacity + page_size - 1) // page_size),
                        device=device,
                        dtype=torch.int64,
                    ),
                )
                self._score_workspaces[layer_name] = workspace
            width = ring.shape[1] * ring.shape[2]
            self._prepared_groups[layer_name] = (
                self._score_inputs.gpu[start * 8 : (start + batch) * 8],
                workspace,
            )
            work = max(max_kept, window * width) if gather_queries else max_kept
            tiles = (max(work, 1) + 255) // 256
            prepare_ngc_score_inputs[(batch * tiles,)](
                ring,
                self._score_inputs.gpu,
                self._score_inputs.gpu[start * 8 :],
                items[0].block_table,
                workspace.queries,
                workspace.query_positions,
                workspace.key_positions,
                workspace.block_ids,
                items[0].block_table.stride(0),
                workspace.key_positions.shape[1],
                workspace.block_ids.shape[1],
                tiles,
                WINDOW=window,
                RING_WINDOW=self._window,
                WIDTH=width,
                PAGE_SIZE=page_size,
                KERNEL_PAGE_SIZE=items[0].kernel_page_size,
                BLOCK=256,
                GATHER_QUERIES=gather_queries,
            )
            for i, item in enumerate(items):
                key = (item.req_id, layer_name)
                if materialize_keys is not None and key not in materialize_keys:
                    continue
                count = min(window, self._seq_lens[item.req_id])
                pages = (item.retained_tokens + page_size - 1) // page_size
                prepared[key] = NGCPreparedScoreInput(
                    workspace.queries[i, :count],
                    workspace.query_positions[i, :count],
                    workspace.key_positions[i, : item.retained_tokens],
                    workspace.block_ids[i, :pages],
                )
            start += batch
        return prepared

    def reset_fused_decode(self) -> None:
        self._fused_decode.clear()
        self.fused_inputs.clear()

    def prepare_fused_decode(
        self, inputs: Sequence[NGCScoreInput], due: dict[tuple[str, str], bool]
    ) -> None:
        """Stage boundary metadata before forward; queries are read from rings later."""
        self.reset_fused_decode()
        self.fused_inputs = self.prepare_score_inputs(
            inputs,
            gather_queries=False,
            materialize_keys={key for key, is_due in due.items() if is_due},
        )
        grouped: dict[str, list[NGCScoreInput]] = {}
        for item in inputs:
            grouped.setdefault(item.layer_name, []).append(item)
        if not grouped:
            return
        max_batch = max(len(items) for items in grouped.values())
        shape = (len(grouped), 1 << (max_batch - 1).bit_length())
        device = self._rings[next(iter(grouped))].device
        if self._fused_due_batch is None or self._fused_due_batch.gpu.shape != shape:
            self._fused_due_batch = CpuGpuBuffer(
                *shape,
                dtype=torch.bool,
                device=device,
                pin_memory=device.type == "cuda",
            )
        due_batch = self._fused_due_batch
        due_batch.np.fill(False)
        for row, items in enumerate(grouped.values()):
            due_batch.np[row, : len(items)] = [
                due[item.req_id, item.layer_name] for item in items
            ]
        due_batch.copy_to_gpu()
        for row, (layer_name, items) in enumerate(grouped.items()):
            descriptors, workspace = self._prepared_groups[layer_name]
            batch = len(items)
            self._fused_due[layer_name] = due_batch.gpu[row, :batch]
            capacity = workspace.key_positions.shape[1]
            pages = (capacity + items[0].page_size - 1) // items[0].page_size
            scores = self._fused_scores.get(
                layer_name, (batch, pages), descriptors.device
            )
            self._fused_decode[layer_name] = NGCFusedDecodeState(
                items,
                descriptors,
                workspace,
                self._fused_due[layer_name],
                score_buffer=scores,
            )

    def fused_graph_inputs(self) -> dict:
        """Dependencies whose addresses and launch shapes must survive replay."""
        return {
            "layers": {
                name: (
                    state.descriptors,
                    state.workspace.key_positions,
                    state.inputs[0].block_table,
                    state.due,
                    state.score_buffer,
                    self._rings[name],
                    self._capture_groups[self._capture_layer_groups[name]].gpu,
                    self._capture_active[self._capture_layer_groups[name]],
                    state.inputs[0].page_size,
                )
                for name, state in self._fused_decode.items()
            },
            "scratch": self._fused_scratch.buffers.copy(),
        }

    def fused_graph_ready(self) -> bool:
        return bool(self._fused_decode) and all(
            state.scores is not None for state in self._fused_decode.values()
        )

    def publish_fused_graph_scores(self) -> None:
        for state in self._fused_decode.values():
            assert state.score_buffer is not None
            state.scores = state.score_buffer

    def fused_decode(
        self,
        layer_name: str,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        softmax_scale: float,
        num_segments: int = 16,
        tile_size: int = 16,
    ) -> bool:
        state = self._fused_decode.get(layer_name)
        if state is None:
            return False
        from vllm.v1.attention.ops.triton_ngc_decode import ngc_decode_and_score

        count = len(state.inputs)
        max_kept = state.workspace.key_positions.shape[1]
        page_size = state.inputs[0].page_size
        scores = self._fused_scores.get(
            layer_name, (count, (max_kept + page_size - 1) // page_size), query.device
        )
        state.scores = ngc_decode_and_score(
            query[:count],
            kv_cache,
            self._rings[layer_name],
            state.descriptors,
            state.workspace.key_positions,
            state.inputs[0].block_table,
            state.due,
            output[:count],
            window=self._windows[layer_name],
            ring_window=self._window,
            page_size=state.inputs[0].page_size,
            max_kept=max_kept,
            softmax_scale=softmax_scale,
            num_segments=num_segments,
            tile_size=tile_size,
            scratch=self._fused_scratch,
            score_buffer=scores,
        )
        return True

    def get_fused_scores(self, req_id: str, layer_name: str) -> torch.Tensor | None:
        state = self._fused_decode.get(layer_name)
        if state is None or state.scores is None:
            return None
        for index, item in enumerate(state.inputs):
            if item.req_id == req_id:
                pages = (item.retained_tokens + item.page_size - 1) // item.page_size
                return state.scores[index, :pages]
        return None

    def _copy_handoff(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Pack on-device, transfer once, and wait before exposing CPU views."""
        device = tensors[0].device
        sizes = [tensor.numel() * tensor.element_size() for tensor in tensors]
        padded = [(size + 7) // 8 * 8 for size in sizes]
        total = sum(padded)
        if (
            self._handoff is None
            or self._handoff.gpu.device != device
            or self._handoff.gpu.numel() < total
        ):
            self._handoff = CpuGpuBuffer(
                max(total, 1),
                dtype=torch.uint8,
                device=device,
                with_numpy=False,
                pin_memory=device.type == "cuda",
            )
        views = []
        offset = 0
        for tensor, size, step in zip(tensors, sizes, padded):
            self._handoff.gpu[offset : offset + size].copy_(
                tensor.contiguous().view(torch.uint8).reshape(-1)
            )
            views.append(self._handoff.cpu[offset : offset + size].view(tensor.dtype))
            offset += step
        self._handoff.copy_to_cpu(total)
        if device.type != "cpu":
            torch.accelerator.synchronize(device.index)
        return views

    def _take_batched_proposals(
        self, table_versions: dict[str, int], layer_names: list[str]
    ) -> list[NGCEvictionProposal]:
        pending = [
            self._pending_scores[req_id, name]
            for req_id in table_versions
            for name in layer_names
            if (req_id, name) in self._pending_scores
        ]
        if not pending:
            return []
        groups: dict[tuple, list[tuple[int, NGCPendingScore]]] = {}
        for index, item in enumerate(pending):
            key = (
                item.scores.device,
                item.scores.dtype,
                item.scores.numel(),
                item.num_victims,
            )
            groups.setdefault(key, []).append((index, item))
        tensors: list[torch.Tensor] = []
        packets = []
        for items in groups.values():
            scores = torch.stack([item.scores for _, item in items])
            count = items[0][1].num_victims
            offset = len(tensors)
            tensors.append(torch.isfinite(scores).all().reshape(1))
            if count or self.include_debug_scores:
                ids = torch.stack([item.block_ids for _, item in items])
                indices = torch.argsort(scores[:, :-1], dim=1, stable=True)[:, :count]
                tensors.extend((indices, ids.gather(1, indices)))
                if self.include_debug_scores:
                    tensors.extend((ids, scores))
            packets.append((items, offset))
        copied = self._copy_handoff(tensors)
        result: list[NGCEvictionProposal | None] = [None] * len(pending)
        for items, offset in packets:
            if not bool(copied[offset].all()):
                raise ValueError("NGC page scores must be finite")
            if not items[0][1].num_victims and not self.include_debug_scores:
                continue
            shape = (len(items), items[0][1].num_victims)
            indices = copied[offset + 1].view(shape).tolist()
            ids = copied[offset + 2].view(shape).tolist()
            page_shape = (len(items), items[0][1].scores.numel())
            debug_ids = (
                copied[offset + 3].view(page_shape).tolist()
                if self.include_debug_scores
                else None
            )
            debug_scores = (
                copied[offset + 4].view(page_shape).tolist()
                if self.include_debug_scores
                else None
            )
            for row, (index, item) in enumerate(items):
                result[index] = NGCEvictionProposal(
                    request_id=item.request_id,
                    layer_name=item.layer_name,
                    group_id=item.group_id,
                    table_version=table_versions[item.request_id],
                    cadence_boundary=item.cadence_boundary,
                    computed_tokens=item.computed_tokens,
                    retained_tokens=item.retained_tokens,
                    page_size=item.page_size,
                    block_ids=tuple(debug_ids[row]) if debug_ids is not None else None,
                    scores=tuple(debug_scores[row])
                    if debug_scores is not None
                    else None,
                    victim_indices=tuple(indices[row]),
                    victim_block_ids=tuple(ids[row]),
                )
        for item in pending:
            del self._pending_scores[item.request_id, item.layer_name]
        return [proposal for proposal in result if proposal is not None]

    def take_proposals(
        self, table_versions: dict[str, int], layer_names: list[str]
    ) -> list[NGCEvictionProposal]:
        batched = self._take_batched_proposals(table_versions, layer_names)
        pending = []
        for req_id, version in table_versions.items():
            for layer_name in layer_names:
                history = self.histories.get((req_id, layer_name))
                if history is None or history.decision is None:
                    continue
                pending.append((req_id, layer_name, version, history.decision))
        if not pending:
            return batched
        tensors: list[torch.Tensor] = [
            torch.stack(
                [
                    torch.isfinite(decision.page_scores.scores).all()
                    for _, _, _, decision in pending
                ]
            )
        ]
        for _, _, _, decision in pending:
            tensors.extend((decision.victim_indices, decision.victim_block_ids))
            if self.include_debug_scores:
                tensors.extend(
                    (decision.page_scores.block_ids, decision.page_scores.scores)
                )
        copied = self._copy_handoff(tensors)
        if not bool(copied[0].all()):
            raise ValueError("NGC page scores must be finite")
        proposals = []
        stride = 4 if self.include_debug_scores else 2
        for i, (req_id, layer_name, version, decision) in enumerate(pending):
            offset = 1 + stride * i
            victims, victim_ids = copied[offset : offset + 2]
            block_ids = (
                tuple(copied[offset + 2].tolist())
                if self.include_debug_scores
                else None
            )
            scores = (
                tuple(copied[offset + 3].tolist())
                if self.include_debug_scores
                else None
            )
            proposals.append(
                NGCEvictionProposal(
                    request_id=req_id,
                    layer_name=layer_name,
                    table_version=version,
                    group_id=decision.group_id,
                    cadence_boundary=decision.cadence_boundary,
                    computed_tokens=decision.computed_tokens,
                    retained_tokens=decision.retained_tokens,
                    page_size=decision.page_size,
                    block_ids=block_ids,
                    scores=scores,
                    victim_indices=tuple(victims.tolist()),
                    victim_block_ids=tuple(victim_ids.tolist()),
                )
            )
        for req_id, layer_name, _, _ in pending:
            self.histories[req_id, layer_name].decision = None
        return batched + proposals

    def get_queries(self, req_id: str, layer_name: str):
        """Return the layer's most recent queries and positions, oldest first."""
        ring = self._rings[layer_name]
        seq_len = self._seq_lens[req_id]
        count = min(self._windows[layer_name], seq_len)
        positions = torch.arange(seq_len - count, seq_len, dtype=torch.int64)
        rows = self._slots[req_id] * self._window + positions % self._window
        positions = positions.to(ring.device)
        return ring.index_select(0, rows.to(ring.device)), positions

    def plan_cadences(
        self,
        req_ids: Sequence[str],
        seq_lens: np.ndarray,
        cadences: set[int],
    ) -> list[NGCCadenceCheck]:
        """Check each request/cadence once without consuming capture boundaries."""
        if not cadences:
            return []
        checks = []
        for index, req_id in enumerate(req_ids):
            seq_len = int(seq_lens[index])
            due = {}
            for cadence in cadences:
                boundary = seq_len // cadence * cadence
                if boundary > self._checked_boundaries.get((req_id, cadence), 0):
                    due[cadence] = boundary
            checks.append(NGCCadenceCheck(index, req_id, seq_len, due))
        return checks

    def mark_cadences_checked(self, check: NGCCadenceCheck) -> None:
        """Consume the planned boundaries after forward has captured queries."""
        for cadence, boundary in check.due_boundaries.items():
            self._checked_boundaries[check.req_id, cadence] = boundary

    def any_due(
        self,
        req_id: str,
        computed_tokens: int,
        cadence: int,
        *,
        mark_checked: bool = True,
    ):
        """Cheap per-request filter before checking each layer's boundary."""
        boundary = computed_tokens // cadence * cadence
        if boundary <= self._checked_boundaries.get((req_id, cadence), 0):
            return False
        if mark_checked:
            self._checked_boundaries[req_id, cadence] = boundary
        return True

    def store_page_scores(
        self,
        req_id: str,
        layer_name: str,
        scores: torch.Tensor,
        block_ids: torch.Tensor,
    ) -> None:
        """Snapshot one score per page and its physical ID for eviction handoff."""
        if scores.ndim != 1 or block_ids.shape != scores.shape:
            raise ValueError("Page scores and block IDs must be matching vectors")
        history = self.histories.setdefault((req_id, layer_name), NGCQueryHistory())
        history.page_scores = NGCPageScores(
            scores.detach().clone(), block_ids.detach().clone()
        )
        history.decision = None

    def due_boundary(
        self, req_id: str, layer_name: str, computed_tokens: int, cadence: int
    ) -> int | None:
        """Coalesce crossed cadence boundaries; replay never repeats an event.

        Prompt and generated tokens with computed KV both count. Eviction does
        not reset the clock. Advancing several intervals in one forward scores the
        latest state once, rather than pretending to score earlier snapshots.
        """
        if cadence <= 0 or computed_tokens < 0:
            raise ValueError("Invalid NGC cadence or computed-token count")
        boundary = computed_tokens // cadence * cadence
        if boundary <= self._scored_boundaries.get((req_id, layer_name), 0):
            return None
        return boundary

    def score_if_due(
        self,
        req_id: str,
        layer_name: str,
        *,
        group_id: int,
        computed_tokens: int,
        cadence: int,
        eviction_rate: float,
        min_retained_tokens: int,
        key_cache: torch.Tensor,
        kernel_block_table: torch.Tensor,
        block_ids: torch.Tensor,
        key_positions: torch.Tensor,
        page_size: int,
        softmax_scale: float,
        score_fn: Callable | None = None,
        reduce_scores: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prepared: NGCPreparedScoreInput | None = None,
        precomputed_scores: torch.Tensor | None = None,
        defer_selection: bool = False,
    ) -> NGCEvictionDecision | None:
        """Score retained keys and save a proposal at the next due boundary.

        block_ids names allocator pages; kernel_block_table can subdivide them.
        key_positions follows retained logical order, preserving original token
        positions. The optional reducer must sum scores across query-head shards
        before ranking. No cache data, block tables, or ownership is changed.
        """
        boundary = self.due_boundary(req_id, layer_name, computed_tokens, cadence)
        if boundary is None:
            return None
        if self._seq_lens.get(req_id, 0) == 0:
            return None
        retained_tokens = key_positions.numel()
        scores = precomputed_scores
        if scores is None:
            if score_fn is None:
                from vllm.v1.attention.ops.triton_ngc_scorer import ngc_page_scores

                score_fn = ngc_page_scores
            if prepared is None:
                queries, query_positions = self.get_queries(req_id, layer_name)
            else:
                queries, query_positions = prepared.queries, prepared.query_positions
            scores = score_fn(
                queries,
                key_cache,
                kernel_block_table,
                retained_tokens,
                softmax_scale,
                scoring_block_size=page_size,
                query_positions=query_positions,
                key_positions=key_positions,
            )
        if reduce_scores is not None:
            scores = reduce_scores(scores)
        if block_ids.device != scores.device:
            raise ValueError("NGC block IDs and scores must share a device")
        if defer_selection:
            if page_size <= 0 or retained_tokens < 0 or min_retained_tokens <= 0:
                raise ValueError("Invalid NGC page size or retained-token budget")
            if not 0 <= eviction_rate < 1:
                raise ValueError("NGC eviction_rate must be in [0, 1)")
            pages = scores.numel()
            if (
                scores.ndim != 1
                or block_ids.shape != scores.shape
                or pages != (retained_tokens + page_size - 1) // page_size
            ):
                raise ValueError("Expected matching page score and block ID vectors")
            count = min(
                floor(eviction_rate * max(pages - 1, 0)),
                max((retained_tokens - min_retained_tokens) // page_size, 0),
            )
            self.histories.pop((req_id, layer_name), None)
            self._pending_scores[req_id, layer_name] = NGCPendingScore(
                req_id,
                layer_name,
                group_id,
                boundary,
                computed_tokens,
                retained_tokens,
                page_size,
                count,
                scores,
                block_ids,
            )
            self._scored_boundaries[req_id, layer_name] = boundary
            return None
        victims = select_ngc_victims(
            scores,
            retained_tokens,
            page_size,
            eviction_rate,
            min_retained_tokens,
            check_finite=False,
        )
        self.store_page_scores(req_id, layer_name, scores, block_ids)
        history = self.histories[req_id, layer_name]
        assert history.page_scores is not None
        decision = NGCEvictionDecision(
            group_id=group_id,
            cadence_boundary=boundary,
            computed_tokens=computed_tokens,
            retained_tokens=retained_tokens,
            page_size=page_size,
            page_scores=history.page_scores,
            victim_indices=victims,
            victim_block_ids=history.page_scores.block_ids[victims],
        )
        history.decision = decision
        self._scored_boundaries[req_id, layer_name] = boundary
        return decision

    def discard_request(self, req_id: str, *, keep_cadence: bool = False) -> None:
        for key in [key for key in self._pending_scores if key[0] == req_id]:
            del self._pending_scores[key]
        for key in list(self.histories):
            if key[0] == req_id:
                del self.histories[key]
        slot = self._slots.pop(req_id, None)
        if slot is not None:
            self._free_slots.append(slot)
        self._seq_lens.pop(req_id, None)
        if not keep_cadence:
            for checked in list(self._checked_boundaries):
                if checked[0] == req_id:
                    del self._checked_boundaries[checked]
            for key in list(self._scored_boundaries):
                if key[0] == req_id:
                    del self._scored_boundaries[key]
