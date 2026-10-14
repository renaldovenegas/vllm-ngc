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


@dataclass
class NGCPageScores:
    scores: torch.Tensor
    block_ids: torch.Tensor


@dataclass
class NGCEvictionDecision:
    """Worker-local proposal; no block has been freed or remapped."""

    group_id: int
    cadence_boundary: int
    computed_generated_tokens: int
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
            computed_generated_tokens=self.computed_generated_tokens,
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
        self._score_workspaces = {}

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
        prompt_lens: Sequence[int] | None = None,
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
                if prompt_lens is None:
                    raise ValueError("Cadenced NGC capture requires prompt lengths")
                prompts = np.asarray(prompt_lens, dtype=np.int64)
                generated = np.asarray(seq_lens, dtype=np.int64) - prompts
                checked = np.fromiter(
                    (
                        self._checked_boundaries.get((req_id, cadence), 0)
                        for req_id in req_ids
                    ),
                    np.int64,
                    len(req_ids),
                )
                due = generated // cadence * cadence > checked
                token_generated = positions + 1 - prompts[req_indices]
                next_boundary = np.maximum(
                    cadence, (token_generated + cadence - 1) // cadence * cadence
                )
                keep &= due[req_indices] | (next_boundary - token_generated < window)
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

    def reset_capture(self) -> None:
        """Send every row to scratch, for dummy runs and graph capture."""
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
        self, inputs: Sequence[NGCScoreInput]
    ) -> dict[tuple[str, str], NGCPreparedScoreInput]:
        """Transfer descriptors once and gather all due requests per layer."""
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
            packed[offset : offset + known] = item.page_positions
            offset += known
        self._score_inputs.copy_to_gpu(size)

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
            tiles = (max(max_kept, window * width) + 255) // 256
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
            )
            for i, item in enumerate(items):
                count = min(window, self._seq_lens[item.req_id])
                pages = (item.retained_tokens + page_size - 1) // page_size
                prepared[item.req_id, layer_name] = NGCPreparedScoreInput(
                    workspace.queries[i, :count],
                    workspace.query_positions[i, :count],
                    workspace.key_positions[i, : item.retained_tokens],
                    workspace.block_ids[i, :pages],
                )
            start += batch
        return prepared

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

    def take_proposals(
        self, table_versions: dict[str, int], layer_names: list[str]
    ) -> list[NGCEvictionProposal]:
        pending = []
        for req_id, version in table_versions.items():
            for layer_name in layer_names:
                history = self.histories.get((req_id, layer_name))
                if history is None or history.decision is None:
                    continue
                pending.append((req_id, layer_name, version, history.decision))
        if not pending:
            return []
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
                    computed_generated_tokens=decision.computed_generated_tokens,
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
        return proposals

    def get_queries(self, req_id: str, layer_name: str):
        """Return the layer's most recent queries and positions, oldest first."""
        ring = self._rings[layer_name]
        seq_len = self._seq_lens[req_id]
        count = min(self._windows[layer_name], seq_len)
        positions = torch.arange(seq_len - count, seq_len, dtype=torch.int64)
        rows = self._slots[req_id] * self._window + positions % self._window
        positions = positions.to(ring.device)
        return ring.index_select(0, rows.to(ring.device)), positions

    def any_due(self, req_id: str, computed_generated_tokens: int, cadence: int):
        """Cheap per-request filter before checking each layer's boundary."""
        boundary = computed_generated_tokens // cadence * cadence
        if boundary <= self._checked_boundaries.get((req_id, cadence), 0):
            return False
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
        self, req_id: str, layer_name: str, computed_generated_tokens: int, cadence: int
    ) -> int | None:
        """Coalesce crossed cadence boundaries; replay never repeats an event.

        Only committed generated tokens with computed KV count. Prompt tokens
        do not count. Advancing several intervals in one forward scores the
        latest state once, rather than pretending to score earlier snapshots.
        """
        if cadence <= 0 or computed_generated_tokens < 0:
            raise ValueError("Invalid NGC cadence or computed generated-token count")
        boundary = computed_generated_tokens // cadence * cadence
        if boundary <= self._scored_boundaries.get((req_id, layer_name), 0):
            return None
        return boundary

    def score_if_due(
        self,
        req_id: str,
        layer_name: str,
        *,
        group_id: int,
        computed_generated_tokens: int,
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
    ) -> NGCEvictionDecision | None:
        """Score retained keys and save a proposal at the next due boundary.

        block_ids names allocator pages; kernel_block_table can subdivide them.
        key_positions follows retained logical order, preserving original token
        positions. The optional reducer must sum scores across query-head shards
        before ranking. No cache data, block tables, or ownership is changed.
        """
        boundary = self.due_boundary(
            req_id, layer_name, computed_generated_tokens, cadence
        )
        if boundary is None:
            return None
        if self._seq_lens.get(req_id, 0) == 0:
            return None
        if score_fn is None:
            from vllm.v1.attention.ops.triton_ngc_scorer import ngc_page_scores

            score_fn = ngc_page_scores
        if prepared is None:
            queries, query_positions = self.get_queries(req_id, layer_name)
        else:
            queries, query_positions = prepared.queries, prepared.query_positions
        retained_tokens = key_positions.numel()
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
            computed_generated_tokens=computed_generated_tokens,
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
