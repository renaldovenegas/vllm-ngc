# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned query history and pending page scores for NGC eviction."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from math import floor

import torch

from vllm.v1.ngc_eviction import NGCEvictionProposal


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
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("NGC page scores must be finite")
    eligible_pages = max(num_pages - 1, 0)
    budget = max((retained_tokens - min_retained_tokens) // page_size, 0)
    num_victims = min(floor(eviction_rate * eligible_pages), budget)
    if num_victims == 0:
        return torch.empty(0, dtype=torch.int64, device=scores.device)
    return torch.argsort(scores[:eligible_pages], stable=True)[:num_victims]


@dataclass
class NGCQueryHistory:
    queries: torch.Tensor
    positions: torch.Tensor
    count: int = 0
    page_scores: NGCPageScores | None = None
    decision: NGCEvictionDecision | None = None


class NGCScoringBuffers:
    """Eager-mode storage keyed by request ID and layer, independent of batch order.

    Query positions are original sequence positions. Capturing another batch
    invalidates that layer's pending scores. Callers must also discard pending
    scores if they change a block table before applying an eviction decision.
    """

    def __init__(self):
        self.histories: dict[tuple[str, str], NGCQueryHistory] = {}
        self._batch: list[tuple[str, int, int, int]] = []
        self._positions: torch.Tensor | None = None
        self._scored_boundaries: dict[tuple[str, str], int] = {}

    def take_proposals(
        self, table_versions: dict[str, int], layer_names: list[str]
    ) -> list[NGCEvictionProposal]:
        proposals = []
        for req_id, version in table_versions.items():
            for layer_name in layer_names:
                history = self.histories.get((req_id, layer_name))
                if history is None or history.decision is None:
                    continue
                proposals.append(
                    history.decision.to_proposal(req_id, layer_name, version)
                )
                history.decision = None
        return proposals

    @contextmanager
    def batch(
        self,
        req_ids: list[str],
        query_lengths: list[int],
        positions: torch.Tensor,
        *,
        computed_generated_tokens: list[int],
    ):
        """Prepare capture using post-forward CPU counts, negative during prefill."""
        if not len(req_ids) == len(query_lengths) == len(computed_generated_tokens):
            raise ValueError("Request IDs, query lengths, and counts must match")
        if self._positions is not None:
            raise RuntimeError("NGC capture batches cannot overlap")
        if positions.ndim != 1 or sum(query_lengths) > positions.numel():
            raise ValueError("NGC capture requires one original position per query")
        if any(length < 0 for length in query_lengths):
            raise ValueError("Query lengths must be nonnegative")
        offset = 0
        for req_id, length, generated in zip(
            req_ids, query_lengths, computed_generated_tokens
        ):
            self._batch.append((req_id, offset, offset + length, generated))
            offset += length
        self._positions = positions
        try:
            yield
        finally:
            self._batch = []
            self._positions = None

    def capture(self, layer, args, kwargs):
        """Capture only the query window needed at the next scoring boundary."""
        if self._positions is None:
            return
        query = args[0] if args else kwargs["query"]
        query = query.reshape(-1, layer.num_heads, layer.head_size)
        if self._batch and self._batch[-1][2] > query.shape[0]:
            raise ValueError(
                "NGC capture query count does not cover the prepared batch"
            )
        capacity = layer.ngc_eviction_config.query_buffer_size
        cadence = layer.ngc_eviction_config.cadence
        for req_id, start, end, generated in self._batch:
            if start == end:
                continue
            key = (req_id, layer.layer_name)
            history = self.histories.get(key)
            if history is not None:
                history.page_scores = None
                history.decision = None
            # A chunk crossing a due boundary is scored at its final state.
            due = self.due_boundary(
                req_id, layer.layer_name, max(generated, 0), cadence
            )
            if due is not None:
                incoming = min(end - start, capacity)
            else:
                next_boundary = max(
                    cadence,
                    ((generated + cadence - 1) // cadence) * cadence,
                    self._scored_boundaries.get(key, 0) + cadence,
                )
                distance = next_boundary - generated
                incoming = min(end - start, max(capacity - distance, 0))
            if incoming == 0:
                if history is not None:
                    history.count = 0
                continue
            if history is None:
                history = NGCQueryHistory(
                    torch.empty(
                        (capacity, *query.shape[1:]),
                        dtype=query.dtype,
                        device=query.device,
                    ),
                    torch.empty(capacity, dtype=torch.int64, device=query.device),
                )
                self.histories[key] = history
            old = min(history.count, capacity - incoming)
            if old and old < history.count:
                history.queries[:old].copy_(
                    history.queries[history.count - old : history.count].clone()
                )
                history.positions[:old].copy_(
                    history.positions[history.count - old : history.count].clone()
                )
            history.queries[old : old + incoming].copy_(query[end - incoming : end])
            history.positions[old : old + incoming].copy_(
                self._positions[end - incoming : end]
            )
            history.count = old + incoming

    def get_queries(self, req_id: str, layer_name: str):
        """Return chronological views, valid until the next capture or removal."""
        history = self.histories[req_id, layer_name]
        return history.queries[: history.count], history.positions[: history.count]

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
        history = self.histories[req_id, layer_name]
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
        history = self.histories.get((req_id, layer_name))
        if history is None or history.count == 0:
            return None
        if score_fn is None:
            from vllm.v1.attention.ops.triton_ngc_scorer import ngc_page_scores

            score_fn = ngc_page_scores
        queries, query_positions = self.get_queries(req_id, layer_name)
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
            scores, retained_tokens, page_size, eviction_rate, min_retained_tokens
        )
        self.store_page_scores(req_id, layer_name, scores, block_ids)
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
        if not keep_cadence:
            for key in list(self._scored_boundaries):
                if key[0] == req_id:
                    del self._scored_boundaries[key]
