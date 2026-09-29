# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only eviction proposals and scheduler completion tracking."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class NGCEvictionProposal:
    """Immutable scoring snapshot with no GPU tensors or cache ownership."""

    request_id: str
    layer_name: str
    group_id: int
    table_version: int
    cadence_boundary: int
    computed_generated_tokens: int
    retained_tokens: int
    page_size: int
    block_ids: tuple[int, ...]
    scores: tuple[float, ...]
    victim_indices: tuple[int, ...]
    victim_block_ids: tuple[int, ...]


class NGCEvictionInbox:
    """Keep proposals only while their scheduled cache snapshot is current.

    Scheduling again invalidates pending proposals, even if no page was added.
    Consumers must apply a proposal before scheduling another read of its cache.
    This class neither frees blocks nor changes block tables.
    """

    def __init__(self) -> None:
        self._next_version = 0
        self._versions: dict[str, int] = {}
        self.pending: dict[tuple[str, int], NGCEvictionProposal] = {}

    def invalidate(self, request_id: str) -> None:
        self._versions.pop(request_id, None)
        for key in list(self.pending):
            if key[0] == request_id:
                del self.pending[key]

    def schedule(self, request_id: str) -> int:
        self.invalidate(request_id)
        self._next_version += 1
        self._versions[request_id] = self._next_version
        return self._next_version

    def receive(
        self,
        proposal: NGCEvictionProposal,
        *,
        scheduled_version: int | None,
        layer_name: str,
        block_ids: list[int],
        page_size: int,
        computed_tokens: int,
        prompt_tokens: int,
        in_flight_tokens: int,
    ) -> bool:
        """Accept at output completion only if no later cache read is in flight."""
        p = proposal
        # Before physical eviction exists, retained and computed lengths match.
        num_pages = (p.retained_tokens + page_size - 1) // page_size
        if (
            scheduled_version is None
            or p.table_version != scheduled_version
            or self._versions.get(p.request_id) != scheduled_version
            or in_flight_tokens != 0
            or p.layer_name != layer_name
            or p.page_size != page_size
            or p.retained_tokens != computed_tokens
            or p.computed_generated_tokens != max(computed_tokens - prompt_tokens, 0)
            or not 0 < p.cadence_boundary <= p.computed_generated_tokens
            or len(p.block_ids) != num_pages
            or p.block_ids != tuple(block_ids[:num_pages])
            or len(p.scores) != num_pages
            or not all(isfinite(score) for score in p.scores)
            or len(set(p.victim_indices)) != len(p.victim_indices)
            or any(i < 0 or i >= num_pages - 1 for i in p.victim_indices)
        ):
            return False
        if p.victim_block_ids != tuple(p.block_ids[i] for i in p.victim_indices):
            return False
        key = (p.request_id, p.group_id)
        if key in self.pending:
            return False
        self.pending[key] = p
        return True
