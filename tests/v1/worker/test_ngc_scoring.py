# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.ngc_eviction import NGCCacheLayout, NGCEvictionInbox, NGCEvictionProposal
from vllm.v1.worker.ngc_scoring import NGCScoringBuffers, select_ngc_victims


def test_query_history_follows_requests_and_layers_across_batch_reordering():
    buffers = NGCScoringBuffers()
    layer = SimpleNamespace(
        layer_name="layer.0",
        num_heads=1,
        head_size=1,
        ngc_eviction_config=SimpleNamespace(query_buffer_size=3, cadence=1),
    )
    with buffers.batch(
        ["a", "b"],
        [4, 1],
        torch.tensor([0, 1, 2, 3, 20]),
        computed_generated_tokens=[4, 21],
    ):
        buffers.capture(layer, (torch.arange(5.0).view(5, 1, 1),), {})
        layer.layer_name = "layer.1"
        buffers.capture(layer, (torch.arange(5.0).view(5, 1, 1) + 100,), {})
    layer.layer_name = "layer.0"
    scores, ids = torch.tensor([0.25, 0.75]), torch.tensor([7, 2])
    buffers.store_page_scores("a", "layer.0", scores, ids)
    scores.zero_()
    ids.zero_()
    saved = buffers.histories["a", "layer.0"].page_scores
    torch.testing.assert_close(saved.scores, torch.tensor([0.25, 0.75]))
    assert saved.block_ids.tolist() == [7, 2]
    with buffers.batch(
        ["b", "a"], [1, 1], torch.tensor([21, 4]), computed_generated_tokens=[22, 5]
    ):
        buffers.capture(layer, (torch.tensor([50.0, 60.0]).view(2, 1, 1),), {})
    query, positions = buffers.get_queries("a", "layer.0")
    assert query.flatten().tolist() == [2, 3, 60]
    assert positions.tolist() == [2, 3, 4]
    assert buffers.histories["a", "layer.0"].page_scores is None
    assert buffers.get_queries("a", "layer.1")[0].flatten().tolist() == [101, 102, 103]
    assert buffers.get_queries("b", "layer.0")[1].tolist() == [20, 21]
    buffers.discard_request("a")
    assert all(req != "a" for req, _ in buffers.histories)
    assert ("b", "layer.0") in buffers.histories


@pytest.mark.parametrize("cadence", [1, 3, 128])
@pytest.mark.parametrize("chunks", [[7] + [1] * 260, [3, 4, 120, 9, 131]])
def test_windowed_capture_scores_same_recent_queries_as_continuous_capture(
    cadence, chunks
):
    """Preserve scorer inputs through prefill, short cadences and crossed boundaries."""
    buffers = NGCScoringBuffers()
    layer = SimpleNamespace(
        layer_name="layer.0",
        num_heads=1,
        head_size=1,
        ngc_eviction_config=SimpleNamespace(query_buffer_size=5, cadence=cadence),
    )
    length = 0
    scored = []

    def score(query, cache, table, kept_len, scale, **kwargs):
        expected = torch.arange(max(0, length - 5), length)
        torch.testing.assert_close(kwargs["query_positions"], expected)
        torch.testing.assert_close(query.flatten(), expected.float())
        scored.append(length)
        return torch.ones((length + 15) // 16)

    for chunk in chunks:
        positions = torch.arange(length, length + chunk)
        length += chunk
        generated = length - 7
        with buffers.batch(
            ["a"], [chunk], positions, computed_generated_tokens=[generated]
        ):
            buffers.capture(layer, (positions.float().view(-1, 1, 1),), {})
        pages = torch.arange((length + 15) // 16)
        buffers.score_if_due(
            "a",
            layer.layer_name,
            group_id=0,
            computed_generated_tokens=max(generated, 0),
            cadence=cadence,
            eviction_rate=0,
            min_retained_tokens=1,
            key_cache=torch.empty(0),
            kernel_block_table=pages,
            block_ids=pages,
            key_positions=torch.arange(length),
            page_size=16,
            softmax_scale=1,
            score_fn=score,
        )
    assert scored


def test_capture_skips_idle_steps_and_invalidates_old_proposals():
    buffers = NGCScoringBuffers()
    layer = SimpleNamespace(
        layer_name="layer.0",
        num_heads=1,
        head_size=1,
        ngc_eviction_config=SimpleNamespace(query_buffer_size=5, cadence=128),
    )

    def capture(count):
        with buffers.batch(
            ["a"], [1], torch.tensor([count + 99]), computed_generated_tokens=[count]
        ):
            buffers.capture(layer, (torch.tensor([float(count)]).view(1, 1, 1),), {})

    capture(123)
    assert not buffers.histories
    for count in range(124, 129):
        capture(count)
    history = buffers.histories["a", layer.layer_name]
    assert history.queries.flatten().tolist() == [124, 125, 126, 127, 128]
    before = history.queries.clone()
    buffers._scored_boundaries["a", layer.layer_name] = 128
    history.page_scores = object()
    history.decision = object()
    for count in range(129, 252):
        capture(count)
    assert history.count == 0
    assert history.page_scores is None and history.decision is None
    torch.testing.assert_close(history.queries, before)
    for count in range(252, 257):
        capture(count)
    assert history.queries.flatten().tolist() == [252, 253, 254, 255, 256]


@pytest.mark.parametrize(
    "scores,length,rate,minimum,expected",
    [
        ([0.2, 0.1, 0.3, 0.0], 64, 0.0, 1, []),
        ([0.2, 0.1, 0.3, 0.0], 64, 0.5, 1, [1]),
        ([0.2, 0.1, 0.3, 0.0], 49, 0.5, 1, [1]),
        ([0.2, 0.1, 0.3, 0.0], 64, 0.9, 48, [1]),
        ([0.2, 0.1, 0.3, 0.0], 64, 0.9, 49, []),
        ([1.0, 1.0, 1.0, 0.0], 64, 0.9, 1, [0, 1]),
        ([0.1, 0.0], 32, 0.1, 1, []),
        ([0.0], 7, 0.9, 1, []),
        ([], 0, 0.5, 1, []),
    ],
)
def test_victims_respect_active_page_rounding_and_retention(
    scores, length, rate, minimum, expected
):
    victims = select_ngc_victims(torch.tensor(scores), length, 16, rate, minimum)
    assert victims.tolist() == expected


def test_scoring_cadence_reduction_and_preemption_replay():
    buffers = NGCScoringBuffers()
    layer = SimpleNamespace(
        layer_name="layer.0",
        num_heads=1,
        head_size=1,
        ngc_eviction_config=SimpleNamespace(query_buffer_size=2, cadence=2),
    )

    def capture():
        with buffers.batch(
            ["a"], [2], torch.tensor([62, 63]), computed_generated_tokens=[2]
        ):
            buffers.capture(layer, (torch.ones(2, 1, 1),), {})

    capture()
    calls = []
    kernel_table = torch.tensor([16, 17, 6, 7, 14, 15, 18, 19])
    original_table = kernel_table.clone()
    key_cache = torch.ones(20, 1, 8, 1)

    def score_fn(q, k, table, length, scale, **kwargs):
        calls.append(length)
        assert q.shape == (2, 1, 1)
        assert k is key_cache
        assert table is kernel_table
        assert length == 64 and scale == 0.5
        assert kwargs["scoring_block_size"] == 16
        assert kwargs["query_positions"].tolist() == [62, 63]
        assert kwargs["key_positions"].tolist() == list(range(64))
        return torch.tensor([3.0, 1.0, 2.0, 0.0])

    args = dict(
        group_id=7,
        cadence=2,
        eviction_rate=0.5,
        min_retained_tokens=16,
        key_cache=key_cache,
        kernel_block_table=kernel_table,
        block_ids=kernel_table[::2] // 2,
        key_positions=torch.arange(64),
        page_size=16,
        softmax_scale=0.5,
        score_fn=score_fn,
        # Another TP rank changes the winner from logical page 1 to page 2.
        reduce_scores=lambda scores: scores + torch.tensor([0.0, 10.0, 0.0, 0.0]),
    )

    for count in [0, 1]:
        assert (
            buffers.score_if_due(
                "a", "layer.0", computed_generated_tokens=count, **args
            )
            is None
        )
    assert not calls
    decision = buffers.score_if_due("a", "layer.0", computed_generated_tokens=2, **args)
    assert decision is not None
    assert decision.cadence_boundary == 2 and decision.group_id == 7
    assert decision.victim_indices.tolist() == [2]
    assert decision.victim_block_ids.tolist() == [7]
    assert decision.page_scores.block_ids.tolist() == [8, 3, 7, 9]
    assert buffers.histories["a", "layer.0"].decision is decision
    (proposal,) = buffers.take_proposals({"a": 42}, ["layer.0"])
    assert proposal.request_id == "a" and proposal.layer_name == "layer.0"
    assert proposal.table_version == 42 and proposal.group_id == 7
    assert proposal.block_ids == (8, 3, 7, 9)
    assert proposal.scores == (3.0, 11.0, 2.0, 0.0)
    assert proposal.victim_indices == (2,) and proposal.victim_block_ids == (7,)
    assert not buffers.take_proposals({"a": 42}, ["layer.0"])
    decision.page_scores.scores.zero_()
    assert proposal.scores == (3.0, 11.0, 2.0, 0.0)
    assert (
        buffers.score_if_due("a", "layer.0", computed_generated_tokens=2, **args)
        is None
    )
    decision = buffers.score_if_due("a", "layer.0", computed_generated_tokens=7, **args)
    assert decision.cadence_boundary == 6
    assert len(calls) == 2
    torch.testing.assert_close(kernel_table, original_table)
    torch.testing.assert_close(key_cache, torch.ones_like(key_cache))

    buffers.discard_request("a", keep_cadence=True)
    capture()
    for count in [1, 2, 6, 7]:
        assert (
            buffers.score_if_due(
                "a", "layer.0", computed_generated_tokens=count, **args
            )
            is None
        )
    assert len(calls) == 2
    assert buffers.due_boundary("a", "layer.0", 8, 2) == 8
    assert buffers.due_boundary("a", "other_layer", 2, 2) == 2
    assert buffers.due_boundary("other_request", "layer.0", 2, 2) == 2
    buffers.discard_request("a")
    assert buffers.due_boundary("a", "layer.0", 2, 2) == 2


def test_repeated_eviction_preserves_original_positions_and_layer_independence():
    first, second = NGCCacheLayout(), NGCCacheLayout()
    for layout in (first, second):
        layout.extend(6, 4)
    first.evict((0, 2), 4)
    first.extend(1, 4)
    first.evict((1,), 4)
    assert first.page_positions == [4, 16, 20, 24]
    assert first.evicted_tokens == 12
    assert second.page_positions == [0, 4, 8, 12, 16, 20]
    assert second.evicted_tokens == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_page_scores_are_not_ranked(value):
    with pytest.raises(ValueError, match="finite"):
        select_ngc_victims(torch.tensor([value, 0.0]), 32, 16, 0.5, 1)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "rescheduled",
        "preempted",
        "reused_id",
        "blocks",
        "in_flight",
        "length",
        "layer",
        "version",
        "victim",
    ],
)
@pytest.mark.parametrize("evicted_tokens", [0, 16])
def test_handoff_accepts_only_completed_current_cache_snapshots(change, evicted_tokens):
    inbox = NGCEvictionInbox()
    version = inbox.schedule("a")
    proposal = NGCEvictionProposal(
        request_id="a",
        layer_name="layer.0",
        group_id=2,
        table_version=version,
        cadence_boundary=2,
        computed_generated_tokens=2,
        retained_tokens=48,
        page_size=16,
        block_ids=(8, 3, 7),
        scores=(0.1, 0.9, 0.0),
        victim_indices=(0,),
        victim_block_ids=(8,),
    )
    args = dict(
        scheduled_version=version,
        layer_name="layer.0",
        block_ids=[8, 3, 7],
        page_size=16,
        computed_tokens=48 + evicted_tokens,
        prompt_tokens=46 + evicted_tokens,
        in_flight_tokens=0,
        evicted_tokens=evicted_tokens,
    )
    if change == "rescheduled":
        inbox.schedule("a")
    elif change in ("preempted", "reused_id"):
        inbox.invalidate("a")
        if change == "reused_id":
            inbox.schedule("a")
    elif change == "blocks":
        args["block_ids"] = [8, 4, 7]
    elif change == "in_flight":
        args["in_flight_tokens"] = 1
    elif change == "length":
        args["computed_tokens"] = 49 + evicted_tokens
    elif change == "layer":
        proposal = replace(proposal, layer_name="layer.1")
    elif change == "version":
        args["scheduled_version"] = version + 1
    elif change == "victim":
        proposal = replace(proposal, victim_block_ids=(3,))
    assert inbox.receive(proposal, **args) == (change == "none")
    if change == "none":
        assert inbox.pending == {("a", 2): proposal}
        assert not inbox.receive(proposal, **args)
        inbox.schedule("unrelated")
        assert inbox.pending == {("a", 2): proposal}
        inbox.schedule("a")
    assert not inbox.pending
