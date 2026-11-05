# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.v1.ngc_eviction import NGCCacheLayout, NGCEvictionInbox, NGCEvictionProposal
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.ngc_scoring import (
    NGCEvictionDecision,
    NGCPageScores,
    NGCQueryHistory,
    NGCScoringBuffers,
    select_ngc_victims,
)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("bad_scores", [False, True])
def test_ngc_handoff_batches_layers_and_preserves_snapshots(
    device, bad_scores, monkeypatch
):
    buffers = NGCScoringBuffers()
    copies = []
    original = CpuGpuBuffer.copy_to_cpu

    def copy_to_cpu(buffer, n=None):
        copies.append(n)
        return original(buffer, n)

    monkeypatch.setattr(CpuGpuBuffer, "copy_to_cpu", copy_to_cpu)
    for layer in range(2):
        ids = torch.tensor([2**54 + layer, 9, 11], device=device)
        scores = torch.tensor([0.25, 0.75, 1.0], device=device)
        if bad_scores and layer == 1:
            scores[0] = float("nan")
        victims = torch.tensor(
            [0] if layer == 0 else [], dtype=torch.int64, device=device
        )
        decision = NGCEvictionDecision(
            group_id=layer,
            cadence_boundary=128,
            computed_tokens=128,
            retained_tokens=48,
            page_size=16,
            page_scores=NGCPageScores(scores, ids),
            victim_indices=victims,
            victim_block_ids=ids[victims],
        )
        buffers.histories["a", str(layer)] = NGCQueryHistory(decision=decision)
    if bad_scores:
        with pytest.raises(ValueError, match="finite"):
            buffers.take_proposals({"a": 42}, ["0", "1"])
        assert all(
            history.decision is not None for history in buffers.histories.values()
        )
        return
    proposals = buffers.take_proposals({"a": 42}, ["0", "1"])
    assert len(copies) == 1
    assert copies == [24], "Transfer only validity flags and the one victim pair"
    assert proposals[0].victim_block_ids == (2**54,)
    assert proposals[1].victim_indices == ()
    assert all(
        proposal.scores is None and proposal.block_ids is None for proposal in proposals
    )
    buffers._handoff.cpu.zero_()
    assert proposals[0].victim_block_ids == (2**54,)
    assert not buffers.take_proposals({"a": 42}, ["0", "1"])


def _ring_buffers(window, layers=("layer.0",), num_slots=4, cadences=None):
    buffers = NGCScoringBuffers()
    buffers.allocate(
        {name: (window, 1, 1) for name in layers},
        num_slots=num_slots,
        max_num_tokens=512,
        dtype=torch.float32,
        device=torch.device("cpu"),
        cadences=cadences,
    )
    return buffers


def _step(buffers, chunks, queries=None):
    """Prepare (request, first position, length) chunks, then capture each layer.

    Every layer captures every step, as in a real forward. ``queries`` maps a
    layer to [num_tokens] values, which may include padded rows; by default a
    query equals its original position.
    """
    lengths = np.array([length for _, _, length in chunks])
    positions = np.concatenate(
        [np.arange(start, start + length) for _, start, length in chunks]
    )
    buffers.prepare_capture(
        [req_id for req_id, _, _ in chunks],
        np.repeat(np.arange(len(chunks)), lengths),
        np.concatenate([np.arange(length) for length in lengths]),
        lengths,
        positions,
        [start + length for _, start, length in chunks],
    )
    for name in buffers._rings:
        values = (
            torch.tensor(positions, dtype=torch.float32)
            if queries is None
            else queries[name]
        )
        buffers.capture(name, values.view(-1, 1, 1))


def _queries(buffers, req_id, layer="layer.0"):
    query, positions = buffers.get_queries(req_id, layer)
    return query.flatten().tolist(), positions.tolist()


def test_query_history_follows_requests_and_layers_across_batch_reordering():
    buffers = _ring_buffers(3, layers=("layer.0", "layer.1"))
    _step(
        buffers,
        [("a", 0, 4), ("b", 0, 1)],
        {
            "layer.0": torch.arange(5.0),
            "layer.1": torch.arange(5.0) + 100,
        },
    )
    scores, ids = torch.tensor([0.25, 0.75]), torch.tensor([7, 2])
    buffers.store_page_scores("a", "layer.0", scores, ids)
    scores.zero_()
    ids.zero_()
    saved = buffers.histories["a", "layer.0"].page_scores
    torch.testing.assert_close(saved.scores, torch.tensor([0.25, 0.75]))
    assert saved.block_ids.tolist() == [7, 2]
    _step(
        buffers,
        [("b", 1, 1), ("a", 4, 1)],
        {
            "layer.0": torch.tensor([50.0, 60.0]),
            "layer.1": torch.tensor([150.0, 160.0]),
        },
    )
    assert _queries(buffers, "a") == ([2, 3, 60], [2, 3, 4])
    assert ("a", "layer.0") not in buffers.histories, "Rescheduling drops scores"
    assert _queries(buffers, "a", "layer.1")[0] == [102, 103, 160]
    assert _queries(buffers, "b") == ([4, 50], [0, 1])
    buffers.discard_request("a")
    assert all(req != "a" for req, _ in buffers.histories)
    # A reused slot exposes only the new request's queries.
    _step(buffers, [("c", 0, 2), ("b", 2, 1)])
    assert _queries(buffers, "c") == ([0, 1], [0, 1])
    assert _queries(buffers, "b") == ([4, 50, 2], [0, 1, 2])


@pytest.mark.parametrize("cadence", [1, 3, 128])
@pytest.mark.parametrize("chunks", [[7] + [1] * 260, [3, 4, 120, 9, 131]])
def test_ring_capture_scores_most_recent_queries(cadence, chunks):
    """Preserve scorer inputs through prefill, short cadences and crossed boundaries."""
    buffers = _ring_buffers(5)
    length = 0
    scored = []

    def score(query, cache, table, kept_len, scale, **kwargs):
        expected = torch.arange(max(0, length - 5), length)
        torch.testing.assert_close(kwargs["query_positions"], expected)
        torch.testing.assert_close(query.flatten(), expected.float())
        scored.append(length)
        return torch.ones((length + 15) // 16)

    for chunk in chunks:
        _step(buffers, [("a", length, chunk)])
        length += chunk
        pages = torch.arange((length + 15) // 16)
        buffers.score_if_due(
            "a",
            "layer.0",
            group_id=0,
            computed_tokens=length,
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


def test_padded_and_dummy_rows_never_reach_request_slots():
    buffers = _ring_buffers(4, num_slots=2)
    _step(buffers, [("a", 0, 6)])
    assert _queries(buffers, "a") == ([2, 3, 4, 5], [2, 3, 4, 5])
    # A shorter batch replayed at the old padded size must not reuse stale rows.
    _step(buffers, [("a", 6, 1)], {"layer.0": torch.tensor([6.0] + [-1.0] * 5)})
    assert _queries(buffers, "a") == ([3, 4, 5, 6], [3, 4, 5, 6])
    buffers.reset_capture()
    buffers.capture("layer.0", torch.full((8, 1, 1), -2.0))
    assert _queries(buffers, "a") == ([3, 4, 5, 6], [3, 4, 5, 6])
    _step(buffers, [("a", 7, 1), ("b", 0, 1)])
    with pytest.raises(RuntimeError, match="slots"):
        _step(buffers, [("c", 0, 1)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_capture_replays_inside_cuda_graph_with_new_rows():
    """A graph recorded once keeps routing each step's queries to new rows."""
    buffers = NGCScoringBuffers()
    buffers.allocate(
        {"layer.0": (3, 2, 4)},
        num_slots=2,
        max_num_tokens=8,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    query = torch.zeros(8, 2, 4, dtype=torch.float16, device="cuda")
    buffers.reset_capture()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buffers.capture("layer.0", query)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        buffers.capture("layer.0", query)

    def replay(chunks):
        lengths = np.array([length for _, _, length in chunks])
        positions = np.concatenate(
            [np.arange(start, start + length) for _, start, length in chunks]
        )
        buffers.prepare_capture(
            [req_id for req_id, _, _ in chunks],
            np.repeat(np.arange(len(chunks)), lengths),
            np.concatenate([np.arange(length) for length in lengths]),
            lengths,
            positions,
            [start + length for _, start, length in chunks],
        )
        query.fill_(-1)
        query[: len(positions)] = torch.tensor(positions, device="cuda")[:, None, None]
        graph.replay()

    def recent(req_id):
        values, positions = buffers.get_queries(req_id, "layer.0")
        assert (values == values[:, :1, :1]).all()
        return values[:, 0, 0].tolist(), positions.tolist()

    replay([("a", 0, 5), ("b", 0, 2)])
    assert recent("a") == ([2, 3, 4], [2, 3, 4])
    assert recent("b") == ([0, 1], [0, 1])
    replay([("b", 2, 1), ("a", 5, 1)])
    assert recent("a") == ([3, 4, 5], [3, 4, 5])
    assert recent("b") == ([0, 1, 2], [0, 1, 2])


def test_due_filter_checks_each_request_boundary_once():
    buffers = _ring_buffers(1)
    assert not buffers.any_due("a", 127, 128)
    assert buffers.any_due("a", 128, 128, mark_checked=False)
    assert buffers.any_due("a", 128, 128, mark_checked=False)
    assert not buffers._checked_boundaries
    assert buffers.any_due("a", 128, 128)
    assert not buffers.any_due("a", 200, 128)
    assert buffers.any_due("a", 256, 128) and buffers.any_due("b", 128, 128)
    buffers.discard_request("a", keep_cadence=True)
    assert not buffers.any_due("a", 256, 128)
    buffers.discard_request("a")
    assert buffers.any_due("a", 128, 128)


def test_cadence_plan_snapshots_requests_without_consuming_boundaries():
    """One shared plan survives length-buffer changes and request reordering."""
    buffers = _ring_buffers(1)
    assert buffers.any_due("a", 8, 8)
    lengths = np.array([17, 9, 7, 999], dtype=np.int32)
    checks = buffers.plan_cadences(["a", "b", "c"], lengths, {8, 16})
    lengths[:] = 100
    assert [
        (check.req_index, check.req_id, check.seq_len, check.due_boundaries)
        for check in checks
    ] == [
        (0, "a", 17, {8: 16, 16: 16}),
        (1, "b", 9, {8: 8}),
        (2, "c", 7, {}),
    ]
    assert buffers.any_due("a", 16, 8, mark_checked=False)
    assert buffers.any_due("b", 8, 8, mark_checked=False)
    for check in checks:
        buffers.mark_cadences_checked(check)
    reordered = buffers.plan_cadences(["b", "a"], np.array([16, 17]), {8, 16})
    assert reordered[0].due_boundaries == {8: 16, 16: 16}
    assert reordered[1].due_boundaries == {}
    assert not buffers.plan_cadences(["a"], np.array([100]), set())


@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE"])
def test_cadence_plan_preserves_crossed_prefill_capture_until_commit(mode, monkeypatch):
    """Planning before forward must not suppress the crossed boundary's queries."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm.v1.worker import ngc_scoring

    monkeypatch.setattr(ngc_scoring, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        ngc_scoring,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=getattr(CUDAGraphMode, mode)),
    )
    buffers = _ring_buffers(3, cadences={"layer.0": 8})
    _step(buffers, [("a", 0, 6)])
    check = buffers.plan_cadences(["a"], np.array([11]), {8})[0]
    assert check.due_boundaries == {8: 8}
    _step(buffers, [("a", 6, 5)])
    assert _queries(buffers, "a") == ([8, 9, 10], [8, 9, 10])
    buffers.mark_cadences_checked(check)
    assert buffers.plan_cadences(["a"], np.array([11]), {8})[0].due_boundaries == {}
    # Request-level consumption must not suppress an unscored layer's safety check.
    assert buffers.due_boundary("a", "layer.0", 11, 8) == 8
    assert buffers.plan_cadences(["a"], np.array([16]), {8})[0].due_boundaries == {
        8: 16
    }


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
    buffers = _ring_buffers(2)
    buffers.include_debug_scores = True

    def capture():
        _step(buffers, [("a", 62, 2)], {"layer.0": torch.ones(2)})

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
            buffers.score_if_due("a", "layer.0", computed_tokens=count, **args) is None
        )
    assert not calls
    decision = buffers.score_if_due("a", "layer.0", computed_tokens=2, **args)
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
    assert buffers.score_if_due("a", "layer.0", computed_tokens=2, **args) is None
    decision = buffers.score_if_due("a", "layer.0", computed_tokens=7, **args)
    assert decision.cadence_boundary == 6
    assert len(calls) == 2
    torch.testing.assert_close(kernel_table, original_table)
    torch.testing.assert_close(key_cache, torch.ones_like(key_cache))

    buffers.discard_request("a", keep_cadence=True)
    capture()
    for count in [1, 2, 6, 7]:
        assert (
            buffers.score_if_due("a", "layer.0", computed_tokens=count, **args) is None
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
@pytest.mark.parametrize("compact", [False, True])
def test_handoff_accepts_only_completed_current_cache_snapshots(
    change, evicted_tokens, compact
):
    inbox = NGCEvictionInbox()
    version = inbox.schedule("a")
    proposal = NGCEvictionProposal(
        request_id="a",
        layer_name="layer.0",
        group_id=2,
        table_version=version,
        cadence_boundary=2,
        computed_tokens=48 + evicted_tokens,
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
        in_flight_tokens=0,
        evicted_tokens=evicted_tokens,
    )
    if compact:
        proposal = replace(proposal, block_ids=None, scores=None)
    if change == "rescheduled":
        inbox.schedule("a")
    elif change in ("preempted", "reused_id"):
        inbox.invalidate("a")
        if change == "reused_id":
            inbox.schedule("a")
    elif change == "blocks":
        args["block_ids"] = [4, 3, 7]
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


@pytest.mark.parametrize("max_query_len", [1, 4])
@pytest.mark.parametrize("segments,max_seqs_64", [(16, 0), (64, 1), (64, 2)])
def test_ngc_shared_metadata_matches_build_with_independent_cache(
    max_query_len, segments, max_seqs_64
):
    """Reusing batch fields preserves per-layer lengths, tables, and scratch."""
    from dataclasses import fields

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    def make_builder():
        builder = object.__new__(TritonAttentionMetadataBuilder)
        builder.seq_threshold_3D = 8
        builder.num_par_softmax_segments = segments
        builder.num_heads_q = 2
        builder.max_seqs_64_segments = max_seqs_64
        builder.softmax_segm_output = torch.empty(2, 2, segments, 8)
        builder.softmax_segm_max = torch.empty(2, 2, segments)
        builder.softmax_segm_expsum = torch.empty(2, 2, segments)
        builder.rswa_window = None
        return builder

    first, second = make_builder(), make_builder()
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, max_query_len, 2 * max_query_len]),
        query_start_loc_cpu=torch.tensor([0, max_query_len, 2 * max_query_len]),
        seq_lens=torch.tensor([17, 9]),
        num_reqs=2,
        num_actual_tokens=2 * max_query_len,
        max_query_len=max_query_len,
        max_seq_len=17,
        block_table_tensor=torch.tensor([[1, 2], [3, 4]]),
        slot_mapping=torch.arange(2 * max_query_len),
        causal=True,
    )
    template = first.build(0, common)
    other = replace(
        common,
        seq_lens=torch.tensor([9, 5]),
        max_seq_len=9,
        block_table_tensor=torch.tensor([[5, 6], [7, 8]]),
        slot_mapping=common.slot_mapping + 16,
    )
    reused = second.reuse_ngc_metadata(template, other)
    expected = second.build(0, other)
    for field in fields(expected):
        actual_value, expected_value = (
            getattr(reused, field.name),
            getattr(expected, field.name),
        )
        if isinstance(expected_value, torch.Tensor):
            assert actual_value.data_ptr() == expected_value.data_ptr()
            assert actual_value.shape == expected_value.shape
        else:
            assert actual_value == expected_value
    assert template.seq_lens is common.seq_lens
    assert template.block_table is common.block_table_tensor
    assert reused.softmax_segm_output is not template.softmax_segm_output
    other.seq_lens[0] = 7
    assert reused.seq_lens[0] == 7
    assert template.seq_lens[0] == 17


@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE"])
def test_ngc_capture_gates_each_cadence_and_crossed_prefill(mode, monkeypatch):
    """Capture only upcoming windows or a due chunk, independently per layer."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm.v1.worker import ngc_scoring

    monkeypatch.setattr(ngc_scoring, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        ngc_scoring,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=getattr(CUDAGraphMode, mode)),
    )
    buffers = NGCScoringBuffers()
    buffers.allocate(
        {"short": (3, 1, 1), "long": (5, 1, 1)},
        num_slots=2,
        max_num_tokens=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
        cadences={"short": 8, "long": 16},
    )
    _step(buffers, [("a", 0, 5)])
    assert not buffers._slots
    assert all(not ring.any() for ring in buffers._rings.values())
    for position in (5, 6, 7):
        _step(buffers, [("a", position, 1)])
    assert _queries(buffers, "a", "short") == ([5, 6, 7], [5, 6, 7])
    assert not buffers._rings["long"].any()
    assert buffers.any_due("a", 8, 8)
    previous = buffers._rings["short"].clone()
    _step(buffers, [("a", 8, 1)])
    torch.testing.assert_close(buffers._rings["short"], previous)
    _step(buffers, [("b", 0, 29)])
    assert _queries(buffers, "b", "short") == ([26, 27, 28], [26, 27, 28])
    assert _queries(buffers, "b", "long") == (
        [24, 25, 26, 27, 28],
        [24, 25, 26, 27, 28],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_ngc_full_graph_capture_switches_inactive_active_inactive(monkeypatch):
    """The masked kernel recorded inactive still captures an active replay."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm.v1.worker import ngc_scoring

    monkeypatch.setattr(ngc_scoring, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        ngc_scoring,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )
    buffers = NGCScoringBuffers()
    buffers.allocate(
        {"layer": (3, 1, 300)},
        num_slots=1,
        max_num_tokens=8,
        dtype=torch.float32,
        device=torch.device("cuda"),
        cadences={"layer": 8},
    )
    query = torch.ones((8, 1, 300), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buffers.capture("layer", query)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buffers.capture("layer", query)
    graph.replay()
    assert not buffers._rings["layer"].any()
    buffers.prepare_capture(
        ["a"],
        np.zeros(3, dtype=np.int64),
        np.arange(3),
        np.array([3]),
        np.arange(5, 8),
        [8],
    )
    query[:3] = torch.arange(5, 8, device="cuda")[:, None, None]
    graph.replay()
    actual, positions = buffers.get_queries("a", "layer")
    torch.testing.assert_close(
        actual, positions[:, None, None].expand_as(actual).float()
    )
    assert buffers.any_due("a", 8, 8)
    previous = buffers._rings["layer"].clone()
    buffers.prepare_capture(
        ["a"], np.array([0]), np.array([0]), np.array([1]), np.array([8]), [9]
    )
    query.fill_(-1)
    graph.replay()
    torch.testing.assert_close(buffers._rings["layer"], previous)


@pytest.fixture
def noop_ngc_gather(monkeypatch):
    import sys
    from types import SimpleNamespace

    launches = []

    class Gather:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((args, kwargs))

            return launch

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.ops.triton_ngc_inputs",
        SimpleNamespace(prepare_ngc_score_inputs=Gather()),
    )
    return launches


def test_persistent_score_inputs_refresh_layout_lengths_and_fused_rows(noop_ngc_gather):
    """Reusing descriptors must still refresh every dynamic decode/scoring input."""
    buffers = _ring_buffers(3)
    _step(buffers, [("a", 0, 64), ("b", 0, 64)])
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    a, b = [
        buffers.update_score_input(req, "layer.0", row, 32, 32, 16, 16, (0, 48), table)
        for row, req in enumerate(("a", "b"))
    ]
    due = {("a", "layer.0"): True, ("b", "layer.0"): True}
    buffers.prepare_fused_decode([a, b], due)
    group = buffers._input_groups["layer.0"]
    buffers.prepare_fused_decode([a, b], due)
    assert buffers._input_groups["layer.0"] is group
    new_table = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], dtype=torch.int32)
    assert (
        buffers.update_score_input(
            "a", "layer.0", 1, 65, 0, 16, 16, (0, 16, 48), new_table
        )
        is a
    )
    assert (
        buffers.update_score_input(
            "b", "layer.0", 0, 33, 32, 16, 16, (16, 48), new_table
        )
        is b
    )
    buffers._seq_lens.update(a=65, b=65)
    buffers.prepare_fused_decode([b, a], due)
    state = buffers._fused_decode["layer.0"]
    descriptors = state.descriptors.reshape(-1, 8)
    assert descriptors[:, 0].tolist() == [65, 65]
    assert descriptors[:, 3:6].tolist() == [[33, 32, 2], [65, 0, 3]]
    assert descriptors[:, 7].tolist() == [0, 1]
    assert noop_ngc_gather[-1][0][3] is new_table
    assert state.workspace.key_positions.shape[1] >= 65
    state.score_buffer[0].fill_(7)
    state.score_buffer[1].fill_(13)
    buffers.publish_fused_graph_scores()
    assert buffers.get_fused_scores("b", "layer.0").tolist() == [7] * 3
    assert buffers.get_fused_scores("a", "layer.0").tolist() == [13] * 5
    assert buffers.get_fused_scores("missing", "layer.0") is None
    group = buffers._input_groups["layer.0"]
    a.retained_tokens = 66
    buffers._seq_lens["a"] = 66
    buffers.prepare_fused_decode([b, a], due)
    assert buffers._input_groups["layer.0"] is group
    assert buffers._fused_decode["layer.0"].descriptors.reshape(-1, 8)[
        :, 0
    ].tolist() == [65, 66]


def test_persistent_execution_discards_request_and_resets_allocation(noop_ngc_gather):
    buffers = _ring_buffers(2)
    _step(buffers, [("a", 0, 32)])
    table = torch.ones((1, 2), dtype=torch.int32)
    args = ("a", "layer.0", 0, 32, 0, 16, 16, (), table)
    original = buffers.update_score_input(*args)
    buffers.prepare_fused_decode([original], {("a", "layer.0"): True})
    buffers.discard_request("a", keep_cadence=True)
    assert buffers.get_fused_scores("a", "layer.0") is None
    replacement = buffers.update_score_input(*args)
    assert replacement is not original
    buffers.allocate(
        {"layer.0": (2, 1, 1)},
        num_slots=4,
        max_num_tokens=512,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert buffers.update_score_input(*args) is not replacement
    assert not buffers._input_groups and not buffers._fused_rows


def test_borrowed_cadence_checks_refresh_in_place_and_reuse_request_ids():
    buffers = _ring_buffers(1)
    first = buffers.plan_cadences(["a", "b"], np.array([8, 3]), {8})
    a, b = first
    due = a.due_boundaries
    buffers.mark_cadences_checked(a)
    reordered = buffers.plan_cadences(["b", "a"], np.array([16, 9]), {8})
    assert reordered[0] is b and reordered[1] is a
    assert a.req_index == 1 and a.seq_len == 9
    assert a.due_boundaries is due and not due
    assert b.due_boundaries == {8: 16}
    buffers.discard_request("a", keep_cadence=True)
    reused = buffers.plan_cadences(["a"], np.array([9]), {8})[0]
    assert reused is not a and not reused.due_boundaries
    assert buffers.plan_cadences(["a"], np.array([16]), {8})[0].due_boundaries == {
        8: 16
    }


def test_ngc_packed_metadata_reuses_positions_and_one_due_upload(monkeypatch):
    """Unchanged layouts upload descriptors only; due masks share one transfer."""
    import sys
    from types import SimpleNamespace

    from vllm.v1.worker.ngc_scoring import NGCScoreInput

    class NoOpGather:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.ops.triton_ngc_inputs",
        SimpleNamespace(prepare_ngc_score_inputs=NoOpGather()),
    )
    buffers = _ring_buffers(2, layers=("layer.0", "layer.1"))
    _step(buffers, [("a", 0, 64), ("b", 0, 64)])
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    inputs = [
        NGCScoreInput(req, layer, row, 32, 32, 16, 16, (0, 48), table)
        for layer in ("layer.0", "layer.1")
        for row, req in enumerate(("a", "b"))
    ]
    calls = []
    original = CpuGpuBuffer.copy_to_gpu

    def copy(buffer, n=None):
        calls.append((buffer, n))
        return original(buffer, n)

    monkeypatch.setattr(CpuGpuBuffer, "copy_to_gpu", copy)
    due = {(i.req_id, i.layer_name): i.req_id == "a" for i in inputs}
    buffers.prepare_fused_decode(inputs, due)
    packed, masks = buffers._score_inputs, buffers._fused_due_batch
    assert [(n) for buffer, n in calls if buffer is packed] == [40]
    assert sum(buffer is masks for buffer, _ in calls) == 1
    pointer = masks.gpu.data_ptr()
    calls.clear()
    due = {key: not value for key, value in due.items()}
    buffers.prepare_fused_decode(inputs, due)
    assert [n for buffer, n in calls if buffer is packed] == [32]
    assert sum(buffer is masks for buffer, _ in calls) == 1
    assert masks.gpu.data_ptr() == pointer
    assert all(
        state.due.tolist() == [False, True] for state in buffers._fused_decode.values()
    )
    calls.clear()
    inputs[0] = replace(inputs[0], page_positions=(16, 48))
    buffers.prepare_fused_decode(inputs, due)
    assert [n for buffer, n in calls if buffer is packed] == [40]
    buffers.prepare_fused_decode([], {})
    assert not buffers._fused_decode and not buffers.fused_inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("kernel_page_size", [8, 16])
def test_ngc_packed_score_inputs_match_individual_gathers(kernel_page_size):
    """Packed gathers preserve per-layer windows and original retained positions."""
    buffers = NGCScoringBuffers()
    layers = {"short": (3, 2, 7), "long": (5, 1, 300)}
    buffers.allocate(
        layers,
        num_slots=2,
        max_num_tokens=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    buffers.prepare_capture(
        ["a", "b"],
        np.repeat(np.arange(2), 5),
        np.tile(np.arange(5), 2),
        np.array([5, 5]),
        np.r_[np.arange(36, 41), np.arange(23, 28)],
        [41, 28],
    )
    for name, (_, heads, width) in layers.items():
        query = (
            torch.arange(10 * heads * width, device="cuda")
            .float()
            .reshape(10, heads, width)
        )
        buffers.capture(name, query)
    ratio = 16 // kernel_page_size
    table = torch.tensor(
        [[6, 7, 14, 15, 18, 19], [20, 21, 24, 25, 28, 29]],
        device="cuda",
        dtype=torch.int32,
    )
    if ratio == 1:
        table = table[:, ::2].contiguous()
    inputs = [
        buffers.update_score_input(
            req, name, row, kept, evicted, 16, kernel_page_size, known, table
        )
        for name in ("long", "short")
        for req, row, kept, evicted, known in (
            ("b", 1, 17, 11, (16,)),
            ("a", 0, 21, 20, (0, 32)),
        )
    ]
    expected_queries = {
        (item.req_id, item.layer_name): buffers.get_queries(
            item.req_id, item.layer_name
        )
        for item in inputs
    }
    prepared = buffers.prepare_score_inputs(inputs)
    for item in inputs:
        actual = prepared[item.req_id, item.layer_name]
        query, positions = expected_queries[item.req_id, item.layer_name]
        torch.testing.assert_close(actual.queries, query)
        torch.testing.assert_close(actual.query_positions, positions)
        keys = [
            item.page_positions[i // 16] + i % 16
            if i // 16 < len(item.page_positions)
            else i + item.evicted_tokens
            for i in range(item.retained_tokens)
        ]
        torch.testing.assert_close(
            actual.key_positions, torch.tensor(keys, device="cuda")
        )
        pages = (item.retained_tokens + 15) // 16
        torch.testing.assert_close(
            actual.block_ids,
            (table[item.req_index, : pages * ratio : ratio] // ratio).long(),
        )
    workspaces = {
        name: ws.queries.data_ptr() for name, ws in buffers._score_workspaces.items()
    }
    again = buffers.prepare_score_inputs(inputs[:1])
    assert (
        again[inputs[0].req_id, inputs[0].layer_name].queries.data_ptr()
        == prepared[inputs[0].req_id, inputs[0].layer_name].queries.data_ptr()
    )
    assert workspaces == {
        name: ws.queries.data_ptr() for name, ws in buffers._score_workspaces.items()
    }
    expected_metadata = {
        key: (item.key_positions.clone(), item.block_ids.clone())
        for key, item in prepared.items()
    }
    for workspace in buffers._score_workspaces.values():
        workspace.queries.fill_(-17)
        workspace.query_positions.fill_(-19)
    no_gather = buffers.prepare_score_inputs(inputs, gather_queries=False)
    for key, item in no_gather.items():
        assert item.queries.eq(-17).all()
        assert item.query_positions.eq(-19).all()
        keys, blocks = expected_metadata[key]
        torch.testing.assert_close(item.key_positions, keys)
        torch.testing.assert_close(item.block_ids, blocks)

    # Filtering views must still refresh the complete batch used by fused decode.
    selected = {("a", "short")}
    for workspace in buffers._score_workspaces.values():
        workspace.key_positions.fill_(-101)
        workspace.block_ids.fill_(-103)
    due = {
        (item.req_id, item.layer_name): (item.req_id, item.layer_name) in selected
        for item in inputs
    }
    buffers.prepare_fused_decode(inputs, due)
    assert set(buffers.fused_inputs) == selected
    for name, state in buffers._fused_decode.items():
        items = [item for item in inputs if item.layer_name == name]
        assert state.inputs == items
        descriptors = state.descriptors.reshape(-1, 8)
        assert descriptors.shape[0] == len(items)
        assert descriptors[:, 3].tolist() == [i.retained_tokens for i in items]
        assert state.due.tolist() == [due[i.req_id, name] for i in items]
        for row, item in enumerate(items):
            keys, blocks = expected_metadata[item.req_id, name]
            torch.testing.assert_close(
                state.workspace.key_positions[row, : len(keys)], keys
            )
            torch.testing.assert_close(
                state.workspace.block_ids[row, : len(blocks)], blocks
            )
    # Standalone fallback still gathers every requested query when no filter is set.
    fallback = buffers.prepare_score_inputs(inputs)
    assert set(fallback) == set(expected_queries)
    for key, item in fallback.items():
        query, positions = expected_queries[key]
        torch.testing.assert_close(item.queries, query)
        torch.testing.assert_close(item.query_positions, positions)

    # Cached identities survive a compacted layout and replacement table.
    replacement = table.clone()
    replacement[1, :ratio] += 2 * ratio
    for item in inputs:
        if item.layer_name != "long":
            continue
        is_b = item.req_id == "b"
        assert (
            buffers.update_score_input(
                item.req_id,
                item.layer_name,
                item.req_index,
                16 if is_b else item.retained_tokens,
                12 if is_b else item.evicted_tokens,
                16,
                kernel_page_size,
                (0,) if is_b else item.page_positions,
                replacement,
            )
            is item
        )
    refreshed = buffers.prepare_score_inputs(inputs)["b", "long"]
    torch.testing.assert_close(refreshed.key_positions, torch.arange(16, device="cuda"))
    assert refreshed.block_ids.tolist() == [int(replacement[1, 0]) // ratio]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "head_ratio,dim,max_kept",
    [(1, 64, 145), (2, 64, 145), (4, 64, 145), (7, 128, 2049), (9, 64, 145)],
)
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("num_segments", [1, 16])
def test_ngc_fused_decode_matches_attention_and_standalone_scores(
    dtype, kv_dtype, head_ratio, dim, max_kept, page_size, num_segments
):
    """Decode/scoring casts KV to query dtype and preserves masks and GQA."""
    from vllm.v1.attention.ops.triton_ngc_decode import ngc_decode_and_score
    from vllm.v1.attention.ops.triton_ngc_scorer import ngc_page_scores
    from vllm.v1.worker.ngc_scoring import NGCScratchBuffers

    generator = torch.Generator(device="cuda").manual_seed(37)
    batch, kv_heads = 3, 2
    heads = kv_heads * head_ratio
    window, ring_window, kernel_page = 5, 7, 16
    kept = [max_kept, 33, 1]
    sequence_lengths, counts = [max_kept + 15, 48, 1], [5, 3, 1]
    table_pages = (max_kept + kernel_page - 1) // kernel_page
    physical_pages = table_pages + 10
    query = torch.randn(
        batch, heads, dim, device="cuda", dtype=dtype, generator=generator
    )
    storage = torch.randn(
        physical_pages * 2,
        kv_heads,
        kernel_page,
        2 * dim,
        device="cuda",
        dtype=kv_dtype,
        generator=generator,
    )
    cache = storage[::2]
    assert not cache.is_contiguous()
    ring = torch.randn(
        batch * ring_window + 1,
        heads,
        dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    table = torch.stack(
        [
            torch.randperm(physical_pages, device="cuda", generator=generator)[
                :table_pages
            ]
            for _ in range(batch)
        ]
    ).int()
    descriptors = torch.tensor(
        [
            [seq, row, count, length, seq - length, 0, 0, row]
            for row, (seq, count, length) in enumerate(
                zip(sequence_lengths, counts, kept)
            )
        ],
        device="cuda",
        dtype=torch.int64,
    )
    positions = torch.zeros((batch, max(kept)), device="cuda", dtype=torch.int64)
    for row, length in enumerate(kept):
        # Old retained tokens can be reordered, while the newest token is present.
        values = torch.arange(
            sequence_lengths[row] - length, sequence_lengths[row], device="cuda"
        )
        permutation = torch.randperm(length, device="cuda", generator=generator)
        positions[row, :length] = values[permutation]
        ring[row * ring_window + (sequence_lengths[row] - 1) % ring_window] = query[row]
    due = torch.tensor([True, False, True], device="cuda")
    output = torch.empty_like(query)
    scratch = NGCScratchBuffers()
    score_buffer = torch.empty(
        (batch, (max(kept) + page_size - 1) // page_size), device="cuda"
    )
    scores = ngc_decode_and_score(
        query,
        cache,
        ring,
        descriptors,
        positions,
        table,
        due,
        output,
        window=window,
        ring_window=ring_window,
        page_size=page_size,
        max_kept=max(kept),
        softmax_scale=dim**-0.5,
        num_segments=num_segments,
        scratch=scratch,
        score_buffer=score_buffer,
    )
    for row, length in enumerate(kept):
        slots = torch.arange(length, device="cuda")
        physical = table[row, slots // kernel_page].long()
        keys = (
            cache[physical, :, slots % kernel_page, :dim]
            .repeat_interleave(head_ratio, 1)
            .to(dtype)
            .float()
        )
        values = (
            cache[physical, :, slots % kernel_page, dim:]
            .repeat_interleave(head_ratio, 1)
            .to(dtype)
            .float()
        )
        logits = torch.einsum("hd,thd->ht", query[row].float(), keys) * dim**-0.5
        reference = torch.einsum("ht,thd->hd", logits.softmax(-1), values)
        tolerance = 0.003 if dtype == torch.float16 else 0.025
        torch.testing.assert_close(
            output[row].float(), reference, atol=tolerance, rtol=tolerance
        )
        if not due[row]:
            assert not scores[row].any()
            continue
        query_positions = torch.arange(
            sequence_lengths[row] - counts[row], sequence_lengths[row], device="cuda"
        )
        recent = ring[row * ring_window + query_positions % ring_window]
        expected = ngc_page_scores(
            recent,
            cache[..., :dim],
            table[row],
            length,
            dim**-0.5,
            page_size,
            query_positions=query_positions,
            key_positions=positions[row, :length],
        )
        torch.testing.assert_close(
            scores[row, : len(expected)], expected, atol=0.003, rtol=0.003
        )
        assert not scores[row, len(expected) :].any()
    assert not scores[2].any(), "A sole retained token is masked from its own score"
    # Reuse dirty scratch after shortening the first row and disabling scoring.
    pointers = {name: value.data_ptr() for name, value in scratch.buffers.items()}
    descriptors[0, 3] = 17
    due.zero_()
    scores.fill_(float("nan"))
    replay = ngc_decode_and_score(
        query,
        cache,
        ring,
        descriptors,
        positions,
        table,
        due,
        output,
        window=window,
        ring_window=ring_window,
        page_size=page_size,
        max_kept=max(kept),
        softmax_scale=dim**-0.5,
        num_segments=num_segments,
        scratch=scratch,
        score_buffer=score_buffer,
    )
    assert replay.data_ptr() == score_buffer.data_ptr()
    assert not replay.any()
    assert pointers == {
        name: value.data_ptr() for name, value in scratch.buffers.items()
    }
    slots = torch.arange(17, device="cuda")
    physical = table[0, slots // kernel_page].long()
    keys = (
        cache[physical, :, slots % kernel_page, :dim]
        .repeat_interleave(head_ratio, 1)
        .to(dtype)
        .float()
    )
    values = (
        cache[physical, :, slots % kernel_page, dim:]
        .repeat_interleave(head_ratio, 1)
        .to(dtype)
        .float()
    )
    logits = torch.einsum("hd,thd->ht", query[0].float(), keys) * dim**-0.5
    reference = torch.einsum("ht,thd->hd", logits.softmax(-1), values)
    torch.testing.assert_close(
        output[0].float(), reference, atol=tolerance, rtol=tolerance
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("num_segments", [16, 64])
def test_ngc_fused_decode_unified_numerics(num_segments, record_property):
    """Measure rounding against the standard decoder at Qwen's first boundary."""
    from vllm.v1.attention.ops.triton_ngc_decode import ngc_decode_and_score
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    torch.manual_seed(37)
    device = "cuda"
    query = torch.randn(2, 28, 128, device=device, dtype=torch.float16)
    cache = torch.randn(16, 4, 16, 256, device=device, dtype=torch.float16)
    table = torch.arange(16, device=device, dtype=torch.int32).reshape(2, 8)
    lengths = torch.tensor([122, 128], device=device, dtype=torch.int32)
    descriptors = torch.tensor(
        [[122, 0, 1, 122, 0, 0, 0, 0], [128, 1, 1, 128, 0, 0, 0, 1]],
        device=device,
        dtype=torch.int64,
    )
    ring = torch.zeros(11, 28, 128, device=device, dtype=torch.float16)
    ring[1], ring[7] = query[0], query[1]
    positions = torch.arange(128, device=device).expand(2, -1).contiguous()
    due = torch.tensor([False, True], device=device)
    fused, standard = torch.empty_like(query), torch.empty_like(query)
    ngc_decode_and_score(
        query,
        cache,
        ring,
        descriptors,
        positions,
        table,
        due,
        fused,
        window=5,
        ring_window=5,
        page_size=16,
        max_kept=128,
        softmax_scale=128**-0.5,
        num_segments=num_segments,
    )
    unified_attention(
        q=query,
        k=cache[..., :128].transpose(1, 2),
        v=cache[..., 128:].transpose(1, 2),
        out=standard,
        cu_seqlens_q=torch.arange(3, device=device, dtype=torch.int32),
        max_seqlen_q=1,
        seqused_k=lengths,
        max_seqlen_k=128,
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=8,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=torch.empty(2, 28, num_segments, 128, device=device),
        softmax_segm_max=torch.empty(2, 28, num_segments, device=device),
        softmax_segm_expsum=torch.empty(2, 28, num_segments, device=device),
    )
    error = (fused.float() - standard.float()).abs()
    record_property("fused_unified_max_abs", error.max().item())
    record_property(
        "fused_unified_equal_fraction", (fused == standard).float().mean().item()
    )
    torch.testing.assert_close(fused, standard, atol=0.003, rtol=0.003)


def _defer_test_scores(buffers, req_id, layer, scores, rate=0.5, **overrides):
    kwargs = dict(
        group_id=int(layer[-1]),
        computed_tokens=64,
        cadence=64,
        eviction_rate=rate,
        min_retained_tokens=16,
        key_cache=torch.empty(0),
        kernel_block_table=torch.arange(len(scores), device=scores.device),
        block_ids=torch.arange(len(scores), device=scores.device) + 100,
        key_positions=torch.arange(len(scores) * 16, device=scores.device),
        page_size=16,
        softmax_scale=1.0,
        precomputed_scores=scores,
        defer_selection=True,
    )
    kwargs.update(overrides)
    return buffers.score_if_due(req_id, layer, **kwargs)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("debug", [False, True])
def test_deferred_selection_matches_scalar_ties_rates_and_variable_lengths(
    debug, device, monkeypatch
):
    """One handoff preserves scalar ranking and request/layer proposal order."""
    layers = ("layer.0", "layer.1")
    scalar, batched = (_ring_buffers(2, layers=layers) for _ in range(2))
    for buffers in (scalar, batched):
        buffers.include_debug_scores = debug
        _step(buffers, [("a", 0, 64), ("b", 0, 64)])
    inputs = [
        ("a", "layer.0", [1.0, 1.0, 0.0, 0.0], 0.5),
        ("a", "layer.1", [2.0, 2.0, 2.0, 0.0], 0.75),
        ("b", "layer.0", [2.0, 1.0, 0.0], 0.75),
        ("b", "layer.1", [1.0, 1.0, 0.0], 0.0),
    ]
    for req, layer, values, rate in inputs:
        scores = torch.tensor(values, device=device)
        _defer_test_scores(scalar, req, layer, scores, rate, defer_selection=False)
        assert _defer_test_scores(batched, req, layer, scores, rate) is None
    copies = []
    original = batched._copy_handoff

    def handoff(tensors):
        copies.append(len(tensors))
        return original(tensors)

    monkeypatch.setattr(batched, "_copy_handoff", handoff)
    versions = {"a": 11, "b": 22}
    expected = scalar.take_proposals(versions, list(layers))
    if not debug:
        expected = [p for p in expected if p.victim_indices]
    assert batched.take_proposals(versions, list(layers)) == expected
    assert len(copies) == 1
    assert not batched.take_proposals(versions, list(layers))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_deferred_zero_rate_still_validates_finite_scores(invalid):
    buffers = _ring_buffers(2)
    _step(buffers, [("a", 0, 64)])
    scores = torch.tensor([invalid, 1.0, 2.0, 0.0])
    _defer_test_scores(buffers, "a", "layer.0", scores, rate=0)
    with pytest.raises(ValueError, match="finite"):
        buffers.take_proposals({"a": 1}, ["layer.0"])
    scores[0] = 0
    assert not buffers.take_proposals({"a": 1}, ["layer.0"])


@pytest.mark.parametrize("invalidate", ["capture", "discard"])
def test_deferred_borrowed_scores_do_not_survive_request_invalidation(invalidate):
    buffers = _ring_buffers(2)
    _step(buffers, [("a", 0, 64)])
    scores = torch.tensor([1.0, 2.0, 3.0, 0.0])
    _defer_test_scores(buffers, "a", "layer.0", scores)
    scores.fill_(float("nan"))
    if invalidate == "capture":
        _step(buffers, [("a", 64, 1)])
    else:
        buffers.discard_request("a")
    assert not buffers.take_proposals({"a": 1}, ["layer.0"])


def test_ngc_precomputed_scores_skip_standalone_scorer():
    """Fused scores still rank pages and commit cadence without scoring again."""
    buffers = _ring_buffers(2)
    _step(buffers, [("a", 62, 2)])
    scores = torch.tensor([3.0, 1.0, 2.0, 0.0])

    def forbidden_scorer(*args, **kwargs):
        raise AssertionError("Precomputed scores must bypass the standalone scorer")

    args = dict(
        group_id=0,
        computed_tokens=128,
        cadence=128,
        eviction_rate=0.5,
        min_retained_tokens=16,
        key_cache=torch.empty(0),
        kernel_block_table=torch.arange(4),
        block_ids=torch.tensor([8, 3, 7, 9]),
        key_positions=torch.arange(64),
        page_size=16,
        softmax_scale=0.5,
        score_fn=forbidden_scorer,
        precomputed_scores=scores,
    )
    decision = buffers.score_if_due("a", "layer.0", **args)
    assert decision is not None
    assert decision.victim_indices.tolist() == [1]
    assert decision.victim_block_ids.tolist() == [3]
    torch.testing.assert_close(decision.page_scores.scores, scores)
    scores.zero_()
    assert decision.page_scores.scores.tolist() == [3.0, 1.0, 2.0, 0.0]
    assert buffers.score_if_due("a", "layer.0", **args) is None


def test_ngc_scratch_reuses_capacity_and_grows_without_aliasing_names():
    """Serial layers reuse flat storage; growth preserves independent workspaces."""
    from vllm.v1.worker.ngc_scoring import NGCScratchBuffers

    scratch = NGCScratchBuffers()
    device = torch.device("cpu")
    first = scratch.get("partials", (3, 5), device)
    first.fill_(7)
    assert first.dtype == torch.float32 and first.is_contiguous()
    assert first.untyped_storage().nbytes() == 16 * first.element_size()
    smaller = scratch.get("partials", (2, 4), device)
    assert smaller.data_ptr() == first.data_ptr()
    assert smaller.shape == (2, 4) and smaller.is_contiguous()
    assert smaller.eq(7).all()
    independent = scratch.get("segment_sum", (2, 4), device)
    independent.fill_(9)
    assert independent.data_ptr() != first.data_ptr()
    grown = scratch.get("partials", (5, 7), device)
    assert grown.data_ptr() != first.data_ptr()
    assert grown.untyped_storage().nbytes() == 64 * grown.element_size()
    grown.fill_(3)
    assert independent.eq(9).all()
    reused = scratch.get("partials", (4, 8), device)
    assert reused.data_ptr() == grown.data_ptr()
    assert reused.eq(3).all()


def test_ngc_fused_layers_share_scratch_but_preserve_scores(monkeypatch):
    """The next serial layer cannot overwrite scores awaiting eviction handoff."""
    import sys
    from types import SimpleNamespace

    buffers = _ring_buffers(2, layers=("first", "second"))
    shared = []

    def decode(*args, scratch, score_buffer, **kwargs):
        shared.append(scratch)
        # Scratch is deliberately overwritten on each serial invocation.
        scratch.get("partials", (3, 5), torch.device("cpu")).fill_(len(shared))
        score_buffer.fill_(len(shared))
        return score_buffer

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.ops.triton_ngc_decode",
        SimpleNamespace(ngc_decode_and_score=decode),
    )
    for layer in ("first", "second"):
        buffers._fused_decode[layer] = SimpleNamespace(
            inputs=[
                SimpleNamespace(
                    retained_tokens=32,
                    page_size=16,
                    block_table=torch.zeros((1, 2), dtype=torch.int32),
                )
            ],
            descriptors=torch.zeros((1, 8), dtype=torch.int64),
            workspace=SimpleNamespace(key_positions=torch.arange(32).view(1, -1)),
            due=torch.tensor([True]),
            scores=None,
        )
    query, output = torch.ones((1, 1, 1)), torch.empty((1, 1, 1))
    assert buffers.fused_decode("first", query, torch.empty(0), output, 1.0)
    first = buffers._fused_decode["first"].scores
    assert buffers.fused_decode("second", query, torch.empty(0), output, 1.0)
    second = buffers._fused_decode["second"].scores
    assert first.data_ptr() != second.data_ptr()
    assert first.eq(1).all() and second.eq(2).all()
    assert shared[0] is shared[1]
    assert buffers.fused_decode("first", query, torch.empty(0), output, 1.0)
    assert buffers._fused_decode["first"].scores.data_ptr() == first.data_ptr()
    assert second.eq(2).all()


def _mock_ngc_graph_streams(monkeypatch):
    import contextlib

    events = []

    class Stream:
        def wait_stream(self, other):
            events.append((self, other))

    caller = Stream()
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: caller)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    return caller, events


def test_ngc_boundary_graph_replays_contents_and_invalidates_layout(monkeypatch):
    """Persistent values replay; storage, layout and scalar changes recapture."""
    import contextlib
    import gc
    import weakref

    from vllm.v1.worker.ngc_cudagraph import NGCBoundaryGraph

    events = []

    class Graph:
        def replay(self):
            events.append("replay")

    @contextlib.contextmanager
    def capture(graph, stream):
        events.append("capture")
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    caller, waits = _mock_ngc_graph_streams(monkeypatch)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))
    cache = NGCBoundaryGraph()
    tensor = torch.ones((2, 3))
    state = {"input": tensor, "bucket": 8}
    forwards = []

    def forward():
        forwards.append(1)
        return {"output": state["input"]}

    dependencies = lambda: state.copy()
    output = cache.run(forward, dependencies)
    assert len(forwards) == 2 and cache.captures == 1 and cache.replays == 0
    tensor.fill_(4)
    assert cache.run(forward, dependencies) is output
    assert len(forwards) == 2 and cache.replays == 1
    state["input"] = tensor.view(3, 2)
    cache.run(forward, dependencies)
    assert cache.captures == 2
    state["input"] = state["input"].clone()
    cache.run(forward, dependencies)
    assert cache.captures == 3
    state["bucket"] = 16
    cache.run(forward, dependencies)
    assert cache.captures == 4
    assert events.count("capture") == 4 and events.count("sync") == 4
    assert events.count("replay") == 5
    assert waits == [(cache.stream, caller), (caller, cache.stream)] * 4
    # The single cached entry owns dependencies until explicitly discarded.
    reference = weakref.ref(state["input"])
    state.clear()
    gc.collect()
    assert reference() is not None
    cache.clear()
    gc.collect()
    assert reference() is None
    assert cache.graph is cache.output is cache.dependencies is cache.key is None
    assert cache.captures == 4 and cache.replays == 1


def test_ngc_boundary_graph_rejects_storage_change_during_capture(monkeypatch):
    """Warmup may grow buffers, but capture cannot change their addresses."""
    import contextlib

    from vllm.v1.worker.ngc_cudagraph import NGCBoundaryGraph

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    _mock_ngc_graph_streams(monkeypatch)
    monkeypatch.setattr(
        torch.cuda, "graph", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    cache = NGCBoundaryGraph()
    values = [torch.ones(2)]

    def forward():
        values[0] = values[0].clone()
        return values[0]

    with pytest.raises(RuntimeError, match="changed during capture"):
        cache.run(forward, lambda: values[0])
    assert cache.graph is None and cache.captures == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_ngc_boundary_kernel_graph_reads_updated_persistent_inputs():
    """Replay observes new due flags, lengths and tables without new addresses."""
    from vllm.v1.attention.ops.triton_ngc_decode import ngc_decode_and_score
    from vllm.v1.worker.ngc_scoring import NGCScratchBuffers

    query = torch.randn((1, 7, 64), dtype=torch.float16, device="cuda")
    cache = torch.randn((4, 1, 16, 128), dtype=torch.float16, device="cuda")
    ring = torch.randn((6, 7, 64), dtype=torch.float16, device="cuda")
    descriptors = torch.tensor([[33, 0, 5, 33, 0, 0, 0, 0]], device="cuda")
    positions = torch.arange(64, device="cuda").view(1, -1)
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    due = torch.tensor([True], device="cuda")
    scratch = NGCScratchBuffers()
    output = torch.empty_like(query)
    scores = torch.empty((1, 4), device="cuda")

    def run(target, storage, destination):
        return ngc_decode_and_score(
            query,
            cache,
            ring,
            descriptors,
            positions,
            table,
            due,
            target,
            window=5,
            ring_window=5,
            page_size=16,
            max_kept=64,
            softmax_scale=0.125,
            scratch=storage,
            score_buffer=destination,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(output, scratch, scores)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(output, scratch, scores)
    for active, kept in ((False, 17), (True, 33)):
        due.fill_(active)
        descriptors[0, 3] = kept
        table.copy_(table.flip(1))
        graph.replay()
        expected_output = torch.empty_like(output)
        expected_scores = torch.empty_like(scores)
        run(expected_output, NGCScratchBuffers(), expected_scores)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(scores, expected_scores)


def test_ngc_boundary_graph_declined_fusion_returns_single_warm_forward(monkeypatch):
    """A backend fallback must not capture or repeat its ordinary forward."""
    from vllm.v1.worker.ngc_cudagraph import NGCBoundaryGraph

    calls = []
    output = torch.ones(2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Declined fusion must not touch CUDA graph APIs")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    caller, waits = _mock_ngc_graph_streams(monkeypatch)
    monkeypatch.setattr(torch.accelerator, "synchronize", forbidden)
    cache = NGCBoundaryGraph()

    def forward():
        calls.append(1)
        return output

    assert cache.run(forward, lambda: output, capture_ready=lambda: False) is output
    assert len(calls) == 1
    assert cache.graph is cache.dependencies is cache.output is None
    assert cache.captures == cache.replays == 0
    assert waits == [(cache.stream, caller), (caller, cache.stream)]


def test_ngc_reserved_score_buffer_is_not_published_before_computation():
    """Reserved output storage is not a completed fused result."""
    from types import SimpleNamespace

    buffers = NGCScoringBuffers()
    reserved = torch.full((1, 2), float("nan"))
    buffers._fused_decode["layer"] = SimpleNamespace(
        inputs=[SimpleNamespace(req_id="a", retained_tokens=32, page_size=16)],
        score_buffer=reserved,
        scores=None,
    )
    buffers._fused_rows["a", "layer"] = 0
    assert buffers.get_fused_scores("a", "layer") is None
    assert not buffers.fused_graph_ready()
    # A successful replay publishes its persistent output without a Python kernel call.
    reserved.fill_(2)
    buffers.publish_fused_graph_scores()
    torch.testing.assert_close(buffers.get_fused_scores("a", "layer"), reserved[0])
    assert buffers.fused_graph_ready()


@pytest.mark.parametrize("num_reqs", [0, 2])
def test_ngc_retained_lengths_share_full_vectors_without_aliasing_backing_rows(
    num_reqs,
):
    """Equal maxima alone cannot merge layers, and later divergence splits views."""
    from vllm.v1.worker.ngc_scoring import share_ngc_retained_lengths

    retained = CpuGpuBuffer(
        3, 4, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False
    )
    retained.np[:] = [[10, 6, 77, 88], [10, 6, 99, 66], [6, 10, 77, 88]]
    retained.copy_to_gpu()
    before = retained.cpu.clone()
    views = share_ngc_retained_lengths(retained, num_reqs)
    assert views[0] is views[1]
    assert views[0][2] == (10 if num_reqs else 0)
    if num_reqs:
        assert views[0] is not views[2], "The entire vector must match, not its maximum"
        assert views[1][0].data_ptr() == retained.gpu[0].data_ptr()
        assert retained.gpu[0].data_ptr() != retained.gpu[1].data_ptr()
        retained.np[1, 1] = 5
        retained.copy_to_gpu()
        split = share_ngc_retained_lengths(retained, num_reqs)
        assert split[0] is not split[1]
        assert split[1][0].data_ptr() == retained.gpu[1].data_ptr()
        assert split[0][1].tolist() == [10, 6]
        assert split[1][1].tolist() == [10, 5]
        retained.np[1, 1] = 6
        retained.copy_to_gpu()
        assert share_ngc_retained_lengths(retained, num_reqs)[0][1].tolist() == [10, 6]
    torch.testing.assert_close(retained.cpu, before)
