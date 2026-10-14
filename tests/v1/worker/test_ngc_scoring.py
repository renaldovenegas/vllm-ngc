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
            computed_generated_tokens=128,
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


def _ring_buffers(window, layers=("layer.0",), num_slots=4):
    buffers = NGCScoringBuffers()
    buffers.allocate(
        {name: (window, 1, 1) for name in layers},
        num_slots=num_slots,
        max_num_tokens=512,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return buffers


def _step(buffers, chunks, queries=None, prompt_lens=None):
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
        prompt_lens,
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
        generated = length - 7
        pages = torch.arange((length + 15) // 16)
        buffers.score_if_due(
            "a",
            "layer.0",
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
    assert buffers.any_due("a", 128, 128)
    assert not buffers.any_due("a", 200, 128)
    assert buffers.any_due("a", 256, 128) and buffers.any_due("b", 128, 128)
    buffers.discard_request("a", keep_cadence=True)
    assert not buffers.any_due("a", 256, 128)
    buffers.discard_request("a")
    assert buffers.any_due("a", 128, 128)


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
def test_ngc_shared_metadata_matches_build_with_independent_cache(max_query_len):
    """Reusing batch fields preserves per-layer lengths, tables, and scratch."""
    from dataclasses import fields

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    def make_builder():
        builder = object.__new__(TritonAttentionMetadataBuilder)
        builder.seq_threshold_3D = 8
        builder.num_par_softmax_segments = 16
        builder.softmax_segm_output = torch.empty(2)
        builder.softmax_segm_max = torch.empty(2)
        builder.softmax_segm_expsum = torch.empty(2)
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
            assert actual_value is expected_value
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
    _step(buffers, [("a", 0, 15)], prompt_lens=[10])
    assert not buffers._slots
    assert all(not ring.any() for ring in buffers._rings.values())
    for position in (15, 16, 17):
        _step(buffers, [("a", position, 1)], prompt_lens=[10])
    assert _queries(buffers, "a", "short") == ([15, 16, 17], [15, 16, 17])
    assert not buffers._rings["long"].any()
    assert buffers.any_due("a", 8, 8)
    previous = buffers._rings["short"].clone()
    _step(buffers, [("a", 18, 1)], prompt_lens=[10])
    torch.testing.assert_close(buffers._rings["short"], previous)
    _step(buffers, [("b", 0, 29)], prompt_lens=[10])
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
        np.arange(15, 18),
        [18],
        [10],
    )
    query[:3] = torch.arange(15, 18, device="cuda")[:, None, None]
    graph.replay()
    actual, positions = buffers.get_queries("a", "layer")
    torch.testing.assert_close(
        actual, positions[:, None, None].expand_as(actual).float()
    )
    assert buffers.any_due("a", 8, 8)
    previous = buffers._rings["layer"].clone()
    buffers.prepare_capture(
        ["a"], np.array([0]), np.array([0]), np.array([1]), np.array([18]), [19], [10]
    )
    query.fill_(-1)
    graph.replay()
    torch.testing.assert_close(buffers._rings["layer"], previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("kernel_page_size", [8, 16])
def test_ngc_packed_score_inputs_match_individual_gathers(kernel_page_size):
    """Packed gathers preserve per-layer windows and original retained positions."""
    from vllm.v1.worker.ngc_scoring import NGCScoreInput

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
        NGCScoreInput(req, name, row, kept, evicted, 16, kernel_page_size, known, table)
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
