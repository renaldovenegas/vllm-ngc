# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NGC component sweep with matched correctness, CUPTI timing and CPU staging."""

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import itertools
import json
import statistics
import time
import warnings
from pathlib import Path


def load_gpu_modules():
    global torch, decode, inputs_module, scorer, unified_attention
    global NGCScoringBuffers, NGCScratchBuffers
    import torch

    from vllm.v1.attention.ops import triton_ngc_decode as decode
    from vllm.v1.attention.ops import triton_ngc_inputs as inputs_module
    from vllm.v1.attention.ops import triton_ngc_scorer as scorer
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.worker.ngc_scoring import NGCScoringBuffers, NGCScratchBuffers


def validate_config(args):
    """Validate sweep dimensions before any CUDA imports or remote submission."""
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    shapes = [tuple(map(int, x.split(":"))) for x in args.shapes.split(",")]
    segments = [int(x) for x in args.segments.split(",")]
    tiles = [int(x) for x in args.tiles.split(",")]
    if (
        min([*batches, *contexts, args.repeat, args.trials, args.cpu_repeats]) <= 0
        or min(contexts) < 5
    ):
        raise ValueError("Positive dimensions/repeats, contexts>=5 required")
    if any(len(v) != 3 or min(v) <= 0 or v[0] % v[1] for v in shapes):
        raise ValueError(
            "Shapes require query_heads:kv_heads:head_dim with divisible heads"
        )
    if any(v <= 0 or v & (v - 1) for v in segments + tiles):
        raise ValueError("Segments and tiles must be positive powers of two")
    if not set(args.due.split(",")) <= {"all", "half", "none"} or not set(
        args.lengths.split(",")
    ) <= {"uniform", "uneven"}:
        raise ValueError("Invalid due/length modes")
    return batches, contexts, shapes


class Replay:
    """Replay captured launches with rotated storage preserving tensor aliases."""

    def __init__(self, calls, zeros=()):
        self.calls = calls
        self.zeros = zeros
        self.inputs = []
        self.indices = {}
        for value in [
            *(v for call in calls for v in (*call["args"], *call["kwargs"].values())),
            *zeros,
        ]:
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                if key not in self.indices:
                    self.indices[key] = len(self.inputs)
                    self.inputs.append(
                        value.as_strided(
                            (storage.nbytes() // value.element_size(),), (1,), 0
                        )
                    )

    def __call__(self, *rotated):
        def replace(value):
            if not isinstance(value, torch.Tensor):
                return value
            key = (str(value.device), value.untyped_storage().data_ptr())
            base = rotated[self.indices[key]]
            if base.dtype != value.dtype:
                base = base.view(value.dtype)
            return base.as_strided(value.shape, value.stride(), value.storage_offset())

        for value in self.zeros:
            replace(value).zero_()
        for call in self.calls:
            call["launch"](
                *(replace(v) for v in call["args"]),
                **{k: replace(v) for k, v in call["kwargs"].items()},
            )

    def original(self):
        return self(*self.inputs)


def latency(replay, repeat):
    """CUPTI graph timing in microseconds; every actual tensor storage rotates."""
    if not replay.inputs:
        raise ValueError("Cannot time an empty operation")
    from flashinfer.testing import bench_gpu_time_with_cupti

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        samples = bench_gpu_time_with_cupti(
            replay,
            input_args=tuple(replay.inputs),
            use_cuda_graph=True,
            cold_l2_cache=True,
            dry_run_iters=5,
            repeat_iters=repeat,
        )
    for warning in emitted:
        if any(
            word in str(warning.message).lower()
            for word in ("fallback", "falling back", "disabled")
        ):
            raise RuntimeError(f"CUPTI/cold-L2 unavailable: {warning.message}")
    return statistics.median(samples) * 1000


@contextlib.contextmanager
def launches(module, names):
    """Record preallocated production launches for stage-only replay."""
    recorded = []
    originals = {name: getattr(module, name) for name in names}

    class Recorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            launch = originals[self.name][grid]

            def run(*args, **kwargs):
                compiled = launch(*args, **kwargs)
                recorded.append(
                    {
                        "name": self.name,
                        "call": functools.partial(launch, *args, **kwargs),
                        "args": args,
                        "kwargs": kwargs,
                        "launch": launch,
                        "resources": {
                            "registers": getattr(compiled, "n_regs", None),
                            "spills": getattr(compiled, "n_spills", None),
                            "shared_memory_bytes": getattr(
                                getattr(compiled, "metadata", None), "shared", None
                            ),
                        },
                    }
                )
                return compiled

            return run

    try:
        for name in names:
            setattr(module, name, Recorder(name))
        yield recorded
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def cpu_latency(fn, repeats):
    samples = []
    fn()
    torch.accelerator.synchronize()
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return {"samples_us": samples, "median_us": statistics.median(samples)}


def run_case(batch, context, shape, uneven, due_mode, args):
    torch.manual_seed(args.seed)
    heads, kv_heads, dim = shape
    page, window = 16, 5
    lengths = [
        max(window, context - ((row % 4) * page + 3 if uneven else 0))
        for row in range(batch)
    ]
    pages = (context + page - 1) // page
    cache = torch.randn(
        batch * pages, kv_heads, page, 2 * dim, device="cuda", dtype=torch.bfloat16
    )
    table = (
        torch.randperm(batch * pages, device="cuda")
        .reshape(batch, pages)
        .to(torch.int32)
    )
    query = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    ring = torch.randn(batch * window + 1, heads, dim, device="cuda", dtype=query.dtype)
    original = torch.arange(context, device="cuda")
    original = (
        original + (original // page) * page
    )  # gaps from prior whole-page eviction
    positions = original.expand(batch, -1).contiguous()
    logical = [int(original[length - 1].item()) + 1 for length in lengths]
    recent_queries, recent_positions = [], []
    for row in range(batch):
        pos = torch.arange(logical[row] - window, logical[row], device="cuda")
        q = torch.randn(window, heads, dim, device="cuda", dtype=query.dtype)
        q[-1] = query[row]
        ring[row * window + pos % window] = q
        recent_queries.append(q)
        recent_positions.append(pos)
    due_host = [
        due_mode == "all" or (due_mode == "half" and row % 2 == 0)
        for row in range(batch)
    ]
    if due_mode == "half" and batch == 1:
        due_host = [False]
    due = torch.tensor(due_host, device="cuda")
    desc = torch.tensor(
        [
            [logical[r], r, window, lengths[r], logical[r] - lengths[r], 0, 0, r]
            for r in range(batch)
        ],
        device="cuda",
        dtype=torch.int64,
    )
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    standard, fused = torch.empty_like(query), torch.empty_like(query)
    scores = torch.empty(batch, pages, device="cuda", dtype=torch.float32)
    scratch = NGCScratchBuffers()
    segments_list = [int(x) for x in args.segments.split(",")]
    tiles = [int(x) for x in args.tiles.split(",")]
    max_segments = max(segments_list)
    seg_out = torch.empty(batch, heads, max_segments, dim, device="cuda")
    seg_max = torch.empty(batch, heads, max_segments, device="cuda")
    seg_sum = torch.empty_like(seg_max)

    def standard_fn(segments):
        unified_attention(
            q=query,
            k=cache[..., :dim].transpose(1, 2),
            v=cache[..., dim:].transpose(1, 2),
            out=standard,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=lens,
            max_seqlen_k=context,
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=batch + 1 if segments > 1 else 0,
            num_par_softmax_segments=segments,
            softmax_segm_output=seg_out[:, :, :segments].contiguous(),
            softmax_segm_max=seg_max[:, :, :segments].contiguous(),
            softmax_segm_expsum=seg_sum[:, :, :segments].contiguous(),
        )

    # Preallocate each segment configuration, including contiguous launch views.
    standard_arms = {}
    standard_launches = {}
    for segments in segments_list:
        with launches(
            __import__(
                "vllm.v1.attention.ops.triton_unified_attention", fromlist=["x"]
            ),
            ["kernel_unified_attention", "reduce_segments"],
        ) as calls:
            standard_fn(segments)
        standard_arms[f"standard_s{segments}"] = Replay(calls)
        standard_launches[f"standard_s{segments}"] = calls
    standard_arms[next(iter(standard_arms))].original()
    expected_decode = standard.clone()
    # Independent FP32 reference for decode, per request to bound reference memory.
    for row, length in enumerate(lengths):
        physical = table[row, torch.arange(length, device="cuda") // page].long()
        slots = torch.arange(length, device="cuda") % page
        keys = (
            cache[physical, :, slots, :dim]
            .repeat_interleave(heads // kv_heads, dim=1)
            .float()
        )
        values = (
            cache[physical, :, slots, dim:]
            .repeat_interleave(heads // kv_heads, dim=1)
            .float()
        )
        probabilities = torch.softmax(
            torch.einsum("hd,nhd->hn", query[row].float(), keys) * dim**-0.5, -1
        )
        expected = torch.einsum("hn,nhd->hd", probabilities, values)
        torch.testing.assert_close(
            expected_decode[row].float(), expected, atol=0.025, rtol=0.025
        )
    standalone_calls = []
    standalone_outputs = []
    for row in range(batch):
        if not due_host[row]:
            continue
        with launches(scorer, ["ngc_scorer_kernel"]) as calls:
            result = scorer.ngc_page_scores(
                recent_queries[row],
                cache[..., :dim],
                table[row],
                lengths[row],
                dim**-0.5,
                page,
                query_positions=recent_positions[row],
                key_positions=positions[row, : lengths[row]],
            )
        standalone_calls.extend(calls)
        standalone_outputs.append((row, result))

    def standalone():
        for _, out in standalone_outputs:
            out.zero_()  # atomic score accumulation requires reset on every replay
        for call in standalone_calls:
            call["call"]()

    standalone_replay = Replay(standalone_calls, [out for _, out in standalone_outputs])
    standalone()
    expected_scores = {row: out.clone() for row, out in standalone_outputs}
    for row, reference in expected_scores.items():
        length = lengths[row]
        indices = torch.arange(length, device="cuda")
        keys = cache[table[row, indices // page].long(), :, indices % page, :dim]
        keys = keys.repeat_interleave(heads // kv_heads, dim=1).float()
        logits = (
            torch.einsum("qhd,nhd->qhn", recent_queries[row].float(), keys) * dim**-0.5
        )
        allowed = (
            positions[row, :length][None, None, :]
            < recent_positions[row][:, None, None]
        )
        masked = logits.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(
            torch.where(allowed.any(-1, keepdim=True), masked, 0.0), dim=-1
        )
        weights = torch.where(allowed, weights, 0.0)
        mass = weights.sum((0, 1))
        padded = torch.nn.functional.pad(mass, (0, (-length) % page))
        torch.testing.assert_close(
            reference, padded.reshape(-1, page).sum(1), atol=0.003, rtol=0.003
        )
    arms = dict(standard_arms)
    if standalone_calls:
        arms["standalone_scoring"] = standalone_replay
    fused_launches = {}
    for segments, tile in itertools.product(segments_list, tiles):
        name = f"fused_s{segments}_t{tile}"

        def invoke(s=segments, t=tile):
            return decode.ngc_decode_and_score(
                query,
                cache,
                ring,
                desc,
                positions,
                table,
                due,
                fused,
                window=window,
                ring_window=window,
                page_size=page,
                max_kept=context,
                softmax_scale=dim**-0.5,
                num_segments=s,
                tile_size=t,
                scratch=scratch,
                score_buffer=scores,
            )

        with launches(
            decode, ["_ngc_decode_score_partials", "_ngc_decode_score_reduce"]
        ) as calls:
            invoke()
        torch.testing.assert_close(fused, expected_decode, atol=0.025, rtol=0.025)
        for row in range(batch):
            n = (lengths[row] + page - 1) // page
            if due_host[row]:
                torch.testing.assert_close(
                    scores[row, :n], expected_scores[row], atol=0.003, rtol=0.003
                )
            else:
                assert torch.count_nonzero(scores[row]) == 0
        fused_launches[name] = calls
        arms[name] = Replay(calls)
    for fn in standard_arms.values():
        fn.original()
        torch.testing.assert_close(standard, expected_decode, atol=0.025, rtol=0.025)
    if standalone_calls:
        for name in standard_arms:
            arms[name + "_plus_scoring"] = Replay(
                standard_launches[name] + standalone_calls,
                [out for _, out in standalone_outputs],
            )
    if args.counters:
        import nvtx

        timing = next(
            c
            for c in args.counter_cases
            if c["batch"] == batch
            and c["max_context"] == context
            and tuple(c["shape"]) == shape
            and c["uneven"] == uneven
            and c["due_mode"] == due_mode
        )
        with nvtx.annotate("NGCKernelCounters"):
            arms[timing["best_standard"]].original()
            arms[timing["best_fused"]].original()
            standalone()
        torch.accelerator.synchronize()
        return {
            "batch": batch,
            "max_context": context,
            "shape": shape,
            "uneven": uneven,
            "due_mode": due_mode,
            "counters_only": True,
            "best_standard": timing["best_standard"],
            "best_fused": timing["best_fused"],
        }
    samples = {name: [] for name in arms}
    for fn in arms.values():
        fn.original()
    torch.accelerator.synchronize()
    # Exclude CUPTI initialization/calibration and alternate order across trials.
    for fn in arms.values():
        latency(fn, args.repeat)
    for trial in range(args.trials):
        order = list(arms.items())
        for name, fn in order if trial % 2 == 0 else reversed(order):
            samples[name].append(latency(fn, args.repeat))
    best_fused = min(fused_launches, key=lambda n: statistics.median(samples[n]))
    best_standard = min(standard_arms, key=lambda n: statistics.median(samples[n]))
    best_separate = (
        min(
            (n for n in arms if n.endswith("_plus_scoring")),
            key=lambda n: statistics.median(samples[n]),
        )
        if standalone_calls
        else best_standard
    )
    stages = []
    for entry in [*standard_launches[best_standard], *fused_launches[best_fused]]:
        stages.append(
            {
                "kernel": entry["name"],
                "resources": entry["resources"],
                "microseconds": latency(Replay([entry]), args.repeat),
            }
        )
    # Component preparation: CPU staging is intentionally timed separately.
    buffers = NGCScoringBuffers()
    buffers.allocate(
        {"layer": (window, heads, dim)},
        num_slots=batch,
        max_num_tokens=batch,
        dtype=query.dtype,
        device=query.device,
        pin_memory=True,
    )
    buffers._rings["layer"].copy_(ring)
    input_objects = []
    for row in range(batch):
        req = str(row)
        buffers._slot(req)
        buffers._seq_lens[req] = logical[row]
        page_positions = tuple(
            range(0, ((lengths[row] + page - 1) // page) * 2 * page, 2 * page)
        )
        input_objects.append(
            buffers.update_score_input(
                req,
                "layer",
                row,
                lengths[row],
                logical[row] - lengths[row],
                page,
                page,
                page_positions,
                table,
            )
        )
    with launches(inputs_module, ["prepare_ngc_score_inputs"]) as prep_calls:
        prepared = buffers.prepare_score_inputs(input_objects)
    for row in range(batch):
        torch.testing.assert_close(
            prepared[str(row), "layer"].key_positions, positions[row, : lengths[row]]
        )
        torch.testing.assert_close(
            prepared[str(row), "layer"].queries, recent_queries[row]
        )
    prep_us = latency(Replay(prep_calls), args.repeat)
    unchanged = cpu_latency(
        lambda: buffers.prepare_score_inputs(input_objects), args.cpu_repeats
    )

    def changed():
        for item in input_objects:
            shift = page if item.page_positions[0] == 0 else -page
            item.page_positions = tuple(p + shift for p in item.page_positions)
        buffers.prepare_score_inputs(input_objects)

    changed_time = cpu_latency(changed, args.cpu_repeats)
    captures = {}
    for active in (True, False):
        scratch_row = batch * window
        rows = torch.arange(batch, device="cuda", dtype=torch.int64) * window
        if not active:
            rows.fill_(scratch_row)
        destination = ring.clone()
        before = destination.clone()

        def capture(rows=rows, destination=destination, scratch_row=scratch_row):
            inputs_module.capture_ngc_queries[(batch, (heads * dim + 255) // 256)](
                query,
                rows,
                destination,
                query.stride(0),
                scratch_row,
                WIDTH=heads * dim,
                BLOCK=256,
            )

        with launches(inputs_module, ["capture_ngc_queries"]) as capture_calls:
            capture()
        if active:
            torch.testing.assert_close(
                destination[torch.arange(batch, device="cuda") * window], query
            )
        else:
            torch.testing.assert_close(destination, before)
        captures[str(active)] = latency(Replay(capture_calls), args.repeat)
    selection = []
    for rate in (0, 0.25, 0.5, 0.75):
        groups = {}
        for row, length in enumerate(lengths):
            if due_host[row]:
                groups.setdefault((length + page - 1) // page, []).append(row)
        matrices = [
            (
                torch.stack([expected_scores[row] for row in rows]),
                min(
                    int(rate * max(n - 1, 0)),
                    max((min(lengths[r] for r in rows) - 32) // page, 0),
                ),
            )
            for n, rows in groups.items()
        ]

        def select(matrices=matrices):
            return [
                torch.argsort(matrix[:, :-1], dim=1, stable=True)[:, :count]
                for matrix, count in matrices
                if count
            ]

        chosen = select()
        for actual, (matrix, count) in zip(chosen, [(m, c) for m, c in matrices if c]):
            expected = torch.tensor(
                [
                    sorted(range(matrix.shape[1] - 1), key=lambda i: float(row[i]))[
                        :count
                    ]
                    for row in matrix.cpu()
                ],
                device="cuda",
            )
            torch.testing.assert_close(actual, expected)
        selection_calls = []
        for matrix, count in matrices:
            if count:
                values = torch.empty_like(matrix[:, :-1])
                indices = torch.empty(
                    matrix.shape[0],
                    matrix.shape[1] - 1,
                    device=matrix.device,
                    dtype=torch.int64,
                )

                def sort(matrix, values, indices):
                    torch.sort(
                        matrix[:, :-1], dim=1, stable=True, out=(values, indices)
                    )

                selection_calls.append(
                    dict(launch=sort, args=(matrix, values, indices), kwargs={})
                )
        selection.append(
            {
                "rate": rate,
                "microseconds": latency(Replay(selection_calls), args.repeat)
                if selection_calls
                else None,
                "scope": (
                    "Prepacked compatible-row stable sort; excludes CPU grouping and "
                    "handoff"
                ),
            }
        )
    kv_bytes = sum(lengths) * kv_heads * dim * 2 * query.element_size()
    return {
        "batch": batch,
        "max_context": context,
        "lengths": lengths,
        "shape": shape,
        "uneven": uneven,
        "due_mode": due_mode,
        "due_requests": sum(due_host),
        "standalone_scoring_status": "timed"
        if standalone_calls
        else "skipped_no_due_rows",
        "page_size": page,
        "query_window": window,
        "dtype": "bfloat16",
        "trials_us": samples,
        "medians_us": {n: statistics.median(s) for n, s in samples.items()},
        "best_standard": best_standard,
        "best_fused": best_fused,
        "best_separate": best_separate,
        "fused_vs_separate_speedup": statistics.median(samples[best_separate])
        / statistics.median(samples[best_fused]),
        "minimum_kv_read_bytes": kv_bytes,
        "best_standard_effective_GB_s": kv_bytes
        / statistics.median(samples[best_standard])
        / 1000,
        "best_fused_effective_GB_s": kv_bytes
        / statistics.median(samples[best_fused])
        / 1000,
        "stage_timings": stages,
        "input_preparation_gpu_us": prep_us,
        "input_preparation_unchanged_wall": unchanged,
        "input_preparation_changed_wall": changed_time,
        "query_capture_kernel_us": captures,
        "victim_selection": selection,
        "correctness": (
            "FP32 decode reference; fused vs standalone score atol/rtol0.003; "
            "metadata/capture exact; stable-sort reference"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,48")
    parser.add_argument("--contexts", default="128,512,2048,8192")
    parser.add_argument("--shapes", default="12:2:128")
    parser.add_argument("--due", default="all,half,none")
    parser.add_argument("--lengths", default="uniform,uneven")
    parser.add_argument("--segments", default="1,4,16")
    parser.add_argument("--tiles", default="16,64")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--cpu-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--counter-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.counters:
        if args.counter_config is None:
            parser.error(
                "Counter pass requires a completed --counter-config timing report"
            )
        args.counter_cases = json.loads(args.counter_config.read_text())["cases"]
    try:
        batches, contexts, shapes = validate_config(args)
    except ValueError as error:
        parser.error(str(error))
    load_gpu_modules()
    versions = {"torch": torch.__version__}
    for package in ("triton", "flashinfer-python", "cupti-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if not args.counters and (
        versions["cupti-python"] != "13.0.1"
        or not torch.version.cuda
        or torch.version.cuda.split(".")[0] != "13"
    ):
        raise RuntimeError("Timing requires CUDA13 with cupti-python13.0.1")
    report = {
        "software": versions,
        "accelerator_type": torch.accelerator.current_accelerator().type,
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "counter_cases"
        },
        "cases": [],
        "timing": (
            "CUPTI CUDA graphs coldL2 rotating all actual tensor storages with "
            "alias-preserving views; allocations/compilation excluded; per-arm "
            "calibration excluded; alternating trials"
        ),
        "scope": (
            "One layer, local head shapes (1.5B TP1;72B TP2). No distributed "
            "communication or model weights. Effective bandwidth is minimum KV "
            "bytes, NOT actual DRAM traffic. CPU input-preparation wall "
            "includes H2D+completion. Capture inactive case launches a masked "
            "kernel; production may skip it entirely."
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch, context, shape, mode, due in itertools.product(
        batches, contexts, shapes, args.lengths.split(","), args.due.split(",")
    ):
        print(f"CASE {batch=} {context=} {shape=} {mode=} {due=}", flush=True)
        report["cases"].append(
            run_case(batch, context, shape, mode == "uneven", due, args)
        )
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
