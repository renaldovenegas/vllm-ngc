# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-only full/NGC attention timing, CUDA traces, and counter analysis."""

import argparse
import contextlib
import csv
import functools
import gzip
import inspect
import io
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ATTENTION_PATTERN = (
    r"(?i)(flash.*(fwd|forward|combine)|unified_attention|reduce_segments|"
    r"_ngc_decode_score)"
)
NCU_ATTENTION_PATTERN = (
    r"(.*[Ff]lash.*([Ff]wd|[Ff]orward|[Cc]ombine).*"
    r"|.*unified_attention.*|.*reduce_segments.*|.*_ngc_decode_score.*)"
)
METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def generation_budget(context, batch, token_budget, warmup, steps):
    """Keep early decoders alive while the remaining requests finish prefill."""
    prefill = math.ceil(context * batch / max(1, token_budget - batch)) + batch
    return warmup + steps + prefill + 2


def classify(name):
    """Keep unknown kernels explicit; GEMMs cannot identify projections vs MLPs."""
    import regex as re

    lower = name.lower()
    if re.search(ATTENTION_PATTERN, name):
        return "attention"
    if "prepare_ngc" in lower or "capture_ngc" in lower:
        return "ngc_input_preparation"
    if "ngc" in lower and "score" in lower:
        return "ngc_scoring"
    if any(word in lower for word in ("nccl", "allreduce", "all_reduce", "allgather")):
        return "communication"
    if any(word in lower for word in ("gemm", "matmul", "cutlass", "cublas")):
        return "gemm_projections_mlp_lm_head"
    if any(word in lower for word in ("rms_norm", "rmsnorm", "layer_norm")):
        return "normalization"
    if any(word in lower for word in ("silu", "gelu", "activation")):
        return "activation"
    if any(word in lower for word in ("reshape_and_cache", "kv_cache_update")):
        return "kv_cache_write"
    return "other_unclassified"


def merge_intervals(spans):
    merged = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def span_ms(spans):
    return sum(right - left for left, right in merge_intervals(spans)) / 1000


def summarize_trace(path):
    with gzip.open(path, "rt") as source:
        events = json.load(source)["traceEvents"]
    windows = [
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("ph") == "X"
        and e.get("name") == "DecodeProfile.window"
        and e.get("cat") == "user_annotation"
    ]
    if len(windows) != 1:
        raise ValueError("Expected one measured decode window")
    left, right = windows[0]
    groups, busy, streams = defaultdict(list), [], defaultdict(list)
    kernels = defaultdict(lambda: {"calls": 0, "summed_device_ms": 0.0})
    host = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        start, end = max(left, event["ts"]), min(right, event["ts"] + event["dur"])
        if end <= start:
            continue
        name, category = event["name"], event.get("cat", "")
        if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            busy.append((start, end))
            streams[str(event.get("args", {}).get("stream", event.get("tid")))].append(
                (start, end)
            )
            group = classify(name) if category == "kernel" else "copies_and_fills"
            groups[group].append((start, end))
            kernels[name]["calls"] += 1
            kernels[name]["summed_device_ms"] += (end - start) / 1000
        elif category in {"cuda_runtime", "cuda_driver"}:
            group = (
                "synchronization"
                if "Synchronize" in name
                else "launch"
                if "Launch" in name
                else "copies"
                if "Memcpy" in name
                else "other_runtime"
            )
            host[group].append((start, end))
    if not busy:
        raise ValueError("Trace contains no GPU activities in the decode window")
    merged = merge_intervals(busy)
    gaps = [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if b[0] > a[1]]
    return {
        "attention_attribution_status": "matched"
        if groups.get("attention")
        else "unavailable_no_matched_attention_kernels",
        "window_ms": (right - left) / 1000,
        "gpu_busy_union_ms": span_ms(busy),
        "gpu_idle_in_window_ms": (right - left) / 1000 - span_ms(busy),
        "between_gpu_activities_gap_ms": span_ms(gaps),
        "largest_gpu_gaps_ms": sorted(((b - a) / 1000 for a, b in gaps), reverse=True)[
            :20
        ],
        "category_union_ms": {key: span_ms(value) for key, value in groups.items()},
        "host_api_union_ms": {key: span_ms(value) for key, value in host.items()},
        "stream_busy_ms": {key: span_ms(value) for key, value in streams.items()},
        "kernels": dict(
            sorted(kernels.items(), key=lambda x: -x[1]["summed_device_ms"])
        ),
        "communication_scope": "TP=1: no inter-GPU model collectives expected",
        "caveats": [
            "GPU categories may overlap on different streams; do not add unions.",
            "CPU API spans overlap GPU work. Idle gaps are not proof of CPU stalls.",
            "Kernel-name attribution is heuristic; raw names and unknown work remain.",
            "Profiler overhead affects this timeline; use timing.json for latency.",
        ],
    }


def summarize_ncu(path):
    lines = path.read_text().splitlines()
    header = next(
        (
            i
            for i, line in enumerate(lines)
            if "Metric Name" in next(csv.reader([line]), [])
        ),
        None,
    )
    if header is None:
        return {
            "status": "unavailable",
            "reason": "No Nsight metric CSV header",
            "diagnostics": lines[:10],
        }
    launches = defaultdict(dict)
    names = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        if not row.get("Metric Name") or row["Metric Name"] == "Metric Name":
            continue
        key = (row.get("Process ID"), row.get("ID"))
        names[key] = row.get("Kernel Name", "")
        try:
            value = float((row.get("Metric Value") or "").replace(",", ""))
            if not math.isfinite(value) or value < 0:
                value = None
        except (ValueError, KeyError):
            value = None
        launches[key][row["Metric Name"]] = {
            "value": value,
            "unit": row.get("Metric Unit"),
        }
    output = []
    for key, metrics in launches.items():
        duration = metrics.get(METRICS[0], {})
        factor = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1}.get(
            duration.get("unit")
        )
        seconds = duration.get("value")
        seconds = seconds * factor if seconds is not None and factor else None
        read = metrics.get(METRICS[1], {})
        write = metrics.get(METRICS[2], {})
        read_bytes = read.get("value") if read.get("unit") == "byte" else None
        write_bytes = write.get("value") if write.get("unit") == "byte" else None
        output.append(
            {
                "kernel": names[key],
                "launch_id": key[1],
                "metrics": metrics,
                "duration_seconds": seconds,
                "dram_read_bytes": read_bytes,
                "dram_write_bytes": write_bytes,
                "read_GB_per_second": read_bytes / seconds / 1e9
                if read_bytes is not None and seconds
                else None,
                "read_write_GB_per_second": (read_bytes + write_bytes) / seconds / 1e9
                if read_bytes is not None and write_bytes is not None and seconds
                else None,
            }
        )
    valid = bool(output) and all(
        row["duration_seconds"]
        and row["dram_read_bytes"] is not None
        and row["dram_write_bytes"] is not None
        for row in output
    )
    missing = sorted(
        {
            metric
            for row in output
            for metric in METRICS
            if row["metrics"].get(metric, {}).get("value") is None
        }
    )
    return {
        "status": "ok" if valid and not missing else "unavailable_or_partial",
        "missing_or_unavailable_metrics": missing,
        "launches": output,
        "note": "Measured DRAM traffic includes all attention kernel accesses, not "
        "only KV. Cache hits can reduce DRAM reads. Nsight replays perturb caches "
        "and serialize kernels; these durations are not end-to-end timings. "
        "SM/tensor activity percentages are not achieved model FLOP/s.",
    }


@contextlib.contextmanager
def attention_policy(rate, cadence, query_window):
    """Apply the same decoder-layer policy as the existing NGC benchmarks."""
    if rate is None:
        yield []
        return
    import regex as re

    from vllm.config.cache import NGCEvictionConfig
    from vllm.model_executor.layers.attention.attention import Attention

    original = Attention.__init__
    signature = inspect.signature(original)
    layers = []

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        prefix = bound.arguments.get("prefix", "")
        if re.search(r"(?:^|\.)layers\.\d+\.", prefix) is None:
            raise RuntimeError(f"Cannot identify decoder layer in {prefix!r}")
        bound.arguments["ngc_eviction_config"] = NGCEvictionConfig(
            cadence=cadence,
            eviction_rate=rate,
            query_buffer_size=query_window,
            min_retained_tokens=32,
        )
        layers.append(prefix)
        original(*bound.args, **bound.kwargs)

    Attention.__init__ = initialize
    try:
        yield layers
        if not layers:
            raise RuntimeError("No NGC attention layers were configured")
    finally:
        Attention.__init__ = original


def retained_snapshot(scheduler, ngc):
    """Read committed CPU layouts without draining cache-update notifications."""
    requests = list(scheduler.requests.values())
    if not ngc:
        return {
            "full_attention_lengths": {
                r.request_id: r.num_computed_tokens for r in requests
            },
            "evicted_tokens_sum_across_groups": 0,
        }
    groups = []
    for group_id, manager in enumerate(
        scheduler.kv_cache_manager.coordinator.single_type_managers
    ):
        if not hasattr(manager, "layouts"):
            continue
        lengths, evicted = {}, {}
        for request in requests:
            layout = manager.layouts.get(request.request_id)
            removed = layout.evicted_tokens if layout is not None else 0
            evicted[request.request_id] = removed
            lengths[request.request_id] = request.num_computed_tokens - removed
        groups.append(
            {
                "group_id": group_id,
                "retained_tokens": lengths,
                "evicted_tokens": evicted,
            }
        )
    if not groups:
        raise RuntimeError("NGC policy requested but no NGC cache managers found")
    return {
        "groups": groups,
        "evicted_tokens_sum_across_groups": sum(
            sum(group["evicted_tokens"].values()) for group in groups
        ),
    }


def retained_range(snapshot):
    values = (
        [n for group in snapshot["groups"] for n in group["retained_tokens"].values()]
        if "groups" in snapshot
        else list(snapshot["full_attention_lengths"].values())
    )
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values)}


@contextlib.contextmanager
def observe_fused_steps(enabled):
    """Observe the existing boundary decision; never advance scorer state."""
    state = {"fused_scoring": False}
    if not enabled:
        yield state
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._prepare_ngc_fused_decode

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        state["fused_scoring"] |= bool(result)
        return result

    GPUModelRunner._prepare_ngc_fused_decode = wrapped
    try:
        yield state
    finally:
        GPUModelRunner._prepare_ngc_fused_decode = original


def run(args):
    import torch

    import vllm
    from vllm import LLM, SamplingParams

    source_root = Path(vllm.__file__).resolve().parent
    if (
        args.expected_source_root
        and source_root != Path(args.expected_source_root).resolve()
    ):
        raise RuntimeError(
            f"Imported vllm from {source_root}; expected {args.expected_source_root}"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    token_budget = max(4096, args.batch_size)
    max_outputs = generation_budget(
        args.context, args.batch_size, token_budget, args.warmup_steps, args.steps
    )
    ngc = args.eviction_rate is not None
    if ngc and args.backend != "TRITON_ATTN":
        raise ValueError("NGC profiling requires TRITON_ATTN")
    with attention_policy(
        args.eviction_rate, args.cadence, args.query_window
    ) as layers:
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            max_num_seqs=args.batch_size,
            max_model_len=args.context + max_outputs + 32,
            max_num_batched_tokens=token_budget,
            attention_config={"backend": args.backend},
            async_scheduling=False,
            distributed_executor_backend="uni",
            tensor_parallel_size=1,
            enable_prefix_caching=False,
            enforce_eager=args.eager,
            gpu_memory_utilization=0.85,
            disable_log_stats=True,
            seed=0,
        )
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode("A traveler follows the river through a quiet valley. ")
    tokens = (seed * ((args.context + len(seed) - 1) // len(seed)))[: args.context]
    params = SamplingParams(
        temperature=0,
        max_tokens=max_outputs,
        ignore_eos=args.ignore_eos,
    )
    serial = 0

    def active_requests():
        requests = list(scheduler.requests.values())
        if any(r.num_preemptions > 0 for r in requests):
            raise RuntimeError(
                "Insufficient KV capacity for fixed batch; reduce "
                "batch/context or use a larger GPU"
            )
        if len(requests) != args.batch_size:
            raise RuntimeError(
                "A request stopped before the decode window; "
                "use --ignore-eos for a fixed-shape profile"
            )
        return requests

    def prepare():
        nonlocal serial
        serial += 1
        for i in range(args.batch_size):
            engine.add_request(f"{serial}-{i}", {"prompt_token_ids": tokens}, params)
        while True:
            engine.step()
            requests = active_requests()
            if all(r.num_computed_tokens >= args.context for r in requests):
                break
        for _ in range(args.warmup_steps):
            engine.step()
            active_requests()
        if len(scheduler.requests) != args.batch_size:
            raise RuntimeError(
                "Early EOS during warmup; no fixed-batch timing available"
            )
        torch.accelerator.synchronize()

    def drain():
        while engine.has_unfinished_requests():
            engine.step()

    def window(label=False):
        lengths = [r.num_computed_tokens for r in active_requests()]
        if len(lengths) != args.batch_size:
            raise RuntimeError("Incomplete batch before measurement")
        initial = sum(r.num_output_tokens for r in scheduler.requests.values())
        annotation = (
            torch.profiler.record_function("DecodeProfile.window")
            if label
            else contextlib.nullcontext()
        )
        before = retained_snapshot(scheduler, ngc)
        step_rows = []
        start = time.perf_counter()
        with annotation, observe_fused_steps(label and ngc) as observed:
            for index in range(args.steps):
                if not label:
                    engine.step()
                    continue
                observed["fused_scoring"] = False
                inbox = getattr(scheduler, "ngc_eviction_inbox", None)
                pending = len(inbox.pending) if inbox is not None else 0
                pre = retained_snapshot(scheduler, ngc)
                tick = time.perf_counter()
                with torch.profiler.record_function(f"DecodeProfile.step.{index}"):
                    engine.step()
                    torch.accelerator.synchronize()
                elapsed = (time.perf_counter() - tick) * 1000
                post = retained_snapshot(scheduler, ngc)
                removed = (
                    post["evicted_tokens_sum_across_groups"]
                    - pre["evicted_tokens_sum_across_groups"]
                )
                kind = (
                    "scoring_and_proposals"
                    if observed["fused_scoring"] and pending
                    else "scoring"
                    if observed["fused_scoring"]
                    else "proposal_application"
                    if pending
                    else "ordinary"
                )
                step_rows.append(
                    {
                        "step": index,
                        "kind": kind,
                        "wall_ms": elapsed,
                        "pending_proposals": pending,
                        "fused_scoring": observed["fused_scoring"],
                        "removed_tokens_sum_across_groups": removed,
                        "retained_after_range": retained_range(post),
                    }
                )
            torch.accelerator.synchronize()
        seconds = time.perf_counter() - start
        requests = active_requests()
        generated = sum(r.num_output_tokens for r in requests) - initial
        if generated != args.steps * args.batch_size:
            raise RuntimeError("Measured window was not one-token full-batch decode")
        after = retained_snapshot(scheduler, ngc)
        removed = (
            after["evicted_tokens_sum_across_groups"]
            - before["evicted_tokens_sum_across_groups"]
        )
        if (
            ngc
            and args.eviction_rate > 0
            and args.steps >= 2 * args.cadence
            and removed <= 0
        ):
            raise RuntimeError("Positive-rate NGC window did not evict any tokens")
        if ngc and args.eviction_rate == 0 and removed != 0:
            raise RuntimeError("Zero-rate NGC unexpectedly evicted tokens")
        kinds = sorted({row["kind"] for row in step_rows})
        return {
            "retained_before": before,
            "retained_after": after,
            "profiled_steps": step_rows,
            "profiled_step_summary": {
                kind: {
                    "count": sum(row["kind"] == kind for row in step_rows),
                    "mean_wall_ms": statistics.mean(
                        row["wall_ms"] for row in step_rows if row["kind"] == kind
                    ),
                }
                for kind in kinds
            },
            "seconds": seconds,
            "decode_steps": args.steps,
            "generated_tokens": generated,
            "ms_per_decode_step": seconds * 1000 / args.steps,
            "ms_per_output_token_aggregate": seconds * 1000 / generated,
            "output_tokens_per_second": generated / seconds,
            "context_before_window_min": min(lengths),
            "context_before_window_max": max(lengths),
            "context_last_attention_min": min(lengths) + args.steps,
            "context_last_attention_max": max(lengths) + args.steps,
            "request_contexts_before_window": lengths,
            "request_contexts_last_attention": [n + args.steps for n in lengths],
        }

    # An entire matching generation warms compilation and graph paths first.
    prepare()
    window()
    drain()
    rows = []
    if args.mode == "timing":
        for _ in range(args.repeats):
            prepare()
            rows.append(window())
            drain()
    elif args.mode == "trace":
        prepare()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            rows.append(window(label=True))
        trace = output / "trace.json"
        prof.export_chrome_trace(str(trace))
        with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
            import shutil

            shutil.copyfileobj(source, target)
        trace.unlink()
        (output / "timeline.json").write_text(
            json.dumps(summarize_trace(Path(str(trace) + ".gz")), indent=2)
        )
        drain()
    else:
        prepare()
        torch.cuda.nvtx.range_push("DecodeProfile")
        try:
            rows.append(window())
        finally:
            torch.cuda.nvtx.range_pop()
        drain()
    report = {
        "settings": vars(args),
        "ngc_layers": layers,
        "vllm_source_root": str(source_root),
        "generation_budget": max_outputs,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "runs": rows,
        "median_ms_per_decode_step": statistics.median(
            row["ms_per_decode_step"] for row in rows
        ),
        "median_ms_per_output_token_aggregate": statistics.median(
            row["ms_per_output_token_aggregate"] for row in rows
        ),
        "timing_valid_for_latency": args.mode == "timing",
        "counter_scope": (
            "First measured step after warmup; may be ordinary decode rather "
            "than a scoring boundary. Not a cadence-cycle average."
            if args.mode == "counters"
            else None
        ),
        "note": "Prefill and warmup excluded. Wall time includes scheduler, "
        "sampling and final GPU completion. Context grows across the window. "
        "TP=1, synchronous scheduling; no speculative decode. Trace step timings "
        "include profiler overhead and per-step synchronization, and are diagnostic "
        "only. Snapshot bookkeeping occurs outside each step timer. Fused scoring "
        "is attributed jointly with attention. Retained lengths are committed CPU "
        "cache state, not allocated capacity. Proposal steps can remove zero tokens.",
    }
    (output / f"{args.mode}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--eviction-rate", type=float, default=None)
    parser.add_argument("--cadence", type=int, default=256)
    parser.add_argument("--query-window", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--backend", choices=["FLASH_ATTN", "TRITON_ATTN"], default="FLASH_ATTN"
    )
    parser.add_argument(
        "--mode", choices=["timing", "trace", "counters"], default="timing"
    )
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-root", default=None)
    args = parser.parse_args()
    if (
        min(args.batch_size, args.context, args.steps, args.repeats) < 1
        or args.warmup_steps < 1
    ):
        parser.error("Batch, context, steps, warmup and repeats must be positive")
    if args.cadence < 1 or not 1 <= args.query_window <= args.cadence:
        parser.error("Require 1 <= query-window <= cadence")
    if args.eviction_rate is not None and not 0 <= args.eviction_rate < 1:
        parser.error("Eviction rate must be in [0, 1)")
    run(args)
