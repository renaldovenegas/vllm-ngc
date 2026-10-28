# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end wall-clock rate sweep inside one Modal GPU allocation."""

import argparse
import bisect
import contextlib
import cProfile
import csv
import functools
import gzip
import hashlib
import inspect
import json
import os
import pstats
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import regex as re


@contextlib.contextmanager
def trace_steps(llm, rows: list[dict], upstream: bool):
    """Time each engine step and record the KV tokens it attended over.

    Evictions are applied while scheduling, so state read after a step is the
    retained cache that step's forward pass used, plus its one new token.
    """
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    managers = []
    if not upstream:
        from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager

        managers = [
            manager
            for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
            if isinstance(manager, NGCEvictionManager)
        ]
    original_step = engine.step
    original_core_step = core.step_fn

    def core_step():
        started = time.perf_counter()
        result = original_core_step()
        rows.append({"core_ms": (time.perf_counter() - started) * 1e3})
        return result

    def step():
        started = time.perf_counter()
        result = original_step()
        elapsed = (time.perf_counter() - started) * 1e3
        request = next(iter(scheduler.requests.values()), None)
        if rows and "total_ms" not in rows[-1] and request is not None:
            evicted = [
                manager.layouts[request.request_id].evicted_tokens
                for manager in managers
                if request.request_id in manager.layouts
            ] or [0]
            context = request.num_computed_tokens
            rows[-1].update(
                total_ms=elapsed,
                context_tokens=context,
                prompt_tokens=request.num_prompt_tokens,
                evicted_tokens_total=sum(evicted),
                retained_mean=context - sum(evicted) / len(evicted),
                retained_min=context - max(evicted),
                retained_max=context - min(evicted),
            )
        return result

    engine.step = step
    core.step_fn = core_step
    try:
        yield
    finally:
        engine.step = original_step
        core.step_fn = original_core_step


def summarize_steps(runs: list[list[dict]], cadence: int) -> list[dict]:
    """Median per-step timings across identically seeded runs."""
    runs = [[row for row in rows if "context_tokens" in row] for rows in runs]
    assert len({len(rows) for rows in runs}) == 1, "Runs took different steps"
    steps = []
    previous = None
    for index, rows in enumerate(zip(*runs)):
        first = rows[0]
        assert all(
            row["context_tokens"] == first["context_tokens"]
            and row["evicted_tokens_total"] == first["evicted_tokens_total"]
            for row in rows
        ), "Runs diverged in cache state"
        generated = first["context_tokens"] - first["prompt_tokens"]
        steps.append(
            {
                "step": index,
                "generated": generated,
                "context_tokens": first["context_tokens"],
                "retained_mean": first["retained_mean"],
                "retained_min": first["retained_min"],
                "retained_max": first["retained_max"],
                "scoring_step": first["context_tokens"] // cadence
                > (previous["context_tokens"] // cadence if previous else 0),
                "eviction_step": previous is not None
                and first["evicted_tokens_total"] > previous["evicted"],
                "core_ms": statistics.median(row["core_ms"] for row in rows),
                "total_ms": statistics.median(row["total_ms"] for row in rows),
                "core_ms_runs": [row["core_ms"] for row in rows],
            }
        )
        previous = {
            "context_tokens": first["context_tokens"],
            "evicted": first["evicted_tokens_total"],
        }
    return steps


def run_case(
    model: str,
    case: str,
    output: Path,
    *,
    profiling: bool = False,
    profile_only: bool = False,
    scheduler_profiling: bool = False,
    step_trace: bool = False,
    num_seqs: int = 1,
    output_tokens: int = 2048,
    warmup_tokens: int = 2048,
    prompt_tokens: int = 0,
    cadence: int = 128,
    query_window: int = 5,
    repeats: int = 3,
    temperature: float = 0.6,
) -> None:
    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.attention.attention import Attention

    # Pristine base commit without NGC code; skip every NGC hook.
    # A "-graph" suffix runs the same case with CUDA graphs and torch.compile.
    use_graphs = case.endswith("-graph")
    base_case = case.removesuffix("-graph").removeprefix("before-")
    upstream = base_case == "upstream"
    rate = None if base_case in ("full", "upstream") else float(base_case)
    if not upstream:
        from vllm.config.cache import NGCEvictionConfig
        from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager
    original_init = Attention.__init__
    signature = inspect.signature(original_init)
    layers = []

    @functools.wraps(original_init)
    def initialize_attention(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        prefix = bound.arguments.get("prefix", "")
        if re.search(r"(?:^|\.)layers\.\d+\.", prefix) is None:
            raise RuntimeError(f"Cannot identify decoder layer in {prefix!r}")
        if rate is not None:
            bound.arguments["ngc_eviction_config"] = NGCEvictionConfig(
                cadence=cadence,
                eviction_rate=rate,
                query_buffer_size=query_window,
                min_retained_tokens=32,
            )
            layers.append(prefix)
        original_init(*bound.args, **bound.kwargs)

    Attention.__init__ = initialize_attention
    try:
        started = time.perf_counter()
        llm = LLM(
            model=model,
            dtype="float16",
            kv_cache_dtype="auto",
            enforce_eager=not use_graphs,
            async_scheduling=False,
            distributed_executor_backend="uni",
            tensor_parallel_size=1,
            attention_config={"backend": "TRITON_ATTN"},
            block_size=16,
            max_model_len=max(
                4096, max(output_tokens, warmup_tokens) + max(prompt_tokens, 1024)
            ),
            max_num_seqs=num_seqs,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.65,
            enable_prefix_caching=False,
            seed=0,
            jit_monitor_verbose=profiling,
        )
        load_seconds = time.perf_counter() - started
    finally:
        Attention.__init__ = original_init
    assert bool(layers) == (rate is not None)
    prompt = "Write a long fantasy story that is approximately 3000 words."
    tokenizer = llm.get_tokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if prompt_tokens:
        before, after = rendered.split(prompt, 1)
        prefix = tokenizer.encode(before, add_special_tokens=False)
        suffix = tokenizer.encode("\n\n" + prompt + after, add_special_tokens=False)
        background = tokenizer.encode(
            "The kingdom of Aster lies between a mountain range and the sea. "
            "Its northern villages guard an ancient forest, while ships bring "
            "travelers to the southern ports. A young mapmaker discovers that "
            "the stars on an old chart describe a hidden road through the mountains. "
            "Use these setting notes as background for the story.\n",
            add_special_tokens=False,
        )
        remaining = prompt_tokens - len(prefix) - len(suffix)
        if remaining <= 0 or not background:
            raise ValueError("Prompt token budget is too small for the chat template")
        context = (background * ((remaining + len(background) - 1) // len(background)))[
            :remaining
        ]
        token_ids = prefix + context + suffix
        assert len(token_ids) == prompt_tokens
    assert token_ids and all(isinstance(token, int) for token in token_ids)
    sampling = dict(
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=0,
        min_tokens=output_tokens,
        max_tokens=output_tokens,
        ignore_eos=True,
    )

    def generate(num_tokens: int = output_tokens):
        # Distinct seeds give the batch distinct samples from one prompt.
        params = [
            SamplingParams(
                **dict(
                    sampling,
                    min_tokens=num_tokens,
                    max_tokens=num_tokens,
                    seed=sampling["seed"] + i,
                )
            )
            for i in range(num_seqs)
        ]
        results = llm.generate(
            [{"prompt_token_ids": token_ids}] * num_seqs, params, use_tqdm=False
        )
        assert len(results) == num_seqs
        assert all(len(r.outputs[0].token_ids) == num_tokens for r in results)
        return results

    warmup_events = []
    if upstream:
        warmup = generate(warmup_tokens)
        torch.accelerator.synchronize()
    else:
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        batched = hasattr(NGCEvictionManager, "detach_eviction_blocks")
        eviction_method = "detach_eviction_blocks" if batched else "apply_eviction"
        original_evict = getattr(NGCEvictionManager, eviction_method)
        original_batch = getattr(KVCacheManager, "apply_ngc_evictions", None)

        @functools.wraps(original_evict)
        def validate_eviction(self, proposal, computed_tokens):
            before = self.block_pool.get_num_free_blocks()
            result = original_evict(self, proposal, computed_tokens)
            applied = result is not None if batched else result
            if applied:
                freed = (
                    len(result)
                    if batched
                    else (self.block_pool.get_num_free_blocks() - before)
                )
                assert freed == len(proposal.victim_indices)
                warmup_events.append(freed)
            return result

        def validate_batch(self, requests, proposals):
            before = self.coordinator.block_pool.get_num_free_blocks()
            start = len(warmup_events)
            result = original_batch(self, requests, proposals)
            assert self.coordinator.block_pool.get_num_free_blocks() - before == sum(
                warmup_events[start:]
            )
            return result

        setattr(NGCEvictionManager, eviction_method, validate_eviction)
        if batched:
            KVCacheManager.apply_ngc_evictions = validate_batch
        try:
            warmup = generate(warmup_tokens)
            torch.accelerator.synchronize()
        finally:
            setattr(NGCEvictionManager, eviction_method, original_evict)
            if batched:
                KVCacheManager.apply_ngc_evictions = original_batch
    assert len(warmup[0].outputs[0].token_ids) == warmup_tokens
    if rate is not None and rate > 0:
        assert sum(warmup_events) > 0, "Warmup must exercise real eviction"
    else:
        assert sum(warmup_events) == 0

    from vllm.utils import jit_monitor

    jit_events = []
    phase = "timed"
    original_jit_event = jit_monitor._handle_jit_event

    def record_jit_event(**kwargs):
        jit_events.append({"phase": phase, **kwargs})
        original_jit_event(**kwargs)

    if profiling:
        jit_monitor._handle_jit_event = record_jit_event
    seconds = []
    output_hashes = []
    step_runs: list[list[dict]] = []
    for repeat in range(0 if profile_only else repeats):
        rows: list[dict] = []
        tracing = (
            trace_steps(llm, rows, upstream) if step_trace else contextlib.nullcontext()
        )
        torch.accelerator.synchronize()
        started = time.perf_counter()
        with tracing:
            result = generate()
        torch.accelerator.synchronize()
        step_runs.append(rows)
        seconds.append(time.perf_counter() - started)
        output_hashes.append(
            hashlib.sha256(
                json.dumps([r.outputs[0].token_ids for r in result]).encode()
            ).hexdigest()
        )
        print(
            f"TIMING {case} repeat={repeat + 1} seconds={seconds[-1]:.6f}", flush=True
        )
    median = statistics.median(seconds) if seconds else None
    report = {
        "model": model,
        "case": case,
        "enforce_eager": not use_graphs,
        "vllm_source": vllm.__file__,
        "source_files_sha256": {
            filename: hashlib.sha256(
                (Path(vllm.__file__).parent / filename).read_bytes()
            ).hexdigest()
            for filename in (
                "v1/core/kv_cache_coordinator.py",
                "v1/core/single_type_kv_cache_manager.py",
                "v1/core/kv_cache_manager.py",
                "v1/core/sched/scheduler.py",
                "v1/worker/gpu_model_runner.py",
                "v1/attention/backends/triton_attn.py",
                "v1/attention/ops/triton_ngc_inputs.py",
                "v1/worker/ngc_scoring.py",
            )
            if (Path(vllm.__file__).parent / filename).is_file()
        },
        "manager_source_sha256": hashlib.sha256(
            (
                Path(vllm.__file__).parent / "v1/core/single_type_kv_cache_manager.py"
            ).read_bytes()
        ).hexdigest(),
        "output_token_hashes": output_hashes,
        "eviction_rate": rate,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": str(
            getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ngc_layers": layers,
        "cadence": cadence,
        "query_buffer_size": query_window,
        "min_retained_tokens": 32,
        "batch_size": num_seqs,
        "prompt_token_ids": token_ids,
        "prompt_tokens": len(token_ids),
        "sampling": sampling,
        "output_tokens": output_tokens,
        "warmup_tokens": warmup_tokens,
        "seconds": seconds,
        "median_seconds": median,
        "tokens_per_second": num_seqs * output_tokens / median if median else None,
        "model_load_seconds": load_seconds,
        "warmup_eviction_decisions": len(warmup_events),
        "warmup_freed_pages": sum(warmup_events),
        "jit_events_after_warmup": jit_events,
    }
    if step_trace:
        report["steps"] = summarize_steps(step_runs, cadence=cadence)
    if scheduler_profiling:
        output.write_text(json.dumps(report, indent=2))
        report["scheduler_profile"] = profile_scheduler(llm, generate)
    if profiling:
        print(f"JIT_TIMED_SUMMARY {case} events={len(jit_events)}", flush=True)
        output.write_text(json.dumps(report, indent=2))
        phase = "profile"
        report["profile"] = (
            profile_generation_4k if profile_only else profile_generation
        )(llm, generate, output, upstream=upstream)
        jit_monitor._handle_jit_event = original_jit_event
    output.write_text(json.dumps(report, indent=2))


def profile_scheduler(llm, generate) -> dict:
    """Time coarse scheduler operations, then profile their Python call trees."""
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    engine = llm.llm_engine
    original_step = engine.step
    original_schedule = scheduler.schedule
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    patches = []
    passes = {}
    profiles = {kind: cProfile.Profile() for kind in ("ordinary", "proposals")}
    active_row = None

    def instrument(owner, name, label):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if active_row is None:
                return original(*args, **kwargs)
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e6
                values = active_row["operations"].setdefault(label, [0, 0.0])
                values[0] += 1
                values[1] += elapsed

        patches.append((owner, name, original))
        setattr(owner, name, wrapped)

    def run_pass(mode):
        step_index = 0
        rows = []

        def schedule(*args, **kwargs):
            nonlocal active_row
            if not 1800 <= step_index < 1944:
                return original_schedule(*args, **kwargs)
            inbox = getattr(scheduler, "ngc_eviction_inbox", None)
            pending = len(inbox.pending) if inbox is not None else 0
            kind = "proposals" if pending else "ordinary"
            row = {
                "step": step_index,
                "kind": kind,
                "pending_proposals": pending,
                "running_requests": len(scheduler.running),
                "operations": {},
            }
            active_row = row if mode == "timed" else None
            started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            if mode == "calls":
                profiles[kind].enable()
            try:
                return original_schedule(*args, **kwargs)
            finally:
                if mode == "calls":
                    profiles[kind].disable()
                row["thread_cpu_ms"] = (time.thread_time_ns() - cpu_started) / 1e6
                row["wall_ms"] = (time.perf_counter_ns() - started) / 1e6
                active_row = None
                rows.append(row)

        def step():
            nonlocal step_index
            result = original_step()
            step_index += 1
            return result

        scheduler.schedule = schedule
        engine.step = step
        generate()
        assert len(rows) == 144, "Scheduler profile missed the late-decode window"
        return rows

    try:
        for mode in ("timed", "calls"):
            if mode == "timed":
                for name in (
                    "allocate_slots",
                    "take_ngc_cache_updates",
                    "apply_ngc_evictions"
                    if hasattr(manager, "apply_ngc_evictions")
                    else "apply_ngc_eviction",
                    "get_num_common_prefix_blocks",
                    "get_block_ids",
                    "take_new_block_ids",
                    "take_kv_cache_block_copies",
                    "take_boundary_state_offloads",
                    "new_step_starts",
                ):
                    instrument(manager, name, "manager." + name)
                for name in (
                    "get_num_blocks_to_allocate",
                    "allocate_new_blocks",
                    "remove_skipped_blocks",
                ):
                    instrument(coordinator, name, "coordinator." + name)
                instrument(
                    scheduler, "_make_cached_request_data", "cached_request_data"
                )
                instrument(scheduler, "_update_after_schedule", "update_after_schedule")
            passes[mode] = run_pass(mode)
            for owner, name, original in reversed(patches):
                setattr(owner, name, original)
            patches.clear()
            print(f"SCHEDULER_PROFILE {mode} steps={len(passes[mode])}", flush=True)
    finally:
        scheduler.schedule = original_schedule
        engine.step = original_step
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
    functions = {}
    for kind, profile in profiles.items():
        rows = []
        if not any(row["kind"] == kind for row in passes["calls"]):
            functions[kind] = rows
            continue
        for (filename, line, name), values in pstats.Stats(profile).stats.items():
            primitive, calls, self_seconds, cumulative_seconds, callers = values
            rows.append(
                {
                    "file": filename,
                    "line": line,
                    "function": name,
                    "primitive_calls": primitive,
                    "calls": calls,
                    "self_ms": self_seconds * 1000,
                    "cumulative_ms": cumulative_seconds * 1000,
                    "callers": [
                        {
                            "file": key[0],
                            "line": key[1],
                            "function": key[2],
                            "stats": val,
                        }
                        for key, val in callers.items()
                    ],
                }
            )
        functions[kind] = sorted(rows, key=lambda row: -row["cumulative_ms"])
    return {
        "start_step": 1800,
        "active_steps": 144,
        "passes": passes,
        "functions": functions,
        "note": "Timed pass uses coarse method wrappers; nested operation spans "
        "overlap. Thread CPU time excludes waiting. Separate cProfile pass adds "
        "per-call overhead: use it for attribution and counts, not absolute latency. "
        "Proposals identifies scheduler steps consuming the NGC inbox, including "
        "zero-victim proposals; ordinary includes block-allocation boundaries.",
    }


def summarize_profile_trace(trace: Path) -> dict:
    """Measure interval unions, including kernels replayed by CUDA graphs."""
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    intervals: dict[str, list[tuple[float, float]]] = {}
    steps = []
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name, category = event["name"], event.get("cat", "")
        span = (event["ts"], event["ts"] + event["dur"])
        if category == "user_annotation" and name.startswith("ProfilerStep#"):
            steps.append((name, *span))
        labels = []
        if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            labels.append("gpu_busy")
            labels.append("gpu_kernels" if category == "kernel" else "gpu_copies_fills")
            if category == "kernel":
                group = (
                    "fused_partials"
                    if "_ngc_decode_score_partials" in name
                    else "fused_reduce"
                    if "_ngc_decode_score_reduce" in name
                    else "ordinary_attention"
                    if "unified_attention" in name or "reduce_segments" in name
                    else "input_preparation"
                    if "prepare_ngc" in name or "capture_ngc" in name
                    else "other_kernels"
                )
                labels.append("gpu_" + group)
        if category in {"cuda_runtime", "cuda_driver"}:
            if "Synchronize" in name:
                labels.append("host_sync_api")
            if "Memcpy" in name:
                labels.append("host_copy_api")
            if "GraphLaunch" in name:
                labels.append("host_graph_launch_api")
            if "Capture" in name or "GraphInstantiate" in name:
                labels.append("host_graph_setup_api")
        if category == "user_annotation" and name.startswith(
            ("worker.", "scheduler.", "NGC.")
        ):
            labels.append(name)
        for label in labels:
            intervals.setdefault(label, []).append(span)

    merged = {}
    for label, spans in intervals.items():
        union = []
        for start, end in sorted(spans):
            if union and start <= union[-1][1]:
                union[-1] = (union[-1][0], max(end, union[-1][1]))
            else:
                union.append((start, end))
        merged[label] = (union, [end for _, end in union])

    def duration(label, start, end):
        spans, ends = merged.get(label, ([], []))
        total = 0.0
        index = bisect.bisect_right(ends, start)
        for left, right in spans[index:]:
            if left >= end:
                break
            total += max(0.0, min(end, right) - max(start, left))
        return total / 1000

    rows = []
    for name, start, end in sorted(steps, key=lambda step: step[1]):
        row = {label + "_ms": duration(label, start, end) for label in merged}
        row.update(step=name, wall_ms=(end - start) / 1000)
        row["gpu_idle_ms"] = row["wall_ms"] - row.get("gpu_busy_ms", 0.0)
        row["scoring_boundary"] = (
            row.get("NGC.select_victims_ms", 0.0) > 0
            or row.get("NGC.boundary_graph_ms", 0.0) > 0
        )
        row["proposal_application"] = row.get("NGC.apply_evictions_ms", 0.0) > 0
        rows.append(row)
    assert rows and "gpu_busy" in merged, "Missing CPU steps or CUDA kernel events"
    summaries = {}
    for kind in ("all", "ordinary", "scoring_boundary"):
        selected = [
            row
            for row in rows
            if kind == "all" or row["scoring_boundary"] == (kind == "scoring_boundary")
        ]
        summaries[kind] = {
            "steps": len(selected),
            "mean_ms": {
                key: statistics.mean(row[key] for row in selected)
                for key in rows[0]
                if key.endswith("_ms") and selected
            },
        }
    window_start = min(step[1] for step in steps)
    window_end = max(step[2] for step in steps)
    return {
        "window_totals_ms": {
            label: duration(label, window_start, window_end) for label in merged
        },
        "host_window_ms": (window_end - window_start) / 1000,
        "summary": summaries,
        "steps": rows,
        "note": "GPU busy is the union of kernels/copies/fills across streams. "
        "Host annotations are inclusive wall spans, including waits, not CPU "
        "utilization. Synchronization API spans overlap host annotations and GPU "
        "work. GPU idle includes profiler overhead and is not all CPU compute.",
    }


def profile_generation(llm, generate, output: Path, *, upstream: bool = False) -> dict:
    """Capture late decode including the 1920-token eviction boundary."""
    import torch

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    patches = []

    def annotate(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)

        patches.append((owner, name, original))
        setattr(owner, name, wrapped)

    capture_hooks = []

    def annotate_capture(model):
        for module in model.modules():
            original = getattr(module, "ngc_query_capture", None)
            if original is None:
                continue

            def wrapped(query, original=original):
                with torch.profiler.record_function("NGC.query_capture"):
                    return original(query)

            capture_hooks.append((module, original))
            module.ngc_query_capture = wrapped

    if not upstream:
        from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager
        from vllm.v1.worker import ngc_scoring

        llm.apply_model(annotate_capture)
        annotate(ngc_scoring.NGCScoringBuffers, "score_if_due", "NGC.score_if_due")
        annotate(ngc_scoring.NGCScoringBuffers, "_copy_handoff", "NGC.proposal_to_cpu")
        annotate(ngc_scoring, "select_ngc_victims", "NGC.select_victims")
        from vllm.v1.core.kv_cache_manager import KVCacheManager

        if hasattr(KVCacheManager, "apply_ngc_evictions"):
            annotate(KVCacheManager, "apply_ngc_evictions", "NGC.apply_evictions")
        else:
            annotate(NGCEvictionManager, "apply_eviction", "NGC.apply_eviction")
        annotate(GPUModelRunner, "_score_ngc_pages", "NGC.score_pages")
    annotate(Scheduler, "schedule", "scheduler.schedule")
    annotate(Scheduler, "update_from_output", "scheduler.update_from_output")
    annotate(GPUModelRunner, "_prepare_inputs", "worker.prepare_inputs")
    annotate(GPUModelRunner, "_update_states", "worker.update_states")
    annotate(GPUModelRunner, "_build_attention_metadata", "worker.attention_metadata")
    engine = llm.llm_engine
    original_step = engine.step
    trace = output.with_suffix(".trace.json")
    rows = []

    def save_profile(prof):
        prof.export_chrome_trace(str(trace))
        for event in prof.key_averages():
            rows.append(
                {
                    "name": event.key,
                    "calls": event.count,
                    "self_cpu_us": event.self_cpu_time_total,
                    "cpu_us": event.cpu_time_total,
                    "self_device_us": event.self_device_time_total,
                    "device_us": event.device_time_total,
                }
            )

    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1792, warmup=8, active=144, repeat=1),
            on_trace_ready=save_profile,
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        ) as prof:

            def step():
                result = original_step()
                prof.step()
                return result

            engine.step = step
            result = generate()
            torch.accelerator.synchronize()
            assert len(result[0].outputs[0].token_ids) == 2048
    finally:
        engine.step = original_step
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        for module, original in capture_hooks:
            module.ngc_query_capture = original
    assert rows and trace.exists(), "Profiler did not capture the decode window"
    timeline = summarize_profile_trace(trace)
    with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
        shutil.copyfileobj(source, target)
    trace.unlink()
    return {
        "wait_steps": 1792,
        "warmup_steps": 8,
        "active_steps": 144,
        "trace": trace.name + ".gz",
        "events": rows,
        "timeline": timeline,
        "note": "Inclusive nested CPU/device totals overlap; do not add them. "
        "Profile instrumentation is excluded from wall-clock samples. "
        "Query-capture Python annotations are absent during CUDA graph replay; "
        "replayed kernels are included in the GPU timeline.",
    }


def profile_generation_4k(llm, generate, output: Path, *, upstream: bool) -> dict:
    """Profile a bounded late-decode window selected by actual generated KV."""
    import threading

    import torch

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    patches, rows, averages = [], [], []
    stacks = threading.local()
    active_row = None
    graph_counts = {"captures": 0, "replays": 0}
    original_step = engine.step
    trace = output.with_suffix(".trace.json")
    started = False
    stopped = False
    profiled_steps = 0
    warmup_steps, active_steps, start_generated = 8, 128, 3000

    def annotate(owner, name, label, graph=False):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            nonlocal active_row
            row = active_row
            previous_counts = None
            if graph:
                previous_counts = (args[0].captures, args[0].replays)
            if row is None:
                result = original(*args, **kwargs)
            else:
                stack = getattr(stacks, "frames", None)
                if stack is None:
                    stack = stacks.frames = []
                frame = [0, 0]
                wall_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
                stack.append(frame)
                try:
                    with torch.profiler.record_function(label):
                        result = original(*args, **kwargs)
                finally:
                    wall = time.perf_counter_ns() - wall_start
                    cpu = time.thread_time_ns() - cpu_start
                    stack.pop()
                    if stack:
                        stack[-1][0] += wall
                        stack[-1][1] += cpu
                    values = row["operations"].setdefault(
                        label,
                        {
                            "calls": 0,
                            "wall_ms": 0.0,
                            "thread_cpu_ms": 0.0,
                            "exclusive_wall_ms": 0.0,
                            "exclusive_thread_cpu_ms": 0.0,
                        },
                    )
                    values["calls"] += 1
                    for key, value in (
                        ("wall_ms", wall),
                        ("thread_cpu_ms", cpu),
                        ("exclusive_wall_ms", wall - frame[0]),
                        ("exclusive_thread_cpu_ms", cpu - frame[1]),
                    ):
                        values[key] += value / 1e6
            if graph:
                for index, key in enumerate(("captures", "replays")):
                    delta = getattr(args[0], key) - previous_counts[index]
                    graph_counts[key] += delta
                    if row is not None:
                        row["boundary_graph_" + key] += delta
            return result

        patches.append((owner, name, original))
        setattr(owner, name, wrapped)

    for owner, name, label in (
        (Scheduler, "schedule", "scheduler.schedule"),
        (Scheduler, "update_from_output", "scheduler.update_from_output"),
        (GPUModelRunner, "_prepare_inputs", "worker.prepare_inputs"),
        (GPUModelRunner, "_update_states", "worker.update_states"),
        (GPUModelRunner, "_build_attention_metadata", "worker.attention_metadata"),
        (GPUModelRunner, "_model_forward", "worker.model_forward"),
    ):
        annotate(owner, name, label)
    if not upstream:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.worker.ngc_cudagraph import NGCBoundaryGraph
        from vllm.v1.worker.ngc_scoring import NGCScoringBuffers

        for owner, name, label in (
            (KVCacheManager, "apply_ngc_evictions", "NGC.apply_evictions"),
            (GPUModelRunner, "_prepare_ngc_fused_decode", "NGC.prepare_fused_decode"),
            (GPUModelRunner, "_prepare_ngc_evicted", "NGC.prepare_offsets"),
            (GPUModelRunner, "_score_ngc_pages", "NGC.score_pages"),
            (NGCScoringBuffers, "prepare_score_inputs", "NGC.prepare_score_inputs"),
            (NGCScoringBuffers, "_copy_handoff", "NGC.proposal_to_cpu"),
            (NGCScoringBuffers, "take_proposals", "NGC.take_proposals"),
        ):
            annotate(owner, name, label)
        annotate(NGCBoundaryGraph, "run", "NGC.boundary_graph", graph=True)

    def save_profile(prof):
        prof.export_chrome_trace(str(trace))
        for event in prof.key_averages():
            averages.append(
                dict(
                    name=event.key,
                    calls=event.count,
                    self_cpu_us=event.self_cpu_time_total,
                    cpu_us=event.cpu_time_total,
                    self_device_us=event.self_device_time_total,
                    device_us=event.device_time_total,
                )
            )

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0, warmup=warmup_steps, active=active_steps, repeat=1
        ),
        on_trace_ready=save_profile,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    )

    def generated():
        request = next(iter(scheduler.requests.values()), None)
        return (
            (request.num_computed_tokens - request.num_prompt_tokens)
            if request is not None
            else -1
        )

    def step():
        nonlocal started, stopped, profiled_steps, active_row
        count = generated()
        if not started and count >= start_generated:
            torch.accelerator.synchronize()
            prof.start()
            started = True
            print(f"PROFILE_START {output.stem} generated={count}", flush=True)
        if started and not stopped:
            row = None
            if profiled_steps >= warmup_steps:
                inbox = getattr(scheduler, "ngc_eviction_inbox", None)
                row = dict(
                    generated_before=count,
                    operations={},
                    pending_proposals=len(inbox.pending) if inbox else 0,
                    boundary_graph_captures=0,
                    boundary_graph_replays=0,
                )
                active_row = row
            start_wall, start_cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                result = original_step()
            finally:
                active_row = None
            if row is not None:
                row.update(
                    generated_after=generated(),
                    wall_ms=(time.perf_counter_ns() - start_wall) / 1e6,
                    thread_cpu_ms=(time.thread_time_ns() - start_cpu) / 1e6,
                )
                row["kind"] = (
                    "scoring_boundary"
                    if row["boundary_graph_captures"] or row["boundary_graph_replays"]
                    else "proposal_application"
                    if row["pending_proposals"]
                    else "ordinary"
                )
                rows.append(row)
            prof.step()
            profiled_steps += 1
            if profiled_steps == warmup_steps + active_steps:
                torch.accelerator.synchronize()
                prof.stop()
                stopped = True
                print(f"PROFILE_END {output.stem} generated={generated()}", flush=True)
            return result
        return original_step()

    try:
        engine.step = step
        result = generate()
        assert all(len(item.outputs[0].token_ids) == 4096 for item in result)
        assert stopped and len(rows) == active_steps
        if not upstream:
            assert any(row["kind"] == "scoring_boundary" for row in rows), (
                "No scoring boundary captured"
            )
            assert any(row["kind"] == "proposal_application" for row in rows), (
                "No proposal step captured"
            )
    finally:
        if started and not stopped:
            prof.stop()
        engine.step = original_step
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
    assert trace.exists() and averages
    timeline = summarize_profile_trace(trace)
    assert len(timeline["steps"]) == len(rows), (
        "Host profiler steps differ from actual decode window"
    )
    for device_row, cpu_row in zip(timeline["steps"], rows):
        device_row["kind"] = cpu_row["kind"]
    timeline["summary"] = {}
    for kind in ("ordinary", "scoring_boundary", "proposal_application"):
        selected = [row for row in timeline["steps"] if row["kind"] == kind]
        timeline["summary"][kind] = {
            "steps": len(selected),
            "mean_ms": {
                key: statistics.mean(row[key] for row in selected)
                for key in timeline["steps"][0]
                if key.endswith("_ms") and selected
            },
        }
    phase_summary = {}
    for kind in ("ordinary", "scoring_boundary", "proposal_application"):
        selected = [row for row in rows if row["kind"] == kind]
        operations = {}
        for row in selected:
            for name, values in row["operations"].items():
                total = operations.setdefault(name, {key: 0.0 for key in values})
                for key, value in values.items():
                    total[key] += value
        phase_summary[kind] = dict(
            steps=len(selected),
            mean_wall_ms=statistics.mean(r["wall_ms"] for r in selected)
            if selected
            else None,
            mean_thread_cpu_ms=statistics.mean(r["thread_cpu_ms"] for r in selected)
            if selected
            else None,
            operations_totals=operations,
        )
    with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
        shutil.copyfileobj(source, target)
    trace.unlink()
    return dict(
        trace=trace.name + ".gz",
        events=averages,
        timeline=timeline,
        generated_window=[rows[0]["generated_before"], rows[-1]["generated_after"]],
        phases=phase_summary,
        steps=rows,
        boundary_graph_totals=graph_counts,
        boundary_graph_window={
            key: sum(r["boundary_graph_" + key] for r in rows) for key in graph_counts
        },
        note=(
            "Profiler-only attribution, not standalone speed. CPU wall spans overlap "
            "device work; thread CPU excludes waits. Exclusive timers subtract "
            "directly nested instrumented calls on the same thread. Device "
            "intervals clipped to host steps have asynchronous attribution limits; "
            "use whole-window GPU interval unions for totals."
        ),
    )


def compare_steps(reports: dict, cases: list[str], result_dir: Path) -> dict:
    """Relate decode-step latency to retained cache size for each case."""
    fields = [
        "step",
        "generated",
        "context_tokens",
        "retained_mean",
        "retained_min",
        "retained_max",
        "scoring_step",
        "eviction_step",
        "core_ms",
        "total_ms",
    ]
    summary = {}
    intervals: dict[int, dict] = {}
    for case in cases:
        steps = reports[case]["steps"]
        with (result_dir / f"steps-{case}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(steps)
        decode = steps[1:]
        plain = [s for s in decode if not s["scoring_step"] and not s["eviction_step"]]
        fit = statistics.linear_regression(
            [s["retained_mean"] for s in plain], [s["core_ms"] for s in plain]
        )
        summary[case] = {
            "decode_steps": len(decode),
            "core_ms_per_1k_retained_tokens": fit.slope * 1000,
            "core_ms_at_zero_retained": fit.intercept,
            "median_core_ms": statistics.median(s["core_ms"] for s in plain),
            "median_scoring_step_core_ms": statistics.median(
                [s["core_ms"] for s in decode if s["scoring_step"]] or [0.0]
            ),
            "median_eviction_step_core_ms": statistics.median(
                [s["core_ms"] for s in decode if s["eviction_step"]] or [0.0]
            ),
            "retained_mean_range": [
                min(s["retained_mean"] for s in decode),
                max(s["retained_mean"] for s in decode),
            ],
        }
        by_interval: dict[int, list[dict]] = {}
        for step in decode:
            by_interval.setdefault(step["generated"] // 128, []).append(step)
        for interval, members in by_interval.items():
            row = intervals.setdefault(interval, {"interval": interval})
            row[f"{case}_retained_mean"] = statistics.mean(
                s["retained_mean"] for s in members
            )
            row[f"{case}_median_core_ms"] = statistics.median(
                s["core_ms"] for s in members
            )
            row[f"{case}_first_step_core_ms"] = members[0]["core_ms"]
    rows = [intervals[key] for key in sorted(intervals)]
    with (result_dir / "steps_by_interval.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def sweep(
    model: str,
    result_dir: Path,
    *,
    profiling: bool = False,
    profile_only: bool = False,
    scheduler_profiling: bool = False,
    cases: list[str] | None = None,
    output_tokens: int = 2048,
    warmup_tokens: int = 2048,
    prompt_tokens: int = 0,
    cadence: int = 128,
    query_window: int = 5,
    repeats: int = 3,
    temperature: float = 0.6,
    step_trace: bool = False,
    num_seqs: int = 1,
) -> None:
    if cases is None:
        cases = (
            ["full", "0", "0.25"]
            if profiling
            else ["full", "0", "0.10", "0.25", "0.50", "0.75"]
        )
    baseline_case = next(
        (
            case
            for case in (
                "before-0-graph",
                "upstream-graph",
                "full-graph",
                "full",
                "upstream",
            )
            if case in cases
        ),
        "full",
    )
    assert baseline_case in cases, "Speedups need a full or upstream baseline"
    order = cases.copy()
    random.Random(0).shuffle(order)
    reports = {}
    upstream_root = os.environ.get("NGC_UPSTREAM_ROOT")
    before_root = os.environ.get("NGC_BEFORE_ROOT")
    if any(case.startswith("upstream") for case in cases) and not upstream_root:
        raise RuntimeError("The upstream case needs NGC_UPSTREAM_ROOT")
    if any(case.startswith("before-") for case in cases) and not before_root:
        raise RuntimeError("The before case needs NGC_BEFORE_ROOT")
    for case in order:
        output = result_dir / f"speed-{case}.json"
        script = __file__
        env = None
        if case.startswith("upstream"):
            # The script's own directory precedes PYTHONPATH on sys.path.
            assert upstream_root is not None
            script = str(Path(upstream_root) / Path(__file__).name)
            env = dict(os.environ, PYTHONPATH=upstream_root)
        elif case.startswith("before-"):
            assert before_root is not None
            script = str(Path(before_root) / Path(__file__).name)
            env = dict(os.environ, PYTHONPATH=before_root)
        subprocess.run(
            [
                sys.executable,
                script,
                "--model",
                model,
                "--result-dir",
                str(result_dir),
                "--case",
                case,
                "--output-tokens",
                str(output_tokens),
                "--warmup-tokens",
                str(warmup_tokens),
                "--num-seqs",
                str(num_seqs),
                "--prompt-tokens",
                str(prompt_tokens),
                "--cadence",
                str(cadence),
                "--query-window",
                str(query_window),
                "--repeats",
                str(repeats),
                "--temperature",
                str(temperature),
            ]
            + (["--profile"] if profiling else [])
            + (["--profile-only"] if profile_only else [])
            + (["--scheduler-profile"] if scheduler_profiling else [])
            + (["--step-trace"] if step_trace else []),
            check=True,
            env=env,
        )
        reports[case] = json.loads(output.read_text())
        expected_root = upstream_root if case.startswith("upstream") else "/opt/vllm"
        if case.startswith("before-"):
            expected_root = before_root
        if expected_root is not None:
            assert reports[case]["vllm_source"].startswith(f"{expected_root}/"), (
                case,
                reports[case]["vllm_source"],
            )
        print(f"NGC_CHECKPOINT {case} {result_dir}", flush=True)
    baseline = reports[baseline_case]
    rows = []
    for case in cases:
        report = reports[case]
        assert report["gpu"] == baseline["gpu"]
        assert report["sampling"] == baseline["sampling"]
        assert report["prompt_token_ids"] == baseline["prompt_token_ids"]
        if profile_only:
            rows.append({"case": case, "profile": report["profile"]["trace"]})
            continue
        rows.append(
            {
                "case": case,
                "median_seconds": report["median_seconds"],
                "min_seconds": min(report["seconds"]),
                "max_seconds": max(report["seconds"]),
                "tokens_per_second": report["tokens_per_second"],
                f"speedup_vs_{baseline_case}": baseline["median_seconds"]
                / report["median_seconds"],
                **(
                    {
                        "speedup_vs_upstream-graph": reports["upstream-graph"][
                            "median_seconds"
                        ]
                        / report["median_seconds"]
                    }
                    if "upstream-graph" in reports
                    else {}
                ),
            }
        )
    summary = {"order": order, "results": rows, "cases": reports}
    if "before-0-graph" in reports and "0-graph" in reports:
        summary["before_after_output_hashes_match"] = (
            reports["before-0-graph"]["output_token_hashes"]
            == reports["0-graph"]["output_token_hashes"]
        )
        assert summary["before_after_output_hashes_match"]
    for rate in ("0", "0.5"):
        before_case, after_case = f"before-{rate}-graph", f"{rate}-graph"
        if before_case in reports and after_case in reports:
            match = (
                reports[before_case]["output_token_hashes"]
                == reports[after_case]["output_token_hashes"]
            )
            summary[f"before_after_{rate}_hashes_match"] = match
            assert match, f"Output hashes differ for {rate}"
    if "upstream-graph" in reports and "0-graph" in reports:
        match = (
            reports["upstream-graph"]["output_token_hashes"]
            == reports["0-graph"]["output_token_hashes"]
        )
        summary["upstream_zero_output_hashes_match"] = match
        assert match, "Pristine upstream and zero-eviction output hashes differ"
    if step_trace:
        summary["step_trace"] = compare_steps(reports, cases, result_dir)
        print(json.dumps(summary["step_trace"], indent=2), flush=True)
    (result_dir / "speed.json").write_text(json.dumps(summary, indent=2))
    with (result_dir / "speed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--scheduler-profile", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--warmup-tokens", type=int, default=2048)
    parser.add_argument("--step-trace", action="store_true")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--cadence", type=int, default=128)
    parser.add_argument("--query-window", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    if (
        args.prompt_tokens < 0
        or args.cadence <= 0
        or args.query_window <= 0
        or args.repeats <= 0
    ):
        parser.error(
            "Prompt length must be nonnegative; cadence and query window positive"
        )
    if (
        (args.profile or args.scheduler_profile)
        and args.output_tokens != 2048
        and not args.profile_only
    ):
        parser.error("The profile window assumes 2048 output tokens")
    if args.case is None:
        sweep(
            args.model,
            args.result_dir,
            profiling=args.profile,
            profile_only=args.profile_only,
            scheduler_profiling=args.scheduler_profile,
            cases=args.cases,
            output_tokens=args.output_tokens,
            warmup_tokens=args.warmup_tokens,
            prompt_tokens=args.prompt_tokens,
            cadence=args.cadence,
            query_window=args.query_window,
            repeats=args.repeats,
            temperature=args.temperature,
            step_trace=args.step_trace,
            num_seqs=args.num_seqs,
        )
    else:
        run_case(
            args.model,
            args.case,
            args.result_dir / f"speed-{args.case}.json",
            profiling=args.profile,
            profile_only=args.profile_only,
            scheduler_profiling=args.scheduler_profile,
            output_tokens=args.output_tokens,
            warmup_tokens=args.warmup_tokens,
            prompt_tokens=args.prompt_tokens,
            cadence=args.cadence,
            query_window=args.query_window,
            repeats=args.repeats,
            temperature=args.temperature,
            step_trace=args.step_trace,
            num_seqs=args.num_seqs,
        )
