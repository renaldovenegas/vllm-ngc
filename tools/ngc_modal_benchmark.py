# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end wall-clock rate sweep inside one Modal GPU allocation."""

import argparse
import csv
import functools
import gzip
import hashlib
import inspect
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import regex as re


def continuous_capture(self, layer, args, kwargs):
    """Previous capture implementation, adapted only to the new batch tuple."""
    import torch

    from vllm.v1.worker.ngc_scoring import NGCQueryHistory

    if self._positions is None:
        return
    query = args[0] if args else kwargs["query"]
    query = query.reshape(-1, layer.num_heads, layer.head_size)
    if self._batch and self._batch[-1][2] > query.shape[0]:
        raise ValueError("NGC capture query count does not cover the prepared batch")
    capacity = layer.ngc_eviction_config.query_buffer_size
    for req_id, start, end, _ in self._batch:
        if start == end:
            continue
        key = (req_id, layer.layer_name)
        history = self.histories.get(key)
        if history is None:
            history = NGCQueryHistory(
                torch.empty(
                    (capacity, *query.shape[1:]), dtype=query.dtype, device=query.device
                ),
                torch.empty(capacity, dtype=torch.int64, device=query.device),
            )
            self.histories[key] = history
        incoming = min(end - start, capacity)
        old = min(history.count, capacity - incoming)
        if old:
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
        history.page_scores = None
        history.decision = None


def run_case(
    model: str,
    case: str,
    output: Path,
    *,
    profiling: bool = False,
    output_tokens: int = 2048,
    warmup_tokens: int = 2048,
) -> None:
    import torch

    from vllm import LLM, SamplingParams
    from vllm.config.cache import NGCEvictionConfig
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager
    from vllm.v1.worker.ngc_scoring import NGCScoringBuffers

    capture_mode, _, rate_text = case.partition("-")
    rate = None if case == "full" else float(rate_text or case)
    original_capture = NGCScoringBuffers.capture
    if capture_mode == "continuous":
        NGCScoringBuffers.capture = continuous_capture
    original_init = Attention.__init__
    signature = inspect.signature(original_init)
    layers = []

    @functools.wraps(original_init)
    def initialize_attention(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        prefix = bound.arguments.get("prefix", "")
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
        if match is None:
            raise RuntimeError(f"Cannot identify layer in {prefix!r}")
        if rate is not None and int(match.group(1)) % 2 == 0:
            bound.arguments["ngc_eviction_config"] = NGCEvictionConfig(
                cadence=128,
                eviction_rate=rate,
                query_buffer_size=5,
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
            enforce_eager=True,
            async_scheduling=False,
            distributed_executor_backend="uni",
            tensor_parallel_size=1,
            attention_config={"backend": "TRITON_ATTN"},
            block_size=16,
            max_model_len=max(4096, output_tokens + 1024),
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.65,
            enable_prefix_caching=False,
            seed=0,
            jit_monitor_verbose=profiling,
        )
        load_seconds = time.perf_counter() - started
    finally:
        Attention.__init__ = original_init
        NGCScoringBuffers.capture = original_capture
    assert bool(layers) == (rate is not None)
    prompt = "Write a long fantasy story that is approximately 3000 words."
    tokenizer = llm.get_tokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert token_ids and all(isinstance(token, int) for token in token_ids)
    sampling = dict(
        temperature=0.6,
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
        params = dict(sampling, min_tokens=num_tokens, max_tokens=num_tokens)
        return llm.generate(
            [{"prompt_token_ids": token_ids}],
            SamplingParams(**params),
            use_tqdm=False,
        )

    original_evict = NGCEvictionManager.apply_eviction
    warmup_events = []

    @functools.wraps(original_evict)
    def validate_eviction(self, proposal, computed_tokens):
        before = self.block_pool.get_num_free_blocks()
        applied = original_evict(self, proposal, computed_tokens)
        if applied:
            assert self.block_pool.get_num_free_blocks() - before == len(
                proposal.victim_indices
            )
            warmup_events.append(len(proposal.victim_indices))
        return applied

    NGCEvictionManager.apply_eviction = validate_eviction
    try:
        warmup = generate(warmup_tokens)
        torch.accelerator.synchronize()
    finally:
        NGCEvictionManager.apply_eviction = original_evict
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
    for repeat in range(3):
        torch.accelerator.synchronize()
        started = time.perf_counter()
        result = generate()
        torch.accelerator.synchronize()
        seconds.append(time.perf_counter() - started)
        assert len(result[0].outputs[0].token_ids) == output_tokens
        output_hashes.append(
            hashlib.sha256(
                json.dumps(result[0].outputs[0].token_ids).encode()
            ).hexdigest()
        )
        print(
            f"TIMING {case} repeat={repeat + 1} seconds={seconds[-1]:.6f}", flush=True
        )
    median = statistics.median(seconds)
    report = {
        "model": model,
        "case": case,
        "capture_mode": "continuous" if capture_mode == "continuous" else "windowed",
        "output_token_hashes": output_hashes,
        "eviction_rate": rate,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ngc_layers": layers,
        "cadence": 128,
        "query_buffer_size": 5,
        "min_retained_tokens": 32,
        "batch_size": 1,
        "prompt_token_ids": token_ids,
        "sampling": sampling,
        "output_tokens": output_tokens,
        "warmup_tokens": warmup_tokens,
        "seconds": seconds,
        "median_seconds": median,
        "tokens_per_second": output_tokens / median,
        "model_load_seconds": load_seconds,
        "warmup_eviction_decisions": len(warmup_events),
        "warmup_freed_pages": sum(warmup_events),
        "jit_events_after_warmup": jit_events,
    }
    if profiling:
        print(f"JIT_TIMED_SUMMARY {case} events={len(jit_events)}", flush=True)
        output.write_text(json.dumps(report, indent=2))
        phase = "profile"
        report["profile"] = profile_generation(llm, generate, output)
        jit_monitor._handle_jit_event = original_jit_event
    output.write_text(json.dumps(report, indent=2))


def profile_generation(llm, generate, output: Path) -> dict:
    """Capture late decode including the 1920-token eviction boundary."""
    import torch

    from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager
    from vllm.v1.worker import ngc_scoring
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
            for key, hook in list(module._forward_pre_hooks.items()):
                if (
                    getattr(hook, "__func__", None)
                    is ngc_scoring.NGCScoringBuffers.capture
                ):

                    def wrap_capture(original):
                        @functools.wraps(original)
                        def wrapped(*args, **kwargs):
                            with torch.profiler.record_function("NGC.query_capture"):
                                return original(*args, **kwargs)

                        return wrapped

                    capture_hooks.append((module, key, hook))
                    module._forward_pre_hooks[key] = wrap_capture(hook)

    llm.apply_model(annotate_capture)
    annotate(ngc_scoring.NGCScoringBuffers, "score_if_due", "NGC.score_if_due")
    annotate(ngc_scoring.NGCEvictionDecision, "to_proposal", "NGC.proposal_to_cpu")
    annotate(ngc_scoring, "select_ngc_victims", "NGC.select_victims")
    annotate(NGCEvictionManager, "apply_eviction", "NGC.apply_eviction")
    annotate(GPUModelRunner, "_score_ngc_pages", "NGC.score_pages")
    annotate(GPUModelRunner, "_prepare_inputs", "worker.prepare_inputs")
    annotate(GPUModelRunner, "_update_states", "worker.update_states")
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
        for module, key, hook in capture_hooks:
            module._forward_pre_hooks[key] = hook
    assert rows and trace.exists(), "Profiler did not capture the decode window"
    if capture_hooks:
        assert any(row["name"] == "NGC.query_capture" for row in rows)
    with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
        shutil.copyfileobj(source, target)
    trace.unlink()
    return {
        "wait_steps": 1792,
        "warmup_steps": 8,
        "active_steps": 144,
        "trace": trace.name + ".gz",
        "events": rows,
        "note": "Inclusive nested CPU/device totals overlap; do not add them. "
        "Profile instrumentation is excluded from wall-clock samples.",
    }


def sweep(
    model: str,
    result_dir: Path,
    *,
    profiling: bool = False,
    compare_capture: bool = False,
    cases: list[str] | None = None,
    output_tokens: int = 2048,
    warmup_tokens: int = 2048,
) -> None:
    if cases is None:
        cases = (
            ["full", "0", "0.25"]
            if profiling
            else ["full", "0", "0.10", "0.25", "0.50", "0.75"]
        )
    assert "full" in cases, "Speedups are relative to full attention"
    if compare_capture:
        cases = [
            "full",
            "continuous-0",
            "windowed-0",
            "continuous-0.25",
            "windowed-0.25",
        ]
    order = cases.copy()
    random.Random(0).shuffle(order)
    reports = {}
    for case in order:
        output = result_dir / f"speed-{case}.json"
        subprocess.run(
            [
                sys.executable,
                __file__,
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
            ]
            + (["--profile"] if profiling else []),
            check=True,
        )
        reports[case] = json.loads(output.read_text())
        print(f"NGC_CHECKPOINT {case} {result_dir}", flush=True)
    baseline = reports["full"]
    rows = []
    for case in cases:
        report = reports[case]
        assert report["gpu"] == baseline["gpu"]
        assert report["sampling"] == baseline["sampling"]
        assert report["prompt_token_ids"] == baseline["prompt_token_ids"]
        rows.append(
            {
                "case": case,
                "median_seconds": report["median_seconds"],
                "min_seconds": min(report["seconds"]),
                "max_seconds": max(report["seconds"]),
                "tokens_per_second": report["tokens_per_second"],
                "speedup_vs_full": baseline["median_seconds"]
                / report["median_seconds"],
            }
        )
    summary = {"order": order, "results": rows, "cases": reports}
    if compare_capture:
        comparisons = []
        for rate in ("0", "0.25"):
            old = reports[f"continuous-{rate}"]
            new = reports[f"windowed-{rate}"]
            comparisons.append(
                {
                    "eviction_rate": float(rate),
                    "continuous_seconds": old["median_seconds"],
                    "windowed_seconds": new["median_seconds"],
                    "speedup": old["median_seconds"] / new["median_seconds"],
                    "latency_reduction_percent": 100
                    * (1 - new["median_seconds"] / old["median_seconds"]),
                    "same_output_tokens": old["output_token_hashes"]
                    == new["output_token_hashes"],
                }
            )
        summary["capture_comparison"] = comparisons
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
    parser.add_argument("--compare-capture", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--warmup-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.profile and args.output_tokens != 2048:
        parser.error("The profile window assumes 2048 output tokens")
    if args.case is None:
        sweep(
            args.model,
            args.result_dir,
            profiling=args.profile,
            compare_capture=args.compare_capture,
            cases=args.cases,
            output_tokens=args.output_tokens,
            warmup_tokens=args.warmup_tokens,
        )
    else:
        run_case(
            args.model,
            args.case,
            args.result_dir / f"speed-{args.case}.json",
            profiling=args.profile,
            output_tokens=args.output_tokens,
            warmup_tokens=args.warmup_tokens,
        )
