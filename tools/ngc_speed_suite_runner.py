# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One warmed model/batch/policy case; launched by ngc_speed_suite_modal.py."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
    }


class Measurements:
    """Observed token emission timings; arrival precedes batch submission."""

    def __init__(self, batch):
        self.times = [[] for _ in range(batch)]
        self.prefill_step_seconds = 0.0
        self.decode_seconds = 0.0
        self.decode_tokens = 0
        self.full_batch_decode_seconds = 0.0
        self.full_batch_decode_steps = 0

    def step(self, elapsed, timestamp, counts, has_prefill, warmed):
        deltas = []
        for row, count in enumerate(counts):
            delta = count - len(self.times[row])
            if delta not in (0, 1):
                raise RuntimeError("Expected at most one generated token/request/step")
            if delta:
                self.times[row].append(timestamp)
            deltas.append(delta)
        if has_prefill:
            self.prefill_step_seconds += elapsed
        elif warmed:
            self.decode_seconds += elapsed
            self.decode_tokens += sum(deltas)
            if all(delta == 1 for delta in deltas):
                self.full_batch_decode_seconds += elapsed
                self.full_batch_decode_steps += 1

    def finish(self, seconds):
        if not self.times or any(not row for row in self.times):
            raise RuntimeError("Every request must emit at least one token")
        first = [row[0] for row in self.times]
        last = [row[-1] for row in self.times]
        itls = [b - a for row in self.times for a, b in zip(row, row[1:])]
        total = sum(map(len, self.times))
        return {
            "total_request_latency_seconds": distribution(last),
            "ttft_seconds": distribution(first),
            "per_request_decode_seconds": distribution(
                [b - a for a, b in zip(first, last)]
            ),
            "inter_token_latency_seconds": distribution(itls),
            "per_request_mean_itl_seconds": distribution(
                [
                    (row[-1] - row[0]) / (len(row) - 1)
                    for row in self.times
                    if len(row) > 1
                ]
            ),
            "batch_completion_seconds": seconds,
            "batch_prefill_span_seconds": max(first),
            "prefill_involving_steps_seconds": self.prefill_step_seconds,
            "output_tokens": total,
            "end_to_end_output_tokens_per_second": total / seconds,
            "warmed_decode_seconds": self.decode_seconds,
            "warmed_decode_output_tokens": self.decode_tokens,
            "warmed_decode_tokens_per_second": self.decode_tokens / self.decode_seconds
            if self.decode_seconds
            else None,
            "warmed_decode_ms_per_output_token": 1000
            * self.decode_seconds
            / self.decode_tokens
            if self.decode_tokens
            else None,
            "full_batch_decode_steps": self.full_batch_decode_steps,
            "full_batch_decode_ms_per_step": 1000
            * self.full_batch_decode_seconds
            / self.full_batch_decode_steps
            if self.full_batch_decode_steps
            else None,
            "request_first_token_seconds": first,
            "request_last_token_seconds": last,
        }


def run(args):
    import torch
    from decode_profile import retained_snapshot

    import vllm
    from vllm import LLM, SamplingParams

    actual = str(Path(vllm.__file__).resolve().parent)
    if actual != os.environ["NGC_SUITE_SOURCE"]:
        raise RuntimeError(f"Wrong vLLM source: {actual}")
    if args.policy != os.environ.get("NGC_SUITE_RATE", "upstream"):
        raise RuntimeError("Requested policy differs from worker policy environment")
    ngc = args.policy != "upstream"
    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        distributed_executor_backend="mp" if args.tp > 1 else "uni",
        worker_extension_cls="ngc_suite_worker.SuiteWorker",
        max_num_seqs=args.batch,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        attention_config={"backend": "TRITON_ATTN"},
        async_scheduling=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        disable_log_stats=True,
        seed=0,
    )
    init_seconds = time.perf_counter() - init_start
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    tokenizer = llm.get_tokenizer()
    seed = tokenizer.encode(
        "Write a detailed fantasy story about a traveler following a river, "
        "including dialogue, discoveries, and a journey through unfamiliar lands. "
    )
    prompt = (seed * (2048 // len(seed) + 1))[:2048]
    params = SamplingParams(
        temperature=0, max_tokens=6144, ignore_eos=True, detokenize=False
    )

    def generation(measure):
        metrics = Measurements(args.batch)
        references = []
        maximum_progress = {}
        recomputed = 0
        occupancy = []
        retained = []
        original_schedule = scheduler.schedule
        current_prefill = False

        def schedule(*schedule_args, **schedule_kwargs):
            nonlocal recomputed, current_prefill
            output = original_schedule(*schedule_args, **schedule_kwargs)
            current_prefill = False
            for req_id, count in output.num_scheduled_tokens.items():
                request = scheduler.requests[req_id]
                start = request.num_computed_tokens - count
                previous = maximum_progress.get(req_id, 0)
                recomputed += max(0, min(start + count, previous) - start)
                maximum_progress[req_id] = max(previous, start + count)
                current_prefill |= start < len(request.prompt_token_ids)
            occupancy.append(scheduler.kv_cache_manager.usage)
            return output

        if measure:
            scheduler.schedule = schedule
        try:
            if measure:
                memory_before = llm.collective_rpc(
                    "suite_memory", kwargs={"reset": True}
                )
            start = time.perf_counter()
            for row in range(args.batch):
                req_id = engine.add_request(
                    f"suite-{measure}-{row}", {"prompt_token_ids": prompt}, params
                )
                references.append(scheduler.requests[req_id])
            step = 0
            while engine.has_unfinished_requests():
                warmed = all(r.num_output_tokens >= 16 for r in references)
                tick = time.perf_counter()
                engine.step()
                now = time.perf_counter()
                if measure:
                    metrics.step(
                        now - tick,
                        now - start,
                        [r.num_output_tokens for r in references],
                        current_prefill,
                        warmed,
                    )
                    if step % 256 == 0:
                        retained.append(
                            {"step": step, "state": retained_snapshot(scheduler, ngc)}
                        )
                step += 1
            seconds = time.perf_counter() - start
        finally:
            scheduler.schedule = original_schedule
        if any(r.num_output_tokens != 6144 for r in references):
            raise RuntimeError("Generation did not produce exactly 6144 tokens/request")
        if not measure:
            return None
        memory_after = llm.collective_rpc("suite_memory")
        result = metrics.finish(seconds)
        result.update(
            valid_fixed_batch=not any(r.num_preemptions for r in references),
            preemptions=sum(r.num_preemptions for r in references),
            recomputed_scheduled_tokens=recomputed,
            kv_occupancy_fraction=distribution(occupancy),
            peak_kv_occupancy_fraction=max(occupancy),
            retained_samples=retained,
            memory_before_by_rank=memory_before,
            memory_after_by_rank=memory_after,
            peak_allocator_allocated_bytes_sum=sum(
                r["peak_allocated_bytes"] for r in memory_after
            ),
            peak_allocator_reserved_bytes_sum=sum(
                r["peak_reserved_bytes"] for r in memory_after
            ),
            output_sha256=[
                hashlib.sha256(
                    json.dumps(list(r.output_token_ids)).encode()
                ).hexdigest()
                for r in references
            ],
        )
        return result

    generation(False)  # Match the complete measured trajectory, including boundaries.
    result = generation(True)
    report = {
        "status": "complete",
        "config": vars(args),
        "source_root": actual,
        "software": {"torch": torch.__version__, "vllm": vllm.__version__},
        "workload": {
            "prompt_tokens": 2048,
            "generated_tokens_per_request": 6144,
            "cadence": 256,
            "query_window": 5,
            "min_retained_tokens": 32,
            "dtype": "bfloat16",
            "attention_backend": "TRITON_ATTN",
            "ignore_eos": True,
            "temperature": 0,
            "prefix_caching": False,
        },
        "model_initialization_seconds": init_seconds,
        "metrics": result,
        "definitions": {
            "observer": (
                "Batch latency includes CPU metric collection and retained-layout "
                "snapshots; warmed decode sums engine.step wall time and excludes "
                "post-step observation. No per-step GPU synchronization is "
                "inserted."
            ),
            "timing": (
                "Warmed in-process offline engine wall time; includes scheduler "
                "and worker completion, excludes model initialization and separate "
                "full-generation warmup."
            ),
            "ttft": (
                "Batch arrival before submission to first observed token; includes "
                "submission overhead and queuing."
            ),
            "prefill": (
                "Sum of steps containing prompt work; mixed steps include "
                "simultaneous decode. Also report time until last request's first "
                "token. Neither is isolated GPU prefill time."
            ),
            "decode": (
                "Steps with no prompt work after every request has "
                "emitted16tokens; may include shrinking batch and recomputation of "
                "generated context. Full-batch steps also reported separately."
            ),
            "itl": (
                "Observed emission intervals, first token excluded; pooled "
                "distribution weights each interval equally."
            ),
            "memory": (
                "Per-rank PyTorch allocator peaks, including resident "
                "weights/KV/graphs; not total device peak. Sum of rank peaks need "
                "not occur simultaneously."
            ),
            "kv": (
                "Occupancy sampled after every scheduler.schedule before "
                "execution/free; retained layouts sampled every256steps."
            ),
            "recomputation": (
                "Scheduled token positions below previously scheduled high-water "
                "mark, including replayed generated context; not additional output "
                "tokens."
            ),
            "tp": (
                "Same TP for all policies within a model. NGC boundary CUDA graphs "
                "are currently TP1-only; ordinary decode graphs remain enabled."
            ),
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 48), required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
