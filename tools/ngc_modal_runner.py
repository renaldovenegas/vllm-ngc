# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote NGC checks; uses the Modal image's editable vLLM installation."""

import argparse
import functools
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import regex as re


def check_environment() -> None:
    import torch
    import triton
    import vllm._C_stable_libtorch  # noqa: F401

    import vllm

    assert Path(vllm.__file__).resolve().parent == Path("/opt/vllm/vllm")
    assert torch.cuda.is_available(), "CUDA is unavailable"
    from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager

    print(
        json.dumps(
            {
                "vllm_source": vllm.__file__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": triton.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "manager": NGCEvictionManager.__name__,
            },
            indent=2,
        ),
        flush=True,
    )


def smoke(
    model: str,
    result_dir: Path,
    *,
    story: bool = False,
    ngc_enabled: bool = True,
    equivalence: bool = False,
    graphs: bool = False,
    eviction_rate: float = 0.25,
    cadence: int = 128,
    gpu_memory_utilization: float = 0.65,
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config.cache import NGCEvictionConfig
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager
    from vllm.v1.worker.ngc_scoring import NGCScoringBuffers

    disable_fusion = os.environ.get("NGC_MODAL_DISABLE_FUSION") == "1"
    original_prepare_fused = None
    if disable_fusion:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        original_prepare_fused = GPUModelRunner._prepare_ngc_fused_decode

        def prepare_without_fusion(self, *args, **kwargs):
            self.ngc_scoring_buffers.reset_fused_decode()
            return False

        GPUModelRunner._prepare_ngc_fused_decode = prepare_without_fusion

    original_init = Attention.__init__
    signature = inspect.signature(original_init)
    batched = hasattr(NGCEvictionManager, "detach_eviction_blocks")
    eviction_method = "detach_eviction_blocks" if batched else "apply_eviction"
    original_evict = getattr(NGCEvictionManager, eviction_method)
    original_batch = getattr(KVCacheManager, "apply_ngc_evictions", None)
    configured_layers = {"ngc": [], "full": []}
    events = []
    zero_decisions = []
    original_score = NGCScoringBuffers.score_if_due
    original_fused = getattr(NGCScoringBuffers, "fused_decode", None)
    try:
        from vllm.v1.worker.ngc_cudagraph import NGCBoundaryGraph
    except ModuleNotFoundError:
        NGCBoundaryGraph = None
    original_graph_run = NGCBoundaryGraph.run if NGCBoundaryGraph is not None else None
    fused_counts = {
        "supported": original_fused is not None,
        "disabled_by_harness": disable_fusion,
        "successful_calls": 0,
        "precomputed_score_decisions": 0,
        "boundary_graph_captures": 0,
        "boundary_graph_replays": 0,
    }

    def record_graph_run(self, *args, **kwargs):
        captures, replays = self.captures, self.replays
        result = original_graph_run(self, *args, **kwargs)
        fused_counts["boundary_graph_captures"] += self.captures - captures
        fused_counts["boundary_graph_replays"] += self.replays - replays
        return result

    def record_fused(self, *args, **kwargs):
        result = original_fused(self, *args, **kwargs)
        if result:
            fused_counts["successful_calls"] += 1
        return result

    @functools.wraps(original_score)
    def record_score(self, req_id, layer_name, **kwargs):
        decision = original_score(self, req_id, layer_name, **kwargs)
        if decision is not None and kwargs.get("precomputed_scores") is not None:
            fused_counts["precomputed_score_decisions"] += 1
        if (equivalence or eviction_rate == 0) and decision is not None:
            assert decision.victim_indices.numel() == 0
            zero_decisions.append((req_id, layer_name, decision.cadence_boundary))
        return decision

    @functools.wraps(original_init)
    def initialize_attention(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        prefix = bound.arguments.get("prefix", "")
        if re.search(r"(?:^|\.)layers\.\d+\.", prefix) is None:
            raise RuntimeError(f"Cannot identify decoder layer in {prefix!r}")
        enabled = ngc_enabled
        if enabled:
            bound.arguments["ngc_eviction_config"] = NGCEvictionConfig(
                cadence=cadence,
                eviction_rate=0.0 if equivalence else eviction_rate,
                query_buffer_size=5,
                min_retained_tokens=32,
            )
        configured_layers["ngc" if enabled else "full"].append(prefix)
        original_init(*bound.args, **bound.kwargs)

    @functools.wraps(original_evict)
    def record_eviction(self, proposal, computed_tokens):
        before = self.block_pool.get_num_free_blocks()
        result = original_evict(self, proposal, computed_tokens)
        applied = result is not None if batched else result
        if equivalence:
            assert not proposal.victim_indices
            assert self.block_pool.get_num_free_blocks() == before
            assert self.layouts[proposal.request_id].evicted_tokens == 0
        if applied and proposal.victim_indices:
            freed = (
                len(result)
                if batched
                else (self.block_pool.get_num_free_blocks() - before)
            )
            assert freed == len(proposal.victim_indices)
            event = {
                "request": proposal.request_id,
                "group": proposal.group_id,
                "computed_tokens": computed_tokens,
                "freed_pages": freed,
                "evicted_tokens": self.layouts[proposal.request_id].evicted_tokens,
            }
            events.append(event)
            with (result_dir / "evictions.jsonl").open("a") as output:
                output.write(json.dumps(event) + "\n")
        return result

    def record_batch(self, requests, proposals):
        before = self.coordinator.block_pool.get_num_free_blocks()
        start = len(events)
        result = original_batch(self, requests, proposals)
        assert self.coordinator.block_pool.get_num_free_blocks() - before == sum(
            event["freed_pages"] for event in events[start:]
        )
        return result

    Attention.__init__ = initialize_attention
    setattr(NGCEvictionManager, eviction_method, record_eviction)
    if batched:
        KVCacheManager.apply_ngc_evictions = record_batch
    NGCScoringBuffers.score_if_due = record_score
    if original_fused is not None:
        NGCScoringBuffers.fused_decode = record_fused
    if original_graph_run is not None:
        NGCBoundaryGraph.run = record_graph_run
    try:
        llm = LLM(
            model=model,
            enforce_eager=not graphs,
            async_scheduling=False,
            distributed_executor_backend="uni",
            tensor_parallel_size=1,
            attention_config={"backend": "TRITON_ATTN"},
            dtype="float16",
            kv_cache_dtype="auto",
            block_size=16,
            max_model_len=8192 if story else 4096,
            max_num_seqs=2,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            seed=0,
        )
        assert bool(configured_layers["ngc"]) == ngc_enabled
        assert bool(configured_layers["full"]) != ngc_enabled, (
            "NGC runs configure every decoder layer"
        )
        prompts = [
            ("The river flows past the village and into the sea. " * 24) + ending
            for ending in ["Continue the story:", "Describe the landscape:"]
        ]
        sampling = dict(
            temperature=0.6,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            max_tokens=6144 if story else 2048,
            min_tokens=2048,
            ignore_eos=not story,
            seed=0,
        )
        if equivalence:
            sampling.update(
                temperature=0.0,
                min_tokens=512,
                max_tokens=512,
                ignore_eos=True,
                logprobs=0,
            )
            prompts = [
                "Write a long fantasy story that is approximately 3000 words.",
                "Explain how a sailing ship navigates using stars. "
                "Include a worked example and discuss sources of error.",
            ]
            outputs = llm.chat(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                SamplingParams(**sampling),
            )
        elif story:
            prompts = ["Write a long fantasy story that is approximately 3000 words."]
            outputs = llm.chat(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                SamplingParams(**sampling),
            )
        else:
            outputs = llm.generate(prompts, SamplingParams(**sampling))
        assert all(
            sampling["min_tokens"]
            <= len(output.outputs[0].token_ids)
            <= sampling["max_tokens"]
            for output in outputs
        )
        cycles: dict[tuple[str, int], int] = {}
        for event in events:
            key = (event["request"], event["group"])
            cycles[key] = cycles.get(key, 0) + 1
        if equivalence:
            assert not events
            expected = {
                (output.request_id, layer, boundary)
                for output in outputs
                for layer in configured_layers["ngc"]
                for boundary in range(
                    cadence,
                    len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                    cadence,
                )
            }
            observed = {
                (req.rsplit("-", 1)[0], layer, boundary)
                for req, layer, boundary in zero_decisions
            }
            assert observed == expected, (observed, expected)
        elif ngc_enabled and eviction_rate == 0:
            assert not events
            assert zero_decisions, "Zero-rate NGC must still score cadence boundaries"
        elif ngc_enabled:
            assert len(cycles) == len(prompts) * len(configured_layers["ngc"])
            assert all(count >= 2 for count in cycles.values())
        else:
            assert not events, "Full attention must not evict"
        report = {
            "fused_decode": fused_counts,
            "model": model,
            "attention_mode": "ngc_all_layers" if ngc_enabled else "full",
            "enforce_eager": not graphs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "chat_template_applied": story or equivalence,
            "zero_decisions": zero_decisions,
            "layers": configured_layers,
            "eviction_events": len(events),
            "sampling": sampling,
            "ngc": {
                "cadence": cadence,
                "eviction_rate": 0.0 if equivalence else eviction_rate,
                "query_buffer_size": 5,
                "min_retained_tokens": 32,
            }
            if ngc_enabled
            else None,
            "prompts": prompts,
            "output_token_counts": [
                len(output.outputs[0].token_ids) for output in outputs
            ],
            "outputs": [output.outputs[0].text for output in outputs],
            "word_counts": [len(output.outputs[0].text.split()) for output in outputs],
            "finish_reasons": [output.outputs[0].finish_reason for output in outputs],
            "prompt_token_ids": [output.prompt_token_ids for output in outputs],
        }
        if equivalence:
            report["token_ids"] = [output.outputs[0].token_ids for output in outputs]
            report["selected_logprobs"] = [
                [
                    step[token].logprob
                    for token, step in zip(
                        output.outputs[0].token_ids, output.outputs[0].logprobs
                    )
                ]
                for output in outputs
            ]
        for index, output in enumerate(outputs):
            (result_dir / f"output-{index}.txt").write_text(output.outputs[0].text)
        (result_dir / "smoke.json").write_text(json.dumps(report, indent=2))
        if graphs and ngc_enabled and equivalence and original_graph_run is not None:
            assert fused_counts["boundary_graph_captures"] > 0
            assert fused_counts["boundary_graph_replays"] > 0
        print(json.dumps(report, indent=2))
    finally:
        (result_dir / "fused-instrumentation.json").write_text(
            json.dumps(fused_counts, indent=2)
        )
        Attention.__init__ = original_init
        if original_prepare_fused is not None:
            GPUModelRunner._prepare_ngc_fused_decode = original_prepare_fused
        setattr(NGCEvictionManager, eviction_method, original_evict)
        if batched:
            KVCacheManager.apply_ngc_evictions = original_batch
        NGCScoringBuffers.score_if_due = original_score
        if original_fused is not None:
            NGCScoringBuffers.fused_decode = original_fused
        if original_graph_run is not None:
            NGCBoundaryGraph.run = original_graph_run


def zero_equivalence(model: str, result_dir: Path, *, graphs: bool = False) -> None:
    """Compare fresh engines on the same GPU across three scoring boundaries.

    Exact greedy tokens and selected-token logprobs test the model's decode
    behavior with active per-layer bookkeeping, without changing its history.
    Both engines use the same execution mode, eager or CUDA graphs.
    """
    reports = []
    suffix = "-graph" if graphs else ""
    for mode in (f"zero-full{suffix}", f"zero-ngc{suffix}"):
        target = result_dir / mode
        target.mkdir()
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--job",
                mode,
                "--model",
                model,
                "--result-dir",
                str(target),
            ],
            check=True,
        )
        report = json.loads((target / "smoke.json").read_text())
        (result_dir / f"{mode}.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
    full, ngc = reports
    errors = [
        abs(a - b)
        for left, right in zip(full["selected_logprobs"], ngc["selected_logprobs"])
        for a, b in zip(left, right)
    ]
    result = {
        "same_prompt_tokens": full["prompt_token_ids"] == ngc["prompt_token_ids"],
        "same_sampling": full["sampling"] == ngc["sampling"],
        "same_output_tokens": full["token_ids"] == ngc["token_ids"],
        "max_selected_logprob_error": max(errors),
        "absolute_tolerance": 0.005,
        "tokens_compared": len(errors),
        "zero_victim_scoring_decisions": len(ngc["zero_decisions"]),
        "eviction_events": ngc["eviction_events"],
    }
    result["passed"] = (
        result["same_prompt_tokens"]
        and result["same_sampling"]
        and result["same_output_tokens"]
        and max(errors) <= 0.005
        and ngc["eviction_events"] == 0
    )
    (result_dir / "equivalence.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], "Zero-eviction equivalence failed; see equivalence.json"


def output_sweep(model: str, result_dir: Path) -> None:
    """Save matched story generations from isolated engines at four rates."""
    reports = {}
    for rate in (0.0, 0.25, 0.5, 0.75):
        name = f"eviction-{int(rate * 100)}"
        target = result_dir / name
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--job",
                "output-case",
                "--model",
                model,
                "--eviction-rate",
                str(rate),
                "--result-dir",
                str(target),
            ],
            check=True,
        )
        reports[name] = json.loads((target / "smoke.json").read_text())
        (result_dir / "outputs.json").write_text(json.dumps(reports, indent=2))
        sections = [
            f"=== {key} ===\n\n{report['outputs'][0]}"
            for key, report in reports.items()
        ]
        (result_dir / "outputs.txt").write_text("\n\n".join(sections))
        print(f"NGC_CHECKPOINT {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        choices=[
            "check",
            "kernels",
            "metadata",
            "smoke",
            "smoke-graph",
            "story-ngc",
            "story-full",
            "output-sweep",
            "output-case",
            "zero-equivalence",
            "zero-equivalence-graph",
            "zero-full",
            "zero-ngc",
            "zero-full-graph",
            "zero-ngc-graph",
            "speed",
            "profile",
            "speed-long",
            "speed-upstream",
            "upstream-check",
            "scheduler-ab",
            "metadata-ab",
            "speed-eviction",
            "speed-4k",
            "validate-inputs",
            "step-trace",
            "speed-batch",
            "speed-batch-graph",
            "profile-batch-graph",
            "profile-4k",
            "profile-scheduler",
            "bookkeeping-ab",
            "graph-baseline",
        ],
        required=True,
    )
    parser.add_argument("--selector", default="ngc")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--eviction-rate", type=float, default=0.25)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.eviction_rate < 1:
        parser.error("--eviction-rate must be in [0, 1)")
    if args.repeats <= 0 or (
        args.repeats != 1 and args.job not in {"metadata-ab", "speed-4k"}
    ):
        parser.error("positive --repeats applies only to metadata-ab and speed-4k")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if (
        args.job not in {"speed-eviction", "speed-4k", "metadata-ab", "profile-4k"}
        and args.batch_size != 50
    ):
        parser.error(
            "--batch-size applies only to speed-eviction, speed-4k and metadata-ab"
        )
    check_environment()
    if args.job == "validate-inputs":
        for job in ("zero-equivalence", "zero-equivalence-graph", "smoke-graph"):
            target = args.result_dir / job
            target.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--job",
                    job,
                    "--model",
                    args.model,
                    "--result-dir",
                    str(target),
                ],
                check=True,
            )
        return
    if args.job == "upstream-check":
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("ngc_modal_benchmark.py")),
                "--model",
                args.model,
                "--result-dir",
                str(args.result_dir),
                "--cases",
                "upstream-graph",
                "0-graph",
                "--num-seqs",
                "2",
                "--output-tokens",
                "512",
                "--warmup-tokens",
                "256",
                "--repeats",
                "1",
                "--temperature",
                "0",
            ],
            check=True,
        )
        return
    if args.job == "graph-baseline":
        # Upstream eager vs CUDA graphs at batch 1 and 50 on one GPU.
        for num_seqs in (1, 50):
            target = args.result_dir / f"batch-{num_seqs}"
            target.mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("ngc_modal_benchmark.py")),
                    "--model",
                    args.model,
                    "--result-dir",
                    str(target),
                    "--cases",
                    "upstream",
                    "upstream-graph",
                    "--num-seqs",
                    str(num_seqs),
                    "--warmup-tokens",
                    "384",
                ],
                check=True,
            )
            print(f"NGC_CHECKPOINT batch-{num_seqs} {target}", flush=True)
        return
    if args.job in {
        "speed",
        "profile",
        "speed-long",
        "speed-upstream",
        "scheduler-ab",
        "metadata-ab",
        "speed-eviction",
        "speed-4k",
        "step-trace",
        "speed-batch",
        "speed-batch-graph",
        "profile-batch-graph",
        "profile-4k",
        "profile-scheduler",
        "bookkeeping-ab",
    }:
        bookkeeping_args = [
            "--scheduler-profile",
            "--cases",
            "before-0-graph",
            "0-graph",
            "--num-seqs",
            "50",
            "--warmup-tokens",
            "384",
        ]
        scheduler_profile_args = [
            "--scheduler-profile",
            "--cases",
            "upstream-graph",
            "0-graph",
            "--num-seqs",
            "50",
            "--warmup-tokens",
            "384",
        ]
        profile_graph_args = [
            "--profile",
            "--cases",
            "upstream-graph",
            "full-graph",
            "0-graph",
            "--num-seqs",
            "50",
            "--warmup-tokens",
            "384",
        ]
        batch_graph_args = [
            "--cases",
            "upstream-graph",
            "0-graph",
            "0.50-graph",
            "--num-seqs",
            "50",
            "--warmup-tokens",
            "384",
        ]
        batch_args = [
            "--cases",
            "upstream",
            "0",
            "0.50",
            "--num-seqs",
            "50",
            "--warmup-tokens",
            "384",
        ]
        trace_args = [
            "--cases",
            "full",
            "0",
            "0.50",
            "--output-tokens",
            "8192",
            "--warmup-tokens",
            "384",
            "--step-trace",
        ]
        upstream_args = [
            "--cases",
            "upstream",
            "full",
            "0",
            "0.50",
            "--warmup-tokens",
            "384",
        ]
        long_args = [
            "--cases",
            "full",
            "0",
            "0.75",
            "--output-tokens",
            "8192",
            "--warmup-tokens",
            "384",
        ]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("ngc_modal_benchmark.py")),
                "--model",
                args.model,
                "--result-dir",
                str(args.result_dir),
            ]
            + (["--profile"] if args.job == "profile" else [])
            + (long_args if args.job == "speed-long" else [])
            + (upstream_args if args.job == "speed-upstream" else [])
            + (trace_args if args.job == "step-trace" else [])
            + (
                [
                    "--cases",
                    "before-0-graph",
                    "0-graph",
                    "upstream-graph",
                    "--num-seqs",
                    "50",
                    "--warmup-tokens",
                    "384",
                ]
                if args.job == "scheduler-ab"
                else []
            )
            + (batch_args if args.job == "speed-batch" else [])
            + (batch_graph_args if args.job == "speed-batch-graph" else [])
            + (profile_graph_args if args.job == "profile-batch-graph" else [])
            + (scheduler_profile_args if args.job == "profile-scheduler" else [])
            + (bookkeeping_args if args.job == "bookkeeping-ab" else [])
            + (
                [
                    "--profile",
                    "--profile-only",
                    "--cases",
                    "upstream-graph",
                    "0-graph",
                    "0.5-graph",
                    "--num-seqs",
                    str(args.batch_size),
                    "--prompt-tokens",
                    "4096",
                    "--output-tokens",
                    "4096",
                    "--warmup-tokens",
                    "768",
                    "--cadence",
                    "256",
                    "--query-window",
                    "5",
                ]
                if args.job == "profile-4k"
                else []
            )
            + (
                [
                    "--cases",
                    "before-0-graph",
                    "0-graph",
                    "before-0.5-graph",
                    "0.5-graph",
                    "--num-seqs",
                    str(args.batch_size),
                    "--prompt-tokens",
                    "4096",
                    "--output-tokens",
                    "4096",
                    "--warmup-tokens",
                    "768",
                    "--cadence",
                    "256",
                    "--query-window",
                    "5",
                    "--repeats",
                    str(args.repeats),
                ]
                if args.job == "metadata-ab"
                else []
            )
            + (
                [
                    "--cases",
                    "upstream-graph",
                    "0-graph",
                    "0.5-graph",
                    "0.75-graph",
                    "--num-seqs",
                    str(args.batch_size),
                    "--warmup-tokens",
                    "384",
                ]
                if args.job == "speed-eviction"
                else []
            )
            + (
                [
                    "--cases",
                    "upstream-graph",
                    "0-graph",
                    "0.5-graph",
                    "0.75-graph",
                    "--num-seqs",
                    str(args.batch_size),
                    "--prompt-tokens",
                    "4096",
                    "--output-tokens",
                    "4096",
                    "--warmup-tokens",
                    "768",
                    "--cadence",
                    "256",
                    "--query-window",
                    "5",
                    "--repeats",
                    str(args.repeats),
                ]
                if args.job == "speed-4k"
                else []
            ),
            check=True,
        )
        return
    if args.job == "metadata":
        subprocess.run([sys.executable, "ngc_metadata_benchmark.py"], check=True)
    elif args.job == "kernels":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "-o",
                "addopts=",
                "tests/kernels/attention/test_triton_unified_attention.py",
                "tests/v1/worker/test_gpu_input_batch.py",
                "tests/v1/worker/test_ngc_scoring.py",
                "-k",
                args.selector,
                "-v",
                "--maxfail=1",
                f"--junitxml={args.result_dir / 'pytest.xml'}",
            ],
            check=True,
        )
    elif args.job in {"smoke", "smoke-graph"}:
        smoke(args.model, args.result_dir, graphs=args.job == "smoke-graph")
    elif args.job == "output-sweep":
        output_sweep(args.model, args.result_dir)
    elif args.job == "output-case":
        smoke(
            args.model,
            args.result_dir,
            story=True,
            graphs=True,
            eviction_rate=args.eviction_rate,
            cadence=256,
            gpu_memory_utilization=0.90,
        )
    elif args.job in {"story-ngc", "story-full"}:
        smoke(
            args.model, args.result_dir, story=True, ngc_enabled=args.job == "story-ngc"
        )
    elif args.job in {"zero-equivalence", "zero-equivalence-graph"}:
        zero_equivalence(
            args.model, args.result_dir, graphs=args.job == "zero-equivalence-graph"
        )
    elif args.job in {"zero-full", "zero-ngc", "zero-full-graph", "zero-ngc-graph"}:
        smoke(
            args.model,
            args.result_dir,
            ngc_enabled=args.job.startswith("zero-ngc"),
            equivalence=True,
            graphs=args.job.endswith("-graph"),
        )


if __name__ == "__main__":
    main()
