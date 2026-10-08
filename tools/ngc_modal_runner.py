# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote NGC checks; uses the Modal image's editable vLLM installation."""

import argparse
import functools
import inspect
import json
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
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.config.cache import NGCEvictionConfig
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.core.single_type_kv_cache_manager import NGCEvictionManager
    from vllm.v1.worker.ngc_scoring import NGCScoringBuffers

    original_init = Attention.__init__
    signature = inspect.signature(original_init)
    original_evict = NGCEvictionManager.apply_eviction
    configured_layers = {"ngc": [], "full": []}
    events = []
    zero_decisions = []
    original_score = NGCScoringBuffers.score_if_due

    @functools.wraps(original_score)
    def record_score(self, req_id, layer_name, **kwargs):
        decision = original_score(self, req_id, layer_name, **kwargs)
        if equivalence and decision is not None:
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
                cadence=128,
                eviction_rate=0.0 if equivalence else 0.25,
                query_buffer_size=5,
                min_retained_tokens=32,
            )
        configured_layers["ngc" if enabled else "full"].append(prefix)
        original_init(*bound.args, **bound.kwargs)

    @functools.wraps(original_evict)
    def record_eviction(self, proposal, computed_tokens):
        before = self.block_pool.get_num_free_blocks()
        applied = original_evict(self, proposal, computed_tokens)
        if equivalence:
            assert not proposal.victim_indices
            assert self.block_pool.get_num_free_blocks() == before
            assert self.layouts[proposal.request_id].evicted_tokens == 0
        if applied and proposal.victim_indices:
            freed = self.block_pool.get_num_free_blocks() - before
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
        return applied

    Attention.__init__ = initialize_attention
    NGCEvictionManager.apply_eviction = record_eviction
    NGCScoringBuffers.score_if_due = record_score
    try:
        llm = LLM(
            model=model,
            enforce_eager=True,
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
            gpu_memory_utilization=0.65,
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
                for boundary in (128, 256, 384)
            }
            observed = {
                (req.rsplit("-", 1)[0], layer, boundary)
                for req, layer, boundary in zero_decisions
            }
            assert observed == expected, (observed, expected)
        elif ngc_enabled:
            assert len(cycles) == len(prompts) * len(configured_layers["ngc"])
            assert all(count >= 2 for count in cycles.values())
        else:
            assert not events, "Full attention must not evict"
        report = {
            "model": model,
            "attention_mode": "ngc_all_layers" if ngc_enabled else "full",
            "chat_template_applied": story or equivalence,
            "zero_decisions": zero_decisions,
            "layers": configured_layers,
            "eviction_events": len(events),
            "sampling": sampling,
            "ngc": {
                "cadence": 128,
                "eviction_rate": 0.0 if equivalence else 0.25,
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
        print(json.dumps(report, indent=2))
    finally:
        Attention.__init__ = original_init
        NGCEvictionManager.apply_eviction = original_evict
        NGCScoringBuffers.score_if_due = original_score


def zero_equivalence(model: str, result_dir: Path) -> None:
    """Compare fresh engines on the same GPU across three scoring boundaries.

    Exact greedy tokens and selected-token logprobs test the model's decode
    behavior with active per-layer bookkeeping, without changing its history.
    """
    reports = []
    for mode in ("zero-full", "zero-ngc"):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        choices=[
            "check",
            "kernels",
            "smoke",
            "story-ngc",
            "story-full",
            "zero-equivalence",
            "zero-full",
            "zero-ngc",
            "speed",
            "profile",
            "capture-speed",
            "speed-long",
            "speed-upstream",
            "step-trace",
        ],
        required=True,
    )
    parser.add_argument("--selector", default="ngc")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    check_environment()
    if args.job in {
        "speed",
        "profile",
        "capture-speed",
        "speed-long",
        "speed-upstream",
        "step-trace",
    }:
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
            + (["--compare-capture"] if args.job == "capture-speed" else [])
            + (long_args if args.job == "speed-long" else [])
            + (upstream_args if args.job == "speed-upstream" else [])
            + (trace_args if args.job == "step-trace" else []),
            check=True,
        )
        return
    if args.job == "kernels":
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
                "-k",
                args.selector,
                "-v",
                "--maxfail=1",
                f"--junitxml={args.result_dir / 'pytest.xml'}",
            ],
            check=True,
        )
    elif args.job == "smoke":
        smoke(args.model, args.result_dir)
    elif args.job in {"story-ngc", "story-full"}:
        smoke(
            args.model, args.result_dir, story=True, ngc_enabled=args.job == "story-ngc"
        )
    elif args.job == "zero-equivalence":
        zero_equivalence(args.model, args.result_dir)
    elif args.job in {"zero-full", "zero-ngc"}:
        smoke(
            args.model,
            args.result_dir,
            ngc_enabled=args.job == "zero-ngc",
            equivalence=True,
        )


if __name__ == "__main__":
    main()
