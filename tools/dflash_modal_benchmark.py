# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare normal-EOS Qwen3 generation with and without a DFlash drafter."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import regex as re

MODEL = "Qwen/Qwen3-8B"
DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
PROMPT = (
    "Write a complete fantasy story of approximately 1000 tokens about a young "
    "mapmaker who discovers a hidden road in the stars. Use ordinary prose, "
    "give the story a clear ending, and stop when the story is complete."
)


def spec_counters(llm):
    result = {}
    for metric in llm.get_metrics():
        if metric.name.startswith("vllm:spec_decode_") and hasattr(metric, "value"):
            result[metric.name] = result.get(metric.name, 0) + metric.value
    return result


DATASET = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def math_problems():
    from vllm.transformers_utils.repo_utils import hf_api

    path = hf_api().hf_hub_download(
        DATASET, "test.jsonl", repo_type="dataset", revision=DATASET_REVISION
    )
    return [json.loads(line) for line in Path(path).read_text().splitlines()[:10]]


def problem_indices(batch, repeat):
    if batch == 1:
        return list(range(10))
    assert batch == 48
    return list(range(10)) * 4 + [(repeat * 8 + i) % 10 for i in range(8)]


def boxed_answer(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    return None


def normalize_answer(text):
    if text is None:
        return None
    for token in (r"\left", r"\right", r"\,", r"\!", "$", "°"):
        text = text.replace(token, "")
    return re.sub(r"\s+", "", text).replace(r"\dfrac", r"\frac")


def run_case(
    result_dir, batch, speculative, workload="story", max_tokens=1024, thinking=False
):
    import torch

    from vllm import LLM, SamplingParams

    problems = math_problems() if workload == "math500" else None
    config = dict(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max(2048, 2 * max_tokens),
        max_num_seqs=batch,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        async_scheduling=False,
        distributed_executor_backend="uni",
        attention_config={"backend": "FLASH_ATTN"},
        seed=0,
        disable_log_stats=False,
    )
    if speculative:
        config["speculative_config"] = dict(
            method="dflash",
            model=DRAFT,
            num_speculative_tokens=15,
            attention_backend="FLASH_ATTN",
        )
    started = time.perf_counter()
    llm = LLM(**config)
    setup_seconds = time.perf_counter() - started
    if speculative:
        resolved = llm.llm_engine.vllm_config.speculative_config
        assert resolved.method == "dflash" and resolved.num_speculative_tokens == 15
    sampling = dict(temperature=0.0, seed=0, max_tokens=max_tokens)

    def generate(repeat=0):
        indices = problem_indices(batch, repeat) if problems else [0] * batch
        groups = [[i] for i in indices] if problems and batch == 1 else [indices]
        outputs, per_problem = [], []
        for group in groups:
            messages = [
                [
                    {
                        "role": "user",
                        "content": (
                            problems[i]["problem"]
                            + "\nSolve the problem and put your final answer "
                            "in \\boxed{...}."
                            if problems
                            else PROMPT
                        ),
                    }
                ]
                for i in group
            ]
            before = spec_counters(llm) if problems and batch == 1 else {}
            torch.accelerator.synchronize()
            started = time.perf_counter()
            result = llm.chat(
                messages,
                SamplingParams(**sampling),
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": thinking},
            )
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - started
            after = spec_counters(llm) if problems and batch == 1 else {}
            if problems and batch == 1:
                counters = {
                    key: value - before.get(key, 0) for key, value in after.items()
                }
                drafts = counters.get("vllm:spec_decode_num_drafts", 0)
                accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0)
                proposed = counters.get("vllm:spec_decode_num_draft_tokens", 0)
                per_problem.append(
                    dict(
                        index=group[0],
                        seconds=elapsed,
                        spec_counters=counters,
                        acceptance_length=1 + accepted / drafts if drafts else None,
                        acceptance_fraction=accepted / proposed if proposed else None,
                    )
                )
            outputs.extend(result)
        return outputs, per_problem, indices

    started = time.perf_counter()
    generate()
    torch.accelerator.synchronize()
    warmup_seconds = time.perf_counter() - started
    runs = []
    for repeat in range(3):
        before = spec_counters(llm)
        torch.accelerator.synchronize()
        started = time.perf_counter()
        outputs, per_problem, indices = generate(repeat)
        torch.accelerator.synchronize()
        seconds = time.perf_counter() - started
        after = spec_counters(llm)
        counters = {key: value - before.get(key, 0) for key, value in after.items()}
        drafts = counters.get("vllm:spec_decode_num_drafts", 0)
        accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0)
        proposed = counters.get("vllm:spec_decode_num_draft_tokens", 0)
        if speculative:
            assert drafts > 0, "DFlash must actually draft during timed generation"
        lengths = [len(item.outputs[0].token_ids) for item in outputs]
        details = []
        if problems:
            for index, output in zip(indices, outputs):
                answer = boxed_answer(output.outputs[0].text)
                details.append(
                    dict(
                        index=index,
                        unique_id=problems[index]["unique_id"],
                        expected_answer=problems[index]["answer"],
                        boxed_answer=answer,
                        strict_normalized_match=normalize_answer(answer)
                        == normalize_answer(problems[index]["answer"]),
                        output=output.outputs[0].text,
                        output_token_ids=output.outputs[0].token_ids,
                    )
                )
        runs.append(
            dict(
                seconds=seconds,
                problem_indices=indices if problems else None,
                per_problem=per_problem,
                problem_results=details,
                capped_responses=sum(
                    item.outputs[0].finish_reason == "length" for item in outputs
                ),
                output_lengths=lengths,
                output_tokens=sum(lengths),
                tokens_per_second=sum(lengths) / seconds,
                finish_reasons=[item.outputs[0].finish_reason for item in outputs],
                output_hash=hashlib.sha256(
                    json.dumps([item.outputs[0].token_ids for item in outputs]).encode()
                ).hexdigest(),
                spec_counters=counters,
                acceptance_length=1 + accepted / drafts if drafts else None,
                acceptance_fraction=accepted / proposed if proposed else None,
            )
        )
        log_summary = {
            key: value
            for key, value in runs[-1].items()
            if key not in ("per_problem", "problem_results")
        }
        print(
            f"DFLASH_TIMING batch={batch} speculative={speculative} "
            f"repeat={repeat + 1} {json.dumps(log_summary)}",
            flush=True,
        )
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    ).strip()
    report = dict(
        model=MODEL,
        draft=DRAFT if speculative else None,
        batch=batch,
        config=config,
        sampling=sampling,
        prompt=PROMPT if not problems else None,
        workload=workload,
        dataset=dict(
            repo=DATASET,
            revision=DATASET_REVISION,
            split="test",
            indices=list(range(10)),
            problems=problems,
        )
        if problems
        else None,
        grading="strict normalized last boxed answer; not symbolic math equivalence"
        if problems
        else None,
        enable_thinking=thinking,
        prompt_lengths=[len(item.prompt_token_ids) for item in outputs],
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        runs=runs,
        median_seconds=statistics.median(item["seconds"] for item in runs),
        median_tokens_per_second=statistics.median(
            item["tokens_per_second"] for item in runs
        ),
        gpu=gpu,
        torch=torch.__version__,
        vllm_source=__import__("vllm").__file__,
        outputs=[item.outputs[0].text for item in outputs],
        output_token_ids=[item.outputs[0].token_ids for item in outputs],
    )
    result_dir.write_text(json.dumps(report, indent=2))


def sweep(result_dir, workload="story", max_tokens=1024, thinking=False):
    reports = {}
    for batch in (1, 48):
        for mode in ("baseline", "dflash"):
            name = f"{mode}-{batch}"
            path = result_dir / f"{name}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--case",
                    mode,
                    "--batch",
                    str(batch),
                    "--result-dir",
                    str(path),
                    "--workload",
                    workload,
                    "--max-tokens",
                    str(max_tokens),
                    *(["--thinking"] if thinking else []),
                ],
                check=True,
            )
            reports[name] = json.loads(path.read_text())
            (result_dir / "comparison.json").write_text(json.dumps(reports, indent=2))
            print(f"NGC_CHECKPOINT {name} {result_dir}", flush=True)
    assert len({item["gpu"] for item in reports.values()}) == 1
    for batch in (1, 48):
        base, draft = reports[f"baseline-{batch}"], reports[f"dflash-{batch}"]
        draft["throughput_speedup_vs_baseline"] = (
            draft["median_tokens_per_second"] / base["median_tokens_per_second"]
        )
        draft["baseline_output_tokens_match"] = (
            draft["output_token_ids"] == base["output_token_ids"]
        )
    (result_dir / "comparison.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--case", choices=["baseline", "dflash"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workload", choices=["story", "math500"], default="story")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.case:
        run_case(
            args.result_dir,
            args.batch,
            args.case == "dflash",
            args.workload,
            args.max_tokens,
            args.thinking,
        )
    else:
        sweep(args.result_dir, args.workload, args.max_tokens, args.thinking)
