# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare production decode and fused decode/scoring on identical synthetic QKV."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm
from vllm.v1.attention.ops.triton_ngc_decode import ngc_decode_and_score
from vllm.v1.attention.ops.triton_ngc_scorer import ngc_page_scores
from vllm.v1.attention.ops.triton_unified_attention import (
    _get_tile_size,
    unified_attention,
)
from vllm.v1.worker.ngc_scoring import NGCScratchBuffers


def latency(fn, repeat, cache):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        samples = bench_gpu_time_with_cupti(
            fn,
            input_args=(cache,),
            use_cuda_graph=True,
            cold_l2_cache=True,
            dry_run_iters=5,
            repeat_iters=repeat,
        )
    for warning in emitted:
        message = str(warning.message)
        if any(
            word in message.lower() for word in ("falling back", "fallback", "disabled")
        ):
            raise RuntimeError(f"Requested CUPTI/cold-L2 timing unavailable: {message}")
    return statistics.median(samples) * 1000


def profile_stages(fn, output):
    """Attribute device stages after an untimed L2 eviction buffer fill."""
    flush = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    flush.zero_()
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn()
        torch.accelerator.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
        with gzip.open(output, "wt") as target:
            json.dump({"traceEvents": events}, target)
    kernels = [
        dict(name=e["name"], us=e["dur"])
        for e in events
        if e.get("cat") == "kernel" and e.get("ph") == "X"
    ]
    return {
        "partials_us": sum(
            e["us"] for e in kernels if "_ngc_decode_score_partials" in e["name"]
        ),
        "reduce_us": sum(
            e["us"] for e in kernels if "_ngc_decode_score_reduce" in e["name"]
        ),
        "standard_main_us": sum(
            e["us"] for e in kernels if "unified_attention" in e["name"]
        ),
        "standard_reduce_us": sum(
            e["us"] for e in kernels if "reduce_segments" in e["name"]
        ),
        "l2_eviction_buffer_bytes": flush.numel(),
        "kernels": kernels,
        "scope": "One compiled invocation after untimed 128MiB L2 eviction, "
        "under torch.profiler outside CUPTI timing; "
        "do not add these times to or substitute for combined latency.",
    }


def run_case(batch, context, args):
    torch.manual_seed(args.seed)
    heads, kv_heads, dim, page, window = 32, 8, 128, 16, 5
    pages = (context + page - 1) // page
    query = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(
        batch * pages, kv_heads, page, 2 * dim, device="cuda", dtype=query.dtype
    )
    table = torch.randperm(batch * pages, device="cuda", dtype=torch.int64)
    table = table.reshape(batch, pages).to(torch.int32)
    ring = torch.randn(batch * window + 1, heads, dim, device="cuda", dtype=query.dtype)
    positions = torch.arange(context, device="cuda").expand(batch, -1).contiguous()
    recent_positions = torch.arange(context - window, context, device="cuda")
    recent_queries = []
    for row in range(batch):
        recent = torch.randn(window, heads, dim, device="cuda", dtype=query.dtype)
        recent[-1] = query[row]
        ring[row * window + recent_positions % window] = recent
        recent_queries.append(recent)
    descriptors = torch.tensor(
        [[context, row, window, context, 0, 0, 0, row] for row in range(batch)],
        device="cuda",
        dtype=torch.int64,
    )
    due = torch.ones(batch, device="cuda", dtype=torch.bool)
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    standard, fused = torch.empty_like(query), torch.empty_like(query)
    score_buffer = torch.empty(batch, pages, device="cuda", dtype=torch.float32)
    scratch = NGCScratchBuffers()
    # The production H100 threshold is 128 / KV heads; no capture-size rounding
    # changes its choice for batch1 or48. Large batches take the 2D path.
    threshold = 128 // kv_heads
    segments = 16 if batch <= threshold else 1
    tile = _get_tile_size(dim, 0, query.element_size(), is_prefill=segments == 1)
    segment_output = torch.empty(batch, heads, 16, dim, device="cuda")
    segment_max = torch.empty(batch, heads, 16, device="cuda")
    segment_sum = torch.empty_like(segment_max)

    def standard_decode(cache):
        unified_attention(
            q=query,
            k=cache[..., :dim].transpose(1, 2),
            v=cache[..., dim:].transpose(1, 2),
            out=standard,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=lengths,
            max_seqlen_k=context,
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=threshold,
            num_par_softmax_segments=16,
            softmax_segm_output=segment_output,
            softmax_segm_max=segment_max,
            softmax_segm_expsum=segment_sum,
        )

    def fused_decode(cache):
        return ngc_decode_and_score(
            query,
            cache,
            ring,
            descriptors,
            positions,
            table,
            due,
            fused,
            window=window,
            ring_window=window,
            page_size=page,
            max_kept=context,
            softmax_scale=dim**-0.5,
            num_segments=segments,
            tile_size=tile,
            scratch=scratch,
            score_buffer=score_buffer,
        )

    standard_decode(cache)
    scores = fused_decode(cache)
    torch.accelerator.synchronize()
    torch.testing.assert_close(fused, standard, atol=0.025, rtol=0.025)
    score_error = 0.0
    for row in range(batch):
        reference = ngc_page_scores(
            recent_queries[row],
            cache[..., :dim],
            table[row],
            context,
            dim**-0.5,
            page,
            query_positions=recent_positions,
            key_positions=positions[row],
        )
        torch.testing.assert_close(scores[row], reference, atol=0.003, rtol=0.003)
        score_error = max(score_error, (scores[row] - reference).abs().max().item())
    errors = (fused.float() - standard.float()).abs()
    for _ in range(5):
        standard_decode(cache)
        fused_decode(cache)
    torch.accelerator.synchronize()
    warm_deadline = time.perf_counter() + 0.5
    while time.perf_counter() < warm_deadline:
        for _ in range(10):
            standard_decode(cache)
            fused_decode(cache)
        torch.accelerator.synchronize()
    calibration = {
        "standard_us": latency(standard_decode, args.repeat, cache),
        "fused_us": latency(fused_decode, args.repeat, cache),
    }
    trials = {"standard_us": [], "fused_us": []}
    for trial in range(args.trials):
        order = [("standard_us", standard_decode), ("fused_us", fused_decode)]
        for name, fn in order if trial % 2 == 0 else reversed(order):
            trials[name].append(latency(fn, args.repeat, cache))
    baseline = statistics.median(trials["standard_us"])
    candidate = statistics.median(trials["fused_us"])
    stages = profile_stages(
        lambda: fused_decode(cache), args.output.parent / f"b{batch}-n{context}.json.gz"
    )
    if stages["partials_us"] <= 0 or stages["reduce_us"] <= 0:
        raise RuntimeError("Profiler failed to find both fused stages")
    standard_trace = profile_stages(
        lambda: standard_decode(cache),
        args.output.parent / f"b{batch}-n{context}-standard.json.gz",
    )
    kv_bytes = batch * context * kv_heads * dim * 2 * query.element_size()
    return {
        "batch": batch,
        "context": context,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": dim,
        "dtype": str(query.dtype),
        "page_size": page,
        "recent_query_window": window,
        "all_requests_due": True,
        "launch": {
            "threshold_3d": threshold,
            "segments": segments,
            "tile_size": tile,
            "standard_path": "3D split" if segments > 1 else "2D",
        },
        "correctness": {
            "decode_atol": 0.025,
            "decode_rtol": 0.025,
            "decode_max_abs": errors.max().item(),
            "score_atol": 0.003,
            "score_rtol": 0.003,
            "score_max_abs": score_error,
        },
        "cupti_calibration_excluded": calibration,
        "trials": trials,
        "standard_median_us": baseline,
        "fused_median_us": candidate,
        "fused_over_standard": candidate / baseline,
        "minimum_kv_read_bytes": kv_bytes,
        "standard_effective_kv_GB_s": kv_bytes / (baseline * 1e3),
        "fused_effective_kv_GB_s": kv_bytes / (candidate * 1e3),
        "fused_profile": stages,
        "standard_profile": standard_trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,48")
    parser.add_argument("--contexts", default="512,2048,8192")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from cupti import cupti  # noqa: F401

    cupti_version = importlib.metadata.version("cupti-python")
    if not cupti_version.startswith("13.0."):
        raise RuntimeError("This CUDA13.0 benchmark requires CUPTI Python13.0.x")
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    if min(*batches, *contexts, args.repeat, args.trials) <= 0 or min(contexts) < 5:
        parser.error("Positive sizes/repetitions and contexts >= 5 are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(vllm.__file__).resolve().parent
    paths = [
        Path(__file__),
        root / "v1/attention/ops/triton_ngc_decode.py",
        root / "v1/attention/ops/triton_unified_attention.py",
    ]
    report = {
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cupti_python": cupti_version,
        "timing_backend_confirmed": "CUPTI (fallback warnings are fatal)",
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "seed": args.seed,
        "timing_sampling": {
            "repeat_iters": args.repeat,
            "trials": args.trials,
            "excluded_calibration_per_arm": 1,
            "paired_warmup_seconds": 0.5,
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "timing": "FlashInfer CUPTI, CUDA graphs, cold L2; allocation/compilation "
        "excluded; alternate arm order per trial; values in microseconds",
        "scope": "Fixed production H100 launch policies, not independently tuned "
        "optima. Shared synthetic unquantized QKV. Fused performs extra "
        "recent-query scoring. Effective KV bandwidth is a theoretical "
        "one-read lower-bound byte model, not measured DRAM traffic.",
        "model_shape_source": "https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json",
        "cases": [],
    }
    for batch in batches:
        for context in contexts:
            print(f"CASE_START batch={batch} context={context}", flush=True)
            report["cases"].append(run_case(batch, context, args))
            args.output.write_text(json.dumps(report, indent=2))
            row = report["cases"][-1]
            print(
                f"CASE_DONE batch={batch} context={context} "
                f"standard_us={row['standard_median_us']:.3f} "
                f"fused_us={row['fused_median_us']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
