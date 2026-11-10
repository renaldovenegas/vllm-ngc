# Latest model speed benchmarks

Measured on November 10, 2026. This folder contains only the latest batch 1 and batch 48 model speed runs.

## Configuration

Qwen/Qwen2.5-1.5B-Instruct, one H100, BF16, TRITON_ATTN, CUDA graphs enabled, tensor parallel size 1. Each request uses exactly 2,048 prompt tokens and generates 6,144 tokens (EOS ignored). NGC uses eviction cadence 256 and query window 5. Each case has one measured repetition after a full matching warmup generation. All ten cases completed without preemption.

## Results

| Batch size | Policy | Aggregate decode tokens/s ↑ | Total generation time (s) ↓ | Decode TPS vs upstream |
|---:|---|---:|---:|---:|
| 1 | Upstream | 282.0 | 21.84 | +0.0% |
| 1 | NGC 0% | 213.1 | 29.01 | -24.4% |
| 1 | NGC 25% | 241.0 | 25.70 | -14.6% |
| 1 | NGC 50% | 246.0 | 25.16 | -12.8% |
| 1 | NGC 75% | 248.6 | 24.91 | -11.9% |
| 48 | Upstream | 6,502.5 | 46.24 | +0.0% |
| 48 | NGC 0% | 5,097.7 | 59.57 | -21.6% |
| 48 | NGC 25% | 6,398.7 | 48.26 | -1.6% |
| 48 | NGC 50% | 7,133.7 | 43.24 | +9.7% |
| 48 | NGC 75% | 7,182.9 | 43.08 | +10.5% |

Decode throughput counts output tokens across all requests divided by the sum of measured engine-step wall times. It excludes prompt-processing steps and the first 16 generated tokens per request; it includes CPU scheduling and GPU execution. Total generation time measures batch submission through completion, including prompt processing, but excluding initialization and warmup. Relative throughput is `(policy TPS / upstream TPS - 1) * 100`, using the upstream case at the same batch size. These are single-run measurements, not repeated-trial averages.

At batch 1, upstream remains fastest. At batch 48, NGC 75% improves decode throughput by 10.5% over upstream and reduces total generation time by 6.8%; its decode throughput is 40.9% above NGC 0%. The gain from 50% to 75% is small in this workload.

## Raw summaries

- [Batch 1](ngc-speed-suite-c97f0ef533d7/summary.json)
- [Batch 48](ngc-speed-suite-1a77d417ab9e/summary.json)

The summaries retain per-case configuration, commands, metrics, memory measurements, retained-cache samples, and output hashes. This folder does not include the separate kernel benchmark or older runs.
