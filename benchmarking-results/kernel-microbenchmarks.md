# Kernel microbenchmarks

Run: `ngc-kernel-suite-5c6c7044724e`, November 10, 2026.

One H100, BF16, Qwen2.5-1.5B attention shape (12 query heads, 2 KV heads, head dimension 128), page size 16, query window 5. All 48 cases passed correctness checks. The sweep covers batch sizes 1 and 48, four context lengths, uniform and uneven lengths, and all/half/no requests due for scoring.

## Decode and scoring timings

The table shows uniform retained contexts with every request due for scoring. All timings are microseconds for one layer. Each arm uses its fastest configuration from the tested segment counts (1, 4, 16) and fused tile sizes (16, 64); these are tuned results, not necessarily production defaults.

| Batch | Retained context | Decode only (µs) | Separate decode + scoring (µs) | Fused decode + scoring (µs) | Fusion speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 7.52 | 15.00 | 8.16 | 1.84× |
| 1 | 512 | 8.64 | 24.83 | 9.86 | 2.52× |
| 1 | 2,048 | 16.82 | 71.50 | 13.63 | 5.25× |
| 1 | 8,192 | 46.74 | 254.80 | 31.10 | 8.19× |
| 48 | 128 | 11.04 | 366.61 | 13.63 | 26.89× |
| 48 | 512 | 26.45 | 815.73 | 26.66 | 30.60× |
| 48 | 2,048 | 64.48 | 2,857.24 | 71.55 | 39.93× |
| 48 | 8,192 | 194.93 | 10,681.88 | 203.76 | 52.42× |

Fusion speedup is separate decode-plus-scoring time divided by fused time. At batch 48 and 8,192 retained tokens, fused scoring adds 8.83 µs (4.5%) over decode alone on a scoring step. Standalone scoring takes roughly 10.5 ms in that case. These measurements do not isolate the contributions of redundant reads, atomics, and other implementation differences.

At batch 1 and longer contexts, the tuned fused implementation beats standard decode. The implementations and winning tile configurations differ; this does not mean adding scoring inherently accelerates decode.

## Other operations

- Batch-48 GPU score-input preparation: approximately 4 µs. Median preparation wall times across cases, including CPU work, H2D transfer, and completion: approximately 509 µs for unchanged layouts and 842 µs for changed layouts. These synchronized measurements do not directly equal added production step latency.
- Active query capture: approximately 1.6–2.0 µs across cases.
- Batch-48 GPU victim selection: approximately 11–99 µs, excluding CPU grouping and handoff.

## Measurement scope and limitations

CUPTI timing uses CUDA graphs and cold-L2 rotating buffers with alias-preserving views. Compilation and allocation are excluded. There is one outer trial with 30 inner timing repetitions; CPU preparation uses 10 repetitions. This is a single-layer kernel benchmark, not an end-to-end model speedup measurement. Hardware counters were disabled: reported effective bandwidth uses minimum KV bytes and is not actual measured DRAM traffic. No distributed communication or model-weight operations are included.

Correctness checks include an FP32 decode reference, fused-versus-standalone scoring comparisons (absolute/relative tolerance 0.003), exact metadata/query capture checks, and a stable-sort victim-selection reference.

[Full raw summary](ngc-kernel-suite-5c6c7044724e/summary.json)
