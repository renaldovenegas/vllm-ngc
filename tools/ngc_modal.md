# NGC GPU tests on Modal

Run from the repository root. The launcher uses Python 3.12, one NVIDIA L4,
an editable installation of this fork, and a persistent cache for model weights,
Triton compilation, and reports. Override the GPU with `NGC_MODAL_GPU=A100`.

## Local setup

```bash
uv pip install modal==1.6.1
# Only needed if this machine does not already have a Modal profile:
.venv/bin/modal token new
```

The commands below upload source to your active Modal workspace and run billable
jobs. Each job uses one GPU, has no automatic retries, and has an execution
timeout of 120 minutes. Image installation is cached separately from Python/Triton
source uploads.

## Run checks

```bash
# Validate source imports, compiled extensions, and GPU availability.
.venv/bin/modal run tools/ngc_modal.py --job check

# Scorer correctness and post-eviction cache writes/attention.
.venv/bin/modal run tools/ngc_modal.py --job kernels

# Narrow the selection while debugging.
.venv/bin/modal run tools/ngc_modal.py --job kernels --selector ngc_compacted

# Two requests, NGC in every decoder layer, repeated eviction.
.venv/bin/modal run tools/ngc_modal.py --job smoke

# Full attention vs zero-rate NGC, fresh engines on the same GPU.
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job zero-equivalence --model Qwen/Qwen2.5-7B-Instruct

# The same checks with torch.compile and CUDA graphs instead of eager mode.
.venv/bin/modal run tools/ngc_modal.py --job smoke-graph
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job zero-equivalence-graph --model Qwen/Qwen2.5-7B-Instruct
```

NGC records each layer's post-RoPE queries inside the attention op by copying
rows between persistent buffers, so it runs in eager mode and under CUDA graphs.
The `-graph` jobs use vLLM's default compilation and CUDA graph settings, and
zero-equivalence compares full attention and NGC in the same execution mode.

The smoke test defaults to the public `Qwen/Qwen3-0.6B` model. It runs the V1
model runner in-process with synchronous eager execution, Triton attention,
FP16 KV, and no prefix caching. Its process-local hooks opt every decoder layer
into NGC without changing the model implementation. It asserts that both requests
generate 2048 tokens and every NGC layer performs multiple nonempty evictions;
each eviction must increase the block pool's free count by the number of victims.
This is a functional smoke test, not a model-quality evaluation.

The zero-equivalence job generates 512 greedy tokens for each of two
chat-formatted prompts. NGC uses every decoder layer, rate zero, and cadence 128.
The eviction clock includes prompt tokens. Prefill chunks stop at cadence
boundaries so eviction takes effect before the next chunk.
Scoring remains active; the job checks zero-victim decisions at every total-token
cadence boundary per request/layer and verifies no pages are freed by eviction. It requires identical prompt/output tokens and selected-token log
probabilities within an absolute tolerance of 0.005. It does not compare the
entire vocabulary distribution. Reports are `equivalence.json`, `zero-full.json`,
and `zero-ngc.json`. Both engines run sequentially in fresh processes on the
same GPU allocation.

For an end-to-end wall-clock rate sweep:

```bash
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job speed --model Qwen/Qwen2.5-7B-Instruct --submit-only
```

This compares full attention with all-layer NGC rates 0, 0.10, 0.25,
0.50, and 0.75 at cadence 128, using the same GPU allocation and a fixed shuffled
configuration order. Each fresh engine runs one 2048-token warmup followed by
three timed 2048-token generations of one chat-formatted fantasy-story prompt.
Temperature is 0.6 and the sampling seed is 0. Prefix caching and CUDA graphs
are disabled. Timing uses `perf_counter` with GPU synchronization and includes
prefill, decode, scheduling, scoring, and eviction, but excludes model loading,
prompt tokenization, and warmup. Warmup validates block freeing; instrumentation
is removed before timing. This measures this fork's eager path, not optimized
upstream serving or output quality. Different rates may generate different text.
Reports include individual samples, medians, ranges, tokens/second, and speedup
relative to full attention in `speed.json` and `speed.csv`.
Each completed configuration commits its report to the volume. With `--detach
--submit-only`, the local launcher exits after submitting the remote call;
retrieve the reports from the volume when the job finishes.

Configuration names ending in `-graph` (for example `0.50-graph` or
`upstream-graph`) run with torch.compile and CUDA graphs. For batch-50
throughput with CUDA graphs for upstream, rate 0, and rate 0.50, use
`--job speed-batch-graph`; `--job speed-batch` is the eager equivalent.

Use `--job profile` with the same command to investigate overhead at full
attention and NGC rates 0 and 0.25. This enables verbose JIT logging, records
post-warmup compilation events separately for timed and profiled generations,
and runs a separate PyTorch CPU/CUDA profile after the timing samples. The
profile captures 144 late-decode engine steps around the 1920-token cadence
boundary, with annotations for query capture, scoring, CPU proposals, eviction,
and worker input/state updates. Inclusive nested timings overlap and must not
be added. Event summaries are in each `speed-<case>.json`; compressed Chrome
traces (`*.trace.json.gz`) stay in the Modal volume and must be downloaded
separately. The original wall-clock samples contain no profiler annotations.

Use `--job profile-batch-graph --model Qwen/Qwen2.5-7B-Instruct` on an A100
to compare upstream, this fork's full attention, and zero-eviction NGC with
CUDA graphs at batch 50. Each case has three uninstrumented timing runs and
a separate CPU/CUDA trace. The `profile.timeline` report measures GPU busy
time as an interval union and separates ordinary steps from scoring boundaries.
Scheduler, metadata, and input preparation annotations are inclusive host wall
spans (including waits), not CPU utilization. They overlap GPU execution;
do not add them to GPU busy time. CUDA graph kernels appear in the GPU trace
even though replay does not invoke the query-capture Python annotation.

Use `--job profile-scheduler --model Qwen/Qwen2.5-7B-Instruct` on an A100
for scheduler attribution at batch 50, comparing upstream and zero-eviction NGC.
After the three ordinary timing runs, separate generations collect coarse method
timings and a scheduler-only Python call profile over 144 late-decode steps.
Reports include calling-thread CPU time, nested KV-manager/coordinator timings,
function call counts, self/cumulative times, and callers. Steps consuming NGC
proposals are separated from ordinary scheduling. Nested times overlap;
the call profiler adds overhead and should not be used for absolute latency.

For a paired bookkeeping comparison on one GPU allocation, set
`NGC_MODAL_BEFORE_MANAGER` to a saved pre-change
`single_type_kv_cache_manager.py` and run `--job bookkeeping-ab`. It compares
`before-0-graph` with `0-graph` at batch 50 using three uninstrumented timing
runs and the same scheduler profiles. Each case runs in a fresh process;
the before checkout differs only in the supplied manager file. Reports record
manager source hashes and whether generated-token hashes match across versions.

For a longer-context comparison of full attention, rate 0, and rate 0.75:

```bash
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job speed-long --model Qwen/Qwen2.5-7B-Instruct
```

This uses the same settings with three timed 8192-token generations and a
384-token warmup, which covers kernel compilation and the first evictions.

Use `--model` for another public, text-only causal model whose attention prefixes
contain `layers.<index>.`. Gated models require additional credential setup;
local Hugging Face credentials are not uploaded.

## Results

For bounded CPU/CUDA attribution of upstream, NGC rate 0, and NGC rate 0.5:

```bash
NGC_MODAL_GPU=A100-80GB .venv/bin/modal run --detach tools/ngc_modal.py \
  --job profile-4k --model Qwen/Qwen2.5-7B-Instruct --batch-size 48
```

This profile-only job uses the same 4096-token prompt/generation, cadence 256,
window 5, and 768-token warmup as `speed-4k`. It skips uninstrumented timing
repetitions. During a separate generation, profiling starts at 3000 actual
computed generated tokens, warms for eight decode steps, and captures 128 steps
including the 3072-token scoring boundary and its following proposal step.
Reports separate ordinary decode, scoring, and proposal application. CPU wall
and thread time include profiler overhead; exclusive timers subtract nested
instrumented calls. Thread time excludes sleeping/descheduling but can include
driver spinning. GPU busy uses interval unions; host/device spans overlap.
Compressed Chrome traces remain in the volume beside the downloaded reports.

For a quick paired check of retained-length sharing and batched eviction, use
`--job metadata-ab --batch-size 48` with `NGC_MODAL_BEFORE_DIR` pointing to a
directory containing the five previous files under their `vllm/` paths:
`core/kv_cache_manager.py`, `core/single_type_kv_cache_manager.py`,
`core/sched/scheduler.py`, `worker/ngc_scoring.py`, and
`worker/gpu_model_runner.py` (each beneath `vllm/v1/`). The job compares before
and current at rates 0 and 0.5 on one GPU, using 4096 prompt tokens, 4096 output
tokens, cadence 256, query window 5, and a 768-token warmup. It takes one timed
sample per case and checks output hashes at both rates; treat timings as a
diagnostic rather than a repeatability study. The underlying benchmark's
`--repeats` option defaults to three for other jobs.

Use `--job validate-inputs` for eager and CUDA graph zero-eviction equivalence
checks followed by a graph smoke test with real eviction. Each check's reports
are downloaded into its own subdirectory.

For non-NGC versus 0%, 50%, and 75% NGC eviction on one GPU allocation:

```bash
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job speed-eviction --model Qwen/Qwen2.5-7B-Instruct
```

This uses 50 requests, 2048 output tokens each, a 384-token warmup, and three
unprofiled timed repetitions per case with unchanged CUDA graph settings.
Reports include warmup eviction decisions and freed pages. Positive eviction
rates can change generated tokens; only rate zero should match non-NGC.

Set `--batch-size` to change both the number of requests and the engine's
`max_num_seqs` for every case (default: 50). For example:

```bash
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job speed-eviction --model Qwen/Qwen2.5-7B-Instruct --batch-size 48
```

For this benchmark's default graph settings, try 48 for full decode graph
coverage close to the previous batch of 50. Other default capture sizes include
8, 16, 24, 32, 40, 56, and 64. Batch 50 falls back to piecewise graphs: the next
capture size is 56, which exceeds its `max_num_seqs=50`. The batch-size option
applies only to `speed-eviction`; the other jobs retain their existing settings.

NGC single-token decode uses two kernels when a request crosses its eviction
cadence. The first groups up to eight query heads sharing a KV head and shares
K loads between decode and recent-query scoring,
writing segment softmax states, partial PV outputs, and page masses. The second
combines segments for decode and normalizes page masses across queries/heads;
it does not reread KV and needs no score atomics or separate output-zeroing
launch. Segment count follows the standard decode dispatch (one or sixteen);
tiles use its selected size, enlarged if needed to contain a scoring page.
Segment/page scratch uses grow-only reusable storage shared by successive layers;
score outputs have separate reusable storage per layer until eviction handoff.
Fused metadata preparation skips historical-query gathers because the kernel
reads the query ring directly after capture.
Other steps use the standard decode path. With graphs enabled, single-GPU dense
models whose attention layers all use the fused boundary path get a separate
full-model boundary graph. It is warmed and captured lazily, then replayed with
updated device descriptors, due masks, query-ring indices, and block tables.
The cache retains one graph and recaptures when buffer addresses, capacities,
batch shapes, or other dispatch dependencies change. Scoring launch dimensions
use persistent capacity buckets; actual retained lengths remain device inputs.
LoRA, offloading, KV connectors, parallel execution, and mixed attention paths
retain eager boundary execution. `enforce_eager` disables boundary graphs too.
Ordinary decode keeps its configured graph mode. Prefill, multi-token batches, and attention
with biases, sinks, soft caps, or sliding windows retain separate scoring.
NGC metadata builds reuse a template per cache spec and backend builder type.
Layers with equal retained lengths for every request share batch-local CPU/GPU
length views and their maximum-length calculation. Block tables, slot mappings,
and backend scratch remain layer-specific. Ordinary graph capture retains
independent length-buffer addresses, and those backing rows are still updated;
sharing does not assume that all layer eviction proposals will be accepted.
Boundary graphs recapture if the shared-view addresses change. Scheduler eviction
checks request state once per batch and frees accepted layer victims in one pool
operation, preserving per-layer validation and the prior free-list order.

Query capture, metadata preparation, and eviction selection remain separate
operations. Fusion does not change the configured scoring window or eviction
policy, but decode floating-point rounding can differ from the standard kernel.
Graph-equivalence reports include boundary capture/replay counts and require
at least one boundary replay, so passing cannot silently exercise only eager
boundaries. Initial captures are setup costs; a benchmark can encounter new
capacity buckets after warmup and should account for those capture costs.

For a longer-context comparison of upstream full attention, NGC zero eviction,
and NGC 50% eviction, run:

```bash
NGC_MODAL_GPU=A100-80GB .venv/bin/modal run --detach tools/ngc_modal.py \
  --job speed-4k --model Qwen/Qwen2.5-7B-Instruct --batch-size 48
```

This uses exactly 4096 prompt tokens (including the chat template), 4096 generated
tokens per request, cadence 256, a five-query window, and minimum retention 32.
The prompt repeats fantasy-setting notes before the story instruction to reach
the fixed token budget. All three cases receive identical token IDs. Temperature
is 0.6, prefix caching is disabled, and graph mode is enabled. Each case gets a
768-token warmup and three timed repetitions on the same GPU allocation. Timing
includes prefill and decode; model loading and warmup are excluded. The 80GB GPU
provides headroom for the batch-48 full-attention KV cache. Results are saved under
`.ngc-modal-results/speed-4k-<id>/`, including `speed.json` and per-case reports.

To check fused kernel outputs against reference attention and standalone scoring:

```bash
NGC_MODAL_GPU=A100 .venv/bin/modal run --detach tools/ngc_modal.py \
  --job kernels --selector fused_decode
```

For a comparison of the batched scheduler changes with the previous NGC
implementation and pristine pinned upstream, place the previous
`kv_cache_coordinator.py`, `gpu_model_runner.py`, and `triton_attn.py` in one
directory, then run:

```bash
NGC_MODAL_GPU=A100 \
  NGC_MODAL_BEFORE_DIR=.ngc-modal-results/batched-management-before \
  .venv/bin/modal run --detach tools/ngc_modal.py \
  --job scheduler-ab --model Qwen/Qwen2.5-7B-Instruct
```

The three cases share one GPU allocation. Each uses 50 requests, a 384-token
warmup, and three timed generations of 2048 tokens per request, with CUDA graphs
enabled and NGC eviction rate zero. The previous implementation retains the
current manager file; the new append helper is unused by the previous
coordinator. Reports include GPU UUID, source file hashes, matching output
hashes for previous/current NGC, and speedups against both baselines. The
upstream baseline is the pinned binary base below, rather than latest upstream.

Logs stream to your terminal. Reports are downloaded to
`.ngc-modal-results/<run-id>/` even when a test process fails. Kernel jobs produce
`pytest.xml`; smoke jobs produce `evictions.jsonl` and, on success, `smoke.json`.
`run.json` identifies the checkout commit and the pinned binary base. Uncommitted
Python/Triton edits are included in each upload.

Reports and caches also persist in the Modal Volume `vllm-ngc-test-cache`, under
`results/`, `huggingface/`, `triton/`, and `vllm/`. The Volume persists after jobs
exit and uses Modal storage. No serving endpoint is deployed.

## Dependency compatibility

The environment installs the upstream source and CUDA 13.0 precompiled artifacts
for stable v0.32.0 commit `5df63a7203cdb9328d824a27b682e77c7b9928be`, then overlays this checkout's
`vllm/` source. Generated native libraries stay in the editable installation;
local `.venv`, `.git`, credentials, and local compiled libraries are not uploaded.

The launcher rejects changes to native code, build metadata, or dependency files
relative to that base. Update the pinned base and its matching wheel when those
inputs change; custom native modifications require a separate source-build image.
Ordinary Python/Triton edits do not rebuild dependencies.

The kernel job deliberately uses `--noconftest -o addopts=` because this test file
is self-contained. This avoids installing unrelated model/evaluation fixtures.

## Compare generated text across eviction rates

Run four isolated engines on the same GPU to generate a story with NGC eviction
rates 0%, 25%, 50%, and 75%:

```bash
NGC_MODAL_GPU=A100-80GB .venv/bin/modal run --detach tools/ngc_modal.py \
  --job output-sweep --model Qwen/Qwen2.5-7B-Instruct
```

Each case uses the chat prompt "Write a long fantasy story that is approximately
3000 words.", temperature 0.6, seed 0, cadence 256, query window 5, and CUDA graphs.
Generation has a 2,048-token minimum and 6,144-token maximum; EOS is allowed after
the minimum. This is a text comparison, not the synthetic 4K-prompt speed test.
The 0% case retains NGC scoring and bookkeeping and evicts no pages.

Results download to `.ngc-modal-results/output-sweep-<run-id>/` when the client
finishes. Read `outputs.txt` for all four stories, or
`eviction-{0,25,50,75}/output-0.txt` for each individual story. `outputs.json` and
each case's `smoke.json` include the prompt, sampling settings, token counts,
finish reasons, and eviction instrumentation. Each completed case is checkpointed
to the Modal volume. With `--detach`, the remote job survives client disconnects;
automatic local download still requires the client to remain connected.

## Full-attention and NGC decode profiling

`tools/decode_profile_modal.py` compares full attention against NGC with 0%, 50%,
and 75% eviction by default. It reuses the pinned Modal environment above and
supports one GPU, synchronous scheduling, FP16/BF16, and CUDA graphs or eager
execution. Speculative decoding is disabled. All comparison cases use Triton
attention, including full attention, to avoid mixing backend effects with eviction.
The full baseline is this checkout with NGC disabled, not a pristine upstream build.

```bash
DECODE_PROFILE_GPU=H100 .venv/bin/modal run --detach tools/decode_profile_modal.py \
  --model Qwen/Qwen3-8B --batch-size 1 --contexts 512,2048,8192 \
  --policies full,0,0.5,0.75 --backend TRITON_ATTN \
  --cadence 256 --query-window 5 \
  --steps 768 --warmup-steps 16 --repeats 3 --ignore-eos
```

The input is a synthetic, exact-length token sequence, not a quality evaluation.
Normal EOS is respected by default. If any sequence finishes inside the selected
window, the script rejects the measurement rather than silently timing a smaller
batch. For controlled fixed-shape profiling, explicitly add `--ignore-eos`.
Use `--batch-size 48` for a larger batch or `--eager` to disable compilation/graphs.
Use `--policies full --backend FLASH_ATTN --steps 64` to reproduce the earlier
full-attention-only setup. The new default Triton baseline is not directly
comparable to the earlier FlashAttention baseline.

Policies are `full` or fractions removed at each boundary, not target cache
occupancies. NGC 0% retains scoring/bookkeeping while removing no tokens. Each
NGC policy applies to all decoder attention layers with a minimum retention of
32 tokens. The cadence and query window are shared across layers. NGC comparisons
require at least two cadences of measured decode; the default 768 steps spans
three cadence intervals. Positive-rate runs fail if no actual eviction occurs.

Each context/policy pair has separate processes for three measurements:

| Artifact | Measurement |
| --- | --- |
| `timing.json` | Unprofiled decode wall time, excluding prefill and warmup; per-step latency and aggregate cost per output token |
| `timeline.json`, `trace.json.gz` | CPU/CUDA timeline; attention, GEMMs, normalization, cache writes, other kernels, synchronization APIs and GPU gaps |
| `hardware.json`, `ncu.csv`, `attention.ncu-rep` | Optional measured attention DRAM traffic and compute/activity counters |

The runner first warms a matching generation, then prefills each new request
outside the measured window and performs the configured decode warmup steps.
Context grows during measurement; reports record the actual context range.
Per-step latency corresponds to one new token per active request. Aggregate
milliseconds per token divides by the total number of tokens across the batch.
Do not confuse the latter with per-request latency. Retained KV lengths are
recorded per request and cache group before and after the measured window;
these count committed tokens, not padded block capacity. Requests start fresh
for each repeat, so this measures cache evolution from the selected prompt,
not a permanently stabilized retained context.

The separate trace pass records per-step diagnostics in `trace.json` and the
summary: ordinary decode, fused scoring, pending proposal application, or both.
Proposal application can remove zero tokens, particularly at 0% eviction.
`profiled_step_summary` reports counts and mean wall time by kind. These diagnostic
timings include profiler overhead and an added GPU synchronization per step;
use the uninstrumented `timing.json` medians for end-to-end comparisons.
Per-step retained-length ranges and actual removal totals are also recorded.
Removal totals sum across cache groups and are not unique token counts.
Fused decode/scoring kernels belong to attention time because their two costs
cannot be separated from the fused launch.

To request hardware counters, install the pinned Nsight Compute CLI in the image
and enable the counter pass:

```bash
DECODE_PROFILE_GPU=H100 DECODE_PROFILE_NCU=1 \
  .venv/bin/modal run --detach tools/decode_profile_modal.py \
  --contexts 512,2048,8192 --batch-size 1 --ignore-eos --counters
```

Hardware collection selects attention kernels inside the first warmed decode
step. For NGC this may be an ordinary step; it does not guarantee capturing a
scoring boundary and must not be interpreted as an average over an eviction cycle.
It requests measured DRAM read/write bytes, kernel duration, DRAM and SM
throughput percentages, tensor-pipeline activity and achieved occupancy. DRAM
traffic includes every access by the selected kernel, not only KV reads; cache
hits matter. Activity percentages are not model FLOP/s. Nsight replay and kernel
serialization perturb execution, so counter-pass durations must not replace
the unprofiled timing results. Cache flushing and clock control are disabled;
the exact invocation is saved beside the report. Collecting all attention layers
can take many minutes even for a single decode step because metrics require
repeated kernel replay.

Modal GPU counter permissions and architecture-specific metric availability are
not guaranteed. Missing counters are marked unavailable/partial with raw logs,
never replaced with theoretical KV bytes or `nvidia-smi` utilization. The timing
and timeline passes remain useful when counters are unavailable.

Kernel categories use explicit name matching and retain all unknown kernels.
Generic GEMMs are grouped together: projections, MLPs and the language-model head
cannot reliably be separated by kernel name. GPU intervals are unioned, but
different categories may still overlap each other. Host API spans overlap GPU
work. Gaps show inactivity, not its cause; inspect the trace before attributing
them to launch starvation. TP=1 has no inter-GPU model communication; this script
does not claim to measure distributed communication performance.

Each completed phase is committed to the `vllm-ngc-test-cache` volume under
`results/decode-profile-<id>/`. The local client saves `summary.json` and prints
the command to download the full directory, including traces. `--submit-only`
returns immediately with a Modal function-call ID; use the Modal app logs to
locate the result directory. Detached execution survives client disconnects,
but automatic summary download requires the local client to remain connected.

## Isolated decode versus fused scoring

`tools/ngc_decode_profile_modal.py` compares standard Triton decode against fused
NGC decode/scoring on identical synthetic Q/K/V tensors and page tables. It uses
Qwen3-8B attention dimensions, BF16, a five-query scoring window, and production
launch settings. It measures one attention layer, excluding the model's other
operations, scheduler, input preparation, and eviction bookkeeping.

```bash
NGC_DECODE_GPU=H100 .venv/bin/modal run --detach tools/ngc_decode_profile_modal.py \
  --batches 1,48 --contexts 512,2048,8192
```

Decode outputs and page scores are checked before timing. GPU timing uses CUPTI
with CUDA graphs and cold L2; a separate profiler pass attributes time to each
kernel stage. Kernel timings exclude compilation and allocations. Estimated KV
traffic is a theoretical lower bound, not a hardware measurement of actual DRAM
bytes. The results test the incremental cost of scoring at a cadence boundary;
they do not directly predict end-to-end model speedup or prove a bandwidth limit.

The client downloads the summary, metadata and log to
`.ngc-modal-results/ngc-decode-kernels-<id>/`; all artifacts also persist in the
`vllm-ngc-test-cache` Modal volume.

## Qwen2.5-0.5B at batch 48 and a 24K prompt

```bash
DECODE_PROFILE_GPU=H100 .venv/bin/modal run --detach tools/ngc_24k_modal.py::compare
```

This preset compares pristine upstream at the pinned base commit against NGC 0%
and NGC 75% with Qwen2.5-0.5B-Instruct, BF16, Triton attention, CUDA graphs,
batch 48, cadence 256 and query window 5. Each request gets exactly 24,576 prompt
tokens. After prefill and 16 warmup steps, every request generates 768 measured
tokens; EOS is ignored to maintain the same fixed batch. Prefill is excluded from
decode timing. `--steps` and `--repeats` override the default 768 and 3. Total output
includes untimed warmup and any decode during other requests' chunked prefill;
actual context ranges are saved in the summary.

The upstream package is copied from the pristine pinned installation before the
fork's Python overlay is applied. Each policy runs in a separate process with an
explicit import path, and the runner checks that the imported package matches the
requested source. This is upstream at the fork's base commit, not latest upstream.
The 24K value is prompt length, not generation length or retained cache size after
eviction. A 75% rate removes eligible pages at each boundary rather than fixing
retention at 6K tokens. The full baseline needs about 13.5 GiB of BF16 KV before
warmup/generation, so an H100 has sufficient headroom for this model.

The preset collects uninstrumented timing and a separate trace for all three
cases. Optional hardware counters remain disabled. Summaries and artifact download
instructions use the same `decode-profile-<id>` output directories described above.

## CPU optimization before/after comparison

The `ngc-cpu-baseline` branch preserves the pre-optimization snapshot, including
shared cadence checks. The `ngc-cpu-optimization` branch batches score selection
and handoff, skips empty non-debug proposals, reuses unchanged position metadata,
and avoids unchanged eviction-offset uploads. Immutable per-layer proposals and
scheduler validation remain intact; allocator freeing was already batched.

```bash
DECODE_PROFILE_GPU=H100 \
DECODE_PROFILE_BEFORE_DIR="$PWD/.ngc-modal-results/worktrees/ngc-cpu-baseline" \
  .venv/bin/modal run --detach tools/ngc_24k_modal.py::compare_ab
```

The comparison uses Qwen2.5-0.5B-Instruct, batch 48, 24,576 prompt tokens, cadence
256, query window 5, and 768 measured decode steps. It alternates before/after
order across three rounds at both 0% and 75% eviction. Each arm has a separate
process and compilation cache, with a matching warmup before measurement.
Generated-token hashes are compared at each rate outside the timing window.
These default AB runs collect unprofiled timing; the generic profiler also
supports optional traces. This measures the same repeated-eviction workload as
the earlier 24K experiment, not fixed 6K retention.

### Persistent execution state comparison

`ngc-persistent-execution` builds on `ngc-cpu-optimization` (`e4926fd48a`). It
reuses request/layer score descriptors and their grouping, batches descriptor
column updates, caches static layer settings, and skips reconstructing fused
inputs after forward. Request removal and cache reinitialization invalidate the
cached state; batch indices, lengths, block-table views, and page positions are
refreshed before use. Scoring and eviction arithmetic are unchanged.

With a clean worktree at the preceding commit, run a matched comparison with an
upstream reference:

```bash
DECODE_PROFILE_GPU=H100 \
DECODE_PROFILE_BEFORE_DIR="$PWD/.ngc-modal-results/worktrees/ngc-persistent-baseline" \
  .venv/bin/modal run --detach tools/decode_profile_modal.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --contexts 24576 --batch-size 48 \
  --steps 768 --warmup-steps 16 --repeats 2 --dtype bfloat16 \
  --backend TRITON_ATTN --policies upstream,before:0,after:0,before:0.75,after:0.75 \
  --cadence 256 --query-window 5 --ignore-eos --interleave-repeats \
  --timing-only --submit-only
```

This submits ten timing windows on one GPU. The local launcher exits after
submission; results persist in the Modal volume. Equal-rate before/after token
hashes must match. Upstream is the pinned v0.32.0 release, not upstream HEAD.

## Model speed suite: Qwen2.5 1.5B

`tools/ngc_speed_suite_modal.py` defaults to two model/batch jobs, queued to run
one at a time on a single H100. Each
job runs the five policies sequentially on the same GPU allocation: pristine
pinned upstream v0.32.0 with `TRITON_ATTN`, then NGC rates 0, 0.25, 0.5 and 0.75.
Order reverses on alternating repetitions. Every case uses a fresh process and a
complete matching generation warmup before the measured generation.

Fixed settings: BF16, cadence 256, query window 5, minimum retention 32 tokens,
2,048 prompt tokens, exactly 6,144 generated tokens per request, greedy sampling,
EOS ignored, prefix caching disabled, synchronous scheduling, and CUDA graphs
enabled. The prompt is a repeated fantasy-story instruction trimmed to exactly
2,048 token IDs; this is a speed workload, not a quality evaluation.

The default model is only `Qwen/Qwen2.5-1.5B-Instruct`, using H100/TP1 for both
batch 1 and batch 48. There is one measured repetition per policy/batch, with a
full warmup retained. No 72B job is submitted unless explicitly selected with
`--models 72b` (that optional path still defaults to two B200s/TP2).

Run the default suite (10 measured cases plus 10 warmups):

```bash
.venv/bin/modal run --detach tools/ngc_speed_suite_modal.py
```

Run a smaller subset first:

```bash
.venv/bin/modal run --detach tools/ngc_speed_suite_modal.py \
  --models 1.5b --batches 1 --repeats 1
```

Hardware can be selected before module import. Keep GPU count equal to TP:

```bash
NGC_SPEED_SMALL_GPU=H100 NGC_SPEED_LARGE_GPU=H100:4 \
  .venv/bin/modal run --detach tools/ngc_speed_suite_modal.py \
  --models 72b --large-tp 4
```

The launcher prints function-call IDs and writes
`.ngc-modal-results/speed-suite-launch-<id>.json`. It exits after spawning; the
GPU jobs continue detached. Every model/batch job checkpoints its `summary.json`
and per-case `metrics.json`/`output.log` to the shared Modal volume:

```bash
.venv/bin/modal volume get vllm-ngc-test-cache \
  results/<run_id> .ngc-modal-results/<run_id>
```

The summary includes exact model revision, source hashes, GPU identity/topology,
command, software versions, per-rank NGC activation and graph counters, and:

- TTFT and complete per-request latency distributions (mean/p50/p95/p99).
- Batch prefill span and wall time of steps containing prompt work. Mixed
  prefill/decode steps are labeled; these are not isolated GPU prefill times.
- Warmed decode wall time and tokens/s, excluding prompt steps and the first
  16 emitted tokens of every request. Full-batch decode ms/step is separate from
  decode throughput over the potentially shrinking batch.
- End-to-end aggregate output tokens/s and batch completion time.
- Pooled inter-token latency and per-request mean inter-token latency. First
  tokens are excluded from ITL; observations are at synchronous engine returns.
- Per-rank peak PyTorch allocated/reserved GPU memory and device-used snapshots.
  Allocator peaks are not total-device peaks; sums of rank peaks need not be
  simultaneous. Memory RPCs occur outside the timed generation.
- KV occupancy sampled after each scheduler allocation, retained-layout samples,
  preemption counts and recomputed scheduled-token counts. Preempted runs are
  marked invalid for a fixed-batch comparison rather than hidden.

Timing includes engine scheduler/worker work. Full latency includes metric
observation overhead; warmed decode sums `engine.step` wall time and excludes
post-step observations. No per-token device synchronization is added. Eviction
also operates during prefill, so always inspect actual retained lengths.

## Component kernel benchmark suite

`tools/ngc_kernel_suite_modal.py` is independent of model loading. It runs
`benchmarks/kernels/benchmark_ngc_suite.py` on one H100 by default:

```bash
.venv/bin/modal run --detach tools/ngc_kernel_suite_modal.py
```

Defaults cover batches 1/48, retained contexts 128/512/2048/8192, uniform and
uneven lengths (including partial pages), gapped original positions, and all,
half, or no requests due. The only default shape is `12:2:128`: Q-head,
KV-head, dimension for Qwen2.5-1.5B TP1. There is one outer timing trial; the
30 inner timing iterations, compilation warmup and calibration remain for
kernel timing. This is a single-rank benchmark, not a TP communication benchmark.

For each shape, it checks correctness before timing and measures:

- Standard decode, standalone scoring, their combined invocation, and fused
  decode/scoring on identical inputs and masks.
- Segment-count sweeps (1/4/16) for both decode paths and fused tile sizes 16/64.
  Results include raw trials and selected configurations; these are coarse
  sweeps, not claims of globally optimal tuning.
- Individual standard/fused stages, including reductions; register/spill/shared
  memory metadata when exposed by the compiler. Stage timings are diagnostics,
  not substitutes for the combined invocation.
- GPU score-input preparation and separate CPU+transfer wall timings for changed
  versus unchanged layouts; active and masked-inactive query-capture kernels.
- Prepacked stable victim ranking at rates 0/0.25/0.5/0.75. CPU grouping and the
  scheduler handoff are excluded from this selection-kernel measurement.

The main device timings use FlashInfer CUPTI, CUDA graphs and cold-L2 timing,
with allocation, compilation and calibration excluded. Estimated effective KV
bandwidth uses a minimum-byte model and is explicitly not actual DRAM traffic.
Correctness includes FP32 decode/scoring references and exact metadata/capture
checks. Non-due standalone scoring has no GPU work and is not timed as a kernel.

A smaller kernel run:

```bash
NGC_KERNEL_GPU=H100 .venv/bin/modal run --detach tools/ngc_kernel_suite_modal.py \
  --batches 1 --contexts 512 --shapes 12:2:128 --due all --lengths uniform \
  --segments 1,16 --tiles 16 --trials 2
```

Optional actual DRAM traffic/utilization counters require Nsight Compute:

```bash
DECODE_PROFILE_NCU=1 .venv/bin/modal run --detach tools/ngc_kernel_suite_modal.py \
  --counters --batches 1 --contexts 2048 --due all --lengths uniform
```

The counter pass is a separate process using winning configurations from the
uninstrumented timing report. It does not invoke CUPTI timing while Nsight is
attached. Counter permission/support failures are recorded explicitly in metadata;
inspect `counters.csv` and `counter-output.log`. They do not invalidate separately
collected timing data. Results and logs use the same volume download command as
the model suite, under `ngc-kernel-suite-<id>`.
