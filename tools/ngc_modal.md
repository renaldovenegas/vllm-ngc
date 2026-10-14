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
Scoring remains active; the job checks zero-victim decisions at 128, 256, and
384 generated tokens per request/layer and verifies no pages are freed by
eviction. It requires identical prompt/output tokens and selected-token log
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
for commit `1f76efaa2195485b92cb04215aba6fb8f5fe523d`, then overlays this checkout's
`vllm/` source. Generated native libraries stay in the editable installation;
local `.venv`, `.git`, credentials, and local compiled libraries are not uploaded.

The launcher rejects changes to native code, build metadata, or dependency files
relative to that base. Update the pinned base and its matching wheel when those
inputs change; custom native modifications require a separate source-build image.
Ordinary Python/Triton edits do not rebuild dependencies.

The kernel job deliberately uses `--noconftest -o addopts=` because this test file
is self-contained. This avoids installing unrelated model/evaluation fixtures.
