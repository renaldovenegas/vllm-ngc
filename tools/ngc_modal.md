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
```

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
