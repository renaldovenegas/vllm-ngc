# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen2.5-0.5B: upstream versus NGC at batch 48 and a 24K-token prompt.

Run: .venv/bin/modal run --detach tools/ngc_24k_modal.py::compare
"""

from tools.decode_profile_modal import app
from tools.decode_profile_modal import main as profile_main


@app.local_entrypoint()
def compare(repeats: int = 3, steps: int = 768, submit_only: bool = False):
    """Force equal prompt/decode token counts; profile three attention policies."""
    profile_main(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        contexts="24576",
        batch_size=48,
        policies="upstream,0,0.75",
        backend="TRITON_ATTN",
        dtype="bfloat16",
        cadence=256,
        query_window=5,
        steps=steps,
        warmup_steps=16,
        repeats=repeats,
        ignore_eos=True,
        submit_only=submit_only,
    )


@app.local_entrypoint()
def compare_ab(
    repeats: int = 3, steps: int = 768, trace: bool = False, upstream: bool = False
):
    """Interleave committed before/current policies on one GPU allocation.

    Set DECODE_PROFILE_BEFORE_DIR to the baseline worktree before launching.
    """
    profile_main(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        contexts="24576",
        batch_size=48,
        policies=("upstream," if upstream else "")
        + "before:0,after:0,before:0.75,after:0.75",
        backend="TRITON_ATTN",
        dtype="bfloat16",
        cadence=256,
        query_window=5,
        steps=steps,
        warmup_steps=16,
        repeats=repeats,
        ignore_eos=True,
        interleave_repeats=True,
        timing_only=not trace,
    )
