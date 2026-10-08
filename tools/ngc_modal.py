# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run this checkout's NGC GPU tests with `modal run tools/ngc_modal.py`."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "1f76efaa2195485b92cb04215aba6fb8f5fe523d"
GPU = os.environ.get("NGC_MODAL_GPU", "L4")
REMOTE_ROOT = Path("/opt/vllm")
PYTHON = str(REMOTE_ROOT / ".venv/bin/python")
BUILD_INPUTS = [
    "csrc",
    "cmake",
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
    "requirements",
    "rust",
    "Cargo.toml",
    "Cargo.lock",
]


def check_checkout() -> str:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", *BUILD_INPUTS],
        cwd=ROOT,
        text=True,
    ).strip()
    new_inputs = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *BUILD_INPUTS],
        cwd=ROOT,
        text=True,
    ).strip()
    if changed or new_inputs:
        raise RuntimeError(
            "The cached binary/dependency environment no longer matches this checkout. "
            "Update BASE_COMMIT and its matching wheel before running:\n"
            + "\n".join(filter(None, [changed, new_inputs]))
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


if modal.is_local():
    check_checkout()

app = modal.App("vllm-ngc-tests")
cache = modal.Volume.from_name("vllm-ngc-test-cache", create_if_missing=True)

# Keep source uploads after the cached installation layers. Runtime copies merge
# with the installed checkout so mounted Python files never hide compiled modules.
image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("git", "libnuma1", "libgomp1", "libgl1", "libglib2.0-0")
    .uv_pip_install("uv==0.10.12")
    .run_commands(
        "git init /opt/vllm",
        "git -C /opt/vllm fetch --depth=1 "
        f"https://github.com/vllm-project/vllm.git {BASE_COMMIT}",
        "git -C /opt/vllm checkout --detach FETCH_HEAD",
        "uv venv --python 3.12 /opt/vllm/.venv",
    )
    .workdir(str(REMOTE_ROOT))
    .env(
        {
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_COMMIT": BASE_COMMIT,
            "VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130",
            "VLLM_MAIN_CUDA_VERSION": "13.0",
        }
    )
    .run_commands(
        "uv pip install --python /opt/vllm/.venv/bin/python "
        "-e /opt/vllm --torch-backend=cu130",
        "uv pip install --python /opt/vllm/.venv/bin/python pytest==9.1.1",
    )
    .apt_install("build-essential")
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TRITON_CACHE_DIR": "/cache/triton",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "NGC_MODAL_GPU": GPU,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        ROOT / "vllm",
        "/ngc-source/vllm",
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.so",
            "**/.env*",
            "**/_version.py",
        ],
    )
    .add_local_file(
        ROOT / "tests/kernels/attention/test_triton_unified_attention.py",
        "/ngc-source/test_triton_unified_attention.py",
    )
    .add_local_file(ROOT / "tools/ngc_modal_runner.py", "/ngc-source/runner.py")
    .add_local_file(ROOT / "tools/ngc_modal_benchmark.py", "/ngc-source/benchmark.py")
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    volumes={"/cache": cache},
    timeout=7200,
    max_containers=1,
    retries=0,
    single_use_containers=True,
)
def run_tests(job: str, selector: str, model: str, revision: str) -> dict:
    source_hash = hashlib.sha256()
    for path in sorted(Path("/ngc-source").rglob("*")):
        if path.is_file():
            source_hash.update(str(path.relative_to("/ngc-source")).encode())
            source_hash.update(path.read_bytes())
    tracked = subprocess.check_output(
        ["git", "ls-files", "vllm"], cwd=REMOTE_ROOT, text=True
    ).splitlines()
    for name in tracked:
        if not (Path("/ngc-source") / name).exists():
            (REMOTE_ROOT / name).unlink(missing_ok=True)
    shutil.copytree("/ngc-source/vllm", REMOTE_ROOT / "vllm", dirs_exist_ok=True)
    shutil.copyfile("/ngc-source/runner.py", REMOTE_ROOT / "ngc_modal_runner.py")
    shutil.copyfile("/ngc-source/benchmark.py", REMOTE_ROOT / "ngc_modal_benchmark.py")
    test_path = REMOTE_ROOT / "tests/kernels/attention/test_triton_unified_attention.py"
    shutil.copyfile("/ngc-source/test_triton_unified_attention.py", test_path)
    run_id = f"{job}-{uuid4().hex[:12]}"
    result_dir = Path("/cache/results") / run_id
    result_dir.mkdir(parents=True)
    command = [
        PYTHON,
        "ngc_modal_runner.py",
        "--job",
        job,
        "--selector",
        selector,
        "--model",
        model,
        "--result-dir",
        str(result_dir),
    ]
    try:
        with (result_dir / "output.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=REMOTE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if line.startswith("NGC_CHECKPOINT "):
                    log.flush()
                    cache.commit()
            returncode = process.wait()
        metadata = {
            "run_id": run_id,
            "job": job,
            "gpu": GPU,
            "base_commit": BASE_COMMIT,
            "checkout_commit": revision,
            "source_sha256": source_hash.hexdigest(),
            "returncode": returncode,
        }
        (result_dir / "run.json").write_text(json.dumps(metadata, indent=2))
        return {
            "metadata": metadata,
            "files": {
                path.name: path.read_text()
                for path in result_dir.iterdir()
                if path.is_file() and path.suffix != ".gz"
            },
        }
    finally:
        cache.commit()


@app.local_entrypoint()
def main(
    job: str = "kernels",
    selector: str = "ngc",
    model: str = "Qwen/Qwen3-0.6B",
    output_dir: str = ".ngc-modal-results",
    submit_only: bool = False,
):
    if job not in {
        "kernels",
        "smoke",
        "check",
        "story-ngc",
        "story-full",
        "zero-equivalence",
        "speed",
        "profile",
        "capture-speed",
        "speed-long",
    }:
        raise ValueError("Unknown job")
    revision = check_checkout()
    if submit_only:
        call = run_tests.spawn(job, selector, model, revision)
        print(f"Submitted function call: {call.object_id}")
        print("Results will persist in vllm-ngc-test-cache under results/.")
        return
    result = run_tests.remote(job, selector, model, revision)
    target = Path(output_dir) / result["metadata"]["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    for name, contents in result["files"].items():
        (target / name).write_text(contents)
    print(f"Results saved to {target.resolve()}")
    if result["metadata"]["returncode"]:
        raise RuntimeError(f"Remote {job} job failed; see {target / 'output.log'}")
