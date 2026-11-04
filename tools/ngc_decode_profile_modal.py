# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile standard and fused NGC decode on identical inputs on Modal."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import modal

GPU = os.environ.get("NGC_DECODE_GPU", "H100")
image = None
if modal.is_local():
    from tools.ngc_modal import (
        BASE_COMMIT,
        PYTHON,
        REMOTE_ROOT,
        ROOT,
        cache,
        runtime_image,
    )

    image = (
        runtime_image.run_commands(
            "uv pip install --python /opt/vllm/.venv/bin/python cupti-python==13.0.1"
        )
        .env({"NGC_DECODE_BASE_COMMIT": BASE_COMMIT})
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
            ROOT / "benchmarks/kernels/benchmark_ngc_decode.py", "/profile/runner.py"
        )
        .add_local_file(ROOT / "tools/ngc_decode_profile_modal.py", "/profile/modal.py")
    )
else:
    BASE_COMMIT = os.environ["NGC_DECODE_BASE_COMMIT"]
    ROOT = REMOTE_ROOT = Path("/opt/vllm")
    PYTHON = str(REMOTE_ROOT / ".venv/bin/python")
    cache = modal.Volume.from_name("vllm-ngc-test-cache", create_if_missing=True)

app = modal.App("ngc-decode-kernel-profile")


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    volumes={"/cache": cache},
    timeout=3600,
    retries=0,
    single_use_containers=True,
)
def profile(config: dict):
    shutil.copytree("/ngc-source/vllm", REMOTE_ROOT / "vllm", dirs_exist_ok=True)
    run_id = "ngc-decode-kernels-" + uuid4().hex[:12]
    folder = Path("/cache/results") / run_id
    folder.mkdir(parents=True)
    print(f"NGC_DECODE_PROFILE_RUN {run_id}", flush=True)
    source = hashlib.sha256()
    for path in sorted(Path("/ngc-source/vllm").rglob("*")):
        if path.is_file():
            source.update(str(path.relative_to("/ngc-source/vllm")).encode())
            source.update(path.read_bytes())
    metadata = {
        "run_id": run_id,
        "config": config,
        "base_commit": BASE_COMMIT,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "source_sha256": source.hexdigest(),
        "runner_sha256": hashlib.sha256(
            Path("/profile/runner.py").read_bytes()
        ).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            Path("/profile/modal.py").read_bytes()
        ).hexdigest(),
        "status": "running",
    }
    command = [
        PYTHON,
        "/profile/runner.py",
        "--batches",
        config["batches"],
        "--contexts",
        config["contexts"],
        "--output",
        str(folder / "summary.json"),
    ]
    metadata["command"] = command
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    cache.commit()
    with (folder / "output.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    metadata.update(returncode=code, status="complete" if code == 0 else "failed")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    cache.commit()
    files = {
        name: (folder / name).read_text()
        for name in ("summary.json", "metadata.json", "output.log")
        if (folder / name).exists()
    }
    return {"run_id": run_id, "returncode": code, "files": files}


@app.local_entrypoint()
def main(batches: str = "1,48", contexts: str = "512,2048,8192"):
    for values in (batches, contexts):
        if any(int(value) <= 0 for value in values.split(",")):
            raise ValueError("Batch sizes and context lengths must be positive")
    config = {
        "batches": batches,
        "contexts": contexts,
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    result = profile.remote(config)
    target = ROOT / ".ngc-modal-results" / result["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    for name, value in result["files"].items():
        (target / name).write_text(value)
    print(f"Results: {target}")
    print(
        f"All artifacts: .venv/bin/modal volume get vllm-ngc-test-cache "
        f"results/{result['run_id']} {target}"
    )
    if result["returncode"]:
        raise RuntimeError(f"Kernel profile failed; see {target / 'output.log'}")
