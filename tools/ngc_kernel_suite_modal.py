# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submit NGC component kernel sweeps independently of model inference."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import modal

GPU = os.environ.get("NGC_KERNEL_GPU", "H100")
image = None
if modal.is_local():
    import importlib.util
    from types import SimpleNamespace

    from tools.ngc_modal import ROOT, cache, runtime_image

    INSTALL_NCU = os.environ.get("DECODE_PROFILE_NCU", "0") == "1"
    image = runtime_image.run_commands(
        "uv pip install --python /opt/vllm/.venv/bin/python cupti-python==13.0.1 nvtx"
    )
    if INSTALL_NCU:
        image = image.apt_install("wget").run_commands(
            "wget -q https://developer.download.nvidia.com/compute/cuda/repos/"
            "debian12/x86_64/nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb "
            "-O /tmp/decode-ncu.deb",
            "echo 'e2d393e1a0ba7ae1c983b1062f6494a8e85a98c987503f71bb73cc615aafe512 "
            " /tmp/decode-ncu.deb' | sha256sum -c -",
            "dpkg-deb -x /tmp/decode-ncu.deb /",
            "ln -s /opt/nvidia/nsight-compute/2025.3.1/ncu /usr/local/bin/ncu",
        )
    image = image.env({"NGC_KERNEL_COUNTERS_INSTALLED": str(int(INSTALL_NCU))})
    image = image.add_local_dir(
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
    image = image.add_local_file(
        ROOT / "benchmarks/kernels/benchmark_ngc_suite.py",
        "/suite/benchmark_ngc_suite.py",
    )

else:
    ROOT = Path("/opt/vllm")
    INSTALL_NCU = os.environ.get("NGC_KERNEL_COUNTERS_INSTALLED") == "1"
    cache = modal.Volume.from_name("vllm-ngc-test-cache", create_if_missing=True)
app = modal.App("ngc-kernel-suite")


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=65536,
    volumes={"/cache": cache},
    timeout=86400,
    retries=0,
    single_use_containers=True,
)
def profile(config: dict):
    shutil.copytree("/ngc-source/vllm", "/opt/vllm/vllm", dirs_exist_ok=True)
    folder = Path("/cache/results") / config["run_id"]
    folder.mkdir(parents=True)
    digest = hashlib.sha256()
    for path in sorted(Path("/ngc-source/vllm").rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to("/ngc-source/vllm")).encode())
            digest.update(path.read_bytes())
    metadata = {
        "config": config,
        "source_sha256": digest.hexdigest(),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "status": "running",
    }
    command = ["/opt/vllm/.venv/bin/python", "/suite/benchmark_ngc_suite.py"]
    for key in (
        "batches",
        "contexts",
        "shapes",
        "due",
        "lengths",
        "segments",
        "tiles",
        "repeat",
        "trials",
        "cpu_repeats",
    ):
        command += ["--" + key.replace("_", "-"), str(config[key])]
    command += ["--output", str(folder / "summary.json")]
    metadata["command"] = command

    def save():
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        cache.commit()

    def execute(cmd, name):
        with (folder / name).open("w") as log:
            process = subprocess.Popen(
                cmd,
                env=dict(os.environ, PYTHONPATH="/suite:/opt/vllm"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()

    save()
    code = execute(command, "output.log")
    metadata.update(status="complete" if code == 0 else "failed", returncode=code)
    save()
    if code == 0 and config["counters"]:
        # Counter collection is a separate, instrumented process. Never use its timings.
        counters_command = command.copy()
        counters_command[counters_command.index("--output") + 1] = str(
            folder / "counter-pass.json"
        )
        counters_command += [
            "--counters",
            "--counter-config",
            str(folder / "summary.json"),
        ]
        metrics = (
            "gpu__time_duration.sum,"
            "dram__bytes_read.sum,"
            "dram__bytes_write.sum,"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        )
        ncu = [
            "ncu",
            "--target-processes",
            "all",
            "--nvtx",
            "--nvtx-include",
            "NGCKernelCounters/",
            "--metrics",
            metrics,
            "--csv",
            "--print-units",
            "base",
            "--log-file",
            str(folder / "counters.csv"),
            "--force-overwrite",
            "--export",
            str(folder / "counters"),
            *counters_command,
        ]
        try:
            status = execute(ncu, "counter-output.log")
        except OSError as error:
            status = -1
            metadata["counter_error"] = str(error)
        metadata["counters"] = {
            "returncode": status,
            "status": "collected" if status == 0 else "unavailable_or_failed",
            "note": (
                "Inspect counters.csv for permissions/unsupported-metric "
                "diagnostics; CUPTI timing results are separate."
            ),
        }
        save()
    return metadata


@app.local_entrypoint()
def main(
    batches: str = "1,48",
    contexts: str = "128,512,2048,8192",
    shapes: str = "12:2:128",
    due: str = "all,half,none",
    lengths: str = "uniform,uneven",
    segments: str = "1,4,16",
    tiles: str = "16,64",
    repeat: int = 30,
    trials: int = 1,
    cpu_repeats: int = 10,
    counters: bool = False,
):
    if counters and not INSTALL_NCU:
        raise ValueError("Launch with DECODE_PROFILE_NCU=1 to install Nsight Compute")
    spec = importlib.util.spec_from_file_location(
        "ngc_kernel_suite_validation",
        ROOT / "benchmarks/kernels/benchmark_ngc_suite.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_config(
        SimpleNamespace(
            batches=batches,
            contexts=contexts,
            shapes=shapes,
            due=due,
            lengths=lengths,
            segments=segments,
            tiles=tiles,
            repeat=repeat,
            trials=trials,
            cpu_repeats=cpu_repeats,
        )
    )
    config = dict(
        batches=batches,
        contexts=contexts,
        shapes=shapes,
        due=due,
        lengths=lengths,
        segments=segments,
        tiles=tiles,
        repeat=repeat,
        trials=trials,
        cpu_repeats=cpu_repeats,
        counters=counters,
        run_id="ngc-kernel-suite-" + uuid4().hex[:12],
        harness_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "tools/ngc_kernel_suite_modal.py",
                ROOT / "benchmarks/kernels/benchmark_ngc_suite.py",
            )
        },
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    call = profile.spawn(config)
    config["call_id"] = call.object_id
    folder = ROOT / ".ngc-modal-results" / config["run_id"]
    folder.mkdir(parents=True)
    (folder / "launch.json").write_text(json.dumps(config, indent=2))
    print(f"Submitted {config['run_id']}: {call.object_id}")
    print(
        "Download: .venv/bin/modal volume get vllm-ngc-test-cache "
        f"results/{config['run_id']} {folder}"
    )
