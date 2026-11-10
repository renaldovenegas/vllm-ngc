# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched upstream/NGC speed sweep; defaults submit detached GPU jobs."""

import hashlib
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from uuid import uuid4

import modal

SMALL_GPU = os.environ.get("NGC_SPEED_SMALL_GPU", "H100")
LARGE_GPU = os.environ.get("NGC_SPEED_LARGE_GPU", "B200:2")
MODELS = {"1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "72b": "Qwen/Qwen2.5-72B-Instruct"}
POLICIES = ["upstream", "0", "0.25", "0.5", "0.75"]
image = None
if modal.is_local():
    from tools.ngc_modal import BASE_COMMIT, ROOT, cache, runtime_image

    image = runtime_image.env({"NGC_SPEED_BASE": BASE_COMMIT}).add_local_dir(
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
    for filename in (
        "decode_profile.py",
        "ngc_suite_worker.py",
        "ngc_speed_suite_runner.py",
    ):
        image = image.add_local_file(ROOT / "tools" / filename, "/suite/" + filename)
else:
    BASE_COMMIT = os.environ["NGC_SPEED_BASE"]
    ROOT = Path("/opt/vllm")
    cache = modal.Volume.from_name("vllm-ngc-test-cache", create_if_missing=True)
app = modal.App("ngc-speed-suite")


def digest_tree(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def gpu_count(spec):
    """Read the device count from a Modal GPU allocation string."""
    parts = spec.rsplit(":", 1)
    return int(parts[1]) if len(parts) == 2 else 1


def resolve_model_revision(model, revision):
    """Use the inference venv, not Modal's separate container interpreter."""
    result = subprocess.run(
        [
            "/opt/vllm/.venv/bin/python",
            "-c",
            "import json, sys; from huggingface_hub import model_info; "
            "print(json.dumps(model_info(sys.argv[1], revision=sys.argv[2]).sha))",
            model,
            revision,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = json.loads(result.stdout)
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError("Hugging Face returned an empty model revision")
    return resolved


def run_group(config):
    root = Path("/cache/results") / config["run_id"]
    root.mkdir(parents=True)
    upstream = Path("/opt/speed-upstream")
    shutil.copytree("/opt/vllm/vllm", upstream / "vllm")
    shutil.copytree("/ngc-source/vllm", "/opt/vllm/vllm", dirs_exist_ok=True)
    report = {
        "status": "running",
        "config": config,
        "resolved_model_revision": None,
        "base_commit": BASE_COMMIT,
        "fork_source_sha256": digest_tree(Path("/ngc-source/vllm")),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "cases": [],
        "harness_sha256": {
            name: hashlib.sha256((Path("/suite") / name).read_bytes()).hexdigest()
            for name in (
                "decode_profile.py",
                "ngc_suite_worker.py",
                "ngc_speed_suite_runner.py",
            )
        },
    }

    def save():
        (root / "summary.json").write_text(json.dumps(report, indent=2))
        cache.commit()

    save()
    try:
        revision = resolve_model_revision(config["model"], config["revision"])
    except Exception as error:
        report.update(
            status="failed",
            phase="resolve_model_revision",
            error=traceback.format_exc(),
            stderr=getattr(error, "stderr", None),
        )
        save()
        raise
    report["resolved_model_revision"] = revision
    save()
    for repeat in range(config["repeats"]):
        policies = POLICIES if repeat % 2 == 0 else list(reversed(POLICIES))
        for policy in policies:
            folder = root / f"round-{repeat + 1}" / policy
            folder.mkdir(parents=True)
            source = upstream if policy == "upstream" else Path("/opt/vllm")
            env = dict(
                os.environ,
                PYTHONPATH=f"/suite:{source}",
                NGC_SUITE_SOURCE=str(source / "vllm"),
                NGC_SUITE_RATE=policy,
                VLLM_CACHE_ROOT=(
                    f"/cache/speed-suite/{config['model_key']}/"
                    f"{config['batch']}/{policy}"
                ),
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                VLLM_ENABLE_V1_MULTIPROCESSING="0",
            )
            command = [
                "/opt/vllm/.venv/bin/python",
                "/suite/ngc_speed_suite_runner.py",
                "--model",
                config["model"],
                "--revision",
                revision,
                "--policy",
                policy,
                "--batch",
                str(config["batch"]),
                "--tp",
                str(config["tp"]),
                "--output",
                str(folder / "metrics.json"),
            ]
            case = {
                "policy": policy,
                "round": repeat + 1,
                "command": command,
                "status": "running",
            }
            report["cases"].append(case)
            save()
            print(
                f"CASE_START {config['run_id']} {policy} round{repeat + 1}", flush=True
            )
            try:
                with (folder / "output.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                case.update(
                    status="complete" if code == 0 else "failed", returncode=code
                )
                if code == 0:
                    case["result"] = json.loads((folder / "metrics.json").read_text())
            except Exception:
                case.update(status="failed", error=traceback.format_exc())
            save()
            if case["status"] == "failed":
                report.update(status="failed", stopped_early=True)
                save()
                return report
    report["status"] = (
        "complete"
        if all(c["status"] == "complete" for c in report["cases"])
        else "failed"
    )
    save()
    return report


@app.function(
    image=image,
    gpu=SMALL_GPU,
    max_containers=1,
    cpu=8,
    memory=65536,
    volumes={"/cache": cache},
    timeout=86400,
    retries=0,
    single_use_containers=True,
)
def small(config: dict):
    return run_group(config)


@app.function(
    image=image,
    gpu=LARGE_GPU,
    cpu=16,
    memory=262144,
    volumes={"/cache": cache},
    timeout=86400,
    retries=0,
    single_use_containers=True,
)
def large(config: dict):
    return run_group(config)


@app.local_entrypoint()
def main(
    models: str = "1.5b",
    batches: str = "1,48",
    repeats: int = 1,
    small_tp: int = 1,
    large_tp: int = 2,
    revision: str = "main",
):
    selected = models.split(",")
    sizes = [int(b) for b in batches.split(",")]
    if not set(selected) <= MODELS.keys() or not set(sizes) <= {1, 48} or repeats < 1:
        raise ValueError("Use models1.5b,72b; batches1,48; positive repeats")
    if small_tp != 1 or large_tp not in (2, 4, 8):
        raise ValueError("Use TP1 for1.5B and TP2/4/8 for72B; match GPU env allocation")
    if ("1.5b" in selected and gpu_count(SMALL_GPU) != small_tp) or (
        "72b" in selected and gpu_count(LARGE_GPU) != large_tp
    ):
        raise ValueError("GPU allocation device count must match tensor parallel size")
    manifest = []
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    for model in selected:
        for batch in sizes:
            config = dict(
                model_key=model,
                model=MODELS[model],
                revision=revision,
                batch=batch,
                repeats=repeats,
                tp=small_tp if model == "1.5b" else large_tp,
                checkout_commit=commit,
                run_id="ngc-speed-suite-" + uuid4().hex[:12],
            )
            call = (small if model == "1.5b" else large).spawn(config)
            manifest.append({**config, "call_id": call.object_id})
            print(f"Submitted {config['run_id']}: {call.object_id}")
    target = (
        ROOT
        / ".ngc-modal-results"
        / ("speed-suite-launch-" + uuid4().hex[:12] + ".json")
    )
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2))
    print(f"Launch manifest: {target}")
    print(
        "Download: .venv/bin/modal volume get vllm-ngc-test-cache "
        "results/<run_id> .ngc-modal-results/<run_id>"
    )
