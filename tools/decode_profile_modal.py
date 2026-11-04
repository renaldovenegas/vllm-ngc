# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Modal full/NGC attention decode profiler; see tools/ngc_modal.md."""

import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import modal

GPU = os.environ.get("DECODE_PROFILE_GPU", "H100")
INSTALL_NCU = os.environ.get("DECODE_PROFILE_NCU", "0") == "1"
image = None
if modal.is_local():
    from tools.ngc_modal import BASE_COMMIT, PYTHON, REMOTE_ROOT, ROOT, cache
    from tools.ngc_modal import runtime_image as base_image

    image = base_image
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
    image = image.env({"DECODE_PROFILE_BASE_COMMIT": BASE_COMMIT})
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
    image = image.add_local_file(ROOT / "tools/decode_profile.py", "/profile/runner.py")
    image = image.add_local_file(
        ROOT / "tools/decode_profile_modal.py", "/profile/modal.py"
    )
    image = image.add_local_file(
        ROOT / "tools/decode_profile_modal.py",
        str(REMOTE_ROOT / "tools/decode_profile_modal.py"),
    )
else:
    BASE_COMMIT = os.environ["DECODE_PROFILE_BASE_COMMIT"]
    REMOTE_ROOT = Path("/opt/vllm")
    PYTHON = str(REMOTE_ROOT / ".venv/bin/python")
    ROOT = REMOTE_ROOT
    cache = modal.Volume.from_name("vllm-ngc-test-cache", create_if_missing=True)

app = modal.App("vllm-decode-profile")


def build_ncu_command(command, folder, module):
    """Collect one matching decode step without profiling setup or prefill."""
    counter_command = command.copy()
    counter_command[counter_command.index("--steps") + 1] = "1"
    return [
        "ncu",
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        "DecodeProfile/",
        "--kernel-name",
        "regex:" + module.NCU_ATTENTION_PATTERN,
        "--kernel-name-base",
        "demangled",
        "--graph-profiling",
        "node",
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--metrics",
        ",".join(module.METRICS),
        "--csv",
        "--print-units",
        "base",
        "--log-file",
        str(folder / "ncu.csv"),
        "--force-overwrite",
        "--export",
        str(folder / "attention"),
        *counter_command,
        "--mode",
        "counters",
    ]


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    volumes={"/cache": cache},
    timeout=14400,
    retries=0,
    single_use_containers=True,
)
def profile(config: dict):
    from importlib.util import module_from_spec, spec_from_file_location

    # Preserve the pristine pinned installation before applying fork sources.
    upstream_root = Path("/opt/decode-profile-upstream")
    if "upstream" in config["policies"]:
        shutil.copytree(REMOTE_ROOT / "vllm", upstream_root / "vllm")
    # Same source-overlay convention as ngc_modal; retain compiled release binaries.
    shutil.copytree("/ngc-source/vllm", REMOTE_ROOT / "vllm", dirs_exist_ok=True)
    spec = spec_from_file_location("decode_profile", "/profile/runner.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    run_id = "decode-profile-" + uuid4().hex[:12]
    root = Path("/cache/results") / run_id
    root.mkdir(parents=True)
    source = hashlib.sha256()
    for path in sorted(Path("/ngc-source/vllm").rglob("*")):
        if path.is_file():
            source.update(str(path.relative_to("/ngc-source/vllm")).encode())
            source.update(path.read_bytes())
    print(f"DECODE_PROFILE_RUN {run_id}", flush=True)
    report = {
        "run_id": run_id,
        "source_sha256": {
            "vllm_tree": source.hexdigest(),
            "runner": hashlib.sha256(
                Path("/profile/runner.py").read_bytes()
            ).hexdigest(),
            "modal_wrapper": hashlib.sha256(
                Path("/profile/modal.py").read_bytes()
            ).hexdigest(),
        },
        "gpu_identity": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
        ).strip(),
        "config": config,
        "base_commit": BASE_COMMIT,
        "gpu_request": GPU,
        "cases": [],
        "status": "running",
    }

    if shutil.which("ncu"):
        version = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
        report["ncu_version"] = {
            "returncode": version.returncode,
            "output": version.stdout + version.stderr,
        }
    else:
        report["ncu_version"] = None

    def save():
        (root / "summary.json").write_text(json.dumps(report, indent=2))
        cache.commit()

    save()
    for context, policy in itertools.product(config["contexts"], config["policies"]):
        rate = None if policy in {"full", "upstream"} else float(policy)
        folder = (
            root / f"context-{context}" / (policy if rate is None else f"ngc-{rate:g}")
        )
        folder.mkdir(parents=True)
        case = {
            "context": context,
            "policy": policy,
            "eviction_rate": rate,
            "status": "running",
        }
        source_root = upstream_root if policy == "upstream" else REMOTE_ROOT
        case["source_root"] = str(source_root / "vllm")
        case["source_commit"] = (
            BASE_COMMIT if policy == "upstream" else config["checkout_commit"]
        )
        case["source_kind"] = (
            "pristine_upstream" if policy == "upstream" else "working_tree_overlay"
        )
        report["cases"].append(case)
        command = [
            PYTHON,
            "/profile/runner.py",
            "--model",
            config["model"],
            "--batch-size",
            str(config["batch_size"]),
            "--context",
            str(context),
            "--steps",
            str(config["steps"]),
            "--warmup-steps",
            str(config["warmup_steps"]),
            "--repeats",
            str(config["repeats"]),
            "--dtype",
            config["dtype"],
            "--backend",
            config["backend"],
            "--output",
            str(folder),
            "--cadence",
            str(config["cadence"]),
            "--query-window",
            str(config["query_window"]),
            "--expected-source-root",
            str(source_root / "vllm"),
        ]
        if rate is not None:
            command.extend(["--eviction-rate", str(rate)])
        if config["eager"]:
            command.append("--eager")
        if config["ignore_eos"]:
            command.append("--ignore-eos")

        process_env = dict(
            os.environ,
            PYTHONPATH=str(source_root),
            VLLM_CACHE_ROOT=f"/cache/vllm-decode-profile/{folder.name}",
        )
        case["process_environment"] = {
            key: process_env[key] for key in ("PYTHONPATH", "VLLM_CACHE_ROOT")
        }

        def execute(argv, name, folder=folder, process_env=process_env):
            (folder / f"{name}.command.json").write_text(json.dumps(argv, indent=2))
            with (folder / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=process_env,
                )
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                return process.wait()

        for mode in ("timing", "trace"):
            code = execute(command + ["--mode", mode], mode)
            if code:
                case.update(status="failed", phase=mode, returncode=code)
                save()
                break
            case[mode] = json.loads((folder / f"{mode}.json").read_text())
            if mode == "trace":
                case["timeline"] = json.loads((folder / "timeline.json").read_text())
            save()
        else:
            case["status"] = "ok"
        if case["status"] == "failed":
            continue
        if not config["counters"]:
            case["counters"] = {"status": "not_requested"}
        elif not shutil.which("ncu"):
            case["counters"] = {
                "status": "unavailable",
                "reason": "Set DECODE_PROFILE_NCU=1 before launching",
            }
        else:
            ncu = build_ncu_command(command, folder, module)
            code = execute(ncu, "ncu")
            case["counters"] = (
                module.summarize_ncu(folder / "ncu.csv")
                if (folder / "ncu.csv").exists()
                else {"status": "unavailable"}
            )
            if (
                code == 0
                and case["counters"].get("status") == "unavailable"
                and (folder / "attention.ncu-rep").exists()
            ):
                imported = execute(
                    [
                        "ncu",
                        "--import",
                        str(folder / "attention.ncu-rep"),
                        "--csv",
                        "--page",
                        "raw",
                        "--print-units",
                        "base",
                        "--log-file",
                        str(folder / "ncu.csv"),
                    ],
                    "ncu-import",
                )
                if imported == 0:
                    case["counters"] = module.summarize_ncu(folder / "ncu.csv")
            case["counters"]["returncode"] = code
            case["counters"]["scope"] = (
                "First measured step after warmup; may be ordinary decode "
                "rather than a scoring boundary. Not a cadence-cycle average."
            )
            if code:
                case["counters"]["status"] = "unavailable_or_failed"
            (folder / "hardware.json").write_text(
                json.dumps(case["counters"], indent=2)
            )
        save()
    report["counter_collection_status"] = (
        "not_requested"
        if not config["counters"]
        else "complete"
        if all(c.get("counters", {}).get("status") == "ok" for c in report["cases"])
        else "partial_or_unavailable"
    )
    report["status"] = (
        "complete" if all(c["status"] == "ok" for c in report["cases"]) else "failed"
    )
    save()
    return {"run_id": run_id, "summary": report}


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B",
    contexts: str = "512,2048,8192",
    batch_size: int = 1,
    steps: int = 768,
    warmup_steps: int = 16,
    repeats: int = 3,
    dtype: str = "bfloat16",
    backend: str = "TRITON_ATTN",
    policies: str = "full,0,0.5,0.75",
    cadence: int = 256,
    query_window: int = 5,
    eager: bool = False,
    counters: bool = False,
    ignore_eos: bool = False,
    submit_only: bool = False,
):
    selected = list(dict.fromkeys(value.strip() for value in policies.split(",")))
    for policy in selected:
        if policy not in {"full", "upstream"} and not 0 <= float(policy) < 1:
            raise ValueError(
                "Policies must be full, upstream, or eviction fractions in [0, 1)"
            )
    ngc = any(policy not in {"full", "upstream"} for policy in selected)
    if cadence <= 0 or not 1 <= query_window <= cadence:
        raise ValueError("Require 1 <= query-window <= cadence")
    if ngc and backend != "TRITON_ATTN":
        raise ValueError("NGC comparison requires --backend TRITON_ATTN for all cases")
    if ngc and steps < 2 * cadence:
        raise ValueError("NGC comparison needs --steps >= 2 * cadence to cover cycles")
    lengths = [int(value) for value in contexts.split(",")]
    if not lengths or min(*lengths, batch_size, steps, warmup_steps, repeats) <= 0:
        raise ValueError("All sizes/counts must be positive")
    if dtype not in {"bfloat16", "float16"} or backend not in {
        "FLASH_ATTN",
        "TRITON_ATTN",
    }:
        raise ValueError("Unsupported dtype or backend")
    if counters and not INSTALL_NCU:
        raise ValueError("Launch with DECODE_PROFILE_NCU=1 to install Nsight Compute")
    config = dict(
        model=model,
        policies=selected,
        cadence=cadence,
        query_window=query_window,
        contexts=lengths,
        batch_size=batch_size,
        steps=steps,
        warmup_steps=warmup_steps,
        repeats=repeats,
        dtype=dtype,
        backend=backend,
        eager=eager,
        counters=counters,
        ignore_eos=ignore_eos,
        checkout_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    if submit_only:
        call = profile.spawn(config)
        print(
            f"Submitted {call.object_id}; results persist in "
            "vllm-ngc-test-cache/results"
        )
        return
    result = profile.remote(config)
    target = ROOT / ".ngc-modal-results" / result["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(result["summary"], indent=2))
    print(f"Summary saved to {target / 'summary.json'}")
    print("Download all logs, traces and counter reports with:")
    print(
        f".venv/bin/modal volume get vllm-ngc-test-cache "
        f"results/{result['run_id']} {target}"
    )
    if result["summary"]["status"] != "complete":
        raise RuntimeError("Some profile cases failed; see summary and remote logs")
