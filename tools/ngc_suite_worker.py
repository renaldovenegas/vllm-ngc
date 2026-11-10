# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only worker extension: install policy in every TP process."""

import os
from pathlib import Path

_POLICY = None
_LAYERS = []
if os.environ.get("NGC_SUITE_RATE", "upstream") != "upstream":
    from decode_profile import attention_policy

    _POLICY = attention_policy(float(os.environ["NGC_SUITE_RATE"]), 256, 5)
    _LAYERS = _POLICY.__enter__()


class SuiteWorker:
    def suite_memory(self, reset=False):
        import torch

        import vllm

        torch.accelerator.synchronize()
        if reset:
            torch.accelerator.reset_peak_memory_stats()
        stats = torch.accelerator.memory_stats()
        free_bytes, total_bytes = torch.accelerator.get_memory_info()
        root = str(Path(vllm.__file__).resolve().parent)
        expected = os.environ["NGC_SUITE_SOURCE"]
        if root != expected:
            raise RuntimeError(f"Worker imported {root}, expected {expected}")
        if _POLICY is not None and not _LAYERS:
            raise RuntimeError("NGC policy was not installed in this worker")
        runner = self.model_runner
        graph = getattr(runner, "_ngc_boundary_graph", None)
        return {
            "rank": self.rank,
            "device_free_bytes": free_bytes,
            "device_total_bytes": total_bytes,
            "device_used_bytes_snapshot": total_bytes - free_bytes,
            "source_root": root,
            "ngc_layers": len(_LAYERS),
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "reserved_bytes": torch.accelerator.memory_reserved(),
            "peak_allocated_bytes": stats["allocated_bytes.all.peak"],
            "peak_reserved_bytes": stats["reserved_bytes.all.peak"],
            "boundary_graph_captures": getattr(graph, "captures", 0),
            "boundary_graph_replays": getattr(graph, "replays", 0),
        }
