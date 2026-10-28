# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A separate, bounded CUDA graph cache for NGC scoring boundaries."""

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import Any

import torch


def _signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            tuple(value.shape),
            value.stride(),
            value.dtype,
            value.device,
        )
    if isinstance(value, dict):
        return tuple((key, _signature(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_signature(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, _signature(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported NGC graph dependency: {type(value)}")


class NGCBoundaryGraph:
    """Cache one boundary graph; retain all external storage until replacement.

    Dependencies describe fixed addresses/layouts, not tensor contents. Callers
    update their persistent input tensors before invoking run. Replacing an entry
    warms the callable before capturing, so compilation and scratch growth occur
    outside capture. Only idempotent, inference-only forwards may be wrapped.
    """

    def __init__(self) -> None:
        self.graph: torch.cuda.CUDAGraph | None = None
        self.key: Any = None
        self.output: Any = None
        self.dependencies: Any = None
        self.stream: torch.cuda.Stream | None = None
        self.captures = 0
        self.replays = 0

    def clear(self) -> None:
        self.graph = None
        self.key = None
        self.output = None
        self.dependencies = None

    def run(
        self,
        forward: Callable[[], Any],
        dependencies: Callable[[], Any],
        *,
        capture_ready: Callable[[], bool] | None = None,
    ) -> Any:
        inputs = dependencies()
        key = _signature(inputs)
        if self.graph is not None and key == self.key:
            self.graph.replay()
            self.replays += 1
            return self.output

        self.clear()
        # Current-token cache/ring writes are idempotent during warmup/capture.
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        caller_stream = torch.cuda.current_stream()
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            warm_output = forward()
        caller_stream.wait_stream(self.stream)
        if capture_ready is not None and not capture_ready():
            return warm_output
        del warm_output
        torch.accelerator.synchronize()
        inputs = dependencies()
        key = _signature(inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            output = forward()
        if _signature(dependencies()) != key:
            raise RuntimeError("NGC graph dependencies changed during capture")
        graph.replay()
        self.graph, self.key = graph, key
        self.output, self.dependencies = output, inputs
        self.captures += 1
        return output
