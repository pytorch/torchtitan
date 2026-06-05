# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight CUDA graph wrapper for eager-mode training steps.

Adapted from ``torchtitan/experiments/graph_trainer/cudagraph.py``.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor.cudagraph_trees import _use_cuda_memory_pool_manager

from torchtitan.tools.logging import logger


class _CUDAGraphManager:
    """Singleton that owns a shared graph pool and stream."""

    def __init__(self) -> None:
        self._initialized = False
        self._wrappers: list["CUDAGraphWrapper"] = []

    def maybe_initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()
        self._dummy_graph = torch.cuda.CUDAGraph()
        with (
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                self._dummy_graph,
                pool=self.graph_pool,
                stream=self.stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass

    def register(self, wrapper: "CUDAGraphWrapper") -> None:
        self._wrappers.append(wrapper)

    def teardown(self) -> None:
        if not self._initialized:
            return
        for w in self._wrappers:
            w.teardown()
        self._wrappers.clear()
        self._dummy_graph = None
        self.stream = None
        self.graph_pool = None


_manager = _CUDAGraphManager()


def cudagraph_teardown() -> None:
    """Destroy all CUDA graphs and release the shared memory pool."""
    _manager.teardown()


class CUDAGraphWrapper:
    """Wrap a callable with CUDA graph capture and replay.

    Args:
        fn: The callable (forward+backward step) to wrap.
        static_input_indices: Indices of inputs whose tensor addresses
            are stable across calls (e.g. model weights/buffers).
    """

    def __init__(
        self,
        fn: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: tuple[int, ...] | None = None,
    ):
        _manager.maybe_initialize()
        _manager.register(self)

        self._fn = fn
        self._static = set(static_input_indices or ())
        self._copy_indices = [
            i
            for i, inp in enumerate(example_inputs)
            if isinstance(inp, torch.Tensor) and i not in self._static
        ]
        self._graph: torch.cuda.CUDAGraph | None = None
        self._warmup_remaining = 1
        self._args: tuple | None = None
        self._output: Any = None

    def __call__(self, *args):
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            device = torch.cuda.current_device()
            with _use_cuda_memory_pool_manager(
                device, _manager.graph_pool, _manager.stream
            ):
                return self._fn(*args)

        if self._graph is None:
            self._args = args
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self._graph,
                pool=_manager.graph_pool,
                stream=_manager.stream,
                capture_error_mode="thread_local",
            ):
                self._output = self._fn(*args)
            logger.info("Recorded CUDA graph")

        for i in self._copy_indices:
            self._args[i].copy_(args[i])
        self._graph.replay()
        return self._output

    def teardown(self) -> None:
        self._graph = None
        self._args = None
        self._output = None
