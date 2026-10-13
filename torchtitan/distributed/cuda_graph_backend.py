# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA graph capture backend with kernel annotations."""

import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch.cuda._graph_annotations import get_kernel_annotations

from torchtitan.distributed.accelerator_graph import (
    GraphBackend,
    register_graph_backend,
)


class CUDAGraphBackend(GraphBackend):
    """Capture with ``torch.cuda.graph`` so kernel annotations are recorded."""

    def __init__(self) -> None:
        self._dummy_graph: torch.cuda.CUDAGraph | None = None

    def unsupported_reason(self) -> str | None:
        if torch.version.hip is not None:
            return "ROCm is not supported"
        return None

    def new_stream(self) -> torch.Stream:
        return torch.cuda.Stream()

    def initialize(self, stream: torch.Stream) -> Any:
        assert isinstance(stream, torch.cuda.Stream)
        pool = torch.cuda.graph_pool_handle()
        self._dummy_graph = torch.cuda.CUDAGraph()
        with (
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                self._dummy_graph,
                pool=pool,
                stream=stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass
        return pool

    def capture(
        self,
        fn: Callable,
        args: tuple[Any, ...],
        *,
        pool: Any,
        stream: torch.Stream,
    ) -> tuple[Any, Any]:
        assert isinstance(stream, torch.cuda.Stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=pool,
            stream=stream,
            enable_annotations=True,
            capture_error_mode="thread_local",
        ):
            output = fn(*args)
        return graph, output

    def annotations(self) -> dict[int, list[Any]]:
        return get_kernel_annotations()

    def teardown(self) -> None:
        self._dummy_graph = None


register_graph_backend("cuda", CUDAGraphBackend())
