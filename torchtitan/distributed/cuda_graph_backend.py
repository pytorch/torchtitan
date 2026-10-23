# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA graph capture backend with kernel annotations."""

import warnings
from collections.abc import Callable
from typing import Any, cast

import torch
from torch.cuda._graph_annotations import get_kernel_annotations

from torchtitan.distributed.accelerator_graph import (
    GraphBackend,
    register_graph_backend,
)


def get_cuspy_cuda_graph_annotation_config(
    *, enable_event_node_ids: bool, record_py_stacks: bool = False
) -> dict[str, Any] | None:
    """``torch.cuda.graph`` annotation options for a Cuspy-profiled run.

    ``backend="cupti"`` finds each ``mark_kernels`` scope's nodes through CUPTI
    node-creation callbacks instead of walking the capture graph's edges, which
    misses nodes created before the stream began capturing. ``key_by="auto"``
    keeps annotations keyed to the capture graph, which Cuspy resolves through
    CUPTI's ``sourceGraphNodeId``, falling back to the exec graph where CUPTI or
    the driver is too old. Cuspy resolves graph event-record nodes (NCCL's under
    ``NCCL_GRAPH_MIXING_SUPPORT``) only by exec-graph id, so ``enable_event_node_ids``
    keys by the exec graph instead. ``record_py_stacks`` records each node's Python
    launch stack, which needs the CUPTI backend.

    Returns ``None`` when graph annotations are unavailable, so capture does not
    take a CUPTI subscription for nothing; the profiler warns about that once.
    """
    if not torch.cuda.graph_annotations.is_available():
        return None
    config: dict[str, Any] = {
        "backend": "cupti",
        "key_by": "exec" if enable_event_node_ids else "auto",
    }
    if record_py_stacks:
        config["record_py_stacks"] = True
    return config


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
        pool = torch.cuda.graph_pool_handle()
        self._dummy_graph = torch.cuda.CUDAGraph()
        with (
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                self._dummy_graph,
                pool=pool,
                stream=cast(torch.cuda.Stream, stream),
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
        annotation_config: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=pool,
            stream=cast(torch.cuda.Stream, stream),
            enable_annotations=True,
            annotation_config=annotation_config,
            capture_error_mode="thread_local",
        ):
            output = fn(*args)
        return graph, output

    def annotations(self) -> dict[int, list[Any]]:
        return get_kernel_annotations()

    def teardown(self) -> None:
        self._dummy_graph = None


register_graph_backend("cuda", CUDAGraphBackend())
