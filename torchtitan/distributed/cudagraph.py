# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight CUDA graph wrapper for training steps."""

import gzip
import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.cuda._annotate_cuda_graph_trace import annotate_trace
from torch.cuda._graph_annotations import get_kernel_annotations
from torch.nn.attention.flex_attention import BlockMask
from torch.utils import _pytree as pytree

from torchtitan.tools import utils
from torchtitan.tools.logging import logger


ForwardBackwardFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
    torch.Tensor,
]


@dataclass(frozen=True)
class _BlockMaskInputSpec:
    num_leaves: int
    context: tuple[Any, ...]


# TODO(@jinsooihm): Remove this class and use standard pytree flattening after
# attention mask creation moves into model code and BlockMask is no longer an input.
class CUDAGraphInputSpec:
    """Flatten structured inputs while exposing tensors stored in ``BlockMask``."""

    def __init__(self, tree: Any) -> None:
        outer_leaves, self._tree_spec = pytree.tree_flatten(
            tree,
            is_leaf=lambda value: isinstance(value, BlockMask),
        )
        self._leaf_specs: list[_BlockMaskInputSpec | None] = []
        self._num_flat_leaves = 0
        for leaf in outer_leaves:
            if isinstance(leaf, BlockMask):
                block_mask_leaves, context = leaf._flatten()
                self._leaf_specs.append(
                    _BlockMaskInputSpec(len(block_mask_leaves), context)
                )
                self._num_flat_leaves += len(block_mask_leaves)
            else:
                self._leaf_specs.append(None)
                self._num_flat_leaves += 1

    def flatten(self, tree: Any) -> list[Any]:
        outer_leaves, tree_spec = pytree.tree_flatten(
            tree,
            is_leaf=lambda value: isinstance(value, BlockMask),
        )
        if tree_spec != self._tree_spec or len(outer_leaves) != len(self._leaf_specs):
            raise ValueError(
                "CUDA graph auxiliary input structure must remain constant across "
                "training steps."
            )

        flat_leaves: list[Any] = []
        for leaf, leaf_spec in zip(outer_leaves, self._leaf_specs, strict=True):
            if leaf_spec is None:
                if isinstance(leaf, BlockMask):
                    raise ValueError(
                        "CUDA graph auxiliary input structure must remain constant "
                        "across training steps."
                    )
                flat_leaves.append(leaf)
                continue

            if not isinstance(leaf, BlockMask):
                raise ValueError(
                    "CUDA graph auxiliary input structure must remain constant "
                    "across training steps."
                )
            block_mask_leaves, context = leaf._flatten()
            if (
                len(block_mask_leaves) != leaf_spec.num_leaves
                or context != leaf_spec.context
            ):
                raise ValueError(
                    "CUDA graph BlockMask structure must remain constant across "
                    "training steps."
                )
            flat_leaves.extend(block_mask_leaves)

        return flat_leaves

    def unflatten(self, flat_leaves: Sequence[Any]) -> Any:
        if len(flat_leaves) != self._num_flat_leaves:
            raise ValueError(
                f"CUDA graph expected {self._num_flat_leaves} auxiliary inputs, "
                f"got {len(flat_leaves)}."
            )

        outer_leaves: list[Any] = []
        flat_index = 0
        for leaf_spec in self._leaf_specs:
            if leaf_spec is None:
                outer_leaves.append(flat_leaves[flat_index])
                flat_index += 1
                continue

            block_mask_end = flat_index + leaf_spec.num_leaves
            block_mask_leaves = tuple(flat_leaves[flat_index:block_mask_end])
            outer_leaves.append(
                BlockMask._unflatten(block_mask_leaves, leaf_spec.context)
            )
            flat_index = block_mask_end

        return pytree.tree_unflatten(outer_leaves, self._tree_spec)


class _CUDAGraphManager:
    """Singleton that owns a shared graph pool, stream, and annotations."""

    def __init__(self) -> None:
        self._initialized = False
        self._wrappers: list["CUDAGraphWrapper"] = []
        self._graph_pool: Any = None
        self._stream: torch.cuda.Stream | None = None
        self._dummy_graph: torch.cuda.CUDAGraph | None = None
        self.all_annotations: dict[int, list[Any]] = {}

    @property
    def graph_pool(self) -> Any:
        assert self._graph_pool is not None
        return self._graph_pool

    @property
    def stream(self) -> torch.cuda.Stream:
        assert self._stream is not None
        return self._stream

    def maybe_initialize(self) -> None:
        if self._initialized:
            return
        graph_pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        dummy_graph = torch.cuda.CUDAGraph()
        self._graph_pool = graph_pool
        self._stream = stream
        self._dummy_graph = dummy_graph
        with (
            warnings.catch_warnings(record=True),
            torch.cuda.graph(
                dummy_graph,
                pool=graph_pool,
                stream=stream,
                capture_error_mode="thread_local",
            ),
        ):
            pass
        self._initialized = True

    def register(self, wrapper: "CUDAGraphWrapper") -> None:
        self._wrappers.append(wrapper)

    def teardown(self) -> None:
        if not self._initialized:
            return
        for wrapper in self._wrappers:
            wrapper.teardown()
        self._wrappers.clear()
        self._dummy_graph = None
        self._stream = None
        self._graph_pool = None
        self._initialized = False


_manager = _CUDAGraphManager()


def cudagraph_teardown() -> None:
    """Destroy all CUDA graphs and release the shared memory pool."""
    _manager.teardown()


def get_cudagraph_annotations() -> dict[int, list[Any]]:
    """Return all kernel annotations accumulated across CUDA graph captures."""
    return _manager.all_annotations


def cudagraph_annotate_trace_post_processor(trace_path: str) -> None:
    """Post-process a profiler trace with captured CUDA graph annotations."""
    annotations = get_cudagraph_annotations()
    if not annotations:
        return

    open_trace = gzip.open if trace_path.endswith(".gz") else open
    with open_trace(trace_path, "rt") as trace_file:
        trace = json.load(trace_file)

    count = annotate_trace(trace, annotations)
    if count > 0:
        with open_trace(trace_path, "wt") as trace_file:
            json.dump(trace, trace_file)
        logger.info(f"Annotated {count} CUDA graph kernel events in profiler trace")


class CUDAGraphWrapper:
    """Wrap a callable with CUDA graph capture and replay.

    Args:
        fn: The callable (forward+backward step) to wrap.
        example_inputs: Inputs that define the fixed input structure and tensor
            metadata for capture and replay.
        static_input_indices: Indices of inputs whose tensor addresses
            are stable across calls (e.g. model weights/buffers).
        should_check_address: Whether to verify static input tensor addresses
            before each replay. This should only be enabled for debugging.
        tensor_input_indices: Indices of inputs that should be copied before
            replay. When omitted, these are inferred from ``example_inputs``.
    """

    def __init__(
        self,
        fn: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: Sequence[int] | None = None,
        should_check_address: bool = False,
        tensor_input_indices: Sequence[int] | None = None,
    ):
        self._fn = fn
        self._num_inputs = len(example_inputs)
        self._static_input_indices = set(static_input_indices or ())
        invalid_static_indices = {
            i for i in self._static_input_indices if i < 0 or i >= self._num_inputs
        }
        if invalid_static_indices:
            raise ValueError(
                "CUDA graph static input indices are out of range: "
                f"{sorted(invalid_static_indices)}"
            )

        if tensor_input_indices is not None:
            self._input_indices_to_copy = [
                i for i in tensor_input_indices if i not in self._static_input_indices
            ]
        else:
            self._input_indices_to_copy = [
                i
                for i, inp in enumerate(example_inputs)
                if isinstance(inp, torch.Tensor) and i not in self._static_input_indices
            ]
        self._tensor_metadata = {
            i: (inp.shape, inp.dtype, inp.device)
            for i, inp in enumerate(example_inputs)
            if isinstance(inp, torch.Tensor)
        }
        self._non_tensor_inputs = {
            i: inp
            for i, inp in enumerate(example_inputs)
            if not isinstance(inp, torch.Tensor)
        }
        self._graph: torch.cuda.CUDAGraph | None = None
        self._warmup_remaining = 1
        self._args: tuple | None = None
        self._output: Any = None
        self._should_check_address = should_check_address
        self._static_input_addresses: dict[int, int] = {}

        _manager.maybe_initialize()
        _manager.register(self)

    def _record_static_input_addresses(self, args: tuple[Any, ...]) -> None:
        for i in self._static_input_indices:
            arg = args[i]
            if isinstance(arg, torch.Tensor):
                self._static_input_addresses[i] = arg.data_ptr()

    def _check_static_input_addresses(self, args: tuple[Any, ...]) -> None:
        for i, expected in self._static_input_addresses.items():
            arg = args[i]
            assert isinstance(
                arg, torch.Tensor
            ), f"Static input at index {i} changed from a tensor to {type(arg)}"
            actual = arg.data_ptr()
            assert expected == actual, (
                "Expected the same static tensor address at index "
                f"{i}, but found {expected} != {actual}"
            )

    def _validate_inputs(self, args: tuple[Any, ...]) -> None:
        if len(args) != self._num_inputs:
            raise ValueError(
                f"CUDA graph expected {self._num_inputs} inputs, got {len(args)}"
            )

        for i, expected_metadata in self._tensor_metadata.items():
            arg = args[i]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"CUDA graph input {i} changed from a tensor to {type(arg)}"
                )
            actual_metadata = (arg.shape, arg.dtype, arg.device)
            if actual_metadata != expected_metadata:
                raise ValueError(
                    "CUDA graph tensor inputs must keep the same shape, dtype, "
                    f"and device, but input {i} changed from "
                    f"{expected_metadata} to {actual_metadata}"
                )

        for i, expected in self._non_tensor_inputs.items():
            actual = args[i]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "CUDA graph non-tensor inputs must remain constant, but input "
                    f"{i} changed from {expected!r} to {actual!r}"
                )

    def __call__(self, *args):
        self._validate_inputs(args)

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            current_stream = torch.cuda.current_stream()
            _manager.stream.wait_stream(current_stream)
            with torch.cuda.stream(_manager.stream):
                output = self._fn(*args)
            current_stream.wait_stream(_manager.stream)
            return output

        if self._graph is None:
            self._args = args
            self._record_static_input_addresses(args)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self._graph,
                pool=_manager.graph_pool,
                stream=_manager.stream,
                enable_annotations=True,
                capture_error_mode="thread_local",
            ):
                self._output = self._fn(*args)
            _manager.all_annotations.update(get_kernel_annotations())
            logger.info("Recorded CUDA graph")

        if self._should_check_address:
            self._check_static_input_addresses(args)

        assert self._args is not None
        assert self._graph is not None
        for i in self._input_indices_to_copy:
            self._args[i].copy_(args[i])
        self._graph.replay()
        return self._output

    def teardown(self) -> None:
        self._graph = None
        self._args = None
        self._output = None
        self._static_input_addresses.clear()
        self._non_tensor_inputs.clear()


def wrap_with_cuda_graph(fwd_bwd_fn: ForwardBackwardFn) -> ForwardBackwardFn:
    """Decorate a callable with CUDA graph capture/replay.

    After capture, the returned loss aliases graph-owned storage that is
    overwritten by the next replay. Callers must preserve it when needed.
    """

    if not (
        utils.device_type == "cuda"
        and torch.cuda.is_available()
        and torch.version.hip is None
    ):
        logger.warning(
            "CUDA graph capture is only supported on NVIDIA CUDA; "
            "using eager execution."
        )
        return fwd_bwd_fn

    # Every wrapper is registered to the manager in this module and persists
    # until cudagraph_teardown is called.
    graph_wrapper: CUDAGraphWrapper | None = None
    # Input handling for dynamic custom type inputs like BlockMask.
    extra_input_spec: CUDAGraphInputSpec | None = None

    def run(
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        nonlocal graph_wrapper, extra_input_spec

        if graph_wrapper is None:
            extra_input_spec = CUDAGraphInputSpec(extra_kwargs)

            def flat_fwd_bwd(
                step_inputs: torch.Tensor,
                step_labels: torch.Tensor,
                step_global_valid_tokens: torch.Tensor,
                *flat_extra_inputs: Any,
            ) -> torch.Tensor:
                assert extra_input_spec is not None
                step_extra_kwargs = cast(
                    dict[str, Any],
                    extra_input_spec.unflatten(flat_extra_inputs),
                )
                return fwd_bwd_fn(
                    step_inputs,
                    step_labels,
                    step_global_valid_tokens,
                    step_extra_kwargs,
                )

            extra_flat = extra_input_spec.flatten(extra_kwargs)
            example_inputs = [
                inputs,
                labels,
                global_valid_tokens,
                *extra_flat,
            ]
            graph_wrapper = CUDAGraphWrapper(
                flat_fwd_bwd,
                example_inputs,
            )
        else:
            assert extra_input_spec is not None
            extra_flat = extra_input_spec.flatten(extra_kwargs)

        return graph_wrapper(
            inputs,
            labels,
            global_valid_tokens,
            *extra_flat,
        )

    return run
