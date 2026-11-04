# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Accelerator graph wrappers with a shared memory pool."""

import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
from torch.cuda._graph_annotations import get_kernel_annotations
from torch.nn.attention.flex_attention import BlockMask
from torch.utils import _pytree as pytree

from torchtitan.tools import utils


logger = logging.getLogger(__name__)

NUM_ACCELERATOR_GRAPH_WARMUP_STEPS = 2


@dataclass(frozen=True)
class _BlockMaskInputSpec:
    num_leaves: int
    context: tuple[Any, ...]


# TODO(@jinsooihm): Remove this class and use standard pytree flattening after
# attention mask creation moves into model code and BlockMask is no longer an input.
class AcceleratorGraphInputSpec:
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
                "Accelerator graph input structure must remain constant across "
                "training steps."
            )

        flat_leaves: list[Any] = []
        for leaf, leaf_spec in zip(outer_leaves, self._leaf_specs, strict=True):
            if leaf_spec is None:
                if isinstance(leaf, BlockMask):
                    raise ValueError(
                        "Accelerator graph input structure must remain constant "
                        "across training steps."
                    )
                flat_leaves.append(leaf)
                continue

            if not isinstance(leaf, BlockMask):
                raise ValueError(
                    "Accelerator graph input structure must remain constant "
                    "across training steps."
                )
            block_mask_leaves, context = leaf._flatten()
            if (
                len(block_mask_leaves) != leaf_spec.num_leaves
                or context != leaf_spec.context
            ):
                raise ValueError(
                    "Accelerator graph BlockMask structure must remain constant across "
                    "training steps."
                )
            flat_leaves.extend(block_mask_leaves)

        return flat_leaves

    def unflatten(self, flat_leaves: Sequence[Any]) -> Any:
        if len(flat_leaves) != self._num_flat_leaves:
            raise ValueError(
                f"Accelerator graph expected {self._num_flat_leaves} inputs, "
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


class _AcceleratorGraphGradientState:
    """Keep capture-created parameter gradients alive across graph replays.

    Why this state can reduce memory use:

    1. ``set_to_none=True`` without this state fails after the
       first capture.
       Autograd allocates a gradient during capture and assigns it to
       ``param.grad``. The next ``zero_grad`` clears that Python reference.
       Graph replay runs the captured device work, but it does not repeat the
       ``param.grad`` assignment. The optimizer then sees ``param.grad is None``
       and skips the parameter update.

    2. ``set_to_none=False`` avoids that failure, but eager warmup allocates the
       gradient before capture, outside the graph memory pool. Capture only
       records writes into that existing gradient buffer. Its lifetime overlaps
       the complete forward-backward graph::

           time           forward             backward          optimizer
           gradient       [===============================================]
           forward temp   [==========]
           backward temp                       [=========]

       The graph allocator cannot reuse the gradient storage for these temporary
       tensors because their lifetimes overlap.

    3. With this state, ``set_to_none=True`` leaves gradients
       absent when capture starts. Autograd allocates them from the graph
       memory pool during backward, after many forward tensors have expired::

           time           forward             backward          optimizer
           forward temp   [==========]
           gradient                            [=========================]
           shared block   [forward temp]       [gradient                 ]

       During capture, the graph pool may reuse storage released by forward
       tensors for later gradients. This state then keeps those gradient tensors
       alive and restores each ``param.grad`` reference after replay. This makes
       ``set_to_none=True`` safe.

    Args:
        parameters: Parameters whose capture-created ``.grad`` tensors are
            recorded and restored. Parameters that do not require gradients and
            duplicate parameters are ignored.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        unique_parameters: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                unique_parameters.append(parameter)
        self._parameters = tuple(unique_parameters)
        self._captured_parameter_gradients: tuple[
            tuple[torch.nn.Parameter, torch.Tensor], ...
        ] | None = None

    def require_cleared(self) -> None:
        if any(parameter.grad is not None for parameter in self._parameters):
            raise RuntimeError(
                "All parameter gradients must be None before accelerator graph capture "
                "and replay."
            )

    def record(self) -> None:
        assert (
            self._captured_parameter_gradients is None
        ), "Accelerator graph gradients were already recorded"
        self._captured_parameter_gradients = tuple(
            (parameter, gradient)
            for parameter in self._parameters
            if (gradient := parameter.grad) is not None
        )

    def restore(self) -> None:
        assert (
            self._captured_parameter_gradients is not None
        ), "Accelerator graph gradients were not recorded"
        for parameter, gradient in self._captured_parameter_gradients:
            parameter.grad = gradient

    def teardown(self) -> None:
        if self._captured_parameter_gradients is None:
            return
        for parameter, gradient in self._captured_parameter_gradients:
            if parameter.grad is gradient:
                parameter.grad = None
        self._captured_parameter_gradients = None


def _unsupported_reason() -> str | None:
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is None:
        return "no accelerator is available"
    if accelerator.type != utils.device_type:
        return (
            f"the active device type is {utils.device_type!r}, "
            f"not {accelerator.type!r}"
        )
    if utils.device_type == "cuda":
        if torch.version.hip is not None:
            return "ROCm is not supported"
        return None
    try:
        torch.accelerator.Graph()
    except RuntimeError as e:
        return str(e)
    return None


class AcceleratorGraphManager:
    """Own the shared graph pool, capture stream, and wrappers."""

    def __init__(self) -> None:
        self._initialized = False
        self._wrappers: list["AcceleratorGraphWrapper"] = []
        self._graph_pool: Any = None
        self._stream: torch.Stream | None = None
        self._dummy_graph: torch.cuda.CUDAGraph | None = None
        self.all_annotations: dict[int, list[Any]] = {}
        self.capture_allowed = True

    @property
    def all_wrappers_warmed_up(self) -> bool:
        """Return whether all registered wrappers have finished eager warmup."""
        return all(wrapper._warmup_remaining == 0 for wrapper in self._wrappers)

    @property
    def graph_pool(self) -> Any:
        assert self._graph_pool is not None
        return self._graph_pool

    @property
    def stream(self) -> torch.Stream:
        assert self._stream is not None
        return self._stream

    def maybe_initialize(self) -> None:
        if self._initialized:
            return
        reason = _unsupported_reason()
        if reason is not None:
            raise RuntimeError(f"Graph capture is unavailable: {reason}")
        if utils.device_type == "cuda":
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
        else:
            self._stream = torch.Stream()
        self._initialized = True

    def capture(
        self,
        fn: Callable,
        args: tuple[Any, ...],
        *,
        annotation_config: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        """Capture ``fn(*args)`` into a new graph and return the graph and output."""
        if utils.device_type == "cuda":
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                cuda_graph,
                pool=self.graph_pool,
                stream=cast(torch.cuda.Stream, self.stream),
                enable_annotations=True,
                annotation_config=annotation_config,
                capture_error_mode="thread_local",
            ):
                output = fn(*args)
            self.all_annotations.update(get_kernel_annotations())
            return cuda_graph, output
        graph = torch.accelerator.Graph(
            pool=self._graph_pool, capture_error_mode="thread_local"
        )
        with self.stream, graph:
            output = fn(*args)
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        return graph, output

    def register(self, wrapper: "AcceleratorGraphWrapper") -> None:
        self._wrappers.append(wrapper)

    def teardown(self) -> None:
        if self._initialized and utils.device_type != "cuda":
            torch.accelerator.synchronize()
        for wrapper in self._wrappers:
            wrapper.teardown()
        self._wrappers.clear()
        self.all_annotations.clear()
        self._dummy_graph = None
        self._stream = None
        self._graph_pool = None
        self._initialized = False
        self.capture_allowed = True

    def log_memory(self, name: str | None, boundary: str) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        stats = torch.accelerator.memory_stats()
        free, total = torch.accelerator.get_memory_info()
        logger.debug(
            "Accelerator graph name=%s pool=%s boundary=%s allocated=%d reserved=%d "
            "active=%d inactive_split=%d peak_allocated=%d peak_reserved=%d "
            "device_free=%d device_total=%d",
            name,
            self._graph_pool,
            boundary,
            stats["allocated_bytes.all.current"],
            stats["reserved_bytes.all.current"],
            stats["active_bytes.all.current"],
            stats["inactive_split_bytes.all.current"],
            stats["allocated_bytes.all.peak"],
            stats["reserved_bytes.all.peak"],
            free,
            total,
        )


accelerator_graph_manager = AcceleratorGraphManager()


def accelerator_graph_teardown() -> None:
    """Destroy all accelerator graphs and release the shared memory pool."""
    accelerator_graph_manager.teardown()


def get_accelerator_graph_annotations() -> dict[int, list[Any]]:
    """Return all kernel annotations accumulated across accelerator graph captures."""
    return accelerator_graph_manager.all_annotations


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


def _run_eager_on_accelerator_graph_stream(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a callable eagerly on the stream reserved for accelerator graph capture."""
    accelerator_graph_manager.maybe_initialize()
    current_stream = torch.accelerator.current_stream()
    accelerator_graph_manager.stream.wait_stream(current_stream)
    with accelerator_graph_manager.stream:
        output = fn(*args, **kwargs)
    current_stream.wait_stream(accelerator_graph_manager.stream)
    return output


class AcceleratorGraphWrapper:
    """Wrap a callable with accelerator graph capture and replay.

    Args:
        fn: Callable to capture.
        example_inputs: Inputs that define the fixed input structure and tensor
            metadata for capture and replay.
        static_input_indices: Indices of inputs whose tensor addresses
            are stable across calls (e.g. model weights/buffers).
        should_check_address: Whether to verify static input tensor addresses
            before each replay. This should only be enabled for debugging.
        tensor_input_indices: Indices of inputs that should be copied before
            replay. When omitted, these are inferred from ``example_inputs``.
        num_warmup_iterations: Number of eager invocations before capture.
        prepare_for_capture: Initialize state that cannot run inside capture.
            Runs once on the shared stream before capture, with the inputs for fn.
        name: Label used in capture logs.
    Raises:
        ValueError: If ``num_warmup_iterations`` is negative.
    """

    def __init__(
        self,
        fn: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: Sequence[int] | None = None,
        should_check_address: bool = False,
        tensor_input_indices: Sequence[int] | None = None,
        *,
        num_warmup_iterations: int,
        prepare_for_capture: Callable[..., None] | None = None,
        name: str | None = None,
        annotation_config: dict[str, Any] | None = None,
    ):
        if num_warmup_iterations < 0:
            raise ValueError("num_warmup_iterations must be non-negative")
        self._fn = fn
        self._prepare_for_capture = prepare_for_capture
        self.name = name
        self._num_inputs = len(example_inputs)
        self._static_input_indices = set(static_input_indices or ())
        invalid_static_indices = {
            i for i in self._static_input_indices if i < 0 or i >= self._num_inputs
        }
        if invalid_static_indices:
            raise ValueError(
                "Accelerator graph static input indices are out of range: "
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
        self._graph: Any = None
        self._warmup_remaining = num_warmup_iterations
        self._annotation_config = annotation_config
        self._args: tuple | None = None
        self._output: Any = None
        self._should_check_address = should_check_address
        self._static_input_addresses: dict[int, int] = {}

        accelerator_graph_manager.maybe_initialize()
        accelerator_graph_manager.register(self)

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
                f"Accelerator graph expected {self._num_inputs} inputs, got {len(args)}"
            )

        for i, expected_metadata in self._tensor_metadata.items():
            arg = args[i]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Accelerator graph input {i} changed from a tensor to {type(arg)}"
                )
            actual_metadata = (arg.shape, arg.dtype, arg.device)
            if actual_metadata != expected_metadata:
                raise ValueError(
                    "Accelerator graph tensor inputs must keep the same shape, dtype, "
                    f"and device, but input {i} changed from "
                    f"{expected_metadata} to {actual_metadata}"
                )

        for i, expected in self._non_tensor_inputs.items():
            actual = args[i]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "Accelerator graph non-tensor inputs must remain constant, but input "
                    f"{i} changed from {expected!r} to {actual!r}"
                )

    def __call__(self, *args):
        self._validate_inputs(args)

        if self._warmup_remaining > 0 or (
            self._graph is None and not accelerator_graph_manager.capture_allowed
        ):
            self._warmup_remaining = max(0, self._warmup_remaining - 1)
            return _run_eager_on_accelerator_graph_stream(self._fn, *args)

        if self._graph is None:
            if self._prepare_for_capture is not None:
                _run_eager_on_accelerator_graph_stream(self._prepare_for_capture, *args)
            accelerator_graph_manager.log_memory(self.name, "before_empty_cache")
            torch.accelerator.empty_cache()
            accelerator_graph_manager.log_memory(self.name, "after_empty_cache")
            self._args = args
            self._record_static_input_addresses(args)
            self._graph, self._output = accelerator_graph_manager.capture(
                self._fn, args, annotation_config=self._annotation_config
            )
            accelerator_graph_manager.log_memory(self.name, "after_capture")
            logger.info(
                "Recorded %s graph%s",
                utils.device_type.upper(),
                f" {self.name}" if self.name else "",
            )

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


class _ForwardBackwardAcceleratorGraphWrapper(AcceleratorGraphWrapper):
    """Preserve capture-created parameter gradients across graph replays."""

    def __init__(
        self,
        fn: Callable,
        example_inputs: Sequence[Any],
        *,
        parameters: Iterable[torch.nn.Parameter],
        num_warmup_iterations: int,
        prepare_for_capture: Callable[..., None] | None = None,
        name: str | None = None,
        annotation_config: dict[str, Any] | None = None,
    ) -> None:
        self._gradient_state = _AcceleratorGraphGradientState(parameters)
        super().__init__(
            fn,
            example_inputs,
            num_warmup_iterations=num_warmup_iterations,
            prepare_for_capture=prepare_for_capture,
            name=name,
            annotation_config=annotation_config,
        )

    def __call__(self, *args: Any) -> Any:
        if self._warmup_remaining > 0 or (
            self._graph is None and not accelerator_graph_manager.capture_allowed
        ):
            return super().__call__(*args)

        self._gradient_state.require_cleared()
        should_record_gradients = self._graph is None
        output = super().__call__(*args)
        if should_record_gradients:
            self._gradient_state.record()
        self._gradient_state.restore()
        return output

    def teardown(self) -> None:
        self._gradient_state.teardown()
        super().teardown()


def accelerator_graphs_supported() -> bool:
    """Whether graph capture can actually run on this build/device."""
    return _unsupported_reason() is None


# TODO: Unify PP and non-PP callable signatures to restore strict input typing.
def _wrap_with_accelerator_graph(
    fn: Callable[..., Any],
    *,
    num_warmup_iterations: int,
    gradient_parameters: tuple[torch.nn.Parameter, ...] | None,
    prepare_for_capture: Callable[..., None] | None,
    name: str | None,
    annotation_config: dict[str, Any] | None,
) -> Callable[..., Any]:
    """Implement structured accelerator graph wrapping."""

    reason = _unsupported_reason()
    if reason is not None:
        logger.warning(
            f"Graph capture is unavailable ({reason}); using eager execution."
        )
        return fn

    # The manager keeps each captured wrapper alive until teardown.
    graph_wrapper: AcceleratorGraphWrapper | None = None
    input_spec: AcceleratorGraphInputSpec | None = None

    def run(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_wrapper, input_spec

        if graph_wrapper is None:
            input_spec = AcceleratorGraphInputSpec((args, kwargs))

            def run_flat(fn: Callable[..., Any], *flat_inputs: Any) -> Any:
                assert input_spec is not None
                step_args, step_kwargs = input_spec.unflatten(flat_inputs)
                return fn(*step_args, **step_kwargs)

            flat_inputs = input_spec.flatten((args, kwargs))
            flat_prepare = (
                partial(run_flat, prepare_for_capture)
                if prepare_for_capture is not None
                else None
            )
            if gradient_parameters is None:
                graph_wrapper = AcceleratorGraphWrapper(
                    partial(run_flat, fn),
                    flat_inputs,
                    num_warmup_iterations=num_warmup_iterations,
                    prepare_for_capture=flat_prepare,
                    name=name,
                    annotation_config=annotation_config,
                )
            else:
                graph_wrapper = _ForwardBackwardAcceleratorGraphWrapper(
                    partial(run_flat, fn),
                    flat_inputs,
                    parameters=gradient_parameters,
                    num_warmup_iterations=num_warmup_iterations,
                    prepare_for_capture=flat_prepare,
                    name=name,
                    annotation_config=annotation_config,
                )
        else:
            assert input_spec is not None
            flat_inputs = input_spec.flatten((args, kwargs))

        return graph_wrapper(*flat_inputs)

    return run


def wrap_with_accelerator_graph(
    fn: Callable[..., Any],
    *,
    num_warmup_iterations: int = 0,
    prepare_for_capture: Callable[..., None] | None = None,
    name: str | None = None,
    annotation_config: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Decorate a structured callable with accelerator graph capture and replay.

    The positional and keyword inputs must keep the same pytree structure and
    tensor metadata across calls. After capture, tensor outputs alias
    graph-owned storage that is overwritten by the next replay.

    Args:
        fn: Callable to capture.
        num_warmup_iterations: Number of eager invocations before capture.
        prepare_for_capture: Initialize state that cannot run inside capture.
            Runs once on the shared stream before capture, with the inputs for fn.
        name: Label used in capture logs.
        annotation_config: Backend-specific annotation options used at capture.
    """

    return _wrap_with_accelerator_graph(
        fn,
        num_warmup_iterations=num_warmup_iterations,
        gradient_parameters=None,
        prepare_for_capture=prepare_for_capture,
        name=name,
        annotation_config=annotation_config,
    )


def wrap_fwd_bwd_with_accelerator_graph(
    fn: Callable[..., Any],
    *,
    parameters: Iterable[torch.nn.Parameter],
    num_warmup_iterations: int = 0,
    prepare_for_capture: Callable[..., None] | None = None,
    name: str | None = None,
    annotation_config: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Wrap forward-backward and preserve gradients allocated during capture.

    Args:
        fn: Forward-backward callable to capture.
        parameters: Parameters whose capture-created gradients must remain alive.
        num_warmup_iterations: Number of eager invocations before capture.
        prepare_for_capture: Initialize state that cannot run inside capture.
            Runs once on the shared stream before capture, with the inputs for fn.
        name: Label used in capture logs.
        annotation_config: Backend-specific annotation options used at capture.
    """

    return _wrap_with_accelerator_graph(
        fn,
        num_warmup_iterations=num_warmup_iterations,
        gradient_parameters=tuple(parameters),
        prepare_for_capture=prepare_for_capture,
        name=name,
        annotation_config=annotation_config,
    )
