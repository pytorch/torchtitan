# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight XPU graph wrapper for eager-mode training steps.

Sibling of ``torchtitan/distributed/cudagraph.py``: same structure, same
public surface, ``torch.xpu`` APIs in place of ``torch.cuda``. The trainer
picks between the two on ``utils.device_type``.

Note [eager XPU graph capture is all-or-nothing]
This wrapper captures the whole eager forward+backward step. Unlike
``torchtitan/experiments/graph_trainer/xpugraph.py``, which analyzes the FX
graph and captures only the capturable subgraphs, there is no node-level
filtering here: a single ``.item()``/``_local_scalar_dense``, ``index_put``,
or a collective the runtime cannot record will fail capture for the entire
step. FSDP2 in particular issues its all-gather/reduce-scatter through
ProcessGroupXCCL's coalesced path, whose ``endCoalescing()`` reaches oneCCL's
``onecclGroupEnd()``; an unpatched oneCCL performs a blocking event wait there
that SYCL command-graph recording rejects outright. See the docstring of
``_is_known_xpugraph_unsafe_target`` in the experiments module for the
root-cause detail.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from torchtitan.distributed.cudagraph import CUDAGraphInputSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


# A build without torch.xpu graph capture falls back to eager rather than failing at import.
_REQUIRED_XPU_GRAPH_APIS = (
    "XPUGraph",
    "graph",
    "graph_pool_handle",
    "Stream",
    "current_stream",
    "stream",
    "synchronize",
)


def _unsupported_reason() -> str | None:
    """Return why XPU graph capture is unavailable, or None if it is usable."""
    if utils.device_type != "xpu":
        return f"the active device type is {utils.device_type!r}, not 'xpu'"
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return "no XPU device is available"
    missing = [name for name in _REQUIRED_XPU_GRAPH_APIS if not hasattr(torch.xpu, name)]
    if missing:
        return "this PyTorch build is missing " + ", ".join(
            f"torch.xpu.{name}" for name in missing
        )
    return None


class _XPUGraphManager:
    """Singleton that owns a shared graph pool and stream."""

    def __init__(self) -> None:
        self._initialized = False
        self._wrappers: list["XPUGraphWrapper"] = []
        self._graph_pool: Any = None
        self._stream: Any = None

    @property
    def graph_pool(self) -> Any:
        assert self._graph_pool is not None
        return self._graph_pool

    @property
    def stream(self) -> Any:
        assert self._stream is not None
        return self._stream

    def maybe_initialize(self) -> None:
        if self._initialized:
            return
        reason = _unsupported_reason()
        if reason is not None:
            raise RuntimeError(f"XPU graph capture is unavailable: {reason}")
        # A shared pool and capture stream let graphs reuse each other's memory; unlike CUDA no priming capture is needed.
        self._graph_pool = torch.xpu.graph_pool_handle()
        self._stream = torch.xpu.Stream()
        self._initialized = True

    def register(self, wrapper: "XPUGraphWrapper") -> None:
        self._wrappers.append(wrapper)

    def teardown(self) -> None:
        """Destroy all XPU graphs and release the shared memory pool.

        Note [explicit XPU graph teardown]
        A recorded XPU graph retains references to communication and runtime
        resources, which can prevent a clean process-group shutdown. Drop the
        graph objects explicitly, after a device synchronize so no replay or
        capture is still in flight.
        """
        if not self._initialized:
            return
        torch.xpu.synchronize()
        for wrapper in self._wrappers:
            wrapper.teardown()
        self._wrappers.clear()
        self._stream = None
        self._graph_pool = None
        self._initialized = False


_manager = _XPUGraphManager()


def xpugraph_teardown() -> None:
    """Destroy all XPU graphs and release the shared memory pool.

    See Note [explicit XPU graph teardown] for more details.
    """
    _manager.teardown()


class XPUGraphWrapper:
    """Wrap a callable with XPU graph capture and replay.

    Args:
        fn: The callable (forward+backward step) to wrap.
        example_inputs: Inputs that define the fixed input structure and tensor
            metadata for capture and replay.
        static_input_indices: Indices of inputs whose tensor addresses
            are stable across calls (e.g. model weights/buffers).
        should_check_address: Whether to verify static input tensor addresses
            before each replay. This should only be enabled for debugging.
    """

    def __init__(
        self,
        fn: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: tuple[int, ...] | None = None,
        should_check_address: bool = False,
        *,
        num_warmup_iterations: int = 1,
    ):
        if num_warmup_iterations < 0:
            raise ValueError("num_warmup_iterations must be non-negative")
        self._fn = fn
        self._num_inputs = len(example_inputs)
        self._static_input_indices = set(static_input_indices or ())
        invalid_static_indices = {
            i for i in self._static_input_indices if i < 0 or i >= self._num_inputs
        }
        if invalid_static_indices:
            raise ValueError(
                "XPU graph static input indices are out of range: "
                f"{sorted(invalid_static_indices)}"
            )

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
                f"XPU graph expected {self._num_inputs} inputs, got {len(args)}"
            )

        for i, expected_metadata in self._tensor_metadata.items():
            arg = args[i]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"XPU graph input {i} changed from a tensor to {type(arg)}"
                )
            actual_metadata = (arg.shape, arg.dtype, arg.device)
            if actual_metadata != expected_metadata:
                raise ValueError(
                    "XPU graph tensor inputs must keep the same shape, dtype, "
                    f"and device, but input {i} changed from "
                    f"{expected_metadata} to {actual_metadata}"
                )

        for i, expected in self._non_tensor_inputs.items():
            actual = args[i]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "XPU graph non-tensor inputs must remain constant, but input "
                    f"{i} changed from {expected!r} to {actual!r}"
                )

    def _capture(self, args: tuple[Any, ...]) -> None:
        self._args = args
        self._record_static_input_addresses(args)
        self._graph = torch.xpu.XPUGraph()
        try:
            # torch.xpu.graph takes no capture_error_mode.
            with torch.xpu.graph(
                self._graph,
                pool=_manager.graph_pool,
                stream=_manager.stream,
            ):
                self._output = self._fn(*args)
        except Exception:
            self._graph = None
            self._args = None
            # The usual causes are a host sync or a non-recordable collective in the step.
            logger.error(
                "XPU graph capture of the training step failed. The whole "
                "forward+backward is captured as one graph, so any host "
                "synchronization (e.g. .item()), index_put, or non-recordable "
                "collective in the step will fail capture. Collective capture "
                "additionally requires a oneCCL whose group_end() does not "
                "block on an event. Set --training.disable_cuda_graphs to run "
                "eager."
            )
            raise
        logger.info("Recorded XPU graph")

    def __call__(self, *args):
        self._validate_inputs(args)

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            # Warm up lazy kernels on the side stream that capture will use.
            current_stream = torch.xpu.current_stream()
            _manager.stream.wait_stream(current_stream)
            with torch.xpu.stream(_manager.stream):
                output = self._fn(*args)
            current_stream.wait_stream(_manager.stream)
            return output

        if self._graph is None:
            self._capture(args)

        if self._should_check_address:
            self._check_static_input_addresses(args)

        assert self._args is not None
        assert self._graph is not None
        for i in self._input_indices_to_copy:
            self._args[i].copy_(args[i])
        self._graph.replay()
        return self._output

    def teardown(self) -> None:
        if self._graph is not None:
            reset = getattr(self._graph, "reset", None)
            if reset is not None:
                reset()
        self._graph = None
        self._args = None
        self._output = None
        self._static_input_addresses.clear()
        self._non_tensor_inputs.clear()


# TODO: Unify PP and non-PP callable signatures to restore strict input typing.
def wrap_with_xpu_graph(
    fn: Callable[..., torch.Tensor],
    *,
    gradient_accumulation_steps: int,
    sdc_num_steps: int,
    sdc_num_replays: int,
    num_warmup_steps: int = 2,
) -> Callable[..., torch.Tensor]:
    """Decorate a structured callable with XPU graph capture and replay.

    The positional and keyword inputs must keep the same pytree structure and
    tensor metadata across calls. After capture, tensor outputs alias
    graph-owned storage that is overwritten by the next replay.

    Args:
        fn: Callable to capture.
        gradient_accumulation_steps: Forward-backward calls per optimizer step.
        sdc_num_steps: Initial optimizer steps checked by SDC, or -1 for all.
        sdc_num_replays: Additional calls for each SDC-checked optimizer step.
        num_warmup_steps: Number of eager optimizer steps before capture.
    """

    reason = _unsupported_reason()
    if reason is not None:
        logger.warning(
            f"XPU graph capture is unavailable ({reason}); using eager execution."
        )
        return fn

    # SDC checks only the first accumulation group of each checked step.
    num_checked_steps = (
        num_warmup_steps
        if sdc_num_steps == -1
        else min(num_warmup_steps, sdc_num_steps)
    )
    num_warmup_iterations = (
        num_warmup_steps * gradient_accumulation_steps
        + num_checked_steps * sdc_num_replays
    )

    # Every wrapper is registered to the manager in this module and persists
    # until xpugraph_teardown is called.
    graph_wrapper: XPUGraphWrapper | None = None
    input_spec: CUDAGraphInputSpec | None = None

    def run(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal graph_wrapper, input_spec

        if graph_wrapper is None:
            input_spec = CUDAGraphInputSpec((args, kwargs))

            def flat_fn(*flat_inputs: Any) -> torch.Tensor:
                assert input_spec is not None
                step_args, step_kwargs = input_spec.unflatten(flat_inputs)
                return fn(*step_args, **step_kwargs)

            flat_inputs = input_spec.flatten((args, kwargs))
            graph_wrapper = XPUGraphWrapper(
                flat_fn,
                flat_inputs,
                num_warmup_iterations=num_warmup_iterations,
            )
        else:
            assert input_spec is not None
            flat_inputs = input_spec.flatten((args, kwargs))

        return graph_wrapper(*flat_inputs)

    return run
