# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.fx as fx
from torch._library.opaque_object import is_opaque_value
from torch.fx.passes.split_module import split_module

from torchtitan.tools.logging import logger


def _get_xpu_graph_cls():
    return getattr(torch.xpu, "XPUGraph", None)


def _xpu_graph(graph, *, pool=None, stream=None):
    if pool is not None and stream is not None:
        return torch.xpu.graph(graph, pool=pool, stream=stream)
    if pool is not None:
        return torch.xpu.graph(graph, pool=pool)
    if stream is not None:
        return torch.xpu.graph(graph, stream=stream)
    return torch.xpu.graph(graph)


def _xpu_graph_pool_handle():
    if hasattr(torch.xpu, "graph_pool_handle"):
        return torch.xpu.graph_pool_handle()
    return None


class _XPUGraphManager:
    def __init__(self) -> None:
        self.pool = None
        self.stream = None
        self.dummy_graph = None
        self.wrappers: list[XPUGraphWrapper] = []

    def initialize(self) -> None:
        if self.stream is None:
            self.stream = torch.xpu.Stream()

        if self.pool is None:
            pool = _xpu_graph_pool_handle()
            if pool is not None:
                self.pool = pool

    def register(self, wrapper: "XPUGraphWrapper") -> None:
        self.wrappers.append(wrapper)

    def teardown(self) -> None:
        for wrapper in self.wrappers:
            wrapper.reset()

        self.wrappers.clear()
        self.pool = None
        self.stream = None
        self.dummy_graph = None


_xg_manager = _XPUGraphManager()


def xpugraph_teardown() -> None:
    """Synchronize XPU work and release manager-owned XPUGraph state.

    GraphTrainer imports and calls this function during shutdown. It is safe
    to call when XPU is unavailable or when no XPUGraphs were captured.
    """
    if torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
        except Exception as exc:
            logger.warning(
                "Failed to synchronize XPU during XPUGraph teardown: %s",
                exc,
            )

    _xg_manager.teardown()
    logger.info("XPUGraph teardown completed.")


def _is_xpu_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "xpu"


def _is_tensor_input(value: Any) -> bool:
    if not isinstance(value, torch.Tensor):
        return False

    try:
        if is_opaque_value(value):
            return False
    except Exception:
        pass

    return value.device.type == "xpu"


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    return getattr(target, "__name__", str(target))


def _target_qualname(target: Any) -> str:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", str(target))
    return f"{module}.{name}" if module else name


def _node_target_text(node: fx.Node) -> str:
    return _target_qualname(node.target)


def _is_c10d_functional_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False

    target_text = _node_target_text(node)

    return (
        "_c10d_functional" in target_text
        or "reduce_scatter_tensor" in target_text
        or "all_gather_into_tensor" in target_text
        or "all_reduce" in target_text
        or "wait_tensor" in target_text
    )


def contains_c10d_functional(gm: fx.GraphModule) -> bool:
    if "_c10d_functional" in gm.code:
        return True

    for node in gm.graph.nodes:
        if _is_c10d_functional_node(node):
            return True

    return False


_XPUGRAPH_UNSUPPORTED_OPS = {
    torch.ops.aten.index_put_.default,
    torch.ops.aten.embedding_dense_backward.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
}


_FLEX_ATTENTION_OPS = {
    "flex_attention",
    "flex_attention_backward",
}


def _is_flex_attention_node(node: fx.Node) -> bool:
    target_text = _node_target_text(node)
    return any(op_name in target_text for op_name in _FLEX_ATTENTION_OPS)


def _is_backward_node(node: fx.Node) -> bool:
    target_text = _node_target_text(node)
    name = node.name

    return (
        "backward" in target_text
        or "backward" in name
        or "embedding_dense_backward" in target_text
        or target_text.endswith("_backward.default")
    )


def _is_backward_graph(gm: fx.GraphModule) -> bool:
    if "backward" in gm.code or "embedding_dense_backward" in gm.code:
        return True

    for node in gm.graph.nodes:
        if _is_backward_node(node):
            return True

    return False


def _is_cpu_xpu_copy_node(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False

    target_text = _node_target_text(node)

    if "_to_copy" not in target_text and "to.device" not in target_text:
        return False

    device = node.kwargs.get("device", None)
    if device is None:
        return False

    device_text = str(device)
    return "cpu" in device_text or "xpu" in device_text


def is_xpugraph_compatible(gm: fx.GraphModule) -> bool:
    if contains_c10d_functional(gm):
        return False

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        if _is_c10d_functional_node(node):
            return False

        if _is_backward_node(node):
            return False

        if _is_flex_attention_node(node):
            return False

        if _is_cpu_xpu_copy_node(node):
            return False

        if node.target in _XPUGRAPH_UNSUPPORTED_OPS:
            return False

        target_text = _node_target_text(node)

        if "embedding_dense_backward" in target_text:
            return False

        if "_scaled_dot_product_flash_attention" in target_text:
            return False

        if "index_put_" in target_text:
            return False

    return True


def has_xpu_kernel_candidate(gm: fx.GraphModule) -> bool:
    compute_keywords = (
        "aten.mm",
        "aten.addmm",
        "aten.bmm",
        "aten.matmul",
        "aten.linear",
    )

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        target_text = _node_target_text(node)

        if any(keyword in target_text for keyword in compute_keywords):
            return True

    return False


def get_static_input_indices(
    gm: fx.GraphModule,
    is_forward: bool,
) -> list[int]:
    try:
        from torch._guards import TracingContext

        tracing_context = TracingContext.try_get()
        if tracing_context is None:
            return []

        fw_metadata = getattr(tracing_context, "fw_metadata", None)
        if fw_metadata is None:
            return []

        static_input_indices = getattr(
            fw_metadata,
            "static_input_indices",
            None,
        )
        if static_input_indices is None:
            return []

        if is_forward:
            return list(static_input_indices)

        count_tangents = getattr(fw_metadata, "count_tangents", 0)
        return [
            idx
            for idx in static_input_indices
            if idx < count_tangents
        ]

    except Exception:
        return []


class XPUGraphWrapper:
    def __init__(
        self,
        runnable: Callable[..., Any],
        example_inputs: Sequence[Any] | None = None,
        static_input_indices: Sequence[int] | None = None,
        should_check_address: bool = True,
        tensor_input_indices: Sequence[int] | None = None,
    ) -> None:
        self._runnable = runnable
        self._example_inputs = (
            tuple(example_inputs)
            if example_inputs is not None
            else None
        )
        self._static_input_indices = set(static_input_indices or [])
        self._should_check_address = should_check_address

        self._tensor_input_indices = (
            list(tensor_input_indices)
            if tensor_input_indices is not None
            else None
        )

        self._input_indices_to_copy: list[int] = []
        self._static_inputs: list[Any] | None = None

        self._xpugraph = None
        self._output = None

        self._warmed_up = False
        self._captured = False
        self._disabled = False

        _xg_manager.register(self)

    def reset(self) -> None:
        graph = self._xpugraph

        if graph is not None:
            reset = getattr(graph, "reset", None)

            if callable(reset):
                try:
                    reset()
                except Exception as exc:
                    logger.warning(
                        "Failed to reset XPUGraph during teardown: %s",
                        exc,
                    )

        self._xpugraph = None
        self._output = None
        self._warmed_up = False
        self._captured = False
        self._static_inputs = None
        self._input_indices_to_copy.clear()

    def _disable(self, reason: str) -> None:
        self._disabled = True
        logger.warning(
            "Disabling XPUGraph wrapper and falling back to eager: %s",
            reason,
        )

    def _fallback_for_capture_error(
        self,
        exc: RuntimeError,
    ) -> bool:
        msg = str(exc)

        fallback_messages = (
            "wait method cannot be used for an event associated "
            "with a command graph",
            "XPU Graph is empty",
            "graph is empty",
        )

        return any(message in msg for message in fallback_messages)

    def _infer_tensor_input_indices(
        self,
        args: Sequence[Any],
    ) -> list[int]:
        if self._tensor_input_indices is not None:
            return list(self._tensor_input_indices)

        return [
            idx
            for idx, value in enumerate(args)
            if _is_tensor_input(value)
        ]

    def _initialize_static_inputs(
        self,
        args: Sequence[Any],
    ) -> None:
        self._tensor_input_indices = self._infer_tensor_input_indices(args)

        static_inputs = list(args)

        self._input_indices_to_copy = [
            idx
            for idx in self._tensor_input_indices
            if idx not in self._static_input_indices
        ]

        for idx in self._input_indices_to_copy:
            static_inputs[idx] = args[idx].clone()

        self._static_inputs = static_inputs

    def _copy_non_static_inputs(
        self,
        args: Sequence[Any],
    ) -> None:
        assert self._static_inputs is not None

        for idx in self._input_indices_to_copy:
            self._static_inputs[idx].copy_(args[idx])

    def _check_static_input_addresses(
        self,
        args: Sequence[Any],
    ) -> bool:
        if not self._should_check_address:
            return True

        if self._static_inputs is None:
            return True

        for idx in self._static_input_indices:
            if idx >= len(args):
                continue

            arg = args[idx]
            static_arg = self._static_inputs[idx]

            if not isinstance(arg, torch.Tensor):
                continue

            if not isinstance(static_arg, torch.Tensor):
                continue

            if arg.data_ptr() != static_arg.data_ptr():
                warnings.warn(
                    "XPUGraph static input address changed. "
                    "Falling back to eager for this wrapper.",
                    stacklevel=2,
                )
                return False

        return True

    def _warmup(self, args: Sequence[Any]) -> Any:
        assert self._static_inputs is not None
        assert _xg_manager.stream is not None

        current_stream = torch.xpu.current_stream()

        _xg_manager.stream.wait_stream(current_stream)

        with torch.xpu.stream(_xg_manager.stream):
            self._output = self._runnable(*self._static_inputs)

        current_stream.wait_stream(_xg_manager.stream)

        self._warmed_up = True
        return self._output

    def _capture(self, args: Sequence[Any]) -> Any:
        assert self._static_inputs is not None
        assert _xg_manager.stream is not None

        graph_cls = _get_xpu_graph_cls()

        if graph_cls is None:
            self._disable("torch.xpu.XPUGraph is not available.")
            return self._runnable(*args)

        self._copy_non_static_inputs(args)

        self._xpugraph = graph_cls()

        current_stream = torch.xpu.current_stream()

        try:
            _xg_manager.stream.wait_stream(current_stream)

            with _xpu_graph(
                self._xpugraph,
                pool=_xg_manager.pool,
                stream=_xg_manager.stream,
            ):
                self._output = self._runnable(*self._static_inputs)

            current_stream.wait_stream(_xg_manager.stream)

            self._captured = True
            return self._output

        except RuntimeError as exc:
            if self._fallback_for_capture_error(exc):
                self._disable(str(exc))
                return self._runnable(*args)

            raise

    def _replay(self, args: Sequence[Any]) -> Any:
        assert self._static_inputs is not None
        assert self._xpugraph is not None
        assert _xg_manager.stream is not None

        if not self._check_static_input_addresses(args):
            self._disable("static input address changed")
            return self._runnable(*args)

        self._copy_non_static_inputs(args)

        current_stream = torch.xpu.current_stream()

        _xg_manager.stream.wait_stream(current_stream)
        self._xpugraph.replay()
        current_stream.wait_stream(_xg_manager.stream)

        return self._output

    def __call__(self, *args: Any) -> Any:
        if self._disabled:
            return self._runnable(*args)

        if not torch.xpu.is_available():
            return self._runnable(*args)

        if self._static_inputs is None:
            init_args = (
                self._example_inputs
                if self._example_inputs is not None
                else args
            )
            self._initialize_static_inputs(init_args)

        _xg_manager.initialize()

        if self._captured:
            return self._replay(args)

        if not self._warmed_up:
            return self._warmup(args)

        return self._capture(args)


def regional_xpugraph_pass(
    gm: fx.GraphModule,
    is_forward: bool,
) -> fx.GraphModule:
    region_idx = 0

    def split_callback(node: fx.Node) -> str:
        nonlocal region_idx

        if node.op in ("placeholder", "output"):
            return "root"

        if _is_c10d_functional_node(node):
            name = f"eager_collective_{region_idx}"
            region_idx += 1
            return name

        return f"xpu_region_{region_idx}"

    split_gm = split_module(gm, gm, split_callback)

    applied_regions = 0
    candidate_regions = 0
    skipped_no_kernel_regions = 0
    skipped_backward_regions = 0
    skipped_incompatible_regions = 0

    for name, submodule in split_gm.named_modules():
        if not isinstance(submodule, fx.GraphModule):
            continue

        if "xpu_region" not in name:
            continue

        candidate_regions += 1

        if _is_backward_graph(submodule):
            logger.info(
                "Skipping regional xpugraph submodule %s "
                "because it is backward.",
                name,
            )
            skipped_backward_regions += 1
            continue

        if not has_xpu_kernel_candidate(submodule):
            logger.info(
                "Skipping regional xpugraph submodule %s because it "
                "has no large XPU kernel candidate.",
                name,
            )
            skipped_no_kernel_regions += 1
            continue

        if not is_xpugraph_compatible(submodule):
            logger.info(
                "Skipping regional xpugraph submodule %s because "
                "it is not xpugraph compatible.",
                name,
            )
            skipped_incompatible_regions += 1
            continue

        submodule.forward = XPUGraphWrapper(
            submodule.forward,
            example_inputs=None,
            static_input_indices=[],
            tensor_input_indices=None,
        )

        applied_regions += 1
        logger.info(
            "Applied xpugraph to regional submodule %s.",
            name,
        )

    logger.info(
        "Applied xpugraph pass to %s regional compute partition(s). "
        "candidate_regions=%s, skipped_no_kernel_regions=%s, "
        "skipped_backward_regions=%s, "
        "skipped_incompatible_regions=%s",
        applied_regions,
        candidate_regions,
        skipped_no_kernel_regions,
        skipped_backward_regions,
        skipped_incompatible_regions,
    )

    return split_gm


def xpugraph_pass(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any] | None = None,
    *,
    is_forward: bool = True,
) -> fx.GraphModule:
    """Apply full-graph or regional XPUGraph capture.

    GraphTrainer supplies ``example_inputs`` to graph passes. Those inputs can
    contain fake tensors, so this pass intentionally does not retain them for
    XPUGraph capture. Static buffers are initialized during the first real
    runtime invocation instead.

    Args:
        gm: FX graph module to transform.
        example_inputs: GraphTrainer pass inputs. These are intentionally
            ignored by the XPUGraph runtime wrapper.
        is_forward: Whether this graph represents a forward graph.
    """
    del example_inputs

    if not isinstance(gm, fx.GraphModule):
        return gm

    if not torch.xpu.is_available():
        logger.warning(
            "Skipping XPUGraph pass because XPU is not available."
        )
        return gm

    if not is_forward:
        logger.info(
            "Skipping XPUGraph pass because this is not a forward graph."
        )
        return gm

    if _is_backward_graph(gm):
        logger.info(
            "Skipping XPUGraph pass because graph appears to be backward."
        )
        return gm

    if contains_c10d_functional(gm):
        logger.info(
            "Graph contains c10d functional collectives; "
            "applying regional xpugraph."
        )
        return regional_xpugraph_pass(
            gm,
            is_forward=is_forward,
        )

    if not has_xpu_kernel_candidate(gm):
        logger.info(
            "Skipping XPUGraph pass because graph has no "
            "large XPU kernel candidate."
        )
        return gm

    if not is_xpugraph_compatible(gm):
        logger.info(
            "Skipping XPUGraph pass because graph is not "
            "XPUGraph compatible."
        )
        return gm

    static_input_indices = get_static_input_indices(
        gm,
        is_forward=is_forward,
    )

    gm.forward = XPUGraphWrapper(
        gm.forward,
        example_inputs=None,
        static_input_indices=static_input_indices,
        tensor_input_indices=None,
    )

    logger.info("Applied XPUGraph pass to full graph.")
    return gm
