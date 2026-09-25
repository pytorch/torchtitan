# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse annotated graph-owned accumulation into WGrad producers.

The annotations identifies the parameter gradient.
The producer must directly feed the accumulation and have no other users.
Each supported producer registers in ``_WGRAD_FUSION_RULES``

Registered rules::

    aten.mm            -> aten.addmm_
    aten._scaled_mm    -> aten._scaled_addmm_
    aten._scaled_mm_v2 -> aten._scaled_addmm_

The MXFP8 rewrite changes the rounding boundary: the unfused graph rounds the
``_scaled_mm`` or ``_scaled_mm_v2`` result to BF16 before accumulating in BF16.
``_scaled_addmm_`` accumulates the GEMM result directly into that accumulator.
Not expected to be bitwise identical to the unfused graph.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
import torch.nn.functional as F

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import graph_outputs
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META

logger = logging.getLogger(__name__)
_STATIC_INPUT_META = "graph_runtime_static_input"
_MISSING_ARGUMENT = object()
_MXFP8_RECIPE = (F.ScalingType.BlockWise1x32.value,)
_MXFP8_SWIZZLE = (F.SwizzleType.SWIZZLE_32_4_4.value,)

_WGradFusion = Callable[[fx.Node, fx.Node, fx.Node], bool]
_WGRAD_FUSION_RULES: dict[Any, tuple[str, _WGradFusion]] = {}


def _register_wgrad_fusion_rule(
    name: str,
    producer_target: Any,
) -> Callable[[_WGradFusion], _WGradFusion]:
    """Register one annotated producer-to-accumulator lowering."""

    def register(lower: _WGradFusion) -> _WGradFusion:
        _WGRAD_FUSION_RULES[producer_target] = (name, lower)
        return lower

    return register


def _new_accumulator(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if value.layout != torch.strided:
        raise NotImplementedError(
            "Graph runtime gradient accumulation requires strided raw gradients, "
            f"got {value.layout}"
        )
    shape = tuple(int(dim) for dim in value.shape)
    stride = tuple(int(dim) for dim in value.stride())
    return torch.empty_strided(
        shape,
        stride,
        dtype=value.dtype,
        device=device,
    )


def _validate_accumulator(
    accumulator: torch.Tensor,
    value: torch.Tensor,
    *,
    index: int,
    device: torch.device,
) -> None:
    if (
        accumulator.shape != value.shape
        or accumulator.stride() != value.stride()
        or accumulator.dtype != value.dtype
        or accumulator.device != device
    ):
        raise ValueError(
            "Gradient accumulator does not match backward output at index " f"{index}"
        )


def insert_graph_gradient_accumulation(
    gm: fx.GraphModule,
    *,
    num_param_grads: int,
    device: torch.device,
    param_grad_output_start: int = 0,
    accumulators: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Make raw parameter gradients accumulate into explicit graph inputs."""
    outputs = graph_outputs(gm.graph)
    if param_grad_output_start < 0:
        raise ValueError(
            "Parameter gradient output start must be non-negative, got "
            f"{param_grad_output_start}"
        )
    param_grad_output_end = param_grad_output_start + num_param_grads
    if param_grad_output_end > len(outputs):
        raise ValueError(
            "Parameter gradient output range exceeds graph outputs: "
            f"[{param_grad_output_start}, {param_grad_output_end}) with "
            f"{len(outputs)} outputs"
        )

    grad_outputs = outputs[param_grad_output_start:param_grad_output_end]
    if accumulators is None:
        accumulator_values_list: list[Any] = []
        accumulator_by_grad: dict[fx.Node, torch.Tensor] = {}
        for index, output in enumerate(grad_outputs):
            if output is None:
                accumulator_values_list.append(None)
                continue
            if not isinstance(output, fx.Node):
                accumulator_values_list.append(output)
                continue
            value = output.meta.get("val")
            if value is None:
                raise ValueError(
                    "Graph runtime parameter gradient output has no metadata at "
                    f"index {index}: {output!r}"
                )
            if not isinstance(value, torch.Tensor):
                accumulator_values_list.append(value)
                continue
            accumulator = accumulator_by_grad.get(output)
            if accumulator is None:
                accumulator = _new_accumulator(value, device=device)
                accumulator_by_grad[output] = accumulator
            accumulator_values_list.append(accumulator)
        accumulator_values = tuple(accumulator_values_list)
    else:
        if len(accumulators) != num_param_grads:
            raise ValueError(
                "Gradient accumulator count does not match graph outputs: "
                f"{len(accumulators)} != {num_param_grads}"
            )
        accumulator_values = accumulators

    first_compute = next(node for node in gm.graph.nodes if node.op != "placeholder")
    accumulator_nodes: list[fx.Node | None] = []
    node_by_accumulator_id: dict[int, tuple[fx.Node, fx.Node]] = {}
    with gm.graph.inserting_before(first_compute):
        for index, (grad, accumulator) in enumerate(
            zip(grad_outputs, accumulator_values, strict=True)
        ):
            grad_value = _tensor_meta(grad) if isinstance(grad, fx.Node) else None
            if grad_value is None:
                accumulator_nodes.append(None)
                continue
            if not isinstance(accumulator, torch.Tensor):
                raise ValueError(
                    "Graph runtime tensor gradient has no matching tensor accumulator "
                    f"at index {index}"
                )
            _validate_accumulator(
                accumulator,
                grad_value,
                index=index,
                device=device,
            )
            existing = node_by_accumulator_id.get(id(accumulator))
            if existing is not None:
                previous_grad, node = existing
                if grad is not previous_grad:
                    raise ValueError(
                        "One graph runtime gradient accumulator cannot represent "
                        f"different graph outputs at index {index}"
                    )
                accumulator_nodes.append(node)
                continue
            node = gm.graph.placeholder(f"graph_runtime_grad_accumulator_{index}")
            node.meta = copy.copy(grad.meta)
            # The persistent accumulator and this microbatch's gradient are
            # separate runtime buffers, so their fake tensors must be distinct.
            node.meta["val"] = grad_value.new_empty_strided(
                grad_value.shape,
                grad_value.stride(),
                requires_grad=grad_value.requires_grad,
            )
            node.meta[_STATIC_INPUT_META] = True
            node_by_accumulator_id[id(accumulator)] = (grad, node)
            accumulator_nodes.append(node)

    accumulated_outputs: list[Any] = []
    sink_by_accumulator: dict[fx.Node, fx.Node] = {}
    output_node = gm.graph.find_nodes(op="output")[0]
    for grad, accumulator in zip(grad_outputs, accumulator_nodes, strict=True):
        if not isinstance(accumulator, fx.Node):
            accumulated_outputs.append(grad)
            continue
        assert isinstance(grad, fx.Node)
        if accumulator in sink_by_accumulator:
            accumulated_outputs.append(sink_by_accumulator[accumulator])
            continue
        with gm.graph.inserting_before(output_node):
            sink = gm.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, grad),
            )
        sink.meta = copy.copy(grad.meta)
        sink_by_accumulator[accumulator] = sink
        accumulated_outputs.append(sink)

    output_node.args = (
        tuple(
            [
                *outputs[:param_grad_output_start],
                *accumulated_outputs,
                *outputs[param_grad_output_end:],
            ]
        ),
    )
    gm.graph.lint()
    gm.recompile()
    return accumulator_values


def _tensor_meta(node: fx.Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _node_argument(
    node: fx.Node,
    name: str,
    position: int,
    default: Any,
) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if position < len(node.args):
        return node.args[position]
    return default


def _parameter_gradient_fqns(node: fx.Node) -> tuple[str, ...]:
    custom = node.meta.get("custom", {})
    return custom.get(PARAMETER_GRADIENT_FQNS_META) or custom.get(
        FSDP_PARAM_FQNS_META, ()
    )


def _sole_user(node: fx.Node, expected: fx.Node) -> bool:
    return len(node.users) == 1 and expected in node.users


def _compatible_bf16_tensors(
    accumulator: fx.Node,
    producer: fx.Node,
) -> bool:
    accumulator_value = _tensor_meta(accumulator)
    producer_value = _tensor_meta(producer)
    if accumulator_value is None or producer_value is None:
        return False
    return (
        accumulator_value.dtype == torch.bfloat16
        and producer_value.dtype == torch.bfloat16
        and accumulator_value.device == producer_value.device
        and accumulator_value.shape == producer_value.shape
        and accumulator_value.stride() == producer_value.stride()
        and accumulator_value.is_contiguous()
    )


def _annotated_wgrad_accumulation(
    sink: fx.Node,
) -> tuple[fx.Node, fx.Node] | None:
    if sink.target != torch.ops.aten.add_.Tensor or len(sink.args) < 2:
        return None
    if _node_argument(sink, "alpha", 2, 1) != 1:
        return None
    accumulator, producer = sink.args[:2]
    if not isinstance(accumulator, fx.Node) or not isinstance(producer, fx.Node):
        return None
    sink_fqns = _parameter_gradient_fqns(sink)
    if not sink_fqns or sink_fqns != _parameter_gradient_fqns(producer):
        return None
    if accumulator.op != "placeholder" or not _sole_user(accumulator, sink):
        return None
    if accumulator.meta.get(_STATIC_INPUT_META) is not True or not _sole_user(
        producer, sink
    ):
        return None
    if not _compatible_bf16_tensors(accumulator, producer):
        return None
    return accumulator, producer


def _replace_sink_with_producer(
    sink: fx.Node,
    producer: fx.Node,
) -> None:
    sink.replace_all_uses_with(producer)
    sink.graph.erase_node(sink)


@_register_wgrad_fusion_rule("BF16", torch.ops.aten.mm.default)
def _fuse_mm_sink(
    sink: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
) -> bool:
    producer_value = _tensor_meta(producer)
    if producer_value is None or producer_value.dim() != 2 or len(producer.args) != 2:
        return False

    producer.target = torch.ops.aten.addmm_.default
    producer.args = (accumulator, *producer.args)
    producer.meta["original_aten"] = torch.ops.aten.addmm_.default
    _replace_sink_with_producer(sink, producer)
    return True


def _scaled_addmm_target() -> Any | None:
    packet = getattr(torch.ops.aten, "_scaled_addmm_", None)
    return None if packet is None else getattr(packet, "default", None)


def _is_mxfp8_recipe(recipe: Any) -> bool:
    return isinstance(recipe, (list, tuple)) and tuple(recipe) == _MXFP8_RECIPE


@_register_wgrad_fusion_rule("MXFP8", torch.ops.aten._scaled_mm.default)
def _fuse_legacy_mxfp8_scaled_mm_sink(
    sink: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
) -> bool:
    scaled_addmm = _scaled_addmm_target()
    if scaled_addmm is None:
        return False

    operand_names = ("self", "mat2", "scale_a", "scale_b")
    operands = tuple(
        _node_argument(producer, name, position, _MISSING_ARGUMENT)
        for position, name in enumerate(operand_names)
    )
    if any(operand is _MISSING_ARGUMENT for operand in operands):
        return False

    bias = _node_argument(producer, "bias", 4, None)
    scale_result = _node_argument(producer, "scale_result", 5, None)
    out_dtype = _node_argument(producer, "out_dtype", 6, _MISSING_ARGUMENT)
    use_fast_accum = _node_argument(producer, "use_fast_accum", 7, False)
    scale_a, scale_b = operands[2:]
    scale_a_value = _tensor_meta(scale_a) if isinstance(scale_a, fx.Node) else None
    scale_b_value = _tensor_meta(scale_b) if isinstance(scale_b, fx.Node) else None
    producer_value = _tensor_meta(producer)
    if (
        bias is not None
        or scale_result is not None
        or out_dtype != torch.bfloat16
        or not isinstance(use_fast_accum, bool)
        or scale_a_value is None
        or scale_b_value is None
        or scale_a_value.dtype != torch.float8_e8m0fnu
        or scale_b_value.dtype != torch.float8_e8m0fnu
        or not scale_a_value.is_contiguous()
        or not scale_b_value.is_contiguous()
        or producer_value is None
        or producer_value.dim() != 2
        or producer_value.device.type != "cuda"
        or torch.version.hip is not None
    ):
        return False

    producer.target = scaled_addmm
    producer.args = (
        accumulator,
        *operands[:2],
        [scale_a],
        _MXFP8_RECIPE,
        _MXFP8_SWIZZLE,
        [scale_b],
        _MXFP8_RECIPE,
        _MXFP8_SWIZZLE,
        [],
    )
    producer.kwargs = {
        "beta": 1,
        "alpha": 1,
        "use_fast_accum": use_fast_accum,
    }
    producer.meta["original_aten"] = scaled_addmm
    _replace_sink_with_producer(sink, producer)
    return True


@_register_wgrad_fusion_rule("MXFP8", torch.ops.aten._scaled_mm_v2.default)
def _fuse_scaled_mm_v2_sink(
    sink: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
) -> bool:
    scaled_addmm = _scaled_addmm_target()
    if scaled_addmm is None:
        return False

    operand_names = (
        "self",
        "mat2",
        "scale_a",
        "recipe_a",
        "swizzle_a",
        "scale_b",
        "recipe_b",
        "swizzle_b",
    )
    operands = tuple(
        _node_argument(producer, name, position, _MISSING_ARGUMENT)
        for position, name in enumerate(operand_names)
    )
    if any(operand is _MISSING_ARGUMENT for operand in operands):
        return False

    bias = _node_argument(producer, "bias", 8, _MISSING_ARGUMENT)
    out_dtype = _node_argument(producer, "out_dtype", 9, _MISSING_ARGUMENT)
    contraction_dim = _node_argument(producer, "contraction_dim", 10, [])
    use_fast_accum = _node_argument(producer, "use_fast_accum", 11, False)
    producer_value = _tensor_meta(producer)
    if (
        bias is not None
        or out_dtype != torch.bfloat16
        or not _is_mxfp8_recipe(operands[3])
        or not _is_mxfp8_recipe(operands[6])
        or not isinstance(use_fast_accum, bool)
        or producer_value is None
        or producer_value.dim() != 2
        or producer_value.device.type != "cuda"
        or torch.version.hip is not None
    ):
        return False

    producer.target = scaled_addmm
    producer.args = (accumulator, *operands, contraction_dim)
    producer.kwargs = {
        "beta": 1,
        "alpha": 1,
        "use_fast_accum": use_fast_accum,
    }
    producer.meta["original_aten"] = scaled_addmm
    _replace_sink_with_producer(sink, producer)
    return True


def fuse_wgrad_accumulation_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Fuse annotated BF16 and MXFP8 WGrad accumulation when supported."""
    del example_inputs
    fusion_counts = {name: 0 for name, _lower in _WGRAD_FUSION_RULES.values()}
    for sink in tuple(gm.graph.nodes):
        matched = _annotated_wgrad_accumulation(sink)
        if matched is None:
            continue
        accumulator, producer = matched
        rule = _WGRAD_FUSION_RULES.get(producer.target)
        if rule is None:
            continue
        name, lower = rule
        if lower(sink, accumulator, producer):
            fusion_counts[name] += 1

    if any(fusion_counts.values()):
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "Fused graph runtime WGrad accumulation: %s",
            ", ".join(f"{name}={count}" for name, count in fusion_counts.items()),
        )
    return gm


__all__ = [
    "fuse_wgrad_accumulation_pass",
    "insert_graph_gradient_accumulation",
]
