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

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
import torch.nn.functional as F

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    _GRAD_ACCUMULATOR_INPUT_META,
    _tensor_meta,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META

logger = logging.getLogger(__name__)
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
    if accumulator.meta.get(_GRAD_ACCUMULATOR_INPUT_META) is not True or not _sole_user(
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
    """Fuse supported annotated WGrad producers with their accumulator updates.

    The WGrad output must have the in-place addition as its sole user. Unsupported
    producers retain the explicit addition. MXFP8 fusion can change rounding.

    Example::

        Input::
            grad = aten.mm(lhs, rhs)
            updated = aten.add_(accumulator, grad)

        Output::
            updated = aten.addmm_(accumulator, lhs, rhs)
    """
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
            "Fused WGrad accumulation: %s",
            ", ".join(f"{name}={count}" for name, count in fusion_counts.items()),
        )
    return gm


__all__ = ["fuse_wgrad_accumulation_pass"]
