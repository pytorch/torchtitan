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
    dist_moe.block_scaled_backward -> dist_moe.block_scaled_backward_accumulate
    dist_moe.bf16_backward         -> dist_moe.bf16_backward_accumulate

The MXFP8 rewrite changes the rounding boundary: the unfused graph rounds the
``_scaled_mm`` or ``_scaled_mm_v2`` result to BF16 before accumulating in BF16.
``_scaled_addmm_`` accumulates the GEMM result directly into that accumulator.
Not expected to be bitwise identical to the unfused graph.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
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
_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.alias.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
)
_DIST_MOE_ACCUMULATION_TARGETS = {
    "dist_moe.block_scaled_backward.default": "block_scaled_backward_accumulate",
    "dist_moe.bf16_backward.default": "bf16_backward_accumulate",
}


@dataclass(frozen=True, slots=True)
class _DistMoeWgradSink:
    sink: fx.Node
    accumulator: fx.Node
    boundary: fx.Node
    getitem: fx.Node
    view_chain: tuple[fx.Node, ...]


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


def _is_storage_alias(node: fx.Node) -> bool:
    if node.target in _VIEW_TARGETS:
        return True
    if node.target != torch.ops.aten.reshape.default:
        return False
    if not node.args or not isinstance(node.args[0], fx.Node):
        return False
    source = _tensor_meta(node.args[0])
    result = _tensor_meta(node)
    return (
        source is not None
        and result is not None
        and source.is_contiguous()
        and result.is_contiguous()
        and source.dtype == result.dtype
        and source.device == result.device
        and source.numel() == result.numel()
    )


def _source_through_views(
    boundary: fx.Node,
    sink: fx.Node,
) -> tuple[fx.Node, tuple[fx.Node, ...]] | None:
    chain: list[fx.Node] = []
    current = boundary
    expected_user = sink
    while _is_storage_alias(current):
        if not _sole_user(current, expected_user):
            return None
        if not current.args or not isinstance(current.args[0], fx.Node):
            return None
        chain.append(current)
        expected_user = current
        current = current.args[0]
    if not _sole_user(current, expected_user):
        return None
    return current, tuple(chain)


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


def _eligible_dist_moe_sink(sink: fx.Node) -> _DistMoeWgradSink | None:
    if sink.target != torch.ops.aten.add_.Tensor or len(sink.args) < 2:
        return None
    if _node_argument(sink, "alpha", 2, 1) != 1:
        return None
    accumulator, boundary = sink.args[:2]
    if not isinstance(accumulator, fx.Node) or not isinstance(boundary, fx.Node):
        return None
    if not _parameter_gradient_fqns(sink):
        return None
    if (
        accumulator.op != "placeholder"
        or accumulator.meta.get(_GRAD_ACCUMULATOR_INPUT_META) is not True
        or not _sole_user(accumulator, sink)
    ):
        return None
    source = _source_through_views(boundary, sink)
    if source is None:
        return None
    getitem, view_chain = source
    if (
        getitem.target is not operator.getitem
        or len(getitem.args) < 2
        or getitem.args[1] not in (2, 3)
        or not isinstance(getitem.args[0], fx.Node)
    ):
        return None
    backward = getitem.args[0]
    if str(backward.target) not in _DIST_MOE_ACCUMULATION_TARGETS:
        return None
    if not _compatible_bf16_tensors(accumulator, boundary):
        return None
    return _DistMoeWgradSink(
        sink=sink,
        accumulator=accumulator,
        boundary=boundary,
        getitem=getitem,
        view_chain=view_chain,
    )


def _dist_moe_accumulation_target(backward: fx.Node) -> Any | None:
    op_name = _DIST_MOE_ACCUMULATION_TARGETS.get(str(backward.target))
    if op_name is None:
        return None
    try:
        return getattr(torch.ops.dist_moe, op_name).default
    except AttributeError:
        return None


def _has_expected_dist_moe_getitems(
    backward: fx.Node,
    matches: dict[int, _DistMoeWgradSink],
) -> bool:
    for user in backward.users:
        if user.target is not operator.getitem or len(user.args) < 2:
            return False
        index = user.args[1]
        if index in (2, 3):
            match = matches.get(index)
            if match is None or user is not match.getitem:
                return False
    return True


def _erase_dist_moe_sink(gm: fx.GraphModule, match: _DistMoeWgradSink) -> None:
    match.sink.replace_all_uses_with(match.accumulator)
    gm.graph.erase_node(match.sink)
    for view in match.view_chain:
        if not view.users:
            gm.graph.erase_node(view)
    gm.graph.erase_node(match.getitem)


def _fuse_dist_moe_backward(
    gm: fx.GraphModule,
    backward: fx.Node,
    matches: dict[int, _DistMoeWgradSink],
) -> bool:
    if set(matches) != {2, 3} or not _has_expected_dist_moe_getitems(backward, matches):
        return False
    target = _dist_moe_accumulation_target(backward)
    if target is None:
        return False
    values = backward.meta.get("val")
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        return False
    backward.target = target
    backward.args = (
        matches[2].accumulator,
        matches[3].accumulator,
        *backward.args,
    )
    backward.meta["val"] = tuple(values[:2])
    backward.meta["original_aten"] = target
    backward.meta["graph_runtime_fused_wgrad_accumulation"] = True
    _erase_dist_moe_sink(gm, matches[2])
    _erase_dist_moe_sink(gm, matches[3])
    return True


def _fuse_dist_moe_sinks(gm: fx.GraphModule) -> int:
    matches_by_backward: dict[fx.Node, dict[int, _DistMoeWgradSink]] = {}
    ambiguous: set[fx.Node] = set()
    for sink in tuple(gm.graph.nodes):
        match = _eligible_dist_moe_sink(sink)
        if match is None:
            continue
        backward = match.getitem.args[0]
        index = match.getitem.args[1]
        assert isinstance(backward, fx.Node) and isinstance(index, int)
        matches = matches_by_backward.setdefault(backward, {})
        if index in matches:
            ambiguous.add(backward)
        else:
            matches[index] = match
    return sum(
        _fuse_dist_moe_backward(gm, backward, matches)
        for backward, matches in matches_by_backward.items()
        if backward not in ambiguous
    )


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
    fusion_counts["DistMoE"] = _fuse_dist_moe_sinks(gm)
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
