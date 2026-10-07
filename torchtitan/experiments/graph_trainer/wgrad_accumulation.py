# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse annotated graph-owned accumulation into WGrad producers.

The annotations identifies the parameter gradient.
The producer may feed the accumulation through storage-only views.
Each supported producer registers in ``_WGRAD_FUSION_RULES``

Registered rules::

    aten.mm            -> aten.addmm_
    aten._scaled_mm    -> aten._scaled_addmm_
    aten._scaled_mm_v2 -> aten._scaled_addmm_
    dist_moe.block_scaled_backward -> dist_moe.block_scaled_backward_accumulate_
    dist_moe.bf16_backward         -> dist_moe.bf16_backward_accumulate_

The MXFP8 rewrite changes the rounding boundary: the unfused graph rounds the
``_scaled_mm`` or ``_scaled_mm_v2`` result to BF16 before accumulating in BF16.
``_scaled_addmm_`` accumulates the GEMM result directly into that accumulator.
Not expected to be bitwise identical to the unfused graph.
"""

from __future__ import annotations

import copy
import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.nn.functional as F

from torchtitan.experiments.graph_trainer.common_utils import (
    node_argument as _node_argument,
    node_tensor_meta as _tensor_meta,
    parameter_gradient_fqns as _parameter_gradient_fqns,
    same_tensor_metadata,
    sole_user,
    walk_up_unary_chain,
)
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    _GRAD_ACCUMULATOR_INPUT_META,
)

logger = logging.getLogger(__name__)
_MISSING_ARGUMENT = object()
_MXFP8_RECIPE = (F.ScalingType.BlockWise1x32.value,)
_MXFP8_SWIZZLE = (F.SwizzleType.SWIZZLE_32_4_4.value,)
_VIEW_TARGETS = frozenset(
    {
        torch.ops.aten.alias.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    }
)
_DIST_MOE_ACCUMULATION_TARGETS = {
    "dist_moe.block_scaled_backward.default": (
        "block_scaled_backward_accumulate_",
        13,
    ),
    "dist_moe.bf16_backward.default": ("bf16_backward_accumulate_", 12),
}

_WGradFusion = Callable[[fx.Node, fx.Node, fx.Node, bool], bool]
_WGradFusionCheck = Callable[[fx.Node, fx.Node], bool]


@dataclass(frozen=True, slots=True)
class _WGradFusionRule:
    name: str
    lower: _WGradFusion
    can_fuse: _WGradFusionCheck
    native_parameter_grad_accumulation: bool = False


_WGRAD_FUSION_RULES: dict[Any, _WGradFusionRule] = {}


@dataclass(frozen=True, slots=True)
class _DistMoeWgradSink:
    sink: fx.Node
    accumulator: fx.Node
    getitem: fx.Node
    postprocess_chain: tuple[fx.Node, ...]


@dataclass(frozen=True, slots=True)
class _NativeParameterGradLeaf:
    boundary: fx.Node
    producer: fx.Node
    rule: _WGradFusionRule


@dataclass(frozen=True, slots=True)
class _NativeParameterGradAdd:
    sink: fx.Node
    accumulator: fx.Node
    leaves: tuple[_NativeParameterGradLeaf, ...]
    add_nodes: tuple[fx.Node, ...]


def _register_wgrad_fusion_rule(
    name: str,
    producer_target: Any,
    *,
    can_fuse: _WGradFusionCheck,
    native_parameter_grad_accumulation: bool = False,
) -> Callable[[_WGradFusion], _WGradFusion]:
    """Register one annotated producer-to-accumulator lowering."""

    def register(lower: _WGradFusion) -> _WGradFusion:
        _WGRAD_FUSION_RULES[producer_target] = _WGradFusionRule(
            name=name,
            lower=lower,
            can_fuse=can_fuse,
            native_parameter_grad_accumulation=(native_parameter_grad_accumulation),
        )
        return lower

    return register


def _compatible_accumulation_tensors(
    accumulator: fx.Node,
    gradient: fx.Node,
) -> bool:
    accumulator_value = _tensor_meta(accumulator)
    return (
        accumulator_value is not None
        and same_tensor_metadata(accumulator, gradient)
        and accumulator_value.is_contiguous()
    )


def _is_alias(node: fx.Node) -> bool:
    if node.target in _VIEW_TARGETS:
        return True
    if node.target is not torch.ops.aten.reshape.default:
        return False
    if not node.args or not isinstance(node.args[0], fx.Node):
        return False
    source_value = _tensor_meta(node.args[0])
    return source_value is not None and source_value.is_contiguous()


def _producer_through_views(
    boundary: fx.Node,
    grad_accum_inplace_add: fx.Node,
) -> fx.Node | None:
    chain = walk_up_unary_chain(boundary, grad_accum_inplace_add, _is_alias)
    return None if chain is None else chain[0]


def _annotated_wgrad_sink_inputs(sink: fx.Node) -> tuple[fx.Node, fx.Node] | None:
    if sink.target != torch.ops.aten.add_.Tensor or len(sink.args) < 2:
        return None
    if _node_argument(sink, "alpha", 2, 1) != 1:
        return None
    accumulator, boundary = sink.args[:2]
    if not isinstance(accumulator, fx.Node) or not isinstance(boundary, fx.Node):
        return None
    if (
        not _parameter_gradient_fqns(sink)
        or accumulator.op != "placeholder"
        or accumulator.meta.get(_GRAD_ACCUMULATOR_INPUT_META) is not True
        or not sole_user(accumulator, sink)
        or not _compatible_accumulation_tensors(accumulator, boundary)
    ):
        return None
    return accumulator, boundary


def _annotated_wgrad_accumulation(
    grad_accum_inplace_add: fx.Node,
) -> tuple[fx.Node, fx.Node] | None:
    inputs = _annotated_wgrad_sink_inputs(grad_accum_inplace_add)
    if inputs is None:
        return None
    accumulator, boundary = inputs
    if _parameter_gradient_fqns(grad_accum_inplace_add) != (
        _parameter_gradient_fqns(boundary)
    ):
        return None
    producer = _producer_through_views(boundary, grad_accum_inplace_add)
    if producer is None:
        return None
    return accumulator, producer


def _replace_grad_accum_inplace_add_with_gradient(
    grad_accum_inplace_add: fx.Node,
) -> None:
    boundary = grad_accum_inplace_add.args[1]
    assert isinstance(boundary, fx.Node)
    grad_accum_inplace_add.replace_all_uses_with(boundary)
    grad_accum_inplace_add.graph.erase_node(grad_accum_inplace_add)


def _view_accumulator_like(
    accumulator: fx.Node,
    reference: fx.Node,
    *,
    before: fx.Node,
) -> fx.Node | None:
    accumulator_value = _tensor_meta(accumulator)
    reference_value = _tensor_meta(reference)
    if (
        accumulator_value is None
        or reference_value is None
        or reference_value.device != accumulator_value.device
        or not reference_value.is_contiguous()
        or reference_value.numel() != accumulator_value.numel()
    ):
        return None
    with reference.graph.inserting_before(before):
        accumulator_view = reference.graph.call_function(
            torch.ops.aten.view.default,
            args=(accumulator, list(reference_value.shape)),
        )
    accumulator_view.meta = copy.copy(reference.meta)
    accumulator_view.meta["val"] = accumulator_value.view(reference_value.shape)
    return accumulator_view


def _accumulator_for_producer(
    grad_accum_inplace_add: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
) -> fx.Node | None:
    boundary = grad_accum_inplace_add.args[1]
    assert isinstance(boundary, fx.Node)
    if boundary is producer or _compatible_accumulation_tensors(accumulator, producer):
        return accumulator
    return _view_accumulator_like(accumulator, producer, before=producer)


def _can_accumulate_into_producer(
    accumulator: fx.Node,
    producer: fx.Node,
) -> bool:
    accumulator_value = _tensor_meta(accumulator)
    producer_value = _tensor_meta(producer)
    return (
        accumulator_value is not None
        and producer_value is not None
        and producer_value.device == accumulator_value.device
        and producer_value.is_contiguous()
        and producer_value.numel() == accumulator_value.numel()
    )


def _can_fuse_mm_wgrad(
    accumulator: fx.Node,
    producer: fx.Node,
) -> bool:
    producer_value = _tensor_meta(producer)
    return (
        producer_value is not None
        and producer_value.dim() == 2
        and len(producer.args) == 2
        and _can_accumulate_into_producer(accumulator, producer)
    )


@_register_wgrad_fusion_rule(
    "MM",
    torch.ops.aten.mm.default,
    can_fuse=_can_fuse_mm_wgrad,
)
def _fuse_mm_grad_accum_inplace_add(
    grad_accum_inplace_add: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
    replace_accumulation: bool = True,
) -> bool:
    """Fuse one MM WGrad into its accumulator.

    Input::
        wgrad = lhs @ rhs
        output = accumulator.add_(wgrad)

    Output::
        output = accumulator.addmm_(lhs, rhs)
    """
    if not _can_fuse_mm_wgrad(accumulator, producer):
        return False

    producer_accumulator = _accumulator_for_producer(
        grad_accum_inplace_add, accumulator, producer
    )
    if producer_accumulator is None:
        return False
    producer.target = torch.ops.aten.addmm_.default
    producer.args = (producer_accumulator, *producer.args)
    producer.meta["original_aten"] = torch.ops.aten.addmm_.default
    if replace_accumulation:
        _replace_grad_accum_inplace_add_with_gradient(grad_accum_inplace_add)
    return True


def _scaled_addmm_target() -> Any | None:
    packet = getattr(torch.ops.aten, "_scaled_addmm_", None)
    return None if packet is None else getattr(packet, "default", None)


def _is_mxfp8_recipe(recipe: Any) -> bool:
    return isinstance(recipe, (list, tuple)) and tuple(recipe) == _MXFP8_RECIPE


def _legacy_mxfp8_wgrad_configuration(
    producer: fx.Node,
) -> tuple[Any, tuple[Any, ...], bool] | None:
    scaled_addmm = _scaled_addmm_target()
    if scaled_addmm is None:
        return None

    operand_names = ("self", "mat2", "scale_a", "scale_b")
    operands = tuple(
        _node_argument(producer, name, position, _MISSING_ARGUMENT)
        for position, name in enumerate(operand_names)
    )
    if any(operand is _MISSING_ARGUMENT for operand in operands):
        return None

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
        return None
    return scaled_addmm, operands, use_fast_accum


@_register_wgrad_fusion_rule(
    "MXFP8",
    torch.ops.aten._scaled_mm.default,
    can_fuse=lambda accumulator, producer: (
        _legacy_mxfp8_wgrad_configuration(producer) is not None
        and _can_accumulate_into_producer(accumulator, producer)
    ),
    native_parameter_grad_accumulation=True,
)
def _fuse_legacy_mxfp8_scaled_mm_grad_accum_inplace_add(
    grad_accum_inplace_add: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
    replace_accumulation: bool = True,
) -> bool:
    """Fuse one legacy MXFP8 WGrad into its accumulator.

    Input::
        wgrad = aten._scaled_mm(lhs, rhs, ...)
        output = accumulator.add_(wgrad)

    Output::
        output = aten._scaled_addmm_(accumulator, lhs, rhs, ...)
    """
    configuration = _legacy_mxfp8_wgrad_configuration(producer)
    if configuration is None or not _can_accumulate_into_producer(
        accumulator, producer
    ):
        return False
    scaled_addmm, operands, use_fast_accum = configuration
    scale_a, scale_b = operands[2:]

    producer_accumulator = _accumulator_for_producer(
        grad_accum_inplace_add, accumulator, producer
    )
    if producer_accumulator is None:
        return False
    producer.target = scaled_addmm
    producer.args = (
        producer_accumulator,
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
    if replace_accumulation:
        _replace_grad_accum_inplace_add_with_gradient(grad_accum_inplace_add)
    return True


def _mxfp8_v2_wgrad_configuration(
    producer: fx.Node,
) -> tuple[Any, tuple[Any, ...], Any, bool] | None:
    scaled_addmm = _scaled_addmm_target()
    if scaled_addmm is None:
        return None

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
        return None

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
        return None
    return scaled_addmm, operands, contraction_dim, use_fast_accum


@_register_wgrad_fusion_rule(
    "MXFP8",
    torch.ops.aten._scaled_mm_v2.default,
    can_fuse=lambda accumulator, producer: (
        _mxfp8_v2_wgrad_configuration(producer) is not None
        and _can_accumulate_into_producer(accumulator, producer)
    ),
    native_parameter_grad_accumulation=True,
)
def _fuse_scaled_mm_v2_grad_accum_inplace_add(
    grad_accum_inplace_add: fx.Node,
    accumulator: fx.Node,
    producer: fx.Node,
    replace_accumulation: bool = True,
) -> bool:
    """Fuse one MXFP8 WGrad into its accumulator.

    Input::
        wgrad = aten._scaled_mm_v2(lhs, rhs, ...)
        output = accumulator.add_(wgrad)

    Output::
        output = aten._scaled_addmm_(accumulator, lhs, rhs, ...)
    """
    configuration = _mxfp8_v2_wgrad_configuration(producer)
    if configuration is None or not _can_accumulate_into_producer(
        accumulator, producer
    ):
        return False
    scaled_addmm, operands, contraction_dim, use_fast_accum = configuration

    producer_accumulator = _accumulator_for_producer(
        grad_accum_inplace_add, accumulator, producer
    )
    if producer_accumulator is None:
        return False
    producer.target = scaled_addmm
    producer.args = (producer_accumulator, *operands, contraction_dim)
    producer.kwargs = {
        "beta": 1,
        "alpha": 1,
        "use_fast_accum": use_fast_accum,
    }
    producer.meta["original_aten"] = scaled_addmm
    if replace_accumulation:
        _replace_grad_accum_inplace_add_with_gradient(grad_accum_inplace_add)
    return True


def _is_add(node: fx.Node) -> bool:
    return (
        node.target is torch.ops.aten.add.Tensor
        and len(node.args) == 2
        and not (set(node.kwargs) - {"alpha"})
        and _node_argument(node, "alpha", 2, 1) == 1
    )


def _native_parameter_grad_leaf(
    boundary: fx.Node,
    expected_user: fx.Node,
) -> _NativeParameterGradLeaf | None:
    chain = walk_up_unary_chain(boundary, expected_user, _is_alias)
    if chain is None:
        return None
    producer = chain[0]
    rule = _WGRAD_FUSION_RULES.get(producer.target)
    if rule is None or not rule.native_parameter_grad_accumulation:
        return None
    return _NativeParameterGradLeaf(
        boundary=boundary,
        producer=producer,
        rule=rule,
    )


def _native_parameter_grad_add(
    sink: fx.Node,
) -> _NativeParameterGradAdd | None:
    if sink.target is not torch.ops.aten.add_.Tensor or len(sink.args) < 2:
        return None
    if _node_argument(sink, "alpha", 2, 1) != 1:
        return None
    accumulator, root = sink.args[:2]
    if not isinstance(accumulator, fx.Node) or not isinstance(root, fx.Node):
        return None
    param_fqns = _parameter_gradient_fqns(sink)
    if (
        len(param_fqns) != 1
        or param_fqns != _parameter_gradient_fqns(root)
        or accumulator.op != "placeholder"
        or accumulator.meta.get(_GRAD_ACCUMULATOR_INPUT_META) is not True
        or not sole_user(accumulator, sink)
        or not _compatible_accumulation_tensors(accumulator, root)
        or not _is_add(root)
    ):
        return None

    reverse_add_nodes: list[fx.Node] = []
    reverse_rhs_boundaries: list[tuple[fx.Node, fx.Node]] = []
    node = root
    expected_user = sink
    while _is_add(node):
        if (
            not sole_user(node, expected_user)
            or _parameter_gradient_fqns(node) != param_fqns
            or not _compatible_accumulation_tensors(root, node)
        ):
            return None
        lhs, rhs = node.args
        if (
            not isinstance(lhs, fx.Node)
            or not isinstance(rhs, fx.Node)
            or lhs is rhs
            or _is_add(rhs)
            or not _compatible_accumulation_tensors(root, lhs)
            or not _compatible_accumulation_tensors(root, rhs)
        ):
            return None
        reverse_add_nodes.append(node)
        reverse_rhs_boundaries.append((rhs, node))
        expected_user = node
        node = lhs

    boundary_entries = (
        (node, expected_user),
        *reversed(reverse_rhs_boundaries),
    )
    leaves: list[_NativeParameterGradLeaf] = []
    seen_boundaries: set[fx.Node] = set()
    seen_producers: set[fx.Node] = set()
    for boundary, user in boundary_entries:
        if boundary in seen_boundaries:
            return None
        leaf = _native_parameter_grad_leaf(boundary, user)
        if leaf is None or leaf.producer in seen_producers:
            return None
        seen_boundaries.add(boundary)
        seen_producers.add(leaf.producer)
        leaves.append(leaf)

    if len(leaves) < 2 or any(leaf.rule is not leaves[0].rule for leaf in leaves[1:]):
        return None
    node_order = {
        graph_node: index for index, graph_node in enumerate(root.graph.nodes)
    }
    producer_order = [node_order[leaf.producer] for leaf in leaves]
    if producer_order != sorted(producer_order) or any(
        node_order[lhs.boundary] >= node_order[rhs.producer]
        for lhs, rhs in zip(leaves, leaves[1:])
    ):
        return None

    return _NativeParameterGradAdd(
        sink=sink,
        accumulator=accumulator,
        leaves=tuple(leaves),
        add_nodes=tuple(reversed(reverse_add_nodes)),
    )


def _fuse_native_parameter_grad_add(
    match: _NativeParameterGradAdd,
) -> int:
    """Fuse an ordered MXFP8 WGrad add into its accumulator.

    Input::
        wgrad0 = aten._scaled_mm_v2(lhs0, rhs0, ...)
        wgrad1 = aten._scaled_mm_v2(lhs1, rhs1, ...)
        output = accumulator.add_(wgrad0 + wgrad1)

    Output::
        first = aten._scaled_addmm_(accumulator, lhs0, rhs0, ...)
        output = aten._scaled_addmm_(first, lhs1, rhs1, ...)
    """
    current_accumulator = match.accumulator
    for leaf in match.leaves:
        if not leaf.rule.can_fuse(
            current_accumulator,
            leaf.producer,
        ):
            return 0
        current_accumulator = leaf.boundary

    current_accumulator = match.accumulator
    for leaf in match.leaves:
        lowered = leaf.rule.lower(
            match.sink,
            current_accumulator,
            leaf.producer,
            False,
        )
        assert lowered, "WGrad fusion changed after successful validation"
        current_accumulator = leaf.boundary

    match.sink.replace_all_uses_with(current_accumulator)
    match.sink.graph.erase_node(match.sink)
    for add_node in reversed(match.add_nodes):
        assert not add_node.users
        match.sink.graph.erase_node(add_node)
    return len(match.leaves)


def _fuse_native_parameter_grad_adds(gm: fx.GraphModule) -> dict[str, int]:
    fusion_counts: dict[str, int] = {}
    for sink in tuple(gm.graph.nodes):
        match = _native_parameter_grad_add(sink)
        if match is None:
            continue
        num_fused = _fuse_native_parameter_grad_add(match)
        if num_fused:
            name = match.leaves[0].rule.name
            fusion_counts[name] = fusion_counts.get(name, 0) + num_fused
    return fusion_counts


def _eligible_dist_moe_sink(sink: fx.Node) -> _DistMoeWgradSink | None:
    inputs = _annotated_wgrad_sink_inputs(sink)
    if inputs is None:
        return None
    accumulator, boundary = inputs
    accumulator_value = _tensor_meta(accumulator)
    assert accumulator_value is not None
    if accumulator_value.dtype not in (torch.bfloat16, torch.float32):
        return None
    postprocess_chain: list[fx.Node] = []
    source_boundary = boundary
    if source_boundary.target == torch.ops.aten._to_copy.default:
        if not sole_user(source_boundary, sink):
            return None
        if not source_boundary.args or not isinstance(source_boundary.args[0], fx.Node):
            return None
        postprocess_chain.append(source_boundary)
        source_boundary = source_boundary.args[0]
    chain = walk_up_unary_chain(
        source_boundary,
        postprocess_chain[-1] if postprocess_chain else sink,
        _is_alias,
    )
    if chain is None:
        return None
    getitem = chain[0]
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
    return _DistMoeWgradSink(
        sink=sink,
        accumulator=accumulator,
        getitem=getitem,
        postprocess_chain=tuple(postprocess_chain) + tuple(reversed(chain[1:])),
    )


def _dist_moe_accumulation_spec(backward: fx.Node) -> tuple[Any, int] | None:
    target_spec = _DIST_MOE_ACCUMULATION_TARGETS.get(str(backward.target))
    if target_spec is None:
        return None
    op_name, wgrad_output_dtype_position = target_spec
    try:
        return (
            getattr(torch.ops.dist_moe, op_name).default,
            wgrad_output_dtype_position,
        )
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


def _erase_dist_moe_sink(match: _DistMoeWgradSink) -> None:
    graph = match.sink.graph
    match.sink.replace_all_uses_with(match.accumulator)
    graph.erase_node(match.sink)
    for postprocess in match.postprocess_chain:
        if not postprocess.users:
            graph.erase_node(postprocess)
    graph.erase_node(match.getitem)


def _fuse_dist_moe_backward(
    backward: fx.Node,
    matches: dict[int, _DistMoeWgradSink],
) -> bool:
    """Move DistMoE WGrad accumulation into its backward operation.

    Input::
        grad_input, grad_gate, wgrad0, wgrad1 = dist_moe_backward(...)
        output0 = accumulator0.add_(wgrad0)
        output1 = accumulator1.add_(wgrad1)

    Output::
        grad_input, grad_gate = dist_moe_backward_accumulate_(
            ..., accumulator0, accumulator1
        )
    """
    if set(matches) != {2, 3} or not _has_expected_dist_moe_getitems(backward, matches):
        return False
    target_spec = _dist_moe_accumulation_spec(backward)
    if target_spec is None:
        return False
    target, wgrad_output_dtype_position = target_spec
    accumulator_values = tuple(
        _tensor_meta(matches[index].accumulator) for index in (2, 3)
    )
    if any(value is None for value in accumulator_values):
        return False
    first_accumulator_value, second_accumulator_value = accumulator_values
    assert first_accumulator_value is not None and second_accumulator_value is not None
    if first_accumulator_value.dtype != second_accumulator_value.dtype:
        return False
    accumulator_dtype = first_accumulator_value.dtype
    values = backward.meta.get("val")
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        return False
    backward_args = list(backward.args)
    backward_kwargs = dict(backward.kwargs)
    if "wgrad_output_dtype" in backward_kwargs:
        backward_kwargs["wgrad_output_dtype"] = accumulator_dtype
    elif wgrad_output_dtype_position < len(backward_args):
        backward_args[wgrad_output_dtype_position] = accumulator_dtype
    else:
        return False
    accumulator_views = {
        index: _view_accumulator_like(
            matches[index].accumulator,
            matches[index].getitem,
            before=backward,
        )
        for index in (2, 3)
    }
    if any(view is None for view in accumulator_views.values()):
        for view in accumulator_views.values():
            if view is not None:
                backward.graph.erase_node(view)
        return False
    backward.target = target
    backward.args = (
        *backward_args[:3],
        accumulator_views[2],
        accumulator_views[3],
        *backward_args[3:],
    )
    backward.kwargs = backward_kwargs
    backward.meta["val"] = tuple(values[:2])
    backward.meta["original_aten"] = target
    _erase_dist_moe_sink(matches[2])
    _erase_dist_moe_sink(matches[3])
    return True


def _fuse_dist_moe_sinks(gm: fx.GraphModule) -> int:
    matches_by_backward: dict[fx.Node, dict[int, _DistMoeWgradSink]] = {}
    for sink in tuple(gm.graph.nodes):
        match = _eligible_dist_moe_sink(sink)
        if match is None:
            continue
        backward = match.getitem.args[0]
        index = match.getitem.args[1]
        assert isinstance(backward, fx.Node) and isinstance(index, int)
        matches = matches_by_backward.setdefault(backward, {})
        matches.setdefault(index, match)
    return sum(
        _fuse_dist_moe_backward(backward, matches)
        for backward, matches in matches_by_backward.items()
    )


def fuse_wgrad_accumulation_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Fuse supported annotated WGrad producers with their accumulator updates.

    The WGrad output may feed the in-place addition through storage-only views.
    A left-linear add chain of MXFP8 WGrads is accumulated into the parameter
    gradient in leaf order because MXFP8 declares native parameter-gradient
    accumulation. Unsupported producers retain the explicit addition. MXFP8
    fusion can change rounding.

    Example::

        Input::
            grad = aten.mm(lhs, rhs)
            updated = aten.add_(accumulator, grad)

        Output::
            updated = aten.addmm_(accumulator, lhs, rhs)
    """
    del example_inputs
    fusion_counts = {rule.name: 0 for rule in _WGRAD_FUSION_RULES.values()}
    # Match ``accumulator.add_(wgrad0 + wgrad1 + ...)``.
    for name, count in _fuse_native_parameter_grad_adds(gm).items():
        fusion_counts[name] += count
    fusion_counts["DistMoE"] = _fuse_dist_moe_sinks(gm)
    for grad_accum_inplace_add in tuple(gm.graph.nodes):
        matched = _annotated_wgrad_accumulation(grad_accum_inplace_add)
        if matched is None:
            continue
        accumulator, producer = matched
        rule = _WGRAD_FUSION_RULES.get(producer.target)
        if rule is None:
            continue
        if rule.lower(grad_accum_inplace_add, accumulator, producer, True):
            fusion_counts[rule.name] += 1

    if any(fusion_counts.values()):
        gm.graph.lint()
        gm.recompile()
        logger.info(
            "Fused WGrad accumulation: %s",
            ", ".join(f"{name}={count}" for name, count in fusion_counts.items()),
        )
    return gm


__all__ = ["fuse_wgrad_accumulation_pass"]
