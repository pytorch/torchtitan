# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in fusion plans for one production DeepSeek V3 MXFP8 MoE graph."""

import math
import operator
from collections.abc import Callable

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.subgraph_rewriter import replace_pattern_with_filters

from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
)

# Shape suffixes: T = tokens, E = experts, K = selected experts, F = FFN width.
_NUM_TOKENS = 4096
_NUM_EXPERTS = 256
_HIDDEN_DIM = 2048


def _selection(scores_TE, expert_bias_E):
    scores_TGP = (scores_TE + expert_bias_E).unflatten(-1, (8, 32))
    group_scores_TG = scores_TGP.topk(2, dim=-1).values.sum(dim=-1)
    selected_group_ids_TL = group_scores_TG.topk(4, dim=-1, sorted=False).indices
    unselected_TG = torch.ones_like(group_scores_TG, dtype=torch.bool)
    unselected_TG.scatter_(-1, selected_group_ids_TL, False)
    masked_scores_TE = scores_TGP.masked_fill(
        unselected_TG.unsqueeze(-1), float("-inf")
    ).flatten(-2)
    return masked_scores_TE.topk(8, dim=-1, sorted=False).indices


def _weights(scores_TE, expert_ids_TK):
    selected_TK = scores_TE.gather(-1, expert_ids_TK)
    denominator_T1 = selected_TK.sum(dim=-1, keepdim=True) + 1e-20
    weights_TK = (selected_TK / denominator_T1) * 2.5
    return weights_TK, selected_TK, denominator_T1


def _weights_backward(grad_weights_TK, selected_TK, denominator_T1, expert_ids_TK):
    grad_scaled_TK = grad_weights_TK * 2.5
    grad_denominator_TK = -grad_scaled_TK * (
        (selected_TK / denominator_T1) / denominator_T1
    )
    grad_selected_TK = grad_scaled_TK / denominator_T1 + grad_denominator_TK.sum(
        dim=1, keepdim=True
    ).expand(_NUM_TOKENS, 8)
    zeros_TE = torch.ops.aten.new_zeros.default(
        grad_selected_TK, [_NUM_TOKENS, _NUM_EXPERTS]
    )
    return zeros_TE.scatter_add(-1, expert_ids_TK, grad_selected_TK)


def _gate_backward_split(aux_grad_TE, router_grad_TE, scores_TE):
    from torchtitan.models.common.hi_mid_lo_linear import _split_into_bf16_pieces

    grad_TE = torch.ops.aten.sigmoid_backward.default(
        aux_grad_TE + router_grad_TE, scores_TE
    )
    return _split_into_bf16_pieces(grad_TE, 3, dim=1)


def _weights_gate_backward(
    grad_weights_TK,
    selected_TK,
    denominator_T1,
    expert_ids_TK,
    aux_grad_TE,
    scores_TE,
):
    router_grad_TE = _weights_backward(
        grad_weights_TK, selected_TK, denominator_T1, expert_ids_TK
    )
    return _gate_backward_split(aux_grad_TE, router_grad_TE, scores_TE)


def _token_counts(scores_TE, expert_ids_TK):
    selected_TE = torch.zeros_like(scores_TE, dtype=torch.bool)
    selected_TE.scatter_(-1, expert_ids_TK, True)
    return selected_TE.sum(dim=0), selected_TE.float().sum(dim=0)


def _aux_forward(scores_TE):
    raw_norm_T1 = torch.linalg.vector_norm(scores_TE, 1, dim=-1, keepdim=True)
    expanded_norm_TE = raw_norm_T1.clamp_min(1e-12).expand(_NUM_TOKENS, _NUM_EXPERTS)
    return scores_TE / expanded_norm_TE, raw_norm_T1, expanded_norm_TE


def _aux_forward_fused(scores_TE):
    from .kernels.router_aux import router_aux_forward

    normalized_TE, raw_norm_T1, clamped_norm_T1 = router_aux_forward(scores_TE)
    return (
        normalized_TE,
        raw_norm_T1,
        clamped_norm_T1.expand(_NUM_TOKENS, _NUM_EXPERTS),
    )


def _aux_backward(grad_per_expert_E, scores_TE, raw_norm_T1, expanded_norm_TE, zero):
    grad_TE = grad_per_expert_E.unsqueeze(0).expand(_NUM_TOKENS, _NUM_EXPERTS)
    grad_norm_TE = -grad_TE * ((scores_TE / expanded_norm_TE) / expanded_norm_TE)
    grad_scores_TE = grad_TE / expanded_norm_TE
    grad_norm_T1 = grad_norm_TE.sum(dim=1, keepdim=True)
    grad_norm_T1 = torch.where(raw_norm_T1 > 1e-12, grad_norm_T1, zero)
    return grad_scores_TE + scores_TE.sgn() * grad_norm_T1


def _aux_backward_fused(grad_per_expert_E, scores_TE, raw_norm_T1, expanded_norm_TE):
    from .kernels.router_aux import router_aux_backward

    clamped_norm_T1 = expanded_norm_TE.select(1, 0).unsqueeze(1)
    return router_aux_backward(
        grad_per_expert_E, scores_TE, raw_norm_T1, clamped_norm_T1
    )


def _router_backward(
    grad_weights_TK,
    selected_TK,
    denominator_T1,
    expert_ids_TK,
    grad_per_expert_E,
    scores_TE,
    raw_norm_T1,
    expanded_norm_TE,
    zero,
    gate_scores_TE,
):
    aux_grad_TE = _aux_backward(
        grad_per_expert_E, scores_TE, raw_norm_T1, expanded_norm_TE, zero
    )
    return _weights_gate_backward(
        grad_weights_TK,
        selected_TK,
        denominator_T1,
        expert_ids_TK,
        aux_grad_TE,
        gate_scores_TE,
    )


def _router_backward_fused(
    grad_weights_TK,
    selected_TK,
    denominator_T1,
    expert_ids_TK,
    grad_per_expert_E,
    scores_TE,
    raw_norm_T1,
    expanded_norm_TE,
):
    from .kernels.router_backward import router_backward

    clamped_norm_T1 = expanded_norm_TE.select(1, 0).unsqueeze(1)
    return router_backward(
        grad_weights_TK,
        selected_TK,
        denominator_T1,
        expert_ids_TK,
        grad_per_expert_E,
        scores_TE,
        raw_norm_T1,
        clamped_norm_T1,
    )


def _swiglu_quantize(gate_up_T2F):
    hidden_TF = torch.ops.torchtitan.silu_and_mul.default(gate_up_T2F)
    operands = torch.ops.torchao.mxfp8_quantize.default(
        hidden_TF, True, True, 32, 32, "e4m3", "rceil"
    )
    return (
        operands[0],
        operands[1],
        torch.ops.torchao.triton_mx_block_rearrange.default(operands[2]),
        torch.ops.torchao.triton_mx_block_rearrange.default(operands[3]),
    )


def _swiglu_backward_quantize(grad_out_TF, gate_up_T2F):
    grad_gate_up_T2F = torch.ops.torchtitan.silu_and_mul_backward.default(
        grad_out_TF, gate_up_T2F
    )
    operands = torch.ops.torchao.mxfp8_quantize.default(
        grad_gate_up_T2F.flatten(1), True, True, 32, 32, "e4m3", "rceil"
    )
    return (
        operands[0],
        operands[1],
        torch.ops.torchao.triton_mx_block_rearrange.default(operands[2]),
        torch.ops.torchao.triton_mx_block_rearrange.default(operands[3]),
    )


def _is_topk_indices(node: torch.fx.Node) -> bool:
    from .kernels.router_selection import select_experts_fused

    if node.op != "call_function":
        return False
    if node.target is select_experts_fused:
        return True
    if node.target is not operator.getitem:
        return False
    source, index = node.args
    if not isinstance(source, torch.fx.Node) or source.op != "call_function":
        return False
    if source.target != torch.ops.aten.topk.default or index != 1:
        return False
    dim = source.args[2] if len(source.args) > 2 else source.kwargs.get("dim", -1)
    value = source.args[0].meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == (_NUM_TOKENS, _NUM_EXPERTS)
        and dim in (-1, 1)
    )


def _aligned_scores(node: torch.fx.Node) -> bool:
    value = node.meta["val"]
    return value.storage_offset() * value.element_size() % 16 == 0


def _aux_backward_supported(inputs: tuple[torch.fx.Node, ...]) -> bool:
    zero = inputs[4]
    return (
        _aligned_scores(inputs[1])
        and zero.op == "call_function"
        and zero.target == torch.ops.aten.scalar_tensor.default
        and zero.args == (0.0,)
        and math.copysign(1.0, zero.args[0]) == 1.0
    )


def _alias_source(node: torch.fx.Node) -> torch.fx.Node:
    while node.op == "call_function" and node.target == torch.ops.aten.alias.default:
        node = node.args[0]
    return node


def _router_backward_supported(inputs: tuple[torch.fx.Node, ...]) -> bool:
    return (
        _is_topk_indices(inputs[3])
        and _aux_backward_supported(inputs[4:9])
        and _alias_source(inputs[5]) is _alias_source(inputs[9])
    )


def _replace_region(
    gm: torch.fx.GraphModule,
    reference: Callable,
    target: Callable,
    input_specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    *,
    target_kwargs: dict | None = None,
    target_input_indices: tuple[int, ...] | None = None,
    input_strides: dict[int, tuple[int, ...]] | None = None,
    node_filter: Callable[[tuple[torch.fx.Node, ...]], bool] | None = None,
) -> int:
    example_inputs = tuple(
        torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in input_specs
    )
    pattern = make_fx(reference)(*example_inputs)
    canonicalize_graph_pass(pattern)
    eliminate_dead_code_pass(pattern)
    placeholders = [node for node in pattern.graph.nodes if node.op == "placeholder"]
    pattern_output = next(node for node in pattern.graph.nodes if node.op == "output")

    def supported(match, original_graph, pattern_graph):
        inputs = tuple(match.nodes_map[node] for node in placeholders)
        for index, (node, (shape, dtype)) in enumerate(
            zip(inputs, input_specs, strict=True)
        ):
            value = node.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device.type != "cuda"
            ):
                return False
            strides = input_strides.get(index) if input_strides else None
            if strides is None:
                if not value.is_contiguous():
                    return False
            elif value.stride() != strides:
                return False
        return node_filter(inputs) if node_filter else True

    def replacement(match, original_graph, pattern_graph):
        graph = torch.fx.Graph()
        inputs = [graph.placeholder(node.name) for node in placeholders]
        args = (
            tuple(inputs)
            if target_input_indices is None
            else tuple(inputs[index] for index in target_input_indices)
        )
        call = graph.call_function(target, args, target_kwargs or {})
        outputs = pattern_output.args[0]
        if isinstance(outputs, torch.fx.Node):
            call.meta = match.nodes_map[outputs].meta.copy()
            graph.output(call)
        else:
            call.meta["val"] = tuple(
                match.nodes_map[node].meta.get("val") for node in outputs
            )
            result = []
            for index, node in enumerate(outputs):
                output = graph.call_function(operator.getitem, (call, index))
                output.meta = match.nodes_map[node].meta.copy()
                result.append(output)
            graph.output(tuple(result))
        return graph

    matches = replace_pattern_with_filters(
        gm,
        pattern,
        match_filters=[supported],
        replacement_callback=replacement,
    )
    return len(matches)


def fuse_production_moe(
    gm: torch.fx.GraphModule,
    *,
    regions: tuple[str, ...] = (
        "selection",
        "weights",
        "gate_backward",
        "counts",
        "aux",
        "swiglu",
    ),
) -> dict[str, int]:
    """Fuse supported regions, preserving graph outputs and external consumers.

    This experiment deliberately supports the measured production shape only.
    Unsupported shapes, dtypes, layouts, and additional intermediate consumers
    leave the original graph intact. GEMMs and Dist-MoE operators remain graph
    boundaries. The return value records the number of matches for each plan.
    """
    unknown = set(regions) - {
        "selection",
        "weights",
        "gate_backward",
        "counts",
        "aux",
        "swiglu",
    }
    if unknown:
        raise ValueError(f"Unknown MoE fusion regions: {sorted(unknown)}")
    canonicalize_graph_pass(gm)
    eliminate_dead_code_pass(gm)
    scores = ((_NUM_TOKENS, _NUM_EXPERTS), torch.float32)
    indices = ((_NUM_TOKENS, 8), torch.int64)
    selected = ((_NUM_TOKENS, 8), torch.float32)
    counts: dict[str, int] = {}
    if "selection" in regions:
        from .kernels.router_selection import select_experts_fused

        counts["selection"] = _replace_region(
            gm,
            _selection,
            select_experts_fused,
            (scores, ((_NUM_EXPERTS,), torch.float32)),
        )
    if "weights" in regions:
        from .kernels.router_weights import router_weights_forward

        counts["weights_forward"] = _replace_region(
            gm, _weights, router_weights_forward, (scores, indices)
        )
    num_joint_backwards = 0
    if {"weights", "gate_backward", "aux"}.issubset(regions):
        num_joint_backwards = _replace_region(
            gm,
            _router_backward,
            _router_backward_fused,
            (
                selected,
                selected,
                ((_NUM_TOKENS, 1), torch.float32),
                indices,
                ((_NUM_EXPERTS,), torch.float32),
                scores,
                ((_NUM_TOKENS, 1), torch.float32),
                scores,
                ((), torch.float32),
                scores,
            ),
            target_input_indices=tuple(range(8)),
            input_strides={7: (1, 0)},
            node_filter=_router_backward_supported,
        )
        if num_joint_backwards:
            counts["router_backward"] = num_joint_backwards
    if "weights" in regions and "gate_backward" in regions:
        from .kernels.router_weights import router_weights_gate_backward

        num_weights_backwards = _replace_region(
            gm,
            _weights_gate_backward,
            router_weights_gate_backward,
            (
                selected,
                selected,
                ((_NUM_TOKENS, 1), torch.float32),
                indices,
                scores,
                scores,
            ),
            node_filter=lambda inputs: _is_topk_indices(inputs[3]),
        )
        if num_weights_backwards or not num_joint_backwards:
            counts["weights_gate_backward"] = num_weights_backwards
    elif "weights" in regions:
        from .kernels.router_weights import router_weights_backward

        counts["weights_backward"] = _replace_region(
            gm,
            _weights_backward,
            router_weights_backward,
            (selected, selected, ((_NUM_TOKENS, 1), torch.float32), indices),
            target_kwargs={"num_experts": _NUM_EXPERTS},
            node_filter=lambda inputs: _is_topk_indices(inputs[3]),
        )
    elif "gate_backward" in regions:
        from .kernels.router_weights import router_gate_backward_split

        counts["gate_backward"] = _replace_region(
            gm,
            _gate_backward_split,
            router_gate_backward_split,
            (scores, scores, scores),
        )
    if "counts" in regions:
        from .kernels.router_counts import token_counts

        counts["counts"] = _replace_region(
            gm,
            _token_counts,
            token_counts,
            (scores, indices),
            target_input_indices=(1,),
            node_filter=lambda inputs: _is_topk_indices(inputs[1]),
        )
    if "aux" in regions:
        counts["aux_forward"] = _replace_region(
            gm,
            _aux_forward,
            _aux_forward_fused,
            (scores,),
            node_filter=lambda inputs: _aligned_scores(inputs[0]),
        )
        num_aux_backwards = _replace_region(
            gm,
            _aux_backward,
            _aux_backward_fused,
            (
                ((_NUM_EXPERTS,), torch.float32),
                scores,
                ((_NUM_TOKENS, 1), torch.float32),
                scores,
                ((), torch.float32),
            ),
            target_input_indices=(0, 1, 2, 3),
            input_strides={3: (1, 0)},
            node_filter=_aux_backward_supported,
        )
        if num_aux_backwards or not num_joint_backwards:
            counts["aux_backward"] = num_aux_backwards
    if "swiglu" in regions:
        from .kernels.shared_expert import (
            swiglu_backward_mxfp8_quantize,
            swiglu_mxfp8_quantize,
        )

        gate_up = ((_NUM_TOKENS, 2, _HIDDEN_DIM), torch.bfloat16)
        counts["swiglu_forward"] = _replace_region(
            gm, _swiglu_quantize, swiglu_mxfp8_quantize._opoverload, (gate_up,)
        )
        counts["swiglu_backward"] = _replace_region(
            gm,
            _swiglu_backward_quantize,
            swiglu_backward_mxfp8_quantize._opoverload,
            (((_NUM_TOKENS, _HIDDEN_DIM), torch.bfloat16), gate_up),
        )
    eliminate_dead_code_pass(gm)
    gm.graph.lint()
    gm.recompile()
    return counts
