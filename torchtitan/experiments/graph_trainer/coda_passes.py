# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CODA-style GEMM fusion passes for GraphTrainer."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable

import torch
from torch._higher_order_ops.flex_gemm import (
    apply_flex_gemm_body_graph_passes,
    flex_gemm_hop,
)

from torchtitan.tools.logging import logger


_MM = torch.ops.aten.mm.default
_ADD = torch.ops.aten.add.Tensor
_ADD_SCALAR = torch.ops.aten.add.Scalar
_FUSED_RMS_NORM = torch.ops.aten._fused_rms_norm.default
_MEAN_DIM = torch.ops.aten.mean.dim
_MUL = torch.ops.aten.mul.Tensor
_POW_SCALAR = torch.ops.aten.pow.Tensor_Scalar
_RESHAPE = torch.ops.aten.reshape.default
_RSQRT = torch.ops.aten.rsqrt.default
_SIGMOID = torch.ops.aten.sigmoid.default
_SILU = torch.ops.aten.silu.default
_SILU_BACKWARD = torch.ops.aten.silu_backward.default
_TO_COPY = torch.ops.aten._to_copy.default
_VIEW = torch.ops.aten.view.default
_COMPILE_WITH_INDUCTOR = "compile_with_inductor"
_CODA_RMSNORM_GROUP = 512


def _node_dtype(node: torch.fx.Node) -> torch.dtype | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype
    tensor_meta = node.meta.get("tensor_meta")
    return getattr(tensor_meta, "dtype", None)


def _node_shape(node: torch.fx.Node) -> tuple | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    tensor_meta = node.meta.get("tensor_meta")
    shape = getattr(tensor_meta, "shape", None)
    return None if shape is None else tuple(shape)


def _module_fqn(node: torch.fx.Node) -> str | None:
    custom = node.meta.get("custom", {})
    module_fqn = custom.get("module_fqn")
    return module_fqn if isinstance(module_fqn, str) else None


def _sole_user(node: torch.fx.Node) -> torch.fx.Node | None:
    if len(node.users) != 1:
        return None
    return next(iter(node.users))


def _is_cast(node: torch.fx.Node, dtype: torch.dtype) -> bool:
    return node.target == _TO_COPY and node.kwargs.get("dtype") == dtype


def _copy_meta(*nodes: torch.fx.Node) -> dict:
    """Merge metadata in dataflow order without copying FakeTensor values."""
    merged: dict = {}
    custom: dict = {}
    for node in nodes:
        merged.update(node.meta)
        custom.update(node.meta.get("custom", {}))
    if custom:
        merged["custom"] = custom
    return merged


def _copy_meta_with_value_from(
    value_source: torch.fx.Node,
    *nodes: torch.fx.Node,
) -> dict:
    merged = _copy_meta(*nodes)
    for key in ("val", "tensor_meta"):
        if key in value_source.meta:
            merged[key] = value_source.meta[key]
        else:
            merged.pop(key, None)
    return merged


def _copy_meta_with_dtype(
    value_source: torch.fx.Node,
    dtype: torch.dtype,
    *nodes: torch.fx.Node,
) -> dict:
    merged = _copy_meta(*nodes)
    value = value_source.meta.get("val")
    if isinstance(value, torch.Tensor):
        merged["val"] = value.to(dtype=dtype)
    else:
        merged.pop("val", None)
    merged.pop("tensor_meta", None)
    return merged


def _copy_meta_with_value(value: torch.Tensor, *nodes: torch.fx.Node) -> dict:
    merged = _copy_meta(*nodes)
    merged["val"] = value
    merged.pop("tensor_meta", None)
    return merged


def _tag_for_regional_inductor(node: torch.fx.Node) -> None:
    node.meta.setdefault("custom", {})[_COMPILE_WITH_INDUCTOR] = {}


def _next_submodule_name(gm: torch.fx.GraphModule, prefix: str) -> str:
    index = 0
    while hasattr(gm, f"{prefix}_{index}"):
        index += 1
    return f"{prefix}_{index}"


def _build_b6_bf16_body(
    mm: torch.fx.Node,
    cast: torch.fx.Node,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, arg in enumerate(mm.args):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(arg, torch.fx.Node):
            placeholder.meta = dict(arg.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    # FlexGEMM exposes its accumulator as FP32. Preserve the original BF16
    # GEMM store rounding before returning the FP32 gradient.
    to_fp32 = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    to_fp32.meta = dict(cast.meta)
    to_bf16 = graph.call_function(_TO_COPY, (to_fp32,), {"dtype": torch.bfloat16})
    to_bf16.meta = dict(mm.meta)
    value = graph.call_function(_TO_COPY, (to_bf16,), {"dtype": torch.float32})
    value.meta = dict(cast.meta)

    # FlexGEMM lowering returns an ordered output tuple. Keeping the singleton
    # tuple explicit gives regional Inductor a getitem node to consume.
    graph.output((value,))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_f6_router_body(
    mm: torch.fx.Node,
    sigmoid: torch.fx.Node,
    bias_2d: torch.fx.Node | None,
    bias_add: torch.fx.Node | None,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, arg in enumerate(mm.args):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(arg, torch.fx.Node):
            placeholder.meta = dict(arg.meta)
        inputs.append(placeholder)

    bias_input = None
    if bias_2d is not None:
        bias_input = graph.placeholder(f"arg{len(inputs)}")
        bias_input.meta = dict(bias_2d.meta)
        inputs.append(bias_input)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    raw_scores = graph.call_function(_SIGMOID, (body_mm,))
    raw_scores.meta = _copy_meta_with_value_from(mm, mm, sigmoid)
    outputs = [raw_scores]
    if bias_input is not None and bias_add is not None:
        biased_scores = graph.call_function(_ADD, (raw_scores, bias_input))
        biased_scores.meta = _copy_meta_with_value_from(mm, mm, sigmoid, bias_add)
        outputs.append(biased_scores)

    graph.output(tuple(outputs))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_f4_silu_body(
    mm: torch.fx.Node,
    reshape: torch.fx.Node,
    silu: torch.fx.Node,
    preserve_preactivation: bool,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, arg in enumerate(mm.args):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(arg, torch.fx.Node):
            placeholder.meta = dict(arg.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    to_fp32 = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    to_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm, reshape, silu)
    rounded = graph.call_function(_TO_COPY, (to_fp32,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm, reshape, silu)
    silu_2d = graph.call_function(_SILU, (rounded,))
    silu_2d.meta = _copy_meta_with_value_from(mm, mm, reshape, silu)

    outputs = (silu_2d, rounded) if preserve_preactivation else (silu_2d,)
    graph.output(outputs)
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_f4_mul_body(
    mm: torch.fx.Node,
    reshape: torch.fx.Node,
    silu_2d: torch.fx.Node,
    mul: torch.fx.Node,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, arg in enumerate(mm.args):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(arg, torch.fx.Node):
            placeholder.meta = dict(arg.meta)
        inputs.append(placeholder)
    silu_input = graph.placeholder(f"arg{len(inputs)}")
    silu_input.meta = dict(silu_2d.meta)
    inputs.append(silu_input)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    to_fp32 = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    to_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm, reshape, mul)
    rounded = graph.call_function(_TO_COPY, (to_fp32,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm, reshape, mul)
    product = graph.call_function(_MUL, (rounded, silu_input))
    product.meta = _copy_meta_with_value_from(mm, mm, reshape, mul)

    graph.output((rounded, product))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_b2_branch_body(
    mm: torch.fx.Node,
    grad_view: torch.fx.Node,
    saved_silu_2d: torch.fx.Node,
    saved_gate_2d: torch.fx.Node,
    saved_preactivation_2d: torch.fx.Node,
    gate_grad: torch.fx.Node,
    silu_grad: torch.fx.Node,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    sources = (
        *mm.args,
        saved_silu_2d,
        saved_gate_2d,
        saved_preactivation_2d,
    )
    for index, source in enumerate(sources):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(source, torch.fx.Node):
            placeholder.meta = dict(source.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    to_fp32 = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    to_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm, grad_view)
    rounded = graph.call_function(_TO_COPY, (to_fp32,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm, grad_view)

    gate_grad_2d = graph.call_function(_MUL, (rounded, inputs[2]))
    gate_grad_2d.meta = _copy_meta_with_value_from(mm, mm, grad_view, gate_grad)
    gated_grad_2d = graph.call_function(_MUL, (rounded, inputs[3]))
    gated_grad_2d.meta = _copy_meta_with_value_from(mm, mm, grad_view)
    silu_grad_2d = graph.call_function(
        _SILU_BACKWARD,
        (gated_grad_2d, inputs[4]),
    )
    silu_grad_2d.meta = _copy_meta_with_value_from(mm, mm, grad_view, silu_grad)

    graph.output((gate_grad_2d, silu_grad_2d))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_b2_input_add_body(
    mm: torch.fx.Node,
    reshape: torch.fx.Node,
    captured_branch_2d: torch.fx.Node,
    add: torch.fx.Node,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    sources = (*mm.args, captured_branch_2d)
    for index, source in enumerate(sources):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(source, torch.fx.Node):
            placeholder.meta = dict(source.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    to_fp32 = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    to_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm, reshape, add)
    rounded = graph.call_function(_TO_COPY, (to_fp32,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm, reshape, add)
    total = graph.call_function(_ADD, (inputs[2], rounded))
    total.meta = _copy_meta_with_value_from(mm, mm, reshape, add)

    graph.output((total,))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_b4_router_input_grad_body(
    mm: torch.fx.Node,
    cast: torch.fx.Node,
    residual_2d: torch.fx.Node,
    add: torch.fx.Node,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, source in enumerate((*mm.args, residual_2d)):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(source, torch.fx.Node):
            placeholder.meta = dict(source.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    rounded = graph.call_function(_TO_COPY, (body_mm,), dict(cast.kwargs))
    rounded.meta = _copy_meta_with_dtype(mm, torch.bfloat16, mm, cast)
    total = graph.call_function(_ADD, (inputs[2], rounded))
    total.meta = _copy_meta_with_dtype(mm, torch.bfloat16, mm, cast, add)

    graph.output((total,))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_f3_residual_rmsnorm_first_body(
    mm: torch.fx.Node,
    addends_2d: tuple[torch.fx.Node, ...],
    gamma_2d: torch.fx.Node,
    group: int,
    accumulated_value_is_lhs: tuple[bool, ...],
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, source in enumerate((*mm.args, *addends_2d, gamma_2d)):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(source, torch.fx.Node):
            placeholder.meta = dict(source.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    accumulator = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    accumulator.meta = _copy_meta_with_dtype(mm, torch.float32, mm)
    rounded = graph.call_function(_TO_COPY, (accumulator,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm)
    total = rounded
    for input_index, (addend, value_is_lhs) in enumerate(
        zip(addends_2d, accumulated_value_is_lhs, strict=True), start=2
    ):
        add_args = (
            (total, inputs[input_index])
            if value_is_lhs
            else (
                inputs[input_index],
                total,
            )
        )
        total = graph.call_function(_ADD, add_args)
        total.meta = _copy_meta_with_value_from(mm, mm, *addends_2d[: input_index - 1])
    total_fp32 = graph.call_function(_TO_COPY, (total,), {"dtype": torch.float32})
    total_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm, *addends_2d)
    weighted_fp32 = graph.call_function(_MUL, (total_fp32, inputs[-1]))
    weighted_fp32.meta = _copy_meta_with_dtype(
        mm, torch.float32, mm, *addends_2d, gamma_2d
    )
    weighted = graph.call_function(
        _TO_COPY, (weighted_fp32,), {"dtype": torch.bfloat16}
    )
    weighted.meta = _copy_meta_with_value_from(mm, mm, *addends_2d, gamma_2d)

    mm_shape = _node_shape(mm)
    assert mm_shape is not None
    grouped = graph.call_function(
        _VIEW,
        (total_fp32, [mm_shape[0], -1, group]),
    )
    total_value = total_fp32.meta.get("val")
    if isinstance(total_value, torch.Tensor):
        grouped.meta = _copy_meta_with_value(
            total_value.reshape(mm_shape[0], -1, group), mm, *addends_2d
        )
    else:
        grouped.meta = _copy_meta(mm, *addends_2d)
    squared = graph.call_function(_POW_SCALAR, (grouped, 2))
    grouped_value = grouped.meta.get("val")
    if isinstance(grouped_value, torch.Tensor):
        squared.meta = _copy_meta_with_value(grouped_value.square(), mm, *addends_2d)
    else:
        squared.meta = _copy_meta(mm, *addends_2d)
    partial_mean_square = graph.call_function(_MEAN_DIM, (squared, [-1]))
    squared_value = squared.meta.get("val")
    if isinstance(squared_value, torch.Tensor):
        partial_mean_square.meta = _copy_meta_with_value(
            squared_value.mean(-1), mm, *addends_2d
        )
    else:
        partial_mean_square.meta = _copy_meta(mm, *addends_2d)

    graph.output((weighted, total, partial_mean_square))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_f2_q_rmsnorm_first_body(
    mm: torch.fx.Node,
    gamma_2d: torch.fx.Node,
    group: int,
    preserve_raw: bool,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, source in enumerate((*mm.args, gamma_2d)):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(source, torch.fx.Node):
            placeholder.meta = dict(source.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    accumulator = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    accumulator.meta = _copy_meta_with_dtype(mm, torch.float32, mm)
    rounded = graph.call_function(_TO_COPY, (accumulator,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm)
    rounded_fp32 = graph.call_function(_TO_COPY, (rounded,), {"dtype": torch.float32})
    rounded_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm)
    weighted_fp32 = graph.call_function(_MUL, (rounded_fp32, inputs[2]))
    weighted_fp32.meta = _copy_meta_with_dtype(mm, torch.float32, mm, gamma_2d)
    weighted = graph.call_function(
        _TO_COPY, (weighted_fp32,), {"dtype": torch.bfloat16}
    )
    weighted.meta = _copy_meta_with_value_from(mm, mm, gamma_2d)

    mm_shape = _node_shape(mm)
    assert mm_shape is not None
    grouped = graph.call_function(
        _VIEW,
        (rounded_fp32, [mm_shape[0], -1, group]),
    )
    rounded_value = rounded_fp32.meta.get("val")
    if isinstance(rounded_value, torch.Tensor):
        grouped.meta = _copy_meta_with_value(
            rounded_value.reshape(mm_shape[0], -1, group), mm
        )
    else:
        grouped.meta = _copy_meta(mm)
    squared = graph.call_function(_POW_SCALAR, (grouped, 2))
    grouped_value = grouped.meta.get("val")
    if isinstance(grouped_value, torch.Tensor):
        squared.meta = _copy_meta_with_value(grouped_value.square(), mm)
    else:
        squared.meta = _copy_meta(mm)
    partial_mean_square = graph.call_function(_MEAN_DIM, (squared, [-1]))
    squared_value = squared.meta.get("val")
    if isinstance(squared_value, torch.Tensor):
        partial_mean_square.meta = _copy_meta_with_value(squared_value.mean(-1), mm)
    else:
        partial_mean_square.meta = _copy_meta(mm)

    outputs = (
        (weighted, rounded, partial_mean_square)
        if preserve_raw
        else (weighted, partial_mean_square)
    )
    graph.output(outputs)
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _build_f2_q_rmsnorm_second_body(
    mm: torch.fx.Node,
    rstd_2d: torch.fx.Node,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    inputs = []
    for index, source in enumerate((*mm.args, rstd_2d)):
        placeholder = graph.placeholder(f"arg{index}")
        if isinstance(source, torch.fx.Node):
            placeholder.meta = dict(source.meta)
        inputs.append(placeholder)

    body_mm = graph.call_function(_MM, tuple(inputs[:2]), dict(mm.kwargs))
    body_mm.meta = dict(mm.meta)
    accumulator = graph.call_function(_TO_COPY, (body_mm,), {"dtype": torch.float32})
    accumulator.meta = _copy_meta_with_dtype(mm, torch.float32, mm)
    scaled = graph.call_function(_MUL, (accumulator, inputs[2]))
    scaled.meta = _copy_meta_with_dtype(mm, torch.float32, mm, rstd_2d)
    rounded = graph.call_function(_TO_COPY, (scaled,), {"dtype": torch.bfloat16})
    rounded.meta = _copy_meta_with_value_from(mm, mm)

    graph.output((rounded,))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)
    apply_flex_gemm_body_graph_passes(body, _MM)
    for node in body.graph.nodes:
        _tag_for_regional_inductor(node)
    return body


def _match_b6_bf16_cast(mm: torch.fx.Node) -> torch.fx.Node | None:
    if mm.target != _MM:
        return None
    first = _sole_user(mm)
    if first is None:
        return None

    if _node_dtype(mm) == torch.bfloat16 and _is_cast(first, torch.float32):
        return first
    return None


def fuse_b6_bf16_weight_grad_cast_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse BF16 weight-gradient ``mm`` plus FP32 cast into FlexGEMM."""
    del example_inputs
    num_fused = 0
    for mm in list(gm.graph.nodes):
        cast = _match_b6_bf16_cast(mm)
        if cast is None:
            continue

        body = _build_b6_bf16_body(mm, cast)
        body_name = _next_submodule_name(gm, "_coda_b6_body")
        gm.add_module(body_name, body)
        with gm.graph.inserting_before(mm):
            body_ref = gm.graph.get_attr(body_name)
            _tag_for_regional_inductor(body_ref)
            fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    body_ref,
                    tuple(mm.args),
                    dict(mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            fused.meta = _copy_meta(mm, cast)
            for key in ("val", "tensor_meta"):
                if key in fused.meta:
                    fused.meta[key] = (fused.meta[key],)
            _tag_for_regional_inductor(fused)
            output = gm.graph.call_function(operator.getitem, (fused, 0))
            output.meta = _copy_meta(mm, cast)
            _tag_for_regional_inductor(output)

        cast.replace_all_uses_with(output)
        gm.graph.erase_node(cast)
        gm.graph.erase_node(mm)
        num_fused += 1

    gm.graph.lint()
    gm.recompile()
    logger.info(f"B6 fused {num_fused} BF16 weight-gradient GEMM cast chains")
    return gm


def _f6_bias_add(
    sigmoid: torch.fx.Node,
    output_width,
) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    matches = []
    for user in sigmoid.users:
        if user.target != _ADD or user.args[0] is not sigmoid:
            continue
        if user.kwargs.get("alpha", 1) != 1 or len(user.args) < 2:
            continue
        bias = user.args[1]
        if not isinstance(bias, torch.fx.Node):
            continue
        if _node_dtype(bias) != torch.float32 or _node_shape(bias) != (output_width,):
            continue
        matches.append((user, bias))
    if len(matches) != 1:
        return None
    return matches[0]


def _match_f6_router_sigmoid(
    sigmoid: torch.fx.Node,
) -> tuple[
    torch.fx.Node, torch.fx.Node, torch.fx.Node | None, torch.fx.Node | None
] | None:
    if sigmoid.target != _SIGMOID or _node_dtype(sigmoid) != torch.float32:
        return None
    if len(sigmoid.args) != 1 or not isinstance(sigmoid.args[0], torch.fx.Node):
        return None
    reshape = sigmoid.args[0]
    if reshape.target not in (_RESHAPE, _VIEW) or _sole_user(reshape) is not sigmoid:
        return None
    if not reshape.args or not isinstance(reshape.args[0], torch.fx.Node):
        return None
    mm = reshape.args[0]
    if mm.target != _MM or _sole_user(mm) is not reshape:
        return None
    if _node_dtype(mm) != torch.float32:
        return None

    mm_shape = _node_shape(mm)
    reshape_shape = _node_shape(reshape)
    if (
        mm_shape is None
        or len(mm_shape) != 2
        or reshape_shape is None
        or len(reshape_shape) < 2
        or reshape_shape[-1] != mm_shape[-1]
    ):
        return None

    bias_match = _f6_bias_add(sigmoid, mm_shape[-1])
    if bias_match is None:
        return mm, reshape, None, None
    bias_add, bias = bias_match
    return mm, reshape, bias_add, bias


def fuse_f6_router_sigmoid_bias_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse router GEMM sigmoid and optional expert bias into FlexGEMM."""
    del example_inputs
    num_fused = 0
    num_bias_fused = 0
    for sigmoid in list(gm.graph.nodes):
        match = _match_f6_router_sigmoid(sigmoid)
        if match is None:
            continue
        mm, reshape, bias_add, bias = match
        mm_shape = _node_shape(mm)
        output_shape = _node_shape(sigmoid)
        assert mm_shape is not None
        assert output_shape is not None

        bias_2d = None
        if bias is not None:
            with gm.graph.inserting_before(mm):
                bias_2d = gm.graph.call_function(
                    _RESHAPE,
                    (bias, [1, mm_shape[-1]]),
                )
                bias_2d.meta = _copy_meta(bias)
                bias_value = bias.meta.get("val")
                if isinstance(bias_value, torch.Tensor):
                    bias_2d.meta["val"] = bias_value.reshape(1, -1)
                bias_2d.meta.pop("tensor_meta", None)
                _tag_for_regional_inductor(bias_2d)

        body = _build_f6_router_body(mm, sigmoid, bias_2d, bias_add)
        body_name = _next_submodule_name(gm, "_coda_f6_body")
        gm.add_module(body_name, body)
        with gm.graph.inserting_before(mm):
            body_ref = gm.graph.get_attr(body_name)
            _tag_for_regional_inductor(body_ref)
            hop_args = tuple(mm.args) + (() if bias_2d is None else (bias_2d,))
            fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    body_ref,
                    hop_args,
                    dict(mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            fused.meta = _copy_meta(mm, reshape, sigmoid)
            value_meta = _copy_meta_with_value_from(mm, mm, reshape, sigmoid)
            output_metas = [value_meta]
            if bias_add is not None:
                output_metas.append(
                    _copy_meta_with_value_from(mm, mm, reshape, sigmoid, bias_add)
                )
            for key in ("val", "tensor_meta"):
                values = [meta[key] for meta in output_metas if key in meta]
                if len(values) == len(output_metas):
                    fused.meta[key] = tuple(values)
                else:
                    fused.meta.pop(key, None)
            _tag_for_regional_inductor(fused)

            raw_2d = gm.graph.call_function(operator.getitem, (fused, 0))
            raw_2d.meta = value_meta
            _tag_for_regional_inductor(raw_2d)
            raw_scores = gm.graph.call_function(
                _RESHAPE,
                (raw_2d, list(output_shape)),
            )
            raw_scores.meta = _copy_meta(mm, reshape, sigmoid)
            _tag_for_regional_inductor(raw_scores)

            biased_scores = None
            if bias_add is not None:
                biased_2d = gm.graph.call_function(operator.getitem, (fused, 1))
                biased_2d.meta = output_metas[1]
                _tag_for_regional_inductor(biased_2d)
                biased_scores = gm.graph.call_function(
                    _RESHAPE,
                    (biased_2d, list(output_shape)),
                )
                biased_scores.meta = _copy_meta(mm, reshape, sigmoid, bias_add)
                _tag_for_regional_inductor(biased_scores)

        sigmoid.replace_all_uses_with(raw_scores)
        if bias_add is not None:
            assert biased_scores is not None
            bias_add.replace_all_uses_with(biased_scores)
            gm.graph.erase_node(bias_add)
            num_bias_fused += 1
        gm.graph.erase_node(sigmoid)
        gm.graph.erase_node(reshape)
        gm.graph.erase_node(mm)
        num_fused += 1

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"F6 fused {num_fused} router GEMM sigmoid chains, "
        f"including {num_bias_fused} expert-bias epilogues"
    )
    return gm


def _match_f4_swiglu(
    mul: torch.fx.Node,
) -> tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
] | None:
    if mul.target != _MUL or len(mul.args) < 2:
        return None
    silu, gate_reshape = mul.args[:2]
    if not isinstance(silu, torch.fx.Node) or silu.target != _SILU:
        return None
    if not isinstance(gate_reshape, torch.fx.Node) or gate_reshape.target not in (
        _RESHAPE,
        _VIEW,
    ):
        return None
    if len(silu.args) != 1 or not isinstance(silu.args[0], torch.fx.Node):
        return None
    silu_reshape = silu.args[0]
    if silu_reshape.target not in (_RESHAPE, _VIEW):
        return None
    if not silu_reshape.args or not isinstance(silu_reshape.args[0], torch.fx.Node):
        return None
    if not gate_reshape.args or not isinstance(gate_reshape.args[0], torch.fx.Node):
        return None
    silu_mm = silu_reshape.args[0]
    gate_mm = gate_reshape.args[0]
    if silu_mm.target != _MM or gate_mm.target != _MM:
        return None
    if _sole_user(silu_mm) is not silu_reshape:
        return None
    if _sole_user(gate_mm) is not gate_reshape:
        return None
    if any(
        _node_dtype(node) != torch.bfloat16
        for node in (silu_mm, silu_reshape, silu, gate_mm, gate_reshape, mul)
    ):
        return None

    silu_mm_shape = _node_shape(silu_mm)
    gate_mm_shape = _node_shape(gate_mm)
    silu_shape = _node_shape(silu)
    gate_shape = _node_shape(gate_reshape)
    if (
        silu_mm_shape is None
        or len(silu_mm_shape) != 2
        or silu_mm_shape != gate_mm_shape
        or silu_shape is None
        or silu_shape != gate_shape
        or silu_shape != _node_shape(mul)
        or silu_shape[-1] != silu_mm_shape[-1]
    ):
        return None
    return silu_mm, silu_reshape, silu, gate_mm, gate_reshape


def fuse_f4_dense_swiglu_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse dense/shared-expert two-GEMM SwiGLU pointwise work."""
    del example_inputs
    num_fused = 0
    for mul in list(gm.graph.nodes):
        match = _match_f4_swiglu(mul)
        if match is None:
            continue
        silu_mm, silu_reshape, silu, gate_mm, gate_reshape = match
        output_shape = _node_shape(mul)
        assert output_shape is not None
        preserve_preactivation = any(user is not silu for user in silu_reshape.users)

        silu_body = _build_f4_silu_body(
            silu_mm,
            silu_reshape,
            silu,
            preserve_preactivation,
        )
        silu_body_name = _next_submodule_name(gm, "_coda_f4_silu_body")
        gm.add_module(silu_body_name, silu_body)
        with gm.graph.inserting_before(silu_mm):
            silu_body_ref = gm.graph.get_attr(silu_body_name)
            _tag_for_regional_inductor(silu_body_ref)
            silu_fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    silu_body_ref,
                    tuple(silu_mm.args),
                    dict(silu_mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            silu_2d_meta = _copy_meta_with_value_from(
                silu_mm, silu_mm, silu_reshape, silu
            )
            silu_fused.meta = _copy_meta(silu_mm, silu_reshape, silu)
            for key in ("val", "tensor_meta"):
                values = [silu_2d_meta[key]] if key in silu_2d_meta else []
                if preserve_preactivation and key in silu_2d_meta:
                    values.append(silu_2d_meta[key])
                if len(values) == 1 + int(preserve_preactivation):
                    silu_fused.meta[key] = tuple(values)
                else:
                    silu_fused.meta.pop(key, None)
            _tag_for_regional_inductor(silu_fused)
            silu_2d = gm.graph.call_function(operator.getitem, (silu_fused, 0))
            silu_2d.meta = silu_2d_meta
            _tag_for_regional_inductor(silu_2d)
            silu_output = gm.graph.call_function(
                _RESHAPE,
                (silu_2d, list(output_shape)),
            )
            silu_output.meta = _copy_meta(silu_mm, silu_reshape, silu)
            _tag_for_regional_inductor(silu_output)

            preactivation_output = None
            if preserve_preactivation:
                preactivation_2d = gm.graph.call_function(
                    operator.getitem,
                    (silu_fused, 1),
                )
                preactivation_2d.meta = silu_2d_meta
                _tag_for_regional_inductor(preactivation_2d)
                preactivation_output = gm.graph.call_function(
                    _RESHAPE,
                    (preactivation_2d, list(output_shape)),
                )
                preactivation_output.meta = _copy_meta(silu_mm, silu_reshape)
                _tag_for_regional_inductor(preactivation_output)

        gate_body = _build_f4_mul_body(
            gate_mm,
            gate_reshape,
            silu_2d,
            mul,
        )
        gate_body_name = _next_submodule_name(gm, "_coda_f4_mul_body")
        gm.add_module(gate_body_name, gate_body)
        with gm.graph.inserting_before(gate_mm):
            gate_body_ref = gm.graph.get_attr(gate_body_name)
            _tag_for_regional_inductor(gate_body_ref)
            gate_fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    gate_body_ref,
                    tuple(gate_mm.args) + (silu_2d,),
                    dict(gate_mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            gate_2d_meta = _copy_meta_with_value_from(gate_mm, gate_mm, gate_reshape)
            product_2d_meta = _copy_meta_with_value_from(
                gate_mm, silu, gate_mm, gate_reshape, mul
            )
            gate_fused.meta = _copy_meta(silu, gate_mm, gate_reshape, mul)
            for key in ("val", "tensor_meta"):
                values = [
                    meta[key] for meta in (gate_2d_meta, product_2d_meta) if key in meta
                ]
                if len(values) == 2:
                    gate_fused.meta[key] = tuple(values)
                else:
                    gate_fused.meta.pop(key, None)
            _tag_for_regional_inductor(gate_fused)

            gate_2d = gm.graph.call_function(operator.getitem, (gate_fused, 0))
            gate_2d.meta = gate_2d_meta
            _tag_for_regional_inductor(gate_2d)
            gate_output = gm.graph.call_function(
                _RESHAPE,
                (gate_2d, list(output_shape)),
            )
            gate_output.meta = _copy_meta(gate_mm, gate_reshape)
            _tag_for_regional_inductor(gate_output)

            product_2d = gm.graph.call_function(operator.getitem, (gate_fused, 1))
            product_2d.meta = product_2d_meta
            _tag_for_regional_inductor(product_2d)
            product = gm.graph.call_function(
                _RESHAPE,
                (product_2d, list(output_shape)),
            )
            product.meta = _copy_meta(silu, gate_mm, gate_reshape, mul)
            _tag_for_regional_inductor(product)

        silu.replace_all_uses_with(silu_output)
        if preactivation_output is not None:
            silu_reshape.replace_all_uses_with(preactivation_output)
        gate_reshape.replace_all_uses_with(gate_output)
        mul.replace_all_uses_with(product)
        gm.graph.erase_node(mul)
        gm.graph.erase_node(silu)
        gm.graph.erase_node(silu_reshape)
        gm.graph.erase_node(silu_mm)
        gm.graph.erase_node(gate_reshape)
        gm.graph.erase_node(gate_mm)
        num_fused += 1

    gm.graph.lint()
    gm.recompile()
    logger.info(f"F4 fused {num_fused} dense/shared-expert SwiGLU chains")
    return gm


def _match_b2_branch_derivatives(
    silu_grad: torch.fx.Node,
) -> tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
] | None:
    if silu_grad.target != _SILU_BACKWARD or len(silu_grad.args) < 2:
        return None
    gated_grad, saved_preactivation = silu_grad.args[:2]
    if not isinstance(gated_grad, torch.fx.Node) or gated_grad.target != _MUL:
        return None
    if not isinstance(saved_preactivation, torch.fx.Node):
        return None
    if _sole_user(gated_grad) is not silu_grad or len(gated_grad.args) < 2:
        return None
    grad_view, saved_gate = gated_grad.args[:2]
    if not isinstance(grad_view, torch.fx.Node) or grad_view.target not in (
        _RESHAPE,
        _VIEW,
    ):
        return None
    if not isinstance(saved_gate, torch.fx.Node):
        return None
    if not grad_view.args or not isinstance(grad_view.args[0], torch.fx.Node):
        return None
    mm = grad_view.args[0]
    if mm.target != _MM or _sole_user(mm) is not grad_view:
        return None

    sibling_grads = [
        user
        for user in grad_view.users
        if user is not gated_grad
        and user.target == _MUL
        and len(user.args) >= 2
        and user.args[0] is grad_view
        and isinstance(user.args[1], torch.fx.Node)
    ]
    if len(grad_view.users) != 2 or len(sibling_grads) != 1:
        return None
    gate_grad = sibling_grads[0]
    saved_silu = gate_grad.args[1]
    assert isinstance(saved_silu, torch.fx.Node)

    nodes = (
        mm,
        grad_view,
        gate_grad,
        gated_grad,
        silu_grad,
        saved_silu,
        saved_gate,
        saved_preactivation,
    )
    if any(_node_dtype(node) != torch.bfloat16 for node in nodes):
        return None
    mm_shape = _node_shape(mm)
    output_shape = _node_shape(grad_view)
    if (
        mm_shape is None
        or len(mm_shape) != 2
        or output_shape is None
        or output_shape[-1] != mm_shape[-1]
        or any(_node_shape(node) != output_shape for node in nodes[2:])
    ):
        return None
    return (
        mm,
        grad_view,
        gate_grad,
        gated_grad,
        silu_grad,
        saved_silu,
        saved_gate,
    )


def _match_b2_input_grad_add(
    add: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node,] | None:
    if add.target != _ADD or len(add.args) < 2:
        return None
    lhs, rhs = add.args[:2]
    if not all(
        isinstance(node, torch.fx.Node) and node.target in (_RESHAPE, _VIEW)
        for node in (lhs, rhs)
    ):
        return None
    assert isinstance(lhs, torch.fx.Node)
    assert isinstance(rhs, torch.fx.Node)
    if not lhs.args or not rhs.args:
        return None
    lhs_mm, rhs_mm = lhs.args[0], rhs.args[0]
    if not all(
        isinstance(node, torch.fx.Node) and node.target == _MM
        for node in (lhs_mm, rhs_mm)
    ):
        return None
    assert isinstance(lhs_mm, torch.fx.Node)
    assert isinstance(rhs_mm, torch.fx.Node)
    if _sole_user(lhs_mm) is not lhs or _sole_user(rhs_mm) is not rhs:
        return None
    if _sole_user(lhs) is not add or _sole_user(rhs) is not add:
        return None

    lhs_fqn = _module_fqn(lhs_mm)
    rhs_fqn = _module_fqn(rhs_mm)
    if lhs_fqn is None or rhs_fqn is None or "." not in lhs_fqn or "." not in rhs_fqn:
        return None
    lhs_parent, lhs_role = lhs_fqn.rsplit(".", 1)
    rhs_parent, rhs_role = rhs_fqn.rsplit(".", 1)
    if lhs_parent != rhs_parent or {lhs_role, rhs_role} != {"w1", "w3"}:
        return None
    if lhs_role == "w3":
        captured_mm, captured_reshape = lhs_mm, lhs
        fused_mm, fused_reshape = rhs_mm, rhs
    else:
        captured_mm, captured_reshape = rhs_mm, rhs
        fused_mm, fused_reshape = lhs_mm, lhs
    seen_captured_mm = False
    for node in add.graph.nodes:
        if node is captured_mm:
            seen_captured_mm = True
        if node is fused_mm:
            break
    if not seen_captured_mm:
        return None

    if any(
        _node_dtype(node) != torch.bfloat16
        for node in (captured_mm, captured_reshape, fused_mm, fused_reshape, add)
    ):
        return None
    if (
        _node_shape(captured_mm) != _node_shape(fused_mm)
        or _node_shape(captured_reshape) != _node_shape(fused_reshape)
        or _node_shape(captured_reshape) != _node_shape(add)
    ):
        return None
    return captured_mm, captured_reshape, fused_mm, fused_reshape


def fuse_b2_dense_swiglu_backward_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse dense/shared-expert SwiGLU backward GEMM epilogues."""
    del example_inputs
    num_branch_fused = 0
    for silu_grad in list(gm.graph.nodes):
        match = _match_b2_branch_derivatives(silu_grad)
        if match is None:
            continue
        (
            mm,
            grad_view,
            gate_grad,
            gated_grad,
            silu_grad,
            saved_silu,
            saved_gate,
        ) = match
        saved_preactivation = silu_grad.args[1]
        assert isinstance(saved_preactivation, torch.fx.Node)
        mm_shape = _node_shape(mm)
        output_shape = _node_shape(grad_view)
        assert mm_shape is not None
        assert output_shape is not None

        captures = []
        with gm.graph.inserting_before(mm):
            for saved in (saved_silu, saved_gate, saved_preactivation):
                captured = gm.graph.call_function(
                    _RESHAPE,
                    (saved, list(mm_shape)),
                )
                captured.meta = _copy_meta_with_value_from(mm, saved)
                _tag_for_regional_inductor(captured)
                captures.append(captured)

        body = _build_b2_branch_body(
            mm,
            grad_view,
            captures[0],
            captures[1],
            captures[2],
            gate_grad,
            silu_grad,
        )
        body_name = _next_submodule_name(gm, "_coda_b2_branch_body")
        gm.add_module(body_name, body)
        with gm.graph.inserting_before(mm):
            body_ref = gm.graph.get_attr(body_name)
            _tag_for_regional_inductor(body_ref)
            fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    body_ref,
                    tuple(mm.args) + tuple(captures),
                    dict(mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            gate_grad_2d_meta = _copy_meta_with_value_from(mm, mm, grad_view, gate_grad)
            silu_grad_2d_meta = _copy_meta_with_value_from(mm, mm, grad_view, silu_grad)
            fused.meta = _copy_meta(mm, grad_view, gate_grad, silu_grad)
            for key in ("val", "tensor_meta"):
                values = [
                    meta[key]
                    for meta in (gate_grad_2d_meta, silu_grad_2d_meta)
                    if key in meta
                ]
                if len(values) == 2:
                    fused.meta[key] = tuple(values)
                else:
                    fused.meta.pop(key, None)
            _tag_for_regional_inductor(fused)

            gate_grad_2d = gm.graph.call_function(operator.getitem, (fused, 0))
            gate_grad_2d.meta = gate_grad_2d_meta
            _tag_for_regional_inductor(gate_grad_2d)
            gate_grad_output = gm.graph.call_function(
                _RESHAPE,
                (gate_grad_2d, list(output_shape)),
            )
            gate_grad_output.meta = _copy_meta(mm, grad_view, gate_grad)
            _tag_for_regional_inductor(gate_grad_output)

            silu_grad_2d = gm.graph.call_function(operator.getitem, (fused, 1))
            silu_grad_2d.meta = silu_grad_2d_meta
            _tag_for_regional_inductor(silu_grad_2d)
            silu_grad_output = gm.graph.call_function(
                _RESHAPE,
                (silu_grad_2d, list(output_shape)),
            )
            silu_grad_output.meta = _copy_meta(mm, grad_view, silu_grad)
            _tag_for_regional_inductor(silu_grad_output)

        gate_grad.replace_all_uses_with(gate_grad_output)
        silu_grad.replace_all_uses_with(silu_grad_output)
        gm.graph.erase_node(silu_grad)
        gm.graph.erase_node(gated_grad)
        gm.graph.erase_node(gate_grad)
        gm.graph.erase_node(grad_view)
        gm.graph.erase_node(mm)
        num_branch_fused += 1

    num_input_add_fused = 0
    for add in list(gm.graph.nodes):
        match = _match_b2_input_grad_add(add)
        if match is None:
            continue
        captured_mm, captured_reshape, fused_mm, fused_reshape = match
        output_shape = _node_shape(add)
        assert output_shape is not None

        body = _build_b2_input_add_body(
            fused_mm,
            fused_reshape,
            captured_mm,
            add,
        )
        body_name = _next_submodule_name(gm, "_coda_b2_input_add_body")
        gm.add_module(body_name, body)
        with gm.graph.inserting_before(fused_mm):
            body_ref = gm.graph.get_attr(body_name)
            _tag_for_regional_inductor(body_ref)
            fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    body_ref,
                    tuple(fused_mm.args) + (captured_mm,),
                    dict(fused_mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            total_2d_meta = _copy_meta_with_value_from(
                fused_mm, captured_reshape, fused_mm, fused_reshape, add
            )
            fused.meta = _copy_meta(captured_reshape, fused_mm, fused_reshape, add)
            for key in ("val", "tensor_meta"):
                if key in total_2d_meta:
                    fused.meta[key] = (total_2d_meta[key],)
                else:
                    fused.meta.pop(key, None)
            _tag_for_regional_inductor(fused)
            total_2d = gm.graph.call_function(operator.getitem, (fused, 0))
            total_2d.meta = total_2d_meta
            _tag_for_regional_inductor(total_2d)
            total = gm.graph.call_function(
                _RESHAPE,
                (total_2d, list(output_shape)),
            )
            total.meta = _copy_meta(captured_reshape, fused_reshape, add)
            _tag_for_regional_inductor(total)

        add.replace_all_uses_with(total)
        gm.graph.erase_node(add)
        gm.graph.erase_node(captured_reshape)
        gm.graph.erase_node(fused_reshape)
        gm.graph.erase_node(fused_mm)
        num_input_add_fused += 1

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"B2 fused {num_branch_fused} branch-derivative GEMMs and "
        f"{num_input_add_fused} input-gradient GEMM adds"
    )
    return gm


def _match_b4_router_input_grad_add(
    add: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node,] | None:
    if add.target != _ADD or len(add.args) < 2:
        return None
    lhs, rhs = add.args[:2]
    cast = next(
        (
            node
            for node in (lhs, rhs)
            if isinstance(node, torch.fx.Node) and _is_cast(node, torch.bfloat16)
        ),
        None,
    )
    if cast is None:
        return None
    residual = rhs if cast is lhs else lhs
    if not isinstance(residual, torch.fx.Node):
        return None
    if _sole_user(cast) is not add or not cast.args:
        return None
    reshape = cast.args[0]
    if not isinstance(reshape, torch.fx.Node) or reshape.target not in (
        _RESHAPE,
        _VIEW,
    ):
        return None
    if _sole_user(reshape) is not cast or not reshape.args:
        return None
    mm = reshape.args[0]
    if not isinstance(mm, torch.fx.Node) or mm.target != _MM:
        return None
    if _sole_user(mm) is not reshape:
        return None

    module_fqn = _module_fqn(mm)
    if module_fqn is None or not module_fqn.endswith(".moe.router.gate"):
        return None
    mm_shape = _node_shape(mm)
    output_shape = _node_shape(add)
    if (
        mm_shape is None
        or len(mm_shape) != 2
        or output_shape is None
        or output_shape[-1] != mm_shape[-1]
        or _node_shape(reshape) != output_shape
        or _node_shape(cast) != output_shape
        or _node_shape(residual) != output_shape
    ):
        return None
    if _node_dtype(mm) != torch.float32 or _node_dtype(reshape) != torch.float32:
        return None
    if any(_node_dtype(node) != torch.bfloat16 for node in (cast, residual, add)):
        return None
    if not all(
        isinstance(node.meta.get("val"), torch.Tensor)
        for node in (mm, cast, residual, add)
    ):
        return None
    return mm, reshape, cast, residual


def fuse_b4_router_input_grad_add_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Fuse the router input-gradient cast and expert-gradient add."""
    del example_inputs
    num_fused = 0
    for add in list(gm.graph.nodes):
        match = _match_b4_router_input_grad_add(add)
        if match is None:
            continue
        mm, reshape, cast, residual = match
        mm_shape = _node_shape(mm)
        output_shape = _node_shape(add)
        assert mm_shape is not None
        assert output_shape is not None
        residual_value = residual.meta["val"]
        cast_value = cast.meta["val"]
        assert isinstance(residual_value, torch.Tensor)
        assert isinstance(cast_value, torch.Tensor)

        with gm.graph.inserting_before(mm):
            residual_2d = gm.graph.call_function(
                _RESHAPE,
                (residual, list(mm_shape)),
            )
            residual_2d.meta = _copy_meta_with_value(
                residual_value.reshape(mm_shape), residual
            )
            _tag_for_regional_inductor(residual_2d)
            body = _build_b4_router_input_grad_body(mm, cast, residual_2d, add)
            body_name = _next_submodule_name(gm, "_coda_b4_router_body")
            gm.add_module(body_name, body)
            body_ref = gm.graph.get_attr(body_name)
            _tag_for_regional_inductor(body_ref)
            fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    body_ref,
                    tuple(mm.args) + (residual_2d,),
                    dict(mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            total_2d_meta = _copy_meta_with_value(
                cast_value.reshape(mm_shape), mm, reshape, cast, residual, add
            )
            fused.meta = _copy_meta(mm, reshape, cast, residual, add)
            fused.meta["val"] = (total_2d_meta["val"],)
            fused.meta.pop("tensor_meta", None)
            _tag_for_regional_inductor(fused)
            total_2d = gm.graph.call_function(operator.getitem, (fused, 0))
            total_2d.meta = total_2d_meta
            _tag_for_regional_inductor(total_2d)
            total = gm.graph.call_function(
                _RESHAPE,
                (total_2d, list(output_shape)),
            )
            total.meta = _copy_meta(mm, reshape, cast, residual, add)
            _tag_for_regional_inductor(total)

        add.replace_all_uses_with(total)
        gm.graph.erase_node(add)
        gm.graph.erase_node(cast)
        gm.graph.erase_node(reshape)
        gm.graph.erase_node(mm)
        num_fused += 1

    gm.graph.lint()
    gm.recompile()
    logger.info(f"B4 fused {num_fused} router input-gradient cast/add chains")
    return gm


def _layer_module_role(module_fqn: str | None) -> tuple[int, str] | None:
    if module_fqn is None:
        return None
    parts = module_fqn.split(".")
    if len(parts) < 3 or parts[0] != "layers" or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def _f3_expected_second_roles(
    first_mm: torch.fx.Node,
    norm: torch.fx.Node,
) -> tuple[int, set[str]] | None:
    first = _layer_module_role(_module_fqn(first_mm))
    normalized = _layer_module_role(_module_fqn(norm))
    if first is None or normalized is None:
        return None
    first_layer, first_role = first
    norm_layer, norm_role = normalized
    if (
        first_role == "attention.wo"
        and norm_layer == first_layer
        and norm_role == "ffn_norm"
    ):
        return first_layer, {"feed_forward.w1", "feed_forward.w3"}
    if (
        first_role == "feed_forward.w2"
        and norm_layer == first_layer + 1
        and norm_role == "attention_norm"
    ):
        return norm_layer, {"attention.wq_a", "attention.wkv_a"}
    if (
        first_role == "moe.shared_experts.w2"
        and norm_layer == first_layer + 1
        and norm_role == "attention_norm"
    ):
        return norm_layer, {"attention.wq_a", "attention.wkv_a"}
    return None


def _match_f3_add_tree(
    add: torch.fx.Node,
    norm: torch.fx.Node,
    depth: int = 1,
) -> list[
    tuple[
        torch.fx.Node,
        torch.fx.Node,
        tuple[torch.fx.Node, ...],
        tuple[bool, ...],
        tuple[torch.fx.Node, ...],
    ]
]:
    if add.target != _ADD or len(add.args) < 2 or add.kwargs.get("alpha", 1) != 1:
        return []

    matches = []
    for index, operand in enumerate(add.args[:2]):
        sibling = add.args[1 - index]
        if not isinstance(operand, torch.fx.Node) or not isinstance(
            sibling, torch.fx.Node
        ):
            continue
        if (
            operand.target in (_RESHAPE, _VIEW)
            and operand.args
            and isinstance(operand.args[0], torch.fx.Node)
        ):
            candidate_mm = operand.args[0]
            if (
                candidate_mm.target == _MM
                and _sole_user(candidate_mm) is operand
                and _sole_user(operand) is add
                and _f3_expected_second_roles(candidate_mm, norm) is not None
            ):
                matches.append(
                    (
                        candidate_mm,
                        operand,
                        (sibling,),
                        (index == 0,),
                        (add,),
                    )
                )
        if depth == 1 or operand.target != _ADD or _sole_user(operand) is not add:
            continue
        for candidate_mm, reshape, addends, orders, add_nodes in _match_f3_add_tree(
            operand, norm, depth - 1
        ):
            matches.append(
                (
                    candidate_mm,
                    reshape,
                    (*addends, sibling),
                    (*orders, index == 0),
                    (*add_nodes, add),
                )
            )
    return matches


def _match_f3_residual_rmsnorm(
    norm: torch.fx.Node,
) -> tuple[
    torch.fx.Node,
    torch.fx.Node,
    tuple[torch.fx.Node, ...],
    tuple[bool, ...],
    tuple[torch.fx.Node, ...],
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
    list[tuple[torch.fx.Node, torch.fx.Node]],
    float,
] | None:
    if norm.target != _FUSED_RMS_NORM or len(norm.args) < 4:
        return None
    add, normalized_shape, gamma, eps = norm.args[:4]
    if (
        not isinstance(add, torch.fx.Node)
        or add.target != _ADD
        or len(add.args) < 2
        or add.kwargs.get("alpha", 1) != 1
        or not isinstance(gamma, torch.fx.Node)
        or not isinstance(normalized_shape, (list, tuple))
        or len(normalized_shape) != 1
        or not isinstance(eps, float)
    ):
        return None

    producer_matches = _match_f3_add_tree(add, norm, depth=2)
    if len(producer_matches) != 1:
        return None
    (
        first_mm,
        first_reshape,
        addends,
        accumulated_value_is_lhs,
        add_nodes,
    ) = producer_matches[0]

    norm_outputs: dict[int, torch.fx.Node] = {}
    for user in norm.users:
        if (
            user.target is not operator.getitem
            or len(user.args) < 2
            or not isinstance(user.args[1], int)
            or user.args[1] in norm_outputs
        ):
            return None
        norm_outputs[user.args[1]] = user
    if set(norm_outputs) not in ({0}, {0, 1}):
        return None
    norm_output = norm_outputs[0]
    rstd_output = norm_outputs.get(1)
    is_recomputed = rstd_output is not None

    second_pairs = []
    for second_input in norm_output.users:
        if (
            second_input.target not in (_RESHAPE, _VIEW)
            or not second_input.args
            or second_input.args[0] is not norm_output
        ):
            return None
        second_mms = [
            user
            for user in second_input.users
            if user.target == _MM and user.args and user.args[0] is second_input
        ]
        if len(second_mms) != 1:
            return None
        second_mm = second_mms[0]
        if not is_recomputed and _sole_user(second_input) is not second_mm:
            return None
        second_pairs.append((second_input, second_mm))

    expected = _f3_expected_second_roles(first_mm, norm)
    assert expected is not None
    second_layer, expected_roles = expected
    actual_roles = set()
    for _, second_mm in second_pairs:
        second = _layer_module_role(_module_fqn(second_mm))
        if second is None or second[0] != second_layer:
            return None
        actual_roles.add(second[1])
    if actual_roles != expected_roles or len(second_pairs) != len(expected_roles):
        return None

    first_shape = _node_shape(first_mm)
    output_shape = _node_shape(add)
    if (
        first_shape is None
        or len(first_shape) != 2
        or output_shape is None
        or output_shape[-1] != first_shape[-1]
        or first_shape[-1] != normalized_shape[0]
        or first_shape[-1] % _CODA_RMSNORM_GROUP != 0
        or _node_shape(first_reshape) != output_shape
        or any(_node_shape(addend) != output_shape for addend in addends)
        or _node_shape(norm_output) != output_shape
        or _node_shape(gamma) != (first_shape[-1],)
        or any(
            _node_shape(second_input) != first_shape for second_input, _ in second_pairs
        )
    ):
        return None
    if rstd_output is not None and (
        _node_dtype(rstd_output) != torch.float32
        or _node_shape(rstd_output) != (*output_shape[:-1], 1)
    ):
        return None
    tensor_nodes = (
        first_mm,
        first_reshape,
        *addends,
        *add_nodes,
        gamma,
        norm_output,
        *(node for pair in second_pairs for node in pair),
    )
    if any(_node_dtype(node) != torch.bfloat16 for node in tensor_nodes):
        return None
    if not all(
        isinstance(node.meta.get("val"), torch.Tensor)
        for node in (first_mm, *addends, gamma, *(mm for _, mm in second_pairs))
    ):
        return None
    return (
        first_mm,
        first_reshape,
        addends,
        accumulated_value_is_lhs,
        add_nodes,
        norm,
        norm_output,
        gamma,
        rstd_output,
        second_pairs,
        eps,
    )


def fuse_f3_residual_rmsnorm_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Reparameterize residual RMSNorm across its neighboring GEMMs."""
    del example_inputs
    num_original_fused = 0
    num_recomputed_fused = 0
    num_projection_fused = 0
    for norm in list(gm.graph.nodes):
        match = _match_f3_residual_rmsnorm(norm)
        if match is None:
            continue
        (
            first_mm,
            first_reshape,
            addends,
            accumulated_value_is_lhs,
            add_nodes,
            norm,
            norm_output,
            gamma,
            rstd_output,
            second_pairs,
            eps,
        ) = match
        add = add_nodes[-1]
        first_shape = _node_shape(first_mm)
        output_shape = _node_shape(add)
        assert first_shape is not None
        assert output_shape is not None
        first_value = first_mm.meta["val"]
        addend_values = tuple(addend.meta["val"] for addend in addends)
        gamma_value = gamma.meta["val"]
        assert isinstance(first_value, torch.Tensor)
        assert all(isinstance(value, torch.Tensor) for value in addend_values)
        assert isinstance(gamma_value, torch.Tensor)
        preserve_saved_values = rstd_output is not None

        # Bucketing may materialize the next layer's RMSNorm weight after the
        # producing GEMM. Insert at the norm boundary where every capture is
        # guaranteed to dominate the new HOP.
        with gm.graph.inserting_before(norm):
            addends_2d = []
            addend_values_2d = []
            for addend, addend_value in zip(addends, addend_values, strict=True):
                addend_2d = gm.graph.call_function(
                    _RESHAPE,
                    (addend, list(first_shape)),
                )
                addend_value_2d = addend_value.reshape(first_shape)
                addend_2d.meta = _copy_meta_with_value(
                    addend_value_2d, addend, *add_nodes
                )
                _tag_for_regional_inductor(addend_2d)
                addends_2d.append(addend_2d)
                addend_values_2d.append(addend_value_2d)
            gamma_2d = gm.graph.call_function(
                _RESHAPE,
                (gamma, [1, first_shape[-1]]),
            )
            gamma_2d_value = gamma_value.reshape(1, first_shape[-1])
            gamma_2d.meta = _copy_meta_with_value(gamma_2d_value, gamma)
            _tag_for_regional_inductor(gamma_2d)

            first_body = _build_f3_residual_rmsnorm_first_body(
                first_mm,
                tuple(addends_2d),
                gamma_2d,
                _CODA_RMSNORM_GROUP,
                accumulated_value_is_lhs,
            )
            first_body_name = _next_submodule_name(gm, "_coda_f3_residual_first_body")
            gm.add_module(first_body_name, first_body)
            first_body_ref = gm.graph.get_attr(first_body_name)
            _tag_for_regional_inductor(first_body_ref)
            first_fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    first_body_ref,
                    tuple(first_mm.args) + (*addends_2d, gamma_2d),
                    dict(first_mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            total_value = first_value
            for addend_value, value_is_lhs in zip(
                addend_values_2d, accumulated_value_is_lhs, strict=True
            ):
                total_value = (
                    total_value + addend_value
                    if value_is_lhs
                    else addend_value + total_value
                )
            weighted_value = (total_value.float() * gamma_2d_value).to(torch.bfloat16)
            partial_value = (
                total_value.float()
                .reshape(first_shape[0], -1, _CODA_RMSNORM_GROUP)
                .square()
                .mean(-1)
            )
            weighted_meta = _copy_meta_with_value(
                weighted_value,
                first_mm,
                first_reshape,
                *addends,
                *add_nodes,
                norm,
                norm_output,
                gamma,
            )
            total_meta = _copy_meta_with_value(
                total_value, first_mm, first_reshape, *addends, *add_nodes
            )
            partial_meta = _copy_meta_with_value(
                partial_value,
                first_mm,
                first_reshape,
                *addends,
                *add_nodes,
                norm,
            )
            first_fused.meta = _copy_meta(
                first_mm,
                first_reshape,
                *addends,
                *add_nodes,
                norm,
                norm_output,
            )
            first_fused.meta["val"] = (
                weighted_meta["val"],
                total_meta["val"],
                partial_meta["val"],
            )
            first_fused.meta.pop("tensor_meta", None)
            _tag_for_regional_inductor(first_fused)

            weighted = gm.graph.call_function(operator.getitem, (first_fused, 0))
            weighted.meta = weighted_meta
            _tag_for_regional_inductor(weighted)
            total_2d = gm.graph.call_function(operator.getitem, (first_fused, 1))
            total_2d.meta = total_meta
            _tag_for_regional_inductor(total_2d)
            partial = gm.graph.call_function(operator.getitem, (first_fused, 2))
            partial.meta = partial_meta
            _tag_for_regional_inductor(partial)
            total = gm.graph.call_function(
                _RESHAPE,
                (total_2d, list(output_shape)),
            )
            total.meta = _copy_meta_with_value(total_value.reshape(output_shape), add)
            _tag_for_regional_inductor(total)
            mean_square = gm.graph.call_function(_MEAN_DIM, (partial, [-1], True))
            mean_square_value = partial_value.mean(-1, keepdim=True)
            mean_square.meta = _copy_meta_with_value(
                mean_square_value, first_mm, add, norm
            )
            _tag_for_regional_inductor(mean_square)
            stabilized = gm.graph.call_function(_ADD_SCALAR, (mean_square, eps))
            stabilized_value = mean_square_value + eps
            stabilized.meta = _copy_meta_with_value(
                stabilized_value, first_mm, add, norm
            )
            _tag_for_regional_inductor(stabilized)
            rstd = gm.graph.call_function(_RSQRT, (stabilized,))
            rstd_value = stabilized_value.rsqrt()
            rstd.meta = _copy_meta_with_value(rstd_value, first_mm, add, norm)
            _tag_for_regional_inductor(rstd)

            saved_norm_output = None
            saved_rstd_output = None
            if preserve_saved_values:
                assert rstd_output is not None
                normalized_fp32 = gm.graph.call_function(
                    _MUL,
                    (total_2d, rstd),
                )
                normalized_value = total_value.float() * rstd_value
                normalized_fp32.meta = _copy_meta_with_value(
                    normalized_value, add, norm, norm_output
                )
                _tag_for_regional_inductor(normalized_fp32)
                normalized_weighted_fp32 = gm.graph.call_function(
                    _MUL,
                    (normalized_fp32, gamma_2d),
                )
                normalized_weighted_value = normalized_value * gamma_2d_value
                normalized_weighted_fp32.meta = _copy_meta_with_value(
                    normalized_weighted_value, add, norm, norm_output, gamma
                )
                _tag_for_regional_inductor(normalized_weighted_fp32)
                saved_norm_output = gm.graph.call_function(
                    _TO_COPY,
                    (normalized_weighted_fp32,),
                    {"dtype": torch.bfloat16},
                )
                saved_norm_output.meta = _copy_meta_with_value(
                    normalized_weighted_value.to(torch.bfloat16), norm_output
                )
                _tag_for_regional_inductor(saved_norm_output)
                rstd_output_shape = _node_shape(rstd_output)
                assert rstd_output_shape is not None
                saved_rstd_output = gm.graph.call_function(
                    _RESHAPE,
                    (rstd, list(rstd_output_shape)),
                )
                saved_rstd_output.meta = _copy_meta_with_value(
                    rstd_value.reshape(rstd_output_shape), rstd_output
                )
                _tag_for_regional_inductor(saved_rstd_output)

        for second_input, second_mm in second_pairs:
            second_value = second_mm.meta["val"]
            assert isinstance(second_value, torch.Tensor)
            second_body = _build_f2_q_rmsnorm_second_body(second_mm, rstd)
            second_body_name = _next_submodule_name(gm, "_coda_f3_residual_second_body")
            gm.add_module(second_body_name, second_body)
            with gm.graph.inserting_before(second_mm):
                second_body_ref = gm.graph.get_attr(second_body_name)
                _tag_for_regional_inductor(second_body_ref)
                second_fused = gm.graph.call_function(
                    flex_gemm_hop,
                    (
                        _MM,
                        second_body_ref,
                        (weighted, second_mm.args[1], rstd),
                        dict(second_mm.kwargs),
                        {"backend": "QUACK"},
                    ),
                )
                second_meta = _copy_meta_with_value(
                    second_value, norm, norm_output, second_input, second_mm
                )
                second_fused.meta = _copy_meta(
                    norm, norm_output, second_input, second_mm
                )
                second_fused.meta["val"] = (second_meta["val"],)
                second_fused.meta.pop("tensor_meta", None)
                _tag_for_regional_inductor(second_fused)
                output = gm.graph.call_function(operator.getitem, (second_fused, 0))
                output.meta = second_meta
                _tag_for_regional_inductor(output)

            second_mm.replace_all_uses_with(output)
            gm.graph.erase_node(second_mm)
            if preserve_saved_values:
                assert saved_norm_output is not None
                second_input.replace_all_uses_with(saved_norm_output)
            gm.graph.erase_node(second_input)
            num_projection_fused += 1

        add.replace_all_uses_with(total)
        if preserve_saved_values:
            assert rstd_output is not None
            assert saved_rstd_output is not None
            rstd_output.replace_all_uses_with(saved_rstd_output)
            gm.graph.erase_node(rstd_output)
            num_recomputed_fused += 1
        else:
            num_original_fused += 1
        gm.graph.erase_node(norm_output)
        gm.graph.erase_node(norm)
        for add_node in reversed(add_nodes):
            gm.graph.erase_node(add_node)
        gm.graph.erase_node(first_reshape)
        gm.graph.erase_node(first_mm)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"F3 fused {num_original_fused} original and "
        f"{num_recomputed_fused} recomputed residual RMSNorm boundaries "
        f"across {num_projection_fused} downstream projections"
    )
    return gm


def _match_f2_q_rmsnorm(
    norm: torch.fx.Node,
) -> tuple[
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node,
    torch.fx.Node | None,
    float,
] | None:
    if norm.target != _FUSED_RMS_NORM or len(norm.args) < 4:
        return None
    norm_input, normalized_shape, gamma, eps = norm.args[:4]
    if (
        not isinstance(norm_input, torch.fx.Node)
        or norm_input.target not in (_RESHAPE, _VIEW)
        or not isinstance(gamma, torch.fx.Node)
        or not isinstance(normalized_shape, (list, tuple))
        or len(normalized_shape) != 1
        or not isinstance(eps, float)
    ):
        return None
    if not norm_input.args or not isinstance(norm_input.args[0], torch.fx.Node):
        return None
    first_mm = norm_input.args[0]
    if first_mm.target != _MM or _sole_user(first_mm) is not norm_input:
        return None

    norm_outputs: dict[int, torch.fx.Node] = {}
    for user in norm.users:
        if (
            user.target is not operator.getitem
            or len(user.args) < 2
            or not isinstance(user.args[1], int)
            or user.args[1] in norm_outputs
        ):
            return None
        norm_outputs[user.args[1]] = user
    if set(norm_outputs) not in ({0}, {0, 1}):
        return None
    norm_output = norm_outputs[0]
    rstd_output = norm_outputs.get(1)
    second_input = _sole_user(norm_output)
    if second_input is None or second_input.target not in (_RESHAPE, _VIEW):
        return None
    second_mms = [
        user
        for user in second_input.users
        if user.target == _MM and user.args and user.args[0] is second_input
    ]
    if len(second_mms) != 1:
        return None
    second_mm = second_mms[0]

    is_recomputed = rstd_output is not None
    if not is_recomputed and (
        _sole_user(norm_input) is not norm or _sole_user(second_input) is not second_mm
    ):
        return None
    if rstd_output is not None:
        rstd_shape = _node_shape(rstd_output)
        norm_input_shape = _node_shape(norm_input)
        if (
            _node_dtype(rstd_output) != torch.float32
            or norm_input_shape is None
            or rstd_shape != (*norm_input_shape[:-1], 1)
        ):
            return None

    fqns = tuple(_module_fqn(node) for node in (first_mm, norm, second_mm))
    if any(fqn is None or "." not in fqn for fqn in fqns):
        return None
    first_fqn, norm_fqn, second_fqn = fqns
    assert first_fqn is not None
    assert norm_fqn is not None
    assert second_fqn is not None
    first_parent, first_role = first_fqn.rsplit(".", 1)
    norm_parent, norm_role = norm_fqn.rsplit(".", 1)
    second_parent, second_role = second_fqn.rsplit(".", 1)
    if (
        first_parent != norm_parent
        or first_parent != second_parent
        or (first_role, norm_role, second_role) != ("wq_a", "q_norm", "wq_b")
    ):
        return None

    first_shape = _node_shape(first_mm)
    norm_input_shape = _node_shape(norm_input)
    second_input_shape = _node_shape(second_input)
    gamma_shape = _node_shape(gamma)
    if (
        first_shape is None
        or len(first_shape) != 2
        or norm_input_shape is None
        or first_shape[-1] != normalized_shape[0]
        or first_shape[-1] % _CODA_RMSNORM_GROUP != 0
        or norm_input_shape[-1:] != first_shape[-1:]
        or _node_shape(norm_output) != _node_shape(norm_input)
        or second_input_shape != first_shape
        or gamma_shape != (first_shape[-1],)
    ):
        return None
    tensor_nodes = (
        first_mm,
        norm_input,
        gamma,
        norm_output,
        second_input,
        second_mm,
    )
    if any(_node_dtype(node) != torch.bfloat16 for node in tensor_nodes):
        return None
    if not all(
        isinstance(node.meta.get("val"), torch.Tensor)
        for node in (first_mm, gamma, second_mm)
    ):
        return None
    return (
        first_mm,
        norm_input,
        norm,
        norm_output,
        second_input,
        second_mm,
        gamma,
        rstd_output,
        eps,
    )


def fuse_f2_q_rmsnorm_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Reparameterize MLA Q RMSNorm across two GEMMs."""
    del example_inputs
    num_original_fused = 0
    num_recomputed_fused = 0
    for norm in list(gm.graph.nodes):
        match = _match_f2_q_rmsnorm(norm)
        if match is None:
            continue
        (
            first_mm,
            norm_input,
            norm,
            norm_output,
            second_input,
            second_mm,
            gamma,
            rstd_output,
            eps,
        ) = match
        first_shape = _node_shape(first_mm)
        assert first_shape is not None
        first_value = first_mm.meta["val"]
        gamma_value = gamma.meta["val"]
        second_value = second_mm.meta["val"]
        assert isinstance(first_value, torch.Tensor)
        assert isinstance(gamma_value, torch.Tensor)
        assert isinstance(second_value, torch.Tensor)
        preserve_saved_values = rstd_output is not None

        with gm.graph.inserting_before(first_mm):
            gamma_2d = gm.graph.call_function(
                _RESHAPE,
                (gamma, [1, first_shape[-1]]),
            )
            gamma_2d.meta = _copy_meta_with_value(
                gamma_value.reshape(1, first_shape[-1]), gamma
            )
            _tag_for_regional_inductor(gamma_2d)

            first_body = _build_f2_q_rmsnorm_first_body(
                first_mm,
                gamma_2d,
                _CODA_RMSNORM_GROUP,
                preserve_saved_values,
            )
            first_body_name = _next_submodule_name(gm, "_coda_f2_q_first_body")
            gm.add_module(first_body_name, first_body)
            first_body_ref = gm.graph.get_attr(first_body_name)
            _tag_for_regional_inductor(first_body_ref)
            first_fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    first_body_ref,
                    tuple(first_mm.args) + (gamma_2d,),
                    dict(first_mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            weighted_meta = _copy_meta_with_value_from(
                first_mm, first_mm, norm_input, norm, norm_output, gamma
            )
            partial_value = (
                first_value.float()
                .reshape(first_shape[0], -1, _CODA_RMSNORM_GROUP)
                .square()
                .mean(-1)
            )
            partial_meta = _copy_meta_with_value(
                partial_value, first_mm, norm_input, norm
            )
            first_fused.meta = _copy_meta(first_mm, norm_input, norm, norm_output)
            raw_meta = _copy_meta_with_value_from(first_mm, first_mm, norm_input, norm)
            first_fused_values = [weighted_meta["val"]]
            if preserve_saved_values:
                first_fused_values.append(raw_meta["val"])
            first_fused_values.append(partial_meta["val"])
            first_fused.meta["val"] = tuple(first_fused_values)
            first_fused.meta.pop("tensor_meta", None)
            _tag_for_regional_inductor(first_fused)

            weighted = gm.graph.call_function(operator.getitem, (first_fused, 0))
            weighted.meta = weighted_meta
            _tag_for_regional_inductor(weighted)
            raw = None
            partial_index = 1
            if preserve_saved_values:
                raw = gm.graph.call_function(operator.getitem, (first_fused, 1))
                raw.meta = raw_meta
                _tag_for_regional_inductor(raw)
                partial_index = 2
            partial = gm.graph.call_function(
                operator.getitem, (first_fused, partial_index)
            )
            partial.meta = partial_meta
            _tag_for_regional_inductor(partial)
            mean_square = gm.graph.call_function(_MEAN_DIM, (partial, [-1], True))
            mean_square_value = partial_value.mean(-1, keepdim=True)
            mean_square.meta = _copy_meta_with_value(mean_square_value, first_mm, norm)
            _tag_for_regional_inductor(mean_square)
            stabilized = gm.graph.call_function(_ADD_SCALAR, (mean_square, eps))
            stabilized_value = mean_square_value + eps
            stabilized.meta = _copy_meta_with_value(stabilized_value, first_mm, norm)
            _tag_for_regional_inductor(stabilized)
            rstd = gm.graph.call_function(_RSQRT, (stabilized,))
            rstd_value = stabilized_value.rsqrt()
            rstd.meta = _copy_meta_with_value(rstd_value, first_mm, norm)
            _tag_for_regional_inductor(rstd)

            saved_norm_input = None
            saved_norm_output = None
            saved_rstd_output = None
            if preserve_saved_values:
                assert raw is not None
                assert rstd_output is not None
                norm_input_shape = _node_shape(norm_input)
                rstd_output_shape = _node_shape(rstd_output)
                assert norm_input_shape is not None
                assert rstd_output_shape is not None

                raw_fp32 = gm.graph.call_function(
                    _TO_COPY, (raw,), {"dtype": torch.float32}
                )
                raw_fp32.meta = _copy_meta_with_dtype(
                    first_mm, torch.float32, first_mm, norm_input, norm
                )
                _tag_for_regional_inductor(raw_fp32)
                normalized_fp32 = gm.graph.call_function(_MUL, (raw_fp32, rstd))
                normalized_fp32_value = first_value.float() * rstd_value
                normalized_fp32.meta = _copy_meta_with_value(
                    normalized_fp32_value, first_mm, norm_input, norm
                )
                _tag_for_regional_inductor(normalized_fp32)
                normalized_weighted_fp32 = gm.graph.call_function(
                    _MUL, (normalized_fp32, gamma_2d)
                )
                normalized_weighted_value = normalized_fp32_value * gamma_value.reshape(
                    1, first_shape[-1]
                )
                normalized_weighted_fp32.meta = _copy_meta_with_value(
                    normalized_weighted_value, first_mm, norm_input, norm, gamma
                )
                _tag_for_regional_inductor(normalized_weighted_fp32)
                saved_norm_output = gm.graph.call_function(
                    _TO_COPY,
                    (normalized_weighted_fp32,),
                    {"dtype": torch.bfloat16},
                )
                saved_norm_output.meta = _copy_meta_with_value(
                    normalized_weighted_value.to(torch.bfloat16),
                    norm_output,
                    second_input,
                )
                _tag_for_regional_inductor(saved_norm_output)
                saved_norm_input = gm.graph.call_function(
                    _RESHAPE, (raw, list(norm_input_shape))
                )
                saved_norm_input.meta = _copy_meta_with_value(
                    first_value.reshape(norm_input_shape), norm_input
                )
                _tag_for_regional_inductor(saved_norm_input)
                saved_rstd_output = gm.graph.call_function(
                    _RESHAPE, (rstd, list(rstd_output_shape))
                )
                saved_rstd_output.meta = _copy_meta_with_value(
                    rstd_value.reshape(rstd_output_shape), rstd_output
                )
                _tag_for_regional_inductor(saved_rstd_output)

        second_body = _build_f2_q_rmsnorm_second_body(second_mm, rstd)
        second_body_name = _next_submodule_name(gm, "_coda_f2_q_second_body")
        gm.add_module(second_body_name, second_body)
        with gm.graph.inserting_before(second_mm):
            second_body_ref = gm.graph.get_attr(second_body_name)
            _tag_for_regional_inductor(second_body_ref)
            second_fused = gm.graph.call_function(
                flex_gemm_hop,
                (
                    _MM,
                    second_body_ref,
                    (weighted, second_mm.args[1], rstd),
                    dict(second_mm.kwargs),
                    {"backend": "QUACK"},
                ),
            )
            second_meta = _copy_meta_with_value(
                second_value, norm, norm_output, second_input, second_mm
            )
            second_fused.meta = _copy_meta(norm, norm_output, second_input, second_mm)
            second_fused.meta["val"] = (second_meta["val"],)
            second_fused.meta.pop("tensor_meta", None)
            _tag_for_regional_inductor(second_fused)
            output = gm.graph.call_function(operator.getitem, (second_fused, 0))
            output.meta = second_meta
            _tag_for_regional_inductor(output)

        second_mm.replace_all_uses_with(output)
        gm.graph.erase_node(second_mm)
        if preserve_saved_values:
            assert saved_norm_input is not None
            assert saved_norm_output is not None
            assert saved_rstd_output is not None
            assert rstd_output is not None
            second_input.replace_all_uses_with(saved_norm_output)
            norm_input.replace_all_uses_with(saved_norm_input)
            rstd_output.replace_all_uses_with(saved_rstd_output)
            gm.graph.erase_node(rstd_output)
            num_recomputed_fused += 1
        else:
            num_original_fused += 1
        gm.graph.erase_node(second_input)
        gm.graph.erase_node(norm_output)
        gm.graph.erase_node(norm)
        gm.graph.erase_node(norm_input)
        gm.graph.erase_node(first_mm)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"F2 fused {num_original_fused} original and "
        f"{num_recomputed_fused} recomputed MLA Q RMSNorm chains"
    )
    return gm


CODA_PATTERN_PASSES: dict[str, Callable] = {
    "b2_dense_swiglu_backward": fuse_b2_dense_swiglu_backward_pass,
    "b4_router_input_grad_add": fuse_b4_router_input_grad_add_pass,
    "b6_bf16_weight_grad_cast": fuse_b6_bf16_weight_grad_cast_pass,
    "f3_residual_rmsnorm": fuse_f3_residual_rmsnorm_pass,
    "f2_q_rmsnorm": fuse_f2_q_rmsnorm_pass,
    "f4_dense_swiglu": fuse_f4_dense_swiglu_pass,
    "f6_router_sigmoid_bias": fuse_f6_router_sigmoid_bias_pass,
}


def get_coda_pattern_passes(patterns: Iterable[str]) -> list[Callable]:
    """Resolve configured CODA pattern names to independently logged passes."""
    pattern_list = list(patterns)
    duplicates = sorted(
        pattern for pattern in set(pattern_list) if pattern_list.count(pattern) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate --compile.coda_patterns entries: {duplicates}")

    unknown = sorted(set(pattern_list) - CODA_PATTERN_PASSES.keys())
    if unknown:
        supported = sorted(CODA_PATTERN_PASSES)
        raise ValueError(
            f"Unknown --compile.coda_patterns entries: {unknown}; "
            f"supported patterns: {supported}"
        )
    return [CODA_PATTERN_PASSES[pattern] for pattern in pattern_list]
