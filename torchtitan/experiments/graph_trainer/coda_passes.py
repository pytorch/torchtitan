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
_RESHAPE = torch.ops.aten.reshape.default
_SIGMOID = torch.ops.aten.sigmoid.default
_TO_COPY = torch.ops.aten._to_copy.default
_VIEW = torch.ops.aten.view.default
_COMPILE_WITH_INDUCTOR = "compile_with_inductor"


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


CODA_PATTERN_PASSES: dict[str, Callable] = {
    "b6_bf16_weight_grad_cast": fuse_b6_bf16_weight_grad_cast_pass,
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
