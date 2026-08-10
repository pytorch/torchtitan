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
_TO_COPY = torch.ops.aten._to_copy.default
_COMPILE_WITH_INDUCTOR = "compile_with_inductor"


def _node_dtype(node: torch.fx.Node) -> torch.dtype | None:
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype
    tensor_meta = node.meta.get("tensor_meta")
    return getattr(tensor_meta, "dtype", None)


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


CODA_PATTERN_PASSES: dict[str, Callable] = {
    "b6_bf16_weight_grad_cast": fuse_b6_bf16_weight_grad_cast_pass,
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
