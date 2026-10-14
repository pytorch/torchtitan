# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover conservative fusion candidates without rewriting an FX graph."""

from __future__ import annotations

import operator
from typing import TypedDict

import torch
import torch.fx as fx
from torch._inductor.fx_passes.fusion_regions import _find_connected_components

from torchtitan.experiments.graph_trainer.common_utils import _get_graph_modules
from torchtitan.experiments.graph_trainer.mutation_utils import is_mutation_node


FUSION_REGIONS_META = "graph_trainer_fusion_regions"


class FusionRegion(TypedDict):
    id: int
    nodes: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class FusionBoundary(TypedDict):
    node: str
    kinds: tuple[str, ...]


class FusionRegionMetadata(TypedDict):
    regions: tuple[FusionRegion, ...]
    boundaries: tuple[FusionBoundary, ...]


_ATEN_GEMMS = frozenset(
    {
        "mm",
        "addmm",
        "addmm_",
        "_addmm_activation",
        "bmm",
        "addbmm",
        "addbmm_",
        "baddbmm",
        "baddbmm_",
        "matmul",
        "matmul_backward",
        "linear",
        "linear_backward",
        "_int_mm",
        "_scaled_mm",
        "_scaled_mm_v2",
        "_scaled_addmm",
        "_scaled_addmm_",
        "_foreach_mm",
        "_grouped_mm",
        "_scaled_grouped_mm",
        "_scaled_grouped_mm_v2",
        "_sparse_mm",
        "_sparse_addmm",
        "_sparse_semi_structured_mm",
        "_sparse_semi_structured_addmm",
    }
)
_DIST_MOE_GEMMS = frozenset(
    {
        "bf16_forward",
        "bf16_forward_with_clip_stats",
        "bf16_backward",
        "bf16_backward_accumulate_",
        "block_scaled_forward",
        "block_scaled_backward",
        "block_scaled_backward_accumulate_",
    }
)
_SYMM_MEM_GEMMS = frozenset(
    {
        "fused_all_gather_matmul",
        "fused_all_gather_scaled_matmul",
        "fused_matmul_reduce_scatter",
        "fused_scaled_matmul_reduce_scatter",
    }
)
_COLLECTIVE_NAMESPACES = frozenset(
    {"_c10d_functional", "_c10d_functional_autograd", "c10d_functional", "c10d"}
)
_GEMM_FUNCTIONS = frozenset(
    {
        operator.matmul,
        torch.mm,
        torch.addmm,
        torch.bmm,
        torch.addbmm,
        torch.baddbmm,
        torch.matmul,
        torch.nn.functional.linear,
    }
)


def _boundary_kinds(node: fx.Node) -> tuple[str, ...]:
    if node.op in {"placeholder", "get_attr", "output"}:
        return ()
    if node.op != "call_function":
        return ("opaque",)

    target = node.target
    kinds = []
    if isinstance(target, torch._ops.OpOverload):
        namespace, name = target._schema.name.split("::", 1)
        if namespace == "aten":
            if name in _ATEN_GEMMS:
                kinds.append("gemm")
            if name in {
                "scaled_dot_product_attention",
                "_native_multi_head_attention",
                "_triton_multi_head_attention",
                "_triton_scaled_dot_attention",
            } or name.startswith(
                (
                    "_scaled_dot_product_",
                    "_flash_attention_",
                    "_efficient_attention_",
                    "_cudnn_attention_",
                )
            ):
                kinds.append("attention")
        elif namespace == "dist_moe" and name in _DIST_MOE_GEMMS:
            kinds.extend(("gemm", "collective"))
        elif namespace == "symm_mem":
            if name in _SYMM_MEM_GEMMS:
                kinds.append("gemm")
            kinds.append("collective")
        elif namespace in _COLLECTIVE_NAMESPACES or (
            namespace in {"deepep", "hybridep"}
            and name in {"dispatch", "combine", "dispatch_bwd", "combine_bwd"}
        ):
            kinds.append("collective")
        elif namespace == "torch_attn" and name.startswith("_varlen_attn"):
            kinds.append("attention")
    elif isinstance(target, torch._ops.HigherOrderOperator):
        if target.name() in {"flex_attention", "flex_attention_backward"}:
            kinds.append("attention")
        else:
            kinds.append("opaque")
    elif target in _GEMM_FUNCTIONS:
        kinds.append("gemm")
    elif target is torch.nn.functional.scaled_dot_product_attention:
        kinds.append("attention")
    elif target is not operator.getitem:
        kinds.append("opaque")

    if is_mutation_node(node):
        kinds.append("mutation")
    elif node.is_impure():
        kinds.append("effect")
    return tuple(kinds)


def _analyze_graph(gm: fx.GraphModule) -> FusionRegionMetadata:
    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    regions: list[FusionRegion] = []
    boundaries: list[FusionBoundary] = []
    span: list[fx.Node] = []

    def flush() -> None:
        for component in _find_connected_components(span):
            nodes = sorted(component, key=graph_order.__getitem__)
            members = set(nodes)
            inputs = sorted(
                {
                    source
                    for node in nodes
                    for source in node.all_input_nodes
                    if source not in members
                },
                key=graph_order.__getitem__,
            )
            outputs = [
                node
                for node in nodes
                if any(user not in members for user in node.users)
            ]
            regions.append(
                {
                    "id": len(regions),
                    "nodes": tuple(node.name for node in nodes),
                    "inputs": tuple(node.name for node in inputs),
                    "outputs": tuple(node.name for node in outputs),
                }
            )
        span.clear()

    for node in gm.graph.nodes:
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        kinds = _boundary_kinds(node)
        if kinds:
            flush()
            boundaries.append({"node": node.name, "kinds": kinds})
        else:
            span.append(node)
    flush()
    return {"regions": tuple(regions), "boundaries": tuple(boundaries)}


def find_fusion_region(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    recurse: bool = True,
) -> fx.GraphModule:
    """Snapshot fusion candidates in ``gm.meta[FUSION_REGIONS_META]``.

    Regions contain graph-local node names and their external input/output
    node names, all in graph order. IDs are local to each GraphModule. Nested
    GraphModules are analyzed independently when ``recurse`` is true; their
    calls remain opaque boundaries in the parent graph.

    GEMM, attention, and collective calls divide the execution order into
    spans. Mutations, effects including RNG, and opaque calls also end a span,
    preventing alias hazards that are absent from FX dataflow edges. Connected
    components within each span are convex candidates, including singletons.
    Views and tuple getitems remain candidates; graph inputs and attributes
    remain external inputs. All external users, including saved values used
    by backward, determine region outputs.

    Only this pass's metadata key is replaced. No operations are moved,
    outlined, or compiled, and existing node annotations remain untouched.
    Metadata is a snapshot: rerun after graph edits. It does not prove kernel
    support, bitwise or memory parity, or estimate memory traffic.
    """
    for module in _get_graph_modules(gm, recurse=recurse, apply_to_root=True):
        module.meta[FUSION_REGIONS_META] = _analyze_graph(module)
    return gm
