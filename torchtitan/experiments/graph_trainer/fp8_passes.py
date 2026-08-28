# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 graph annotation and validation for GraphTrainer compilation."""

from __future__ import annotations

import warnings
from collections import defaultdict

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _QUANTIZATION_EMULATE,
    _QUANTIZATION_KIND,
)
from torchtitan.tools.logging import logger

_FP8_META = "fp8"


def _available_scaled_mm_targets() -> frozenset[object]:
    targets = []
    for op_name in ("_scaled_mm", "_scaled_grouped_mm"):
        op = getattr(torch.ops.aten, op_name, None)
        target = getattr(op, "default", None)
        if target is not None:
            targets.append(target)
    return frozenset(targets)


_SCALED_MM_TARGETS = _available_scaled_mm_targets()
_FP8_DATA_DTYPES = frozenset(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)

# Each quantization kind declares the operations that prove its compute was
# lowered to a supported FP8 implementation. Add a new kind here together
# with its lowering targets when extending GraphTrainer FP8 support.
FP8_COMPUTE_TARGETS: dict[str, frozenset[object]] = {
    "float8_linear": _SCALED_MM_TARGETS,
    "mxfp8_linear": _SCALED_MM_TARGETS,
    "float8_grouped_experts": _SCALED_MM_TARGETS,
    "mxfp8_grouped_experts": _SCALED_MM_TARGETS,
}


def _value_contains_fp8_tensor(value: object) -> bool:
    if isinstance(value, torch.fx.Node):
        return _value_contains_fp8_tensor(value.meta.get("val"))
    if isinstance(value, torch.Tensor):
        return value.dtype in _FP8_DATA_DTYPES
    if isinstance(value, (tuple, list)):
        return any(_value_contains_fp8_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_value_contains_fp8_tensor(item) for item in value.values())
    return False


def _has_fp8_data_operand(node: torch.fx.Node) -> bool:
    # The first two operands of the supported scaled matrix multiplication
    # overloads are the input matrices. Scale operands may use FP8 formats too,
    # so they cannot establish that the compute itself is FP8.
    return any(_value_contains_fp8_tensor(operand) for operand in node.args[:2])


def _classify_fp8_node(node: torch.fx.Node, quantization_kind: str) -> str:
    if (
        node.target in FP8_COMPUTE_TARGETS.get(quantization_kind, frozenset())
        and _has_fp8_data_operand(node)
    ):
        return "compute"

    # GraphPP may compute shared backward quantization in bw_di and pass the
    # resulting FP8 tensor to bw_dw. Treat that callable input as an explicit
    # boundary, not as a missing cast. The local region starts at its consumer
    # (for example, _scaled_mm), while the placeholder remains interpreted.
    if node.op == "placeholder" and _value_contains_fp8_tensor(
        node.meta.get("val")
    ):
        return "input_boundary"

    target_name = str(node.target)
    if "amax" in target_name or node.target == torch.ops.aten.amax.default:
        return "amax"
    if "_to_copy" in target_name or "convert_element_type" in target_name:
        return "cast"

    value = node.meta.get("val")
    if _value_contains_fp8_tensor(value):
        return "cast"
    return "other"


def _inspect_fp8_regions(
    gm: torch.fx.GraphModule,
    *,
    strict: bool,
    annotate: bool,
) -> torch.fx.GraphModule:
    """Inspect FP8 regions and optionally annotate their nodes."""
    regions: dict[tuple[str, str], dict[str, int]] = {}
    emulated_regions: set[tuple[str, str]] = set()
    target_inventory: defaultdict[str, set[str]] = defaultdict(set)

    for node in gm.graph.nodes:
        custom = node.meta.get("custom", {})
        quantization_kind = custom.get(_QUANTIZATION_KIND)
        if quantization_kind is None:
            continue

        module_fqn = custom.get(_MODULE_FQN, "")
        region_key = (module_fqn, quantization_kind)
        region = regions.setdefault(
            region_key,
            {"forward_compute_ops": 0, "backward_compute_ops": 0},
        )
        if custom.get(_QUANTIZATION_EMULATE, False):
            emulated_regions.add(region_key)
        target_inventory[quantization_kind].add(str(node.target))

        role = _classify_fp8_node(node, quantization_kind)
        if annotate:
            node.meta.setdefault("custom", {})[_FP8_META] = {
                "op_role": role,
            }
        if role == "compute":
            phase = "backward" if node.meta.get("autograd_backward", False) else "forward"
            region[f"{phase}_compute_ops"] += 1

    if strict and not regions:
        raise RuntimeError(
            "GraphTrainer did not find quantized module regions while "
            "--compile.fp8.enabled is set."
        )

    missing_regions = [
        region_key
        for region_key, region in regions.items()
        if region_key not in emulated_regions
        if not region["forward_compute_ops"] and not region["backward_compute_ops"]
    ]
    if strict and missing_regions:
        raise RuntimeError(
            "GraphTrainer found quantized module regions without a supported "
            f"FP8 compute operation: {missing_regions}. Observed targets: "
            f"{dict(target_inventory)}."
        )

    summary = {
        "regions": {str(key): value for key, value in regions.items()},
        "target_inventory": {
            kind: sorted(targets) for kind, targets in target_inventory.items()
        },
    }
    logger.info("GraphTrainer FP8 analysis: %s", summary)
    return gm


def _is_regional_fp8_compute_node(
    node: torch.fx.Node,
    *,
    module_fqn: str,
    quantization_kind: str,
) -> bool:
    if node.op != "call_function":
        return False
    custom = node.meta.get("custom", {})
    if (
        custom.get(_MODULE_FQN) != module_fqn
        or custom.get(_QUANTIZATION_KIND) != quantization_kind
    ):
        return False
    target_name = str(node.target)
    if not target_name.startswith("aten."):
        return False
    value = node.meta.get("val")
    return isinstance(value, torch.Tensor) and value.device.type == "cuda"


def _identify_fp8_regional_components(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Identify maximal dense FP8 compute components for regional Inductor.

    Each component is seeded by a supported FP8 compute operation and expands
    only through CUDA aten nodes with the same module and quantization
    provenance. FP8 placeholders are legal callable boundaries, notably when
    GraphPP passes shared grad-output quantization from bw_di to bw_dw; they
    prove the compute operand dtype but are not compiled as part of the local
    region. Communication and host work remain outside these regions.
    Grouped-expert FP8 is not supported by regional Inductor compilation.
    """
    candidate_nodes: set[torch.fx.Node] = set()
    seeds: list[torch.fx.Node] = []

    for node in gm.graph.nodes:
        custom = node.meta.get("custom", {})
        fp8 = custom.get(_FP8_META)
        if fp8 is None:
            continue
        quantization_kind = custom.get(_QUANTIZATION_KIND)
        module_fqn = custom.get(_MODULE_FQN, "")
        if quantization_kind is None:
            continue
        if quantization_kind.endswith("grouped_experts"):
            raise ValueError(
                "FP8 regional compilation does not support grouped experts. "
                "Use full Inductor for grouped-expert FP8 graphs."
            )
        if _is_regional_fp8_compute_node(
            node,
            module_fqn=module_fqn,
            quantization_kind=quantization_kind,
        ):
            candidate_nodes.add(node)
            if fp8["op_role"] == "compute":
                seeds.append(node)

    if not seeds:
        return gm

    identified_nodes: set[torch.fx.Node] = set()
    num_regions = 0
    for seed in seeds:
        if seed in identified_nodes:
            continue
        seed_custom = seed.meta["custom"]
        module_fqn = seed_custom[_MODULE_FQN]
        quantization_kind = seed_custom[_QUANTIZATION_KIND]
        component = {seed}
        pending = [seed]
        while pending:
            node = pending.pop()
            neighbors = (*node.all_input_nodes, *node.users)
            for neighbor in neighbors:
                if (
                    neighbor in component
                    or neighbor in identified_nodes
                    or neighbor not in candidate_nodes
                ):
                    continue
                if not _is_regional_fp8_compute_node(
                    neighbor,
                    module_fqn=module_fqn,
                    quantization_kind=quantization_kind,
                ):
                    continue
                component.add(neighbor)
                pending.append(neighbor)

        for node in component:
            custom = node.meta.setdefault("custom", {})
            fp8 = custom[_FP8_META]
            fp8["regional_region_id"] = num_regions
            fp8["regional_region_num_nodes"] = len(component)
        identified_nodes.update(component)
        num_regions += 1

    summary = {
        "num_regions": num_regions,
        "num_region_nodes": len(identified_nodes),
    }
    logger.info("GraphTrainer FP8 regional annotation: %s", summary)
    return gm


def annotate_complete_fp8_regions_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Tag only complete FP8 regions identified in the current callable.

    GraphPP re-identifies regions after extracting each callable, so an FP8
    placeholder is a valid local boundary. The node-count check still protects
    against graph rewrites between identification and tagging. Other regional
    annotations are not modified.

    Each tagged node gets ``compile_with_inductor["inductor_region"]`` set to
    its ``regional_region_id``. PyTorch's ``regional_inductor`` uses that key
    to keep separately identified FP8 regions from being scooped into one
    giant default partition.
    """
    del example_inputs
    regions: dict[tuple[str, str, int], list[torch.fx.Node]] = defaultdict(list)
    expected_num_nodes: dict[tuple[str, str, int], int] = {}

    for node in gm.graph.nodes:
        custom = node.meta.get("custom", {})
        fp8 = custom.get(_FP8_META)
        if fp8 is None:
            continue
        region_id = fp8.get("regional_region_id")
        region_num_nodes = fp8.get("regional_region_num_nodes")
        if not isinstance(region_id, int) or not isinstance(region_num_nodes, int):
            continue
        region_key = (
            custom.get(_MODULE_FQN, ""),
            custom.get(_QUANTIZATION_KIND, ""),
            region_id,
        )
        regions[region_key].append(node)
        expected_num_nodes[region_key] = region_num_nodes

    num_tagged_nodes = 0
    incomplete_regions: list[tuple[str, str, int]] = []
    for region_key, nodes in regions.items():
        if len(nodes) != expected_num_nodes[region_key]:
            incomplete_regions.append(region_key)
            continue
        if not any(
            node.meta["custom"][_FP8_META].get("op_role") == "compute"
            for node in nodes
        ):
            continue
        _, _, region_id = region_key
        for node in nodes:
            custom = node.meta.setdefault("custom", {})
            compile_annotation = custom.setdefault("compile_with_inductor", {})
            compile_annotation["inductor_region"] = region_id
            num_tagged_nodes += 1

    if incomplete_regions:
        warnings.warn(
            "GraphTrainer skipped incomplete FP8 regional Inductor regions: "
            f"{incomplete_regions}. The affected nodes will run eagerly.",
            stacklevel=2,
        )
    if num_tagged_nodes:
        gm.meta["fp8_regional_tagged_complete_nodes"] = num_tagged_nodes
        logger.info(
            "Tagged %d complete FP8 regional Inductor nodes",
            num_tagged_nodes,
        )
    return gm


def validate_fp8_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    strict: bool,
) -> torch.fx.GraphModule:
    """Validate FP8 lowering without changing node-level compilation metadata."""
    del example_inputs
    return _inspect_fp8_regions(
        gm,
        strict=strict,
        annotate=False,
    )


def annotate_fp8_regions_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    strict: bool,
) -> torch.fx.GraphModule:
    """Validate FP8 lowering and identify dense regions for regional Inductor."""
    del example_inputs
    gm = _inspect_fp8_regions(
        gm,
        strict=strict,
        annotate=True,
    )
    return _identify_fp8_regional_components(gm)
