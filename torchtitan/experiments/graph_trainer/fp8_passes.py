# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 graph analysis for GraphTrainer's Phase 1 full-compile path."""

from __future__ import annotations

from collections import defaultdict

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _QUANTIZATION_KIND,
)
from torchtitan.tools.logging import logger

_FP8_META = "fp8"


def _available_fp8_gemm_targets() -> frozenset[object]:
    targets = []
    for op_name in ("_scaled_mm", "_scaled_grouped_mm"):
        op = getattr(torch.ops.aten, op_name, None)
        target = getattr(op, "default", None)
        if target is not None:
            targets.append(target)
    return frozenset(targets)


FP8_GEMM_TARGETS = _available_fp8_gemm_targets()


def _classify_fp8_node(node: torch.fx.Node) -> str:
    if node.target in FP8_GEMM_TARGETS:
        return "gemm"

    target_name = str(node.target)
    if "amax" in target_name or node.target == torch.ops.aten.amax.default:
        return "amax"
    if "_to_copy" in target_name or "convert_element_type" in target_name:
        return "cast"

    value = node.meta.get("val")
    if isinstance(value, torch.Tensor) and value.dtype in {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }:
        return "cast"
    return "other"


def analyze_fp8_regions_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    strict: bool,
) -> torch.fx.GraphModule:
    """Record and validate FP8 regions in a traced joint forward/backward graph."""
    del example_inputs
    regions: dict[tuple[str, str], dict[str, int]] = {}
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
            {"forward_gemms": 0, "backward_gemms": 0},
        )
        target_inventory[quantization_kind].add(str(node.target))

        role = _classify_fp8_node(node)
        node.meta.setdefault("custom", {})[_FP8_META] = {
            "format": "mxfp8" if quantization_kind.startswith("mxfp8") else "float8",
            "module_kind": (
                "grouped_experts"
                if quantization_kind.endswith("grouped_experts")
                else "linear"
            ),
            "op_role": role,
        }
        if role == "gemm":
            phase = "backward" if node.meta.get("autograd_backward", False) else "forward"
            region[f"{phase}_gemms"] += 1

    if strict and not regions:
        raise RuntimeError(
            "GraphTrainer did not find quantized module regions while "
            "--compile.fp8.enabled is set."
        )

    missing_regions = [
        region_key
        for region_key, region in regions.items()
        if not region["forward_gemms"] and not region["backward_gemms"]
    ]
    if strict and missing_regions:
        raise RuntimeError(
            "GraphTrainer found quantized module regions without a recognized "
            f"FP8 GEMM: {missing_regions}. Observed targets: "
            f"{dict(target_inventory)}."
        )

    summary = {
        "regions": {str(key): value for key, value in regions.items()},
        "target_inventory": {
            kind: sorted(targets) for kind, targets in target_inventory.items()
        },
    }
    gm.meta["fp8_summary"] = summary
    logger.info("GraphTrainer FP8 analysis: %s", summary)
    return gm
