# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve compressed-tensors MXFP4 policy against a model's HF Linear weights."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MXFP4CheckpointPolicy:
    weight_fqns: frozenset[str]
    block_size: int = 32

    @classmethod
    def from_config(
        cls, quantization: Mapping[str, Any], linear_weights: Iterable[str]
    ) -> "MXFP4CheckpointPolicy":
        """Resolve an exact set; the adapter supplies actual HF Linear weights.

        Only the released static, symmetric 1x32 E2M1/E8M0 format is supported.
        Names alone cannot distinguish Linear weights from embeddings or norms.
        """
        if (
            quantization.get("format") != "mxfp4-pack-quantized"
            or quantization.get("quant_method") != "compressed-tensors"
        ):
            raise ValueError("Checkpoint must declare compressed-tensors MXFP4 weights")
        groups = quantization.get("config_groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("MXFP4 quantization_config has no config groups")
        targets = []
        expected = {
            "num_bits": 4,
            "type": "float",
            "strategy": "group",
            "group_size": 32,
            "symmetric": True,
            "dynamic": False,
            "scale_dtype": "torch.uint8",
        }
        for group in groups.values():
            if not isinstance(group, dict):
                raise ValueError("MXFP4 config group must be an object")
            if group.get("format", "mxfp4-pack-quantized") != "mxfp4-pack-quantized":
                raise ValueError("All groups must use mxfp4-pack-quantized")
            weights = group.get("weights")
            if not isinstance(weights, dict) or any(
                weights.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(
                    "MXFP4 weights require static symmetric float4, group_size=32 and uint8 E8M0 scales"
                )
            if any(
                weights.get(key) is not None
                for key in ("actorder", "block_structure", "zp_dtype")
            ):
                raise ValueError(
                    "MXFP4 activation ordering, block structures and zero points are unsupported"
                )
            if any(
                group.get(key) is not None
                for key in ("input_activations", "output_activations")
            ):
                raise ValueError(
                    "Checkpoint activation quantization is unsupported; configure runtime QAT separately"
                )
            group_targets = group.get("targets")
            if not isinstance(group_targets, list) or not group_targets:
                raise ValueError("MXFP4 group targets must be a nonempty list")
            targets.extend(group_targets)
        ignore = quantization.get("ignore", [])
        if not isinstance(ignore, list) or not all(
            isinstance(p, str) for p in targets + ignore
        ):
            raise ValueError("MXFP4 targets and ignore entries must be strings")

        def matches(pattern: str, module: str) -> bool:
            if pattern.startswith("re:"):
                return re.match(pattern[3:], module) is not None
            return pattern == "Linear" or pattern == module

        return cls(
            frozenset(
                name
                for name in linear_weights
                if any(
                    matches(pattern, name.removesuffix(".weight"))
                    for pattern in targets
                )
                and not any(
                    matches(pattern, name.removesuffix(".weight")) for pattern in ignore
                )
            )
        )


def decode_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Adapt the HF byte representation to TorchAO's numerical codec."""
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    return MXTensor(
        qdata=packed,
        scale=scales.view(torch.float8_e8m0fnu),
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=block_size,
        orig_dtype=target_dtype,
        kernel_preference=None,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
    ).dequantize(target_dtype)
