# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for HF ↔ titan MoE checkpoint conversion.

After the MoE replacement, the model's state dict has titan keys for MoE
layers (``mlp.experts.<colwise_name>``, ``mlp.router.gate.weight``) while HF
checkpoints use HF keys (``mlp.experts.gate_up_proj``, ``mlp.gate.weight``).
Non-MoE layers (attention, norms, embeddings) share the same keys.

The key transformation is the fused ``gate_up_proj`` ↔ separate gate/up
split, plus key renames for the router and shared experts.

Parameter names (e.g. ``w1`` vs ``w1_EFD``) are discovered dynamically from
``GroupedExperts`` so this adapter stays compatible across naming conventions.
"""

import re

import spmd_types as spmd
import torch

from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
    _get_expert_param_info,
)

# Keys that had transposed gate_up_proj layout (E, H, 2*I) instead of
# (E, 2*I, H). Populated by ``hf_to_titan_moe_state_dict`` so
# ``titan_to_hf_moe_state_dict`` can reverse the transpose.
_TRANSPOSED_GATE_UP_PROJ_KEYS: set[str] = set()

# Mapping from titan key → original HF key for keys where the reverse
# regex would produce a different result than the original (e.g. Llama4's
# ``router.weight`` vs the default ``gate.weight``).
_TITAN_TO_ORIGINAL_HF_KEY: dict[str, str] = {}


def _expert_names() -> tuple[str, str, str]:
    """Return ``(gate_name, down_name, up_name)`` for routed expert params.

    Maps canonical roles to actual ``GroupedExperts`` parameter names
    (e.g. ``w1`` or ``w1_EFD`` depending on the torchtitan version).
    """
    _, layout = _get_expert_param_info()
    colwise = [n for n, p in layout.items() if p == spmd.S(1)]
    rowwise = [n for n, p in layout.items() if p == spmd.S(2)]
    return colwise[0], rowwise[0], colwise[1]


def _build_hf_to_titan_patterns() -> list[tuple[str, str, bool]]:
    """Build regex patterns using actual expert parameter names."""
    _, down, _ = _expert_names()
    return [
        (r"^(.*\.)gate\.weight$", r"\1router.gate.weight", False),
        (r"^(.*\.)router\.weight$", r"\1router.gate.weight", False),
        (r"^(.*\.)router\.proj\.weight$", r"\1router.gate.weight", False),
        (r"^(.*\.)gate\.e_score_correction_bias$", r"\1expert_bias", False),
        (r"^(.*\.experts)\.down_proj$", rf"\1.{down}", False),
        # Shared experts use FeedForward (w1/w2/w3 attribute names, not params)
        (r"^(.*\.shared_experts)\.gate_proj\.weight$", r"\1.w1.weight", False),
        (r"^(.*\.shared_experts)\.up_proj\.weight$", r"\1.w3.weight", False),
        (r"^(.*\.shared_experts)\.down_proj\.weight$", r"\1.w2.weight", False),
        (
            r"^(.*\.)shared_expert\.gate_proj\.weight$",
            r"\1shared_experts.w1.weight",
            False,
        ),
        (
            r"^(.*\.)shared_expert\.up_proj\.weight$",
            r"\1shared_experts.w3.weight",
            False,
        ),
        (
            r"^(.*\.)shared_expert\.down_proj\.weight$",
            r"\1shared_experts.w2.weight",
            False,
        ),
        (r"^(.*\.)shared_expert_gate\.weight$", r"\1shared_experts.gate.weight", False),
    ]


def _build_titan_to_hf_patterns() -> list[tuple[str, str, bool]]:
    """Build reverse regex patterns using actual expert parameter names."""
    _, down, _ = _expert_names()
    return [
        (r"^(.*\.)router\.gate\.weight$", r"\1gate.weight", False),
        (r"^(.*\.)expert_bias$", r"\1gate.e_score_correction_bias", False),
        (rf"^(.*\.experts)\.{re.escape(down)}$", r"\1.down_proj", False),
        (
            r"^(.*\.)shared_experts\.w1\.weight$",
            r"\1shared_experts.gate_proj.weight",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.w3\.weight$",
            r"\1shared_experts.up_proj.weight",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.w2\.weight$",
            r"\1shared_experts.down_proj.weight",
            False,
        ),
        (
            r"^(.*\.)shared_experts\.gate\.weight$",
            r"\1shared_expert_gate.weight",
            False,
        ),
    ]


def hf_to_titan_moe_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert HF MoE state dict keys/values to titan format.

    Handles fused ``gate_up_proj`` → gate/up split, router and shared
    expert key renames, and expert bias mapping. Non-MoE keys pass through.

    Args:
        hf_state_dict: State dict with HF-format keys.

    Returns:
        State dict with titan-format keys and split expert weights.
    """
    gate_name, down_name, up_name = _expert_names()
    patterns = _build_hf_to_titan_patterns()
    titan_state_dict = {}

    for key, value in hf_state_dict.items():
        # Handle fused gate_up_proj → gate + up
        if key.endswith("experts.gate_up_proj"):
            prefix = key[: -len("experts.gate_up_proj")]
            down_key = f"{prefix}experts.down_proj"
            transposed = False
            if down_key in hf_state_dict:
                dp = hf_state_dict[down_key]
                if value.shape[1] == dp.shape[2] and value.shape[1] != dp.shape[1]:
                    transposed = True
            if transposed:
                value = value.transpose(1, 2).contiguous()
                _TRANSPOSED_GATE_UP_PROJ_KEYS.add(key)
            intermediate_size = value.shape[1] // 2
            titan_state_dict[f"{prefix}experts.{gate_name}"] = value[
                :, :intermediate_size, :
            ]
            titan_state_dict[f"{prefix}experts.{up_name}"] = value[
                :, intermediate_size:, :
            ]
            continue

        # Handle expert down_proj with potential transpose
        if key.endswith("experts.down_proj"):
            prefix = key[: -len("experts.down_proj")]
            gate_key = f"{prefix}experts.gate_up_proj"
            new_key = f"{prefix}experts.{down_name}"
            if gate_key in _TRANSPOSED_GATE_UP_PROJ_KEYS:
                value = value.transpose(1, 2).contiguous()
                _TITAN_TO_ORIGINAL_HF_KEY[new_key] = key
            titan_state_dict[new_key] = value
            continue

        # Try regex patterns
        converted = False
        for pattern, replacement, _ in patterns:
            new_key = re.sub(pattern, replacement, key)
            if new_key != key:
                titan_state_dict[new_key] = value
                converted = True
                _TITAN_TO_ORIGINAL_HF_KEY[new_key] = key
                break

        if not converted:
            titan_state_dict[key] = value

    return titan_state_dict


def titan_to_hf_moe_state_dict(
    titan_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert titan MoE state dict keys/values to HF format.

    Inverse of ``hf_to_titan_moe_state_dict``. Fuses gate/up back
    into ``gate_up_proj`` and reverses all key renames.

    Args:
        titan_state_dict: State dict with titan-format keys.

    Returns:
        State dict with HF-format keys and fused expert weights.
    """
    gate_name, down_name, up_name = _expert_names()
    gate_suffix = f"experts.{gate_name}"
    up_suffix = f"experts.{up_name}"
    down_suffix = f"experts.{down_name}"
    patterns = _build_titan_to_hf_patterns()
    hf_state_dict = {}

    # Collect gate/up pairs for fusing back into gate_up_proj
    gate_keys: dict[str, str] = {}
    up_keys: dict[str, str] = {}

    for key in titan_state_dict:
        if key.endswith(gate_suffix):
            prefix = key[: -len(gate_suffix)]
            gate_keys[prefix] = key
        elif key.endswith(up_suffix):
            prefix = key[: -len(up_suffix)]
            up_keys[prefix] = key

    fused = set()
    for prefix, g_key in gate_keys.items():
        u_key = up_keys.get(prefix)
        if u_key is not None:
            hf_key = f"{prefix}experts.gate_up_proj"
            fused_value = torch.cat(
                [titan_state_dict[g_key], titan_state_dict[u_key]], dim=1
            )
            if hf_key in _TRANSPOSED_GATE_UP_PROJ_KEYS:
                fused_value = fused_value.transpose(1, 2).contiguous()
            hf_state_dict[hf_key] = fused_value
            fused.add(g_key)
            fused.add(u_key)

    for key, value in titan_state_dict.items():
        if key in fused:
            continue

        if key in _TITAN_TO_ORIGINAL_HF_KEY:
            original_key = _TITAN_TO_ORIGINAL_HF_KEY[key]
            if key.endswith(down_suffix) and original_key.endswith("experts.down_proj"):
                value = value.transpose(1, 2).contiguous()
            hf_state_dict[original_key] = value
            continue

        converted = False
        for pattern, replacement, _ in patterns:
            new_key = re.sub(pattern, replacement, key)
            if new_key != key:
                hf_state_dict[new_key] = value
                converted = True
                break

        if not converted:
            hf_state_dict[key] = value

    return hf_state_dict
