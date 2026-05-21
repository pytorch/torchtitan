# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for HF ↔ titan MoE checkpoint conversion.

After the MoE replacement, the model's state dict has titan keys for MoE
layers (``mlp.experts.w1``, ``mlp.router.gate.weight``) while HF checkpoints
use HF keys (``mlp.experts.gate_up_proj``, ``mlp.gate.weight``). Non-MoE
layers (attention, norms, embeddings) share the same keys in both formats.

The key transformation is the fused ``gate_up_proj`` ↔ separate ``w1``/``w3``
split, plus key renames for the router and shared experts.
"""

import re

import torch

# Patterns match MoE keys at any depth (with or without layer prefix).
# Each entry: (hf_pattern_regex, titan_replacement, needs_split)
_HF_TO_TITAN_PATTERNS = [
    # Fused gate_up_proj → w1 + w3 (handled specially, not via regex)
    # Router: gate.weight → router.gate.weight
    # Match "gate.weight" only when preceded by "experts" sibling (MoE context)
    (r"^(.*\.)gate\.weight$", r"\1router.gate.weight", False),
    # Expert bias
    (r"^(.*\.)gate\.e_score_correction_bias$", r"\1expert_bias", False),
    # Expert down_proj → w2
    (r"^(.*\.experts)\.down_proj$", r"\1.w2", False),
    # Shared experts (plural form)
    (r"^(.*\.shared_experts)\.gate_proj\.weight$", r"\1.w1.weight", False),
    (r"^(.*\.shared_experts)\.up_proj\.weight$", r"\1.w3.weight", False),
    (r"^(.*\.shared_experts)\.down_proj\.weight$", r"\1.w2.weight", False),
    # Shared expert (singular form, DeepSeek V2/V3)
    (r"^(.*\.)shared_expert\.gate_proj\.weight$", r"\1shared_experts.w1.weight", False),
    (r"^(.*\.)shared_expert\.up_proj\.weight$", r"\1shared_experts.w3.weight", False),
    (r"^(.*\.)shared_expert\.down_proj\.weight$", r"\1shared_experts.w2.weight", False),
    # Shared expert gate (Qwen2 MoE, Qwen3.5)
    (r"^(.*\.)shared_expert_gate\.weight$", r"\1shared_experts.gate.weight", False),
]

_TITAN_TO_HF_PATTERNS = [
    # Router
    (r"^(.*\.)router\.gate\.weight$", r"\1gate.weight", False),
    # Expert bias
    (r"^(.*\.)expert_bias$", r"\1gate.e_score_correction_bias", False),
    # Expert w2 → down_proj
    (r"^(.*\.experts)\.w2$", r"\1.down_proj", False),
    # Shared experts → singular form
    (
        r"^(.*\.)shared_experts\.w1\.weight$",
        r"\1shared_experts.gate_proj.weight",
        False,
    ),
    (r"^(.*\.)shared_experts\.w3\.weight$", r"\1shared_experts.up_proj.weight", False),
    (
        r"^(.*\.)shared_experts\.w2\.weight$",
        r"\1shared_experts.down_proj.weight",
        False,
    ),
    # Shared expert gate
    (r"^(.*\.)shared_experts\.gate\.weight$", r"\1shared_expert_gate.weight", False),
]


def hf_to_titan_moe_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert HF MoE state dict keys/values to titan format.

    Handles fused ``gate_up_proj`` → ``w1``/``w3`` split, router and shared
    expert key renames, and expert bias mapping. Non-MoE keys pass through.

    Args:
        hf_state_dict: State dict with HF-format keys.

    Returns:
        State dict with titan-format keys and split expert weights.
    """
    titan_state_dict = {}

    for key, value in hf_state_dict.items():
        # Handle fused gate_up_proj → w1 + w3
        if key.endswith("experts.gate_up_proj"):
            prefix = key[: -len("experts.gate_up_proj")]
            intermediate_size = value.shape[1] // 2
            titan_state_dict[f"{prefix}experts.w1"] = value[:, :intermediate_size, :]
            titan_state_dict[f"{prefix}experts.w3"] = value[:, intermediate_size:, :]
            continue

        # Try regex patterns
        converted = False
        for pattern, replacement, _ in _HF_TO_TITAN_PATTERNS:
            new_key = re.sub(pattern, replacement, key)
            if new_key != key:
                titan_state_dict[new_key] = value
                converted = True
                break

        if not converted:
            titan_state_dict[key] = value

    return titan_state_dict


def titan_to_hf_moe_state_dict(
    titan_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert titan MoE state dict keys/values to HF format.

    Inverse of ``hf_to_titan_moe_state_dict``. Fuses ``w1``/``w3`` back
    into ``gate_up_proj`` and reverses all key renames.

    Args:
        titan_state_dict: State dict with titan-format keys.

    Returns:
        State dict with HF-format keys and fused expert weights.
    """
    hf_state_dict = {}

    # Collect w1/w3 pairs for fusing back into gate_up_proj
    w1_keys = {}
    w3_keys = {}

    for key in titan_state_dict:
        if key.endswith("experts.w1"):
            prefix = key[: -len("experts.w1")]
            w1_keys[prefix] = key
        elif key.endswith("experts.w3"):
            prefix = key[: -len("experts.w3")]
            w3_keys[prefix] = key

    fused = set()
    for prefix, w1_key in w1_keys.items():
        w3_key = w3_keys.get(prefix)
        if w3_key is not None:
            hf_key = f"{prefix}experts.gate_up_proj"
            hf_state_dict[hf_key] = torch.cat(
                [titan_state_dict[w1_key], titan_state_dict[w3_key]], dim=1
            )
            fused.add(w1_key)
            fused.add(w3_key)

    for key, value in titan_state_dict.items():
        if key in fused:
            continue

        # Try regex patterns
        converted = False
        for pattern, replacement, _ in _TITAN_TO_HF_PATTERNS:
            new_key = re.sub(pattern, replacement, key)
            if new_key != key:
                hf_state_dict[new_key] = value
                converted = True
                break

        if not converted:
            hf_state_dict[key] = value

    return hf_state_dict
