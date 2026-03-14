# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for Qwen3.5 MoE: HF <-> torchtitan conversion.

Key difference from Qwen3 MoE: HF stores expert weights as already-fused 3D
tensors (``gate_up_proj [E, 2*I, D]``), not per-expert 2D tensors. Conversion
is a simple ``chunk``/``cat`` along dim=1 — no per-expert stacking needed.
"""

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Model

# Prefix on all text-model keys in the HF checkpoint
_HF_PREFIX = "model.language_model."


class Qwen35MoEStateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        # Non-expert key mapping: HF (with prefix) -> torchtitan
        # Expert keys (gate_up_proj / down_proj) are handled separately
        # because they need tensor manipulation (split / concat).
        self.from_hf_map = {
            # --- top-level ---
            f"{_HF_PREFIX}embed_tokens.weight": "tok_embeddings.weight",
            f"{_HF_PREFIX}norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            # --- per-layer norms ---
            f"{_HF_PREFIX}layers.{{}}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            f"{_HF_PREFIX}layers.{{}}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # --- MoE router + shared expert ---
            f"{_HF_PREFIX}layers.{{}}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            f"{_HF_PREFIX}layers.{{}}.mlp.shared_expert.gate_proj.weight": "layers.{}.shared_ffn.w1.weight",
            f"{_HF_PREFIX}layers.{{}}.mlp.shared_expert.up_proj.weight": "layers.{}.shared_ffn.w3.weight",
            f"{_HF_PREFIX}layers.{{}}.mlp.shared_expert.down_proj.weight": "layers.{}.shared_ffn.w2.weight",
            f"{_HF_PREFIX}layers.{{}}.mlp.shared_expert_gate.weight": "layers.{}.shared_gate.weight",
            # --- GatedDeltaNet (linear_attn) ---
            f"{_HF_PREFIX}layers.{{}}.linear_attn.in_proj_qkv.weight": "layers.{}.attn.in_proj_qkv.weight",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.in_proj_z.weight": "layers.{}.attn.in_proj_z.weight",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.in_proj_a.weight": "layers.{}.attn.in_proj_a.weight",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.in_proj_b.weight": "layers.{}.attn.in_proj_b.weight",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.conv1d.weight": "layers.{}.attn.conv1d.weight",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.A_log": "layers.{}.attn.A_log",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.dt_bias": "layers.{}.attn.dt_bias",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.norm.weight": "layers.{}.attn.norm.weight",
            f"{_HF_PREFIX}layers.{{}}.linear_attn.out_proj.weight": "layers.{}.attn.out_proj.weight",
            # --- Full attention (self_attn) ---
            f"{_HF_PREFIX}layers.{{}}.self_attn.q_proj.weight": "layers.{}.attn.wq.weight",
            f"{_HF_PREFIX}layers.{{}}.self_attn.k_proj.weight": "layers.{}.attn.wk.weight",
            f"{_HF_PREFIX}layers.{{}}.self_attn.v_proj.weight": "layers.{}.attn.wv.weight",
            f"{_HF_PREFIX}layers.{{}}.self_attn.o_proj.weight": "layers.{}.attn.wo.weight",
            f"{_HF_PREFIX}layers.{{}}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.weight",
            f"{_HF_PREFIX}layers.{{}}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HF format.

        Fuses ``moe.experts.w1`` + ``moe.experts.w3`` into ``gate_up_proj``,
        and maps ``moe.experts.w2`` to ``down_proj``.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            # Skip training-only buffers
            if key.endswith(".expert_bias") or key.endswith(".tokens_per_expert"):
                continue
            if key == "freqs_cis":
                continue

            # --- Expert w1 + w3 -> gate_up_proj ---
            if ".moe.experts.w1" in key:
                layer_num = re.search(r"layers\.(\d+)\.", key).group(1)
                w3_key = key.replace(".moe.experts.w1", ".moe.experts.w3")
                w3 = state_dict[w3_key]
                gate_up = torch.cat([value, w3], dim=1)
                hf_key = f"{_HF_PREFIX}layers.{layer_num}.mlp.experts.gate_up_proj"
                hf_state_dict[hf_key] = gate_up
                continue

            # w3 already handled together with w1
            if ".moe.experts.w3" in key:
                continue

            # --- Expert w2 -> down_proj ---
            if ".moe.experts.w2" in key:
                layer_num = re.search(r"layers\.(\d+)\.", key).group(1)
                hf_key = f"{_HF_PREFIX}layers.{layer_num}.mlp.experts.down_proj"
                hf_state_dict[hf_key] = value
                continue

            # --- Standard key mapping ---
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key].format(layer_num)
                hf_state_dict[new_key] = value
            else:
                if key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[key]] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to torchtitan format.

        Splits fused ``gate_up_proj [E, 2*I, D]`` into ``w1`` and ``w3``,
        and maps ``down_proj`` to ``w2``.
        """
        state_dict: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            # Skip visual encoder, MTP, and rotary embedding keys
            if key.startswith("model.visual.") or key.startswith("mtp."):
                continue
            if "rotary_emb.inv_freq" in key:
                continue

            # --- Expert gate_up_proj -> w1 + w3 ---
            if ".mlp.experts.gate_up_proj" in key:
                layer_num = re.search(r"layers\.(\d+)\.", key).group(1)
                # shape: [num_experts, 2 * intermediate, dim]
                w1, w3 = value.chunk(2, dim=1)
                state_dict[f"layers.{layer_num}.moe.experts.w1"] = w1
                state_dict[f"layers.{layer_num}.moe.experts.w3"] = w3
                continue

            # --- Expert down_proj -> w2 ---
            if ".mlp.experts.down_proj" in key:
                layer_num = re.search(r"layers\.(\d+)\.", key).group(1)
                state_dict[f"layers.{layer_num}.moe.experts.w2"] = value
                continue

            # --- Standard key mapping ---
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                state_dict[new_key.format(layer_num)] = value
            else:
                if key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                state_dict[new_key] = value

        # Populate expert_bias as zeros for each MoE layer.
        # HF checkpoints don't store this training-only buffer, but the
        # torchtitan model registers it as a persistent buffer (when
        # load_balance_coeff is not None), so DCP checkpoints need it.
        for key in list(state_dict.keys()):
            if key.endswith(".moe.router.gate.weight"):
                layer_prefix = key.rsplit(".moe.router.gate.weight", 1)[0]
                num_experts = state_dict[key].shape[0]
                bias_key = f"{layer_prefix}.moe.expert_bias"
                if bias_key not in state_dict:
                    state_dict[bias_key] = torch.zeros(num_experts, dtype=torch.float32)

        return state_dict
