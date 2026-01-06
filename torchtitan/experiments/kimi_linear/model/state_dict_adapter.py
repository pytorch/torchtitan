# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Kimi Linear model.

Converts between HuggingFace checkpoint format and torchtitan format.

Note: HF uses `self_attn` for ALL attention layers (both MLA and KDA).
The layer type determines which parameters are present:
- MLA layers have: q_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj
- KDA layers have: q_proj, k_proj, v_proj, *_conv1d, A_log, f_*, dt_bias, b_proj, g_*, o_norm, o_proj

The o_proj mapping differs: MLA -> wo, KDA -> o_proj

Layer type determination:
- full_attn_layers uses 1-based indexing for HF compatibility
- Layer i (0-indexed) is MLA if (i + 1) in full_attn_layers, otherwise KDA

MLA dimension parameters (important for kv_b_proj shape):
- kv_b_proj output dim = n_heads * (qk_nope_head_dim + v_head_dim)
- v_head_dim is separate from head_dim in HF configs
"""

import re
from typing import Any

from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter

from .args import KimiLinearModelArgs


class KimiLinearStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: KimiLinearModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        # Common mappings (not layer-specific)
        self.common_from_hf_map = {
            # Embeddings
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "lm_head.weight": "output.weight",
            # Final norm
            "model.norm.weight": "norm.weight",
            # Layer norms (same for both attention types)
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # Dense MLP
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE experts
            "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "layers.{}.moe.experts.w3",
            # MoE shared expert
            "model.layers.{}.block_sparse_moe.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.block_sparse_moe.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.block_sparse_moe.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            # MoE gate
            "model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.moe.router.gate.weight",
            # e_score_correction_bias in HF maps to expert_bias in torchtitan
            "model.layers.{}.block_sparse_moe.gate.e_score_correction_bias": "layers.{}.moe.expert_bias",
        }

        # MLA (Full Attention) specific mappings
        self.mla_from_hf_map = {
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.q_proj.weight",
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.kv_a_proj_with_mqa.weight",
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_a_layernorm.weight",
            "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.kv_b_proj.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        }

        # KDA (Linear Attention) specific mappings
        self.kda_from_hf_map = {
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.k_proj.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.v_proj.weight",
            # Conv weights - fla's ShortConvolution inherits from nn.Conv1d,
            # so weight is directly on module (.weight not .conv.weight)
            "model.layers.{}.self_attn.q_conv1d.weight": "layers.{}.attention.q_conv1d.weight",
            "model.layers.{}.self_attn.k_conv1d.weight": "layers.{}.attention.k_conv1d.weight",
            "model.layers.{}.self_attn.v_conv1d.weight": "layers.{}.attention.v_conv1d.weight",
            "model.layers.{}.self_attn.A_log": "layers.{}.attention.A_log",
            "model.layers.{}.self_attn.f_a_proj.weight": "layers.{}.attention.f_a_proj.weight",
            "model.layers.{}.self_attn.f_b_proj.weight": "layers.{}.attention.f_b_proj.weight",
            "model.layers.{}.self_attn.dt_bias": "layers.{}.attention.dt_bias",
            "model.layers.{}.self_attn.b_proj.weight": "layers.{}.attention.b_proj.weight",
            "model.layers.{}.self_attn.g_a_proj.weight": "layers.{}.attention.g_a_proj.weight",
            "model.layers.{}.self_attn.g_b_proj.weight": "layers.{}.attention.g_b_proj.weight",
            "model.layers.{}.self_attn.o_norm.weight": "layers.{}.attention.o_norm.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.o_proj.weight",
        }

        # Build from_hf_map for backward compatibility (used by parent class)
        # This is a combined map - layer-specific resolution happens in from_hf/to_hf
        self.from_hf_map = {**self.common_from_hf_map}

    def _is_kda_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses KDA (linear attention)."""
        return self.model_args.is_kda_layer(layer_idx)

    def _get_layer_attention_map(self, layer_idx: int, from_hf: bool = True) -> dict:
        """Get the appropriate attention mapping for a layer."""
        if self._is_kda_layer(layer_idx):
            return self.kda_from_hf_map if from_hf else {v: k for k, v in self.kda_from_hf_map.items()}
        else:
            return self.mla_from_hf_map if from_hf else {v: k for k, v in self.mla_from_hf_map.items()}

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HuggingFace format."""
        hf_state_dict = {}

        # Build reverse map for common keys
        common_to_hf_map = {v: k for k, v in self.common_from_hf_map.items()}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in common_to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = common_to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[abstract_key] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    split_values = self._split_experts_weights(
                        value, self.model_args.moe_args.num_experts
                    )
                    for expert_num in range(self.model_args.moe_args.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = int(re.search(r"\d+", key).group(0))

                # Try common map first
                if abstract_key in common_to_hf_map:
                    new_key = common_to_hf_map[abstract_key].format(layer_num)
                    hf_state_dict[new_key] = value
                    continue

                # Try layer-specific attention map
                attn_to_hf_map = self._get_layer_attention_map(layer_num, from_hf=False)
                if abstract_key in attn_to_hf_map:
                    new_key = attn_to_hf_map[abstract_key].format(layer_num)
                    hf_state_dict[new_key] = value
            else:
                if key in common_to_hf_map:
                    new_key = common_to_hf_map[key]
                    hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace state dict to torchtitan format."""
        state_dict = {}
        expert_weights_by_layer = {}

        for key, value in hf_state_dict.items():
            if "block_sparse_moe.experts." in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)[:2]
                if abstract_key not in self.common_from_hf_map:
                    continue
                titan_abstract_key = self.common_from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                # Use int key to ensure correct sorting order when stacking
                expert_weights_by_layer[layer_num][titan_abstract_key][int(expert_num)] = value

                if isinstance(value, DTensor):
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        value.device_mesh,
                    )
                else:
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        self.model_args.moe_args.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = int(re.search(r"\d+", key).group(0))

                # Try common map first
                if abstract_key in self.common_from_hf_map:
                    new_key = self.common_from_hf_map[abstract_key].format(layer_num)
                    state_dict[new_key] = value
                    continue

                # Try layer-specific attention map
                attn_from_hf_map = self._get_layer_attention_map(layer_num, from_hf=True)
                if abstract_key in attn_from_hf_map:
                    new_key = attn_from_hf_map[abstract_key].format(layer_num)
                    state_dict[new_key] = value
            else:
                if key in self.common_from_hf_map:
                    new_key = self.common_from_hf_map[key]
                    state_dict[new_key] = value

        return state_dict
