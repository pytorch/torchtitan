# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 State Dict Adapter
# Converts between HuggingFace Transformers and TorchTitan checkpoint formats

import logging
import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger()

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.utils import MoEStateDictAdapter
from .model import Gemma4Model


class Gemma4StateDictAdapter(MoEStateDictAdapter):
    """Adapter for converting Gemma-4 checkpoints between HF and TorchTitan formats.

    Handles the mapping between HuggingFace Transformers model keys and TorchTitan's
    expected parameter names, including all four layer norms, Q/K norms, MoE experts, and CosSinRoPE.
    """

    def __init__(
        self,
        model_config: Gemma4Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        # Mapping from HuggingFace keys to TorchTitan keys
        # Supports both official nested (model.language_model.*) and flat (model.*) prefixes
        self.to_hf_map = {
            "tok_embeddings.weight": "model.language_model.embed_tokens.weight",
            "ple.tok_embeddings_per_layer.weight": "model.language_model.embed_tokens_per_layer.weight",
            "ple.per_layer_model_projection.weight": "model.language_model.per_layer_model_projection.weight",
            "ple.per_layer_projection_norm.weight": "model.language_model.per_layer_projection_norm.weight",
            "layers.{}.attention.qkv_linear.wq.weight": "model.language_model.layers.{}.self_attn.q_proj.weight",
            "layers.{}.attention.qkv_linear.wk.weight": "model.language_model.layers.{}.self_attn.k_proj.weight",
            "layers.{}.attention.qkv_linear.wv.weight": "model.language_model.layers.{}.self_attn.v_proj.weight",
            "layers.{}.attention.wo.weight": "model.language_model.layers.{}.self_attn.o_proj.weight",
            "layers.{}.attention.q_norm.weight": "model.language_model.layers.{}.self_attn.q_norm.weight",
            "layers.{}.attention.k_norm.weight": "model.language_model.layers.{}.self_attn.k_norm.weight",
            # PLE Layer Modules
            "layers.{}.ple.per_layer_input_gate.weight": "model.language_model.layers.{}.per_layer_input_gate.weight",
            "layers.{}.ple.per_layer_projection.weight": "model.language_model.layers.{}.per_layer_projection.weight",
            "layers.{}.ple.post_per_layer_input_norm.weight": "model.language_model.layers.{}.post_per_layer_input_norm.weight",
            # Non-MoE MLP
            "layers.{}.feed_forward.w1.weight": "model.language_model.layers.{}.mlp.gate_proj.weight",
            "layers.{}.feed_forward.w3.weight": "model.language_model.layers.{}.mlp.up_proj.weight",
            "layers.{}.feed_forward.w2.weight": "model.language_model.layers.{}.mlp.down_proj.weight",
            # MoE MLP
            "layers.{}.moe.routed_experts.inner_experts.w1_EFD": "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight",
            "layers.{}.moe.routed_experts.inner_experts.w3_EFD": "model.language_model.layers.{}.mlp.experts.{}.up_proj.weight",
            "layers.{}.moe.routed_experts.inner_experts.w2_EDF": "model.language_model.layers.{}.mlp.experts.{}.down_proj.weight",
            "layers.{}.moe.router.gate.weight": "model.language_model.layers.{}.mlp.gate.weight",
            # Layer norms
            "layers.{}.attention_norm.weight": "model.language_model.layers.{}.input_layernorm.weight",
            "layers.{}.post_attention_norm.weight": "model.language_model.layers.{}.post_attention_layernorm.weight",
            "layers.{}.ffn_norm.weight": "model.language_model.layers.{}.pre_feedforward_layernorm.weight",
            "layers.{}.post_ffn_norm.weight": "model.language_model.layers.{}.post_feedforward_layernorm.weight",
            "layers.{}.layer_scalar": "model.language_model.layers.{}.layer_scalar",
            "norm.weight": "model.language_model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        self.from_hf_map = {
            # Canonical nested format (model.language_model.*)
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            "model.language_model.embed_tokens_per_layer.weight": "ple.tok_embeddings_per_layer.weight",
            "model.language_model.per_layer_model_projection.weight": "ple.per_layer_model_projection.weight",
            "model.language_model.per_layer_projection_norm.weight": "ple.per_layer_projection_norm.weight",
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.language_model.layers.{}.per_layer_input_gate.weight": "layers.{}.ple.per_layer_input_gate.weight",
            "model.language_model.layers.{}.per_layer_projection.weight": "layers.{}.ple.per_layer_projection.weight",
            "model.language_model.layers.{}.post_per_layer_input_norm.weight": "layers.{}.ple.post_per_layer_input_norm.weight",
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "model.language_model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            "model.language_model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            "model.language_model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
            "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
            "model.language_model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
            "model.language_model.norm.weight": "norm.weight",

            # Flat format (model.*)
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.embed_tokens_per_layer.weight": "ple.tok_embeddings_per_layer.weight",
            "model.per_layer_model_projection.weight": "ple.per_layer_model_projection.weight",
            "model.per_layer_projection_norm.weight": "ple.per_layer_projection_norm.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.per_layer_input_gate.weight": "layers.{}.ple.per_layer_input_gate.weight",
            "model.layers.{}.per_layer_projection.weight": "layers.{}.ple.per_layer_projection.weight",
            "model.layers.{}.post_per_layer_input_norm.weight": "layers.{}.ple.post_per_layer_input_norm.weight",
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
            "model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
            "model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
            "model.norm.weight": "norm.weight",

            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan checkpoint to HuggingFace format."""
        hf_state_dict = {}
        to_hf_map = self.to_hf_map

        for key, value in state_dict.items():
            if "moe.routed_experts.inner_experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in key '{key}'")
                layer_num = m.group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[abstract_key] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    moe_layer = next(
                        l for l in self.model_config.layers if getattr(l, "moe", None) is not None
                    )
                    split_values = self._split_experts_weights(
                        value,
                        moe_layer.moe.num_experts,
                    )
                    for expert_num in range(moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in key '{key}'")
                layer_num = m.group(0)

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value
            else:
                if key == "lm_head.weight" and getattr(self.model_config, "enable_weight_tying", True):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue
                hf_state_dict[new_key] = value

        # Prune fqn_to_index_mapping to strictly match exported tensors
        # (prevents HuggingFaceStorageWriter from creating empty/corrupt header slots for missing multimodal keys)
        if self.fqn_to_index_mapping is not None:
            self.fqn_to_index_mapping = {
                k: v for k, v in self.fqn_to_index_mapping.items() if k in hf_state_dict
            }

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace checkpoint to TorchTitan format."""
        self._validate_hf_rope_config(CosSinRoPE.Config)
        state_dict = {}
        expert_weights_by_layer = {}

        # 1. Handle Weight Tying (lm_head is tied to embed_tokens if omitted in HF checkpoint)
        if "lm_head.weight" not in hf_state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
            if embed_key not in hf_state_dict and "model.embed_tokens.weight" in hf_state_dict:
                embed_key = "model.embed_tokens.weight"
            if embed_key in hf_state_dict:
                hf_state_dict["lm_head.weight"] = hf_state_dict[embed_key]

        # 2. Map Keys
        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                indices = re.findall(r"\d+", key)
                if len(indices) < 2:
                    continue
                layer_num, expert_num = indices[0], indices[1]
                titan_abstract_key = self.from_hf_map.get(abstract_key)
                if titan_abstract_key is None:
                    continue
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][int(expert_num)] = value

                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:
                    moe_layer = next(
                        l for l in self.model_config.layers if getattr(l, "moe", None) is not None
                    )
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        moe_layer.moe.num_experts,
                    )
                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in HF key '{key}'")
                layer_num = m.group(0)

                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue
                state_dict[new_key] = value

        return state_dict
