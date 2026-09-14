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
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
            "model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
            "model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
            "model.norm.weight": "norm.weight",

            "lm_head.weight": "lm_head.weight",
        }

        # Dynamically populate MLP and MoE mappings per-layer based on the config
        for i, layer in enumerate(self.model_config.layers):
            if getattr(layer, "moe", None) is not None:
                # Routed Experts (Gemma-4 HF stores them as stacked tensors gate_up_proj and down_proj)
                self.to_hf_map[f"layers.{i}.moe.routed_experts.inner_experts.w1_EFD"] = f"model.language_model.layers.{i}.experts.gate_up_proj"
                self.to_hf_map[f"layers.{i}.moe.routed_experts.inner_experts.w3_EFD"] = f"model.language_model.layers.{i}.experts.gate_up_proj"
                self.to_hf_map[f"layers.{i}.moe.routed_experts.inner_experts.w2_EDF"] = f"model.language_model.layers.{i}.experts.down_proj"
                self.to_hf_map[f"layers.{i}.moe.router.gate.weight"] = f"model.language_model.layers.{i}.router.proj.weight"
                self.to_hf_map[f"layers.{i}.moe.router.scale"] = f"model.language_model.layers.{i}.router.scale"
                self.to_hf_map[f"layers.{i}.moe.router.per_expert_scale"] = f"model.language_model.layers.{i}.router.per_expert_scale"
                
                # We handle splitting/concatenating gate_up_proj in from_hf/to_hf custom logic.
                self.from_hf_map[f"model.language_model.layers.{i}.experts.down_proj"] = f"layers.{i}.moe.routed_experts.inner_experts.w2_EDF"
                self.from_hf_map[f"model.layers.{i}.experts.down_proj"] = f"layers.{i}.moe.routed_experts.inner_experts.w2_EDF"
                self.from_hf_map[f"model.language_model.layers.{i}.experts.gate_up_proj"] = f"layers.{i}.moe.routed_experts.inner_experts.gate_up_proj"
                self.from_hf_map[f"model.layers.{i}.experts.gate_up_proj"] = f"layers.{i}.moe.routed_experts.inner_experts.gate_up_proj"
                self.from_hf_map[f"model.language_model.layers.{i}.router.proj.weight"] = f"layers.{i}.moe.router.gate.weight"
                self.from_hf_map[f"model.layers.{i}.router.proj.weight"] = f"layers.{i}.moe.router.gate.weight"
                
                self.from_hf_map[f"model.language_model.layers.{i}.router.scale"] = f"layers.{i}.moe.router.scale"
                self.from_hf_map[f"model.layers.{i}.router.scale"] = f"layers.{i}.moe.router.scale"
                self.from_hf_map[f"model.language_model.layers.{i}.router.per_expert_scale"] = f"layers.{i}.moe.router.per_expert_scale"
                self.from_hf_map[f"model.layers.{i}.router.per_expert_scale"] = f"layers.{i}.moe.router.per_expert_scale"
                
                # MoE Normalization Mappings
                self.to_hf_map[f"layers.{i}.moe.routed_experts.inner_experts.moe_ffn_norm.weight"] = f"model.language_model.layers.{i}.pre_feedforward_layernorm_2.weight"
                self.to_hf_map[f"layers.{i}.post_ffn_norm_1.weight"] = f"model.language_model.layers.{i}.post_feedforward_layernorm_1.weight"
                self.to_hf_map[f"layers.{i}.moe_post_ffn_norm.weight"] = f"model.language_model.layers.{i}.post_feedforward_layernorm_2.weight"

                self.from_hf_map[f"model.language_model.layers.{i}.pre_feedforward_layernorm_2.weight"] = f"layers.{i}.moe.routed_experts.inner_experts.moe_ffn_norm.weight"
                self.from_hf_map[f"model.layers.{i}.pre_feedforward_layernorm_2.weight"] = f"layers.{i}.moe.routed_experts.inner_experts.moe_ffn_norm.weight"
                self.from_hf_map[f"model.language_model.layers.{i}.post_feedforward_layernorm_2.weight"] = f"layers.{i}.moe_post_ffn_norm.weight"
                self.from_hf_map[f"model.layers.{i}.post_feedforward_layernorm_2.weight"] = f"layers.{i}.moe_post_ffn_norm.weight"
                self.from_hf_map[f"model.language_model.layers.{i}.post_feedforward_layernorm_1.weight"] = f"layers.{i}.post_ffn_norm_1.weight"
                self.from_hf_map[f"model.layers.{i}.post_feedforward_layernorm_1.weight"] = f"layers.{i}.post_ffn_norm_1.weight"

            if getattr(layer, "feed_forward", None) is not None:
                # Regular Dense MLP (used for both dense models and shared experts in MoE)
                self.to_hf_map[f"layers.{i}.feed_forward.w1.weight"] = f"model.language_model.layers.{i}.mlp.gate_proj.weight"
                self.to_hf_map[f"layers.{i}.feed_forward.w3.weight"] = f"model.language_model.layers.{i}.mlp.up_proj.weight"
                self.to_hf_map[f"layers.{i}.feed_forward.w2.weight"] = f"model.language_model.layers.{i}.mlp.down_proj.weight"
                self.from_hf_map[f"model.language_model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.feed_forward.w1.weight"
                self.from_hf_map[f"model.language_model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.feed_forward.w3.weight"
                self.from_hf_map[f"model.language_model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.feed_forward.w2.weight"
                self.from_hf_map[f"model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.feed_forward.w1.weight"
                self.from_hf_map[f"model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.feed_forward.w3.weight"
                self.from_hf_map[f"model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.feed_forward.w2.weight"

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan checkpoint to HuggingFace format."""
        hf_state_dict = {}
        to_hf_map = self.to_hf_map

        temp_experts = {}
        for key, value in state_dict.items():
            if "moe.routed_experts.inner_experts" in key:
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue

                if "w1_EFD" in key or "w3_EFD" in key:
                    if new_key not in temp_experts:
                        temp_experts[new_key] = {}
                    which = "w1" if "w1_EFD" in key else "w3"
                    temp_experts[new_key][which] = value
                    if "w1" in temp_experts[new_key] and "w3" in temp_experts[new_key]:
                        w1_val = temp_experts[new_key].pop("w1")
                        w3_val = temp_experts[new_key].pop("w3")
                        del temp_experts[new_key]
                        if isinstance(w1_val, DTensor):
                            assert isinstance(w3_val, DTensor)
                            cat_local = torch.cat(
                                [w1_val.to_local(), w3_val.to_local()], dim=1
                            )
                            hf_state_dict[new_key] = DTensor.from_local(
                                cat_local,
                                device_mesh=w1_val.device_mesh,
                                placements=w1_val.placements,
                                run_check=False,
                            )
                        else:
                            hf_state_dict[new_key] = torch.cat([w1_val, w3_val], dim=1)
                elif "w2_EDF" in key:
                    hf_state_dict[new_key] = value
                elif "moe_ffn_norm.weight" in key:
                    hf_state_dict[new_key] = value
                else:
                    raise ValueError(f"Unexpected inner_experts parameter: {key}")

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in key '{key}'")
                layer_num = m.group(0)

                new_key = to_hf_map.get(key)
                if new_key is None:
                    new_key = to_hf_map.get(abstract_key)
                    if new_key is not None:
                        new_key = new_key.format(layer_num)
                
                if new_key is None:
                    continue
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

            elif "experts.gate_up_proj" in key:
                # Custom handling for Gemma-4 MoE routed experts gate_up_proj
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    continue
                layer_num = m.group(0)

                # Split gate_up_proj into w1 and w3
                # Shape is [num_experts, 2 * hidden_dim, dim]
                # We chunk along dim 1
                if isinstance(value, DTensor):
                    w1_local, w3_local = torch.chunk(value.to_local(), 2, dim=1)
                    w1_val = DTensor.from_local(
                        w1_local.contiguous(),
                        device_mesh=value.device_mesh,
                        placements=value.placements,
                        run_check=False,
                    )
                    w3_val = DTensor.from_local(
                        w3_local.contiguous(),
                        device_mesh=value.device_mesh,
                        placements=value.placements,
                        run_check=False,
                    )
                else:
                    w1_val, w3_val = torch.chunk(value, 2, dim=1)
                state_dict[f"layers.{layer_num}.moe.routed_experts.inner_experts.w1_EFD"] = w1_val
                state_dict[f"layers.{layer_num}.moe.routed_experts.inner_experts.w3_EFD"] = w3_val

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in HF key '{key}'")
                layer_num = m.group(0)

                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    new_key = self.from_hf_map.get(abstract_key)
                    if new_key is not None:
                        new_key = new_key.format(layer_num)
                
                if new_key is None:
                    continue
                state_dict[new_key] = value
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue
                state_dict[new_key] = value

        return state_dict
