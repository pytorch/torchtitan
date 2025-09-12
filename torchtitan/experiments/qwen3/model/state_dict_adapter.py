# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is adapted from torchtitan/models/llama3/model/state_dict_adapter.py.

We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model and vice versa.
This can enable us to do a parity test with the HF implementation and make sure that our results are
aligned with the HF implementation.

"""
import re
import torch
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import Qwen3ModelArgs


class Qwen3StateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: Qwen3ModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        if model_args.moe_enabled:
            self.from_hf_map.update({
                # MoE gating (token router)
                "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
                # Experts
                "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
                "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
                "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            })

    def _split_experts_weights(
        self, weight: torch.Tensor, n_experts: int,
    ) -> list[torch.Tensor]:
        """
        Split the weights of the experts into a list of tensors.
        """
        split_weight = torch.split(weight, weight.shape[0] // n_experts, dim=0)
        return split_weight

    def _concatenate_expert_weights(
        self, expert_weights_by_layer: dict[str, Any], n_experts: int
    ) -> torch.Tensor:
        """
        Concatenate the weights of seprate experts into GroupedExpert weights.
        """
        for layer, abstract_keys in list(expert_weights_by_layer.items()):
            for abstract_key, experts in list(abstract_keys.items()):
                # If we have all the experts for this abstract_key, concatenate them
                if len(experts) == n_experts:
                    sorted_expert_ids = sorted(experts.keys())
                    sorted_experts = [experts[i] for i in sorted_expert_ids]
                    stacked_tensor = torch.stack(sorted_experts, dim=0)

                    # Remove these experts from the tracking dict to free memory
                    del expert_weights_by_layer[layer][abstract_key]
                    if not expert_weights_by_layer[layer]:
                        del expert_weights_by_layer[layer]

                    return stacked_tensor

        return None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:

        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                # Split expert weights into seperate expert weights
                # This will cause all_gather and OOM
                split_values = self._split_experts_weights(
                    value, self.model_args.moe_args.num_experts
                )

                for expert_num in range(0, self.model_args.moe_args.num_experts):
                    new_key = new_abstract_key.format(layer_num, expert_num)
                    hf_state_dict[new_key] = split_values[expert_num].squeeze()
            
            elif "expert_bias" in key:
                continue
        
            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][abstract_key] = {}
                expert_weights_by_layer[layer_num][abstract_key][expert_num] = value

                # try to concat the expert's weight into GroupedExperts' weight.
                stacked_value = self._concatenate_expert_weights(
                    expert_weights_by_layer, self.model_args.moe_args.num_experts
                )
                if stacked_value is not None:
                    state_dict[new_key] = stacked_value
            
            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        return state_dict
