# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import DeepSeekV3ModelArgs
import torch


class DeepSeekV3StateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        """
        StateDictAdapter for DeepSeekV3 model.
        NOTE: Now we observed the rotary embedding difference in torchtitan and huggingface. And this need to 
        be fixed to make the numerical results consistent between torchtitan and huggingface.
        """
        self.model_args = model_args
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention Module
            "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
            "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
            "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # MLP Module
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transfomer Layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE Module
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_expert.w1",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_expert.w3",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_expert.w2",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    
    def _split_experts_weights(self, weight: torch.Tensor, n_experts: int) -> list[torch.Tensor]:
        """
        Split the weights of the experts into a list of tensors.
        """
        split_weight = torch.split(weight, weight.shape[0] // n_experts, dim=0)
        return split_weight
    
    def _concatenate_expert_weights(self, expert_weights_by_layer: dict[str, Any], n_experts: int) -> torch.Tensor:
        """
        Concatenate the weights of seprate experts into GroupedExpert weights.
        """
        for layer, abstract_keys in list(expert_weights_by_layer.items()):
            for abstract_key, experts in list(abstract_keys.items()):
                # If we have all the experts for this abstract_key, concatenate them
                if len(experts) == n_experts:
                    sorted_expert_ids = sorted(experts.keys())
                    sorted_experts = [experts[i] for i in sorted_expert_ids]

                    # Here we need transpose because the torchtitan used nn.Linear() while HF used nn.Parameter
                    stacked_tensor = torch.stack(sorted_experts, dim=0).transpose(
                        1, 2
                    )

                    # Remove these experts from the tracking dict to free memory
                    del expert_weights_by_layer[layer][abstract_key]
                    if not expert_weights_by_layer[layer]:
                        del expert_weights_by_layer[layer]
                    
                    return stacked_tensor
        
        return None

    def _quantization(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Quantize the weights from float32 to float8. Export to HF f
        """
        pass

    def _dequantization(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Dequantize the weights from float8 to float32.
        """
        pass

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Quantize the weights from float32 to float8.
        2. Convert between the HF shape and the torchtitan shape.
        3. Split the GroupedExperts' weight into seprate expert's wegiht.
        """

        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.expert_bias" in key or "moe.tokens_per_expert" in key:
                continue
            
            if "moe.experts" in key:
                print("In to_hf, the key is: ", key, " value is: ", value.shape, "\n")
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                
                # Split expert weights into seperate expert weights
                split_values = self._split_experts_weights(value, self.model_args.n_routed_experts)
                for expert_num in range(0, self.model_args.n_routed_experts):                
                    new_key = new_key.format(layer_num, expert_num)
                    hf_state_dict[new_key] = split_values[expert_num].transpose(0, 1)

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                
                # Special case for `shared_expert`: torchtitan uses nn.Linear, and HF uses nn.Parameter
                # torchtitan shape: (1, s[1], s[2]) -> HF shape: (s[2], s[1])
                if "shared_expert" in key:
                    value = value.squeeze(0).transpose(0, 1)
                    print("shared_expert value ", value.shape)
                    
                hf_state_dict[new_key] = value
            
            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value
            
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concate seprate expert's wegiht into GroupedExperts' weight.
        """

        print("In from_hf, the state_dict key is: ", hf_state_dict.keys(), "\n")
        state_dict = {}
        
        expert_weights_by_layer = {} # {layer: {abstract_key: {expert_id: tensor}}}

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
                stacked_value = self._concatenate_expert_weights(expert_weights_by_layer, self.model_args.n_routed_experts)
                if stacked_value is not None:
                    state_dict[new_key] = value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                
                # Special case for `shared_expert`: torchtitan uses nn.Linear, and HF uses nn.Parameter
                # HF shape: (s[1], s[2]) -> torchtitan shape: (1, s[2], s[1])
                if "shared_experts" in key:
                    value = value.transpose(0, 1).unsqueeze(0)
                    
                state_dict[new_key] = value
            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        return state_dict
