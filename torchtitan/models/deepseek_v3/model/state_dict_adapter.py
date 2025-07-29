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
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.kv_norm.weight",
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
        return torch.split(weight, weight.shape[0] // n_experts, dim=0)
    
    def _concate_experts_weights(self, weights: list[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate the weights of seprate experts into GroupedExpert weights.
        """
        pass

    def _quantization(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Quantize the weights from float32 to float8.
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
            if "layers" in key:
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
        """
        1. Dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concate seprate expert's wegiht into GroupedExperts' weight.
        """
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            state_dict[new_key] = value
        return state_dict
