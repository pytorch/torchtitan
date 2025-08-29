# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger()

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import TransformerModelArgs


class Llama4StateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: TransformerModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
            "language_model.model.norm.weight": "norm.weight",
            "language_model.lm_head.weight": "output.weight",
            "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "language_model.model.layers.{}.feed_forward.router.weight": "layers.{}.moe.router.gate.weight",
            "language_model.model.layers.{}.feed_forward.experts.down_proj": "layers.{}.moe.experts.w2",
            None: "layers.{}.moe.expert_bias",
            "language_model.model.layers.{}.feed_forward.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "language_model.model.layers.{}.feed_forward.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "language_model.model.layers.{}.feed_forward.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "vision_model.model.layers.{}.self_attn.k_proj.weight": None,
            "vision_model.model.layers.{}.self_attn.v_proj.bias": None,
            "vision_model.model.layers.{}.mlp.fc2.weight": None,
            "vision_model.model.layers.{}.post_attention_layernorm.bias": None,
            "vision_model.model.layers.{}.post_attention_layernorm.weight": None,
            "vision_model.class_embedding": None,
            "vision_model.model.layers.{}.self_attn.v_proj.weight": None,
            "vision_model.model.layers.{}.self_attn.o_proj.bias": None,
            "vision_model.layernorm_post.bias": None,
            "vision_model.layernorm_pre.bias": None,
            "vision_model.positional_embedding_vlm": None,
            "vision_model.model.layers.{}.input_layernorm.weight": None,
            "multi_modal_projector.linear_1.weight": None,
            "vision_model.layernorm_pre.weight": None,
            "vision_model.model.layers.{}.self_attn.k_proj.bias": None,
            "vision_model.model.layers.{}.self_attn.q_proj.bias": None,
            "vision_model.model.layers.{}.input_layernorm.bias": None,
            "vision_model.patch_embedding.linear.weight": None,
            "vision_model.layernorm_post.weight": None,
            "vision_model.model.layers.{}.mlp.fc1.weight": None,
            "vision_model.model.layers.{}.mlp.fc2.bias": None,
            "vision_model.model.layers.{}.self_attn.q_proj.weight": None,
            "vision_model.model.layers.{}.self_attn.o_proj.weight": None,
            "vision_model.model.layers.{}.mlp.fc1.bias": None,
            "vision_model.vision_adapter.mlp.fc1.weight": None,
            "vision_model.vision_adapter.mlp.fc2.weight": None,
        }

        self.combination_plan = {
            "language_model.model.layers.{}.feed_forward.experts.gate_up_proj": [
                "layers.{}.moe.experts.w1",
                "layers.{}.moe.experts.w3",
            ]
        }

        # reverse of combination plan: maps fqns to the fqn they are combined into
        self.reverse_combination_plan = {
            value: key
            for key, value_list in self.combination_plan.items()
            for value in value_list
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}

        to_combine = defaultdict(dict)
        for key, value in state_dict.items():
            if "layers" in key:
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in to_hf_map:
                # do direct mapping
                if key in "layers.{}.moe.experts.w2":
                    value = value.transpose(-1, -2)

                new_key = to_hf_map[key]
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value
            elif key in self.reverse_combination_plan:
                # handle collecting values to combine
                hf_abstract_key = self.reverse_combination_plan[key]
                if hf_abstract_key is None:
                    continue
                to_combine[hf_abstract_key.format(layer_num)][
                    key.format(layer_num)
                ] = value

        # combine collected values
        for hf_fqn, tt_fqn_map in to_combine.items():
            layer_num = re.search(r"\d+", hf_fqn).group(0)
            hf_abstract_key = re.sub(r"(\d+)", "{}", hf_fqn, count=1)
            combine_values = []
            # use combination_plan to ensure correct order before concatenation
            for tt_abstract_key in self.combination_plan[hf_abstract_key]:
                tt_key = tt_abstract_key.format(layer_num)
                print("tt_key", tt_key, "shape", tt_fqn_map[tt_key].shape)
                combine_values.append(tt_fqn_map[tt_key].transpose(-1, -2))

            value = torch.cat(combine_values, dim=-1)
            hf_state_dict[hf_fqn] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in self.from_hf_map:
                # do direct mapping
                if key in [
                    "language_model.model.layers.{}.feed_forward.experts.down_proj",
                ]:
                    value = value.transpose(-1, -2)

                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                state_dict[new_key] = value
            elif key in [
                "language_model.model.layers.{}.feed_forward.experts.gate_up_proj"
            ]:
                # handle splitting values
                split_vals = value.chunk(2, dim=-1)
                split_vals = [val.transpose(-1, -2) for val in split_vals]
                for new_key, split_val in zip(self.combination_plan[key], split_vals):
                    new_key = new_key.format(layer_num)
                    state_dict[new_key] = split_val

        return state_dict
