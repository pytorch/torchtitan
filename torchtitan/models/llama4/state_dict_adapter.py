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

from .model import Llama4Model


class Llama4StateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: Llama4Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
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
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}

        # Keeps track of TT fqn values to combine into one HF fqn later
        # {hf_fqn : {tt_fqn1 : value}, {tt_fqn2 : value}, ...}
        to_combine = defaultdict(dict)
        for key, value in state_dict.items():
            if "layers" in key:
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in to_hf_map:
                # do direct mapping
                if key in "layers.{}.moe.experts.w2":
                    # we transpose the expert weights for torchtitan optimization purpose
                    value = value.transpose(-1, -2)

                new_key = to_hf_map[key]
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value
            elif key in [
                "layers.{}.moe.experts.w1",
                "layers.{}.moe.experts.w3",
            ]:
                # handle collecting values to combine
                hf_abstract_key = (
                    "language_model.model.layers.{}.feed_forward.experts.gate_up_proj"
                )
                # pyrefly: ignore [unnecessary-comparison]
                if hf_abstract_key is None:
                    continue
                to_combine[hf_abstract_key.format(layer_num)][
                    key.format(layer_num)
                ] = value

        # combine collected values
        for hf_fqn, tt_fqn_map in to_combine.items():
            # pyrefly: ignore [missing-attribute]
            layer_num = re.search(r"\d+", hf_fqn).group(0)
            combine_values = []
            # put into correct order to combine
            for tt_abstract_key in [
                "layers.{}.moe.experts.w1",
                "layers.{}.moe.experts.w3",
            ]:
                tt_key = tt_abstract_key.format(layer_num)
                # we transpose the expert weights for torchtitan optimization purpose
                combine_values.append(tt_fqn_map[tt_key].transpose(-1, -2))

            value = torch.cat(combine_values, dim=-1)
            hf_state_dict[hf_fqn] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in self.from_hf_map:
                # do direct mapping
                if (
                    key
                    == "language_model.model.layers.{}.feed_forward.experts.down_proj"
                ):
                    # we transpose the expert weights for torchtitan optimization purpose
                    value = value.transpose(-1, -2)

                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                state_dict[new_key] = value
            elif (
                key
                == "language_model.model.layers.{}.feed_forward.experts.gate_up_proj"
            ):
                # handle splitting values
                w1, w3 = value.chunk(2, dim=-1)
                # we transpose the expert weights for torchtitan optimization purpose
                w1, w3 = w1.transpose(-1, -2), w3.transpose(-1, -2)
                # split_vals = [val.transpose(-1, -2) for val in split_vals]
                state_dict["layers.{}.moe.experts.w1".format(layer_num)] = w1
                state_dict["layers.{}.moe.experts.w3".format(layer_num)] = w3

        return state_dict
