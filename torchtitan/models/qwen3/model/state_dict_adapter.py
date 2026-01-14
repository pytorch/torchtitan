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
from typing import Any

from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter

from .args import Qwen3ModelArgs


class Qwen3StateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: Qwen3ModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MLP module for non-MoE
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's wegiht.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                # Store the GroupedExperts Weight metadata for from_hf()
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    # Split GroupedExperts weight to local individual expert weights
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)

                else:
                    # keep this path for offline conversion
                    split_values = self._split_experts_weights(
                        value,
                        # pyrefly: ignore [missing-attribute]
                        self.model_args.moe_args.num_experts,
                    )

                    # pyrefly: ignore [missing-attribute]
                    for expert_num in range(self.model_args.moe_args.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                if self.model_args.enable_weight_tying and key == "output.weight":
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Concate separate expert's wegiht into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        if (
            # pyrefly: ignore [missing-attribute]
            self.model_args.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                assert titan_abstract_key is not None
                new_key = titan_abstract_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                if isinstance(value, DTensor):
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:  # keep this path to be compatible with offline conversion
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        # pyrefly: ignore [missing-attribute]
                        self.model_args.moe_args.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                # pyrefly: ignore [missing-attribute]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                # pyrefly: ignore [unsupported-operation]
                state_dict[new_key] = value

        return state_dict
