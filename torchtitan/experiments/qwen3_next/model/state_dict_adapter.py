# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is adapted from torchtitan/models/qwen3/model/state_dict_adapter.py for Qwen3-Next.

We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model and vice versa.
This can enable us to do a parity test with the HF implementation and make sure that our results are
aligned with the HF implementation.

"""
import re
from typing import Any

from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter

from .args import Qwen3NextModelArgs


class Qwen3NextStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: Qwen3NextModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Full Attention module
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Linear Attention (GatedDeltaNet) module
            "model.layers.{}.linear_attn.in_proj_qkvz.weight": "layers.{}.attention.in_proj_qkvz.weight",
            "model.layers.{}.linear_attn.in_proj_ba.weight": "layers.{}.attention.in_proj_ba.weight",
            "model.layers.{}.linear_attn.conv1d.weight": "layers.{}.attention.conv1d.weight",
            "model.layers.{}.linear_attn.dt_bias": "layers.{}.attention.dt_bias",
            "model.layers.{}.linear_attn.A_log": "layers.{}.attention.A_log",
            "model.layers.{}.linear_attn.norm.weight": "layers.{}.attention.norm.weight",
            "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.attention.out_proj.weight",
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
            "model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1",
            "model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3",
            "model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2",
            "model.layers.{}.mlp.shared_expert_gate.weight": "layers.{}.moe.shared_gate.weight",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's weight.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key or "moe.shared_experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                # Store the GroupedExperts Weight metadata for from_hf()
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape

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
                        value, self.model_args.moe_args.num_experts
                    )

                    for expert_num in range(self.model_args.moe_args.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Concat separate expert's weight into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key or "mlp.shared_expert" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2 if "experts" in key else 1)
                if "experts" in key:
                    layer_num, expert_num = re.findall(r"\d+", key)
                else:
                    layer_num = re.search(r"\d+", key).group(0)
                    expert_num = None  # For shared
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    expert_num
                ] = value

                if isinstance(value, DTensor):
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        value.device_mesh,
                    )
                else:  # keep this path to be compatibile with offline conversion
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        self.model_args.moe_args.num_experts if "experts" in key else self.model_args.moe_args.num_shared_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        return state_dict