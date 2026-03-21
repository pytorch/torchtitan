# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapt checkpoints between HF Mixtral-8x7B-v0.1 format and torchtitan format.

HF uses per-expert 2D weights under block_sparse_moe.experts.{j}.w1/w2/w3.
TorchTitan uses stacked 3D GroupedExperts tensors under moe.experts.w1/w2/w3.
"""

import re
from typing import Any

from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter

from .model import MixtralModel


class MixtralStateDictAdapter(MoEStateDictAdapter):
    def __init__(
        self, model_config: MixtralModel.Config, hf_assets_path: str | None
    ):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # Norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE experts (per-expert 2D → stacked 3D GroupedExperts)
            "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "layers.{}.moe.experts.w3",
            # Router
            "model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.moe.router.gate.weight",
            # Output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HF format.

        Splits 3D GroupedExperts tensors into per-expert 2D weights.
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

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[
                        abstract_key
                    ] = value.device_mesh

                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    split_values = self._split_experts_weights(
                        value,
                        # pyrefly: ignore [missing-attribute]
                        self.model_config.layer.moe.num_experts,
                    )
                    for expert_num in range(
                        # pyrefly: ignore [missing-attribute]
                        self.model_config.layer.moe.num_experts
                    ):
                        new_key = new_abstract_key.format(
                            layer_num, expert_num
                        )
                        hf_state_dict[new_key] = (
                            split_values[expert_num].squeeze()
                        )

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
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to torchtitan format.

        Stacks per-expert 2D weights into 3D GroupedExperts tensors.
        """
        state_dict = {}
        expert_weights_by_layer = {}

        for key, value in hf_state_dict.items():
            if "block_sparse_moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                # w1/w2/w3 contain digits — take only first two matches
                nums = re.findall(r"\d+", key)
                layer_num, expert_num = nums[0], nums[1]
                titan_abstract_key = self.from_hf_map[abstract_key]
                assert titan_abstract_key is not None
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        # pyrefly: ignore [missing-attribute]
                        self.model_config.layer.moe.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                # pyrefly: ignore [unsupported-operation]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                # pyrefly: ignore [unsupported-operation]
                state_dict[new_key] = value

        return state_dict
