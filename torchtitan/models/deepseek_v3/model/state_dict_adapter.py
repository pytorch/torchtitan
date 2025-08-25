# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import DeepSeekV3ModelArgs
from .key_mappings import get_hf_to_tt_map


class DeepSeekV3StateDictAdapter(StateDictAdapter):
    """
    StateDictAdapter for DeepSeekV3 model.
    """

    def __init__(self, model_args: DeepSeekV3ModelArgs, hf_assets_path: str | None):
        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = get_hf_to_tt_map()

    def _split_experts_weights(
        self, weight: torch.Tensor, n_experts: int
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
        Concatenate the weights of separate experts into GroupedExpert weights.
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
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's wegiht.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                # Split expert weights into seperate expert weights
                split_values = self._split_experts_weights(
                    value, self.model_args.moe_args.num_experts
                )

                for expert_num in range(0, self.model_args.moe_args.num_experts):
                    new_key = new_abstract_key.format(layer_num, expert_num)
                    hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert between the HF shape and the torchtitan shape.

        Note: When loading from HF checkpoints via DCP, the conversion from HF format
        to TorchTitan format (including expert concatenation and dequantization) is
        handled by DeepSeekV3HuggingFaceStorageReader. This method is primarily used
        for direct state dict conversions outside of DCP loading.
        """
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key and "mlp.experts" not in key:
                # Handle non-expert layer parameters
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value
            elif "mlp.experts" not in key:
                # Handle non-layer, non-expert parameters
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value
            # Expert parameters are handled by the storage reader during DCP loading

        return state_dict
