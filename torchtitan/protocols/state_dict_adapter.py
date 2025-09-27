# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from .quantization import calculate_scale_shape, dequantize_from_fp8

logger = logging.getLogger()

from .model import BaseModelArgs


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    Args:
        model_args: for initializing the model's memory space
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    @abstractmethod
    def __init__(
        self,
        model_args: BaseModelArgs,
        hf_assets_path: str | None,
    ):
        pass

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        pass


class StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter base class which provides convenient default behavior to build fqn_to_index_mapping"""

    def __init__(
        self,
        model_args: BaseModelArgs,
        hf_assets_path: str | None,
    ):
        if hf_assets_path:
            mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
            try:
                with open(mapping_path, "r") as f:
                    hf_safetensors_indx = json.load(f)
            except FileNotFoundError:
                logger.warning(
                    f"model.safetensors.index.json not found at hf_assets_path: {mapping_path}. \
                    Defaulting to saving a single safetensors file if checkpoint is saved in HF format"
                )
                hf_safetensors_indx = None

            if hf_safetensors_indx:
                self.fqn_to_index_mapping = {}
                for hf_key, raw_indx in hf_safetensors_indx["weight_map"].items():
                    indx = re.search(r"\d+", raw_indx).group(0)
                    self.fqn_to_index_mapping[hf_key] = int(indx)
            else:
                self.fqn_to_index_mapping = None

    def _calculate_strided_shard_shard_indices(
        self,
        strided_shard_dim_degree: int,
        strided_shard_dim_rank: int,
        shard_dim_degree: int,
        shard_dim_rank: int,
        dim_size_to_split: int,
    ) -> tuple[int, int]:
        """
        Given a [StridedShard(dim=i), Shard(dim=i)] placement, caculate the start index
        and end index on dim-i for GPU rank (strided_shard_dim_degree, shard_dim_rank)

        GPU Layout (strided_shard_rank, shard_rank):

        StridedShard Rank                  Shard rank
                        ┌─────────────────┐
                    0   │    GPU(0, 0)    │  0
                    ────┼─────────────────┤
                    1   │    GPU(1, 0)    │
                    ────┼─────────────────┤
                    2   │    GPU(2, 0)    │
                  ──────┼─────────────────┼────
                    0   │    GPU(0, 1)    │  1
                    ────┼─────────────────┤
                    1   │    GPU(1, 1)    │
                    ────┼─────────────────┤
                    2   │    GPU(2, 1)    │
                        └─────────────────┘

        Calulate the start_index from inner dimesion (Shard(dim=i)) to outer demension (StridedShard(dim=i)).
        """

        block_size = dim_size_to_split // (strided_shard_dim_degree * shard_dim_degree)

        # Error out if can not evenly divded
        if (
            block_size * (strided_shard_dim_degree * shard_dim_degree)
            != dim_size_to_split
        ):
            raise ValueError(
                f"Not supported split for strided_shard_dim_degree {strided_shard_dim_degree}, "
                f"shard_dim_degree {shard_dim_degree}, dim_size_to_split {dim_size_to_split}"
            )

        start_index = block_size * (
            strided_shard_dim_degree * shard_dim_rank + strided_shard_dim_rank
        )
        end_index = start_index + block_size

        return start_index, end_index

    def _caculate_indices_from_placements(
        self,
        dim: int,
        dim_size: int,
        dtensor_placements: tuple,
        device_mesh: "DeviceMesh",
    ):

        mesh_names = []
        dim_i_placements = []

        # Find all the device mesh dimensios that shard on dim-i
        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if placement.dim == dim:
                mesh_names.append(name)
                dim_i_placements.append(placement)

        # Calculate local expert indices based on sharding strategy
        start_index, end_index = None, None
        if len(dim_i_placements) == 2:
            # Handle StridedShard(i) + Shard(i) case
            assert isinstance(
                dim_i_placements[0], _StridedShard
            ), "Expected StridedShard as first placement"

            strided_shard_mesh = device_mesh[mesh_names[0]]
            shard_mesh = device_mesh[mesh_names[1]]

            strided_degree = strided_shard_mesh.size()
            strided_rank = strided_shard_mesh.get_local_rank()
            shard_degree = shard_mesh.size()
            shard_rank = shard_mesh.get_local_rank()

            start_index, end_index = self._calculate_strided_shard_shard_indices(
                strided_degree, strided_rank, shard_degree, shard_rank, dim_size
            )

        elif len(dim_i_placements) == 1:
            # Handle single Shard(i) case
            assert not isinstance(
                dim_i_placements[0], _StridedShard
            ), "Expected regular Shard, not StridedShard"

            shard_mesh = device_mesh[mesh_names[0]]
            shard_degree = shard_mesh.size()
            shard_rank = shard_mesh.get_local_rank()

            block_size = dim_size // shard_degree
            if block_size * shard_degree != dim_size:
                raise ValueError(
                    f"Dim {dim} size ({dim_size}) cannot be evenly divided by shard degree ({shard_degree})"
                )

            start_index = block_size * shard_rank
            end_index = start_index + block_size

        elif len(dim_i_placements) == 0:
            # No need to split on this dimension
            return start_index, end_index

        else:
            raise NotImplementedError(
                f"Unsupported DTensor placements for GroupedExperts: {dtensor_placements} {dim_i_placements} {mesh_names}"
            )

        return start_index, end_index

    def _get_local_experts_weights(
        self,
        abstract_key: str,
        titan_abstract_key: str,
        layer_id: str,
        grouped_expert_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Split GroupedExperts weight into individual expert weights for local processing.

        This method handles various sharding strategies for expert weights:
        - FSDP + EP: StridedShard(0)Shard(0) or Shard(0)
        - FSDP + ETP + EP: StridedShard(0)Shard(0)Shard(1/2) or StridedShard(1)Shard(0)Shard(1/2)

        Args:
            abstract_key: HuggingFace templage key with {} placeholders for layer and expert IDs
            titan_abstract_key: TorchTitan templage key with {} placeholders for layer and expert IDs
            layer_id: Layer identifier
            grouped_expert_weight: DTensor containing all experts' weights

        Returns:
            Dictionary mapping individual expert keys to their DTensor weights
        """
        device_mesh = grouped_expert_weight.device_mesh
        dtensor_placements = grouped_expert_weight.placements

        # Step 1: Extract dimension-0 placement information
        num_experts = grouped_expert_weight.shape[0]
        start_index, end_index = self._caculate_indices_from_placements(
            dim=0,
            dim_size=num_experts,
            dtensor_placements=dtensor_placements,
            device_mesh=device_mesh,
        )
        assert (
            start_index is not None and end_index is not None
        ), "Start index and end index can not be None on dim-0!"

        # Step 2: Store indices for potential future use in from_hf()
        self.local_experts_indices[titan_abstract_key] = (start_index, end_index)

        # Step 3: Create new placements for individual expert weights
        new_placements = []
        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if placement.dim == 0:
                # Convert dim-0 sharding to replication for individual experts
                new_placements.append(Replicate())
            elif isinstance(placement, Shard):
                # Keep other shard dimensions (individual expert weight has 2D)
                new_placements.append(Shard(placement.dim))
            elif isinstance(placement, _StridedShard):
                # Keep strided shard with same parameters
                new_placements.append(
                    _StridedShard(placement.dim, placement.split_factor)
                )
            else:
                raise ValueError(f"Unsupported placement type: {type(placement)}")

        # Step 4: Create individual expert DTensors
        assert isinstance(
            grouped_expert_weight, DTensor
        ), "Expected DTensor for grouped expert weight"

        local_grouped_weights = grouped_expert_weight._local_tensor
        expected_local_experts = end_index - start_index

        if local_grouped_weights.shape[0] != expected_local_experts:
            raise ValueError(
                f"Local tensor shape mismatch: expected {expected_local_experts} experts, "
                f"got {local_grouped_weights.shape[0]}"
            )

        local_expert_tensors = {}
        for expert_id in range(start_index, end_index):
            expert_key = abstract_key.format(layer_id, expert_id)
            local_expert_index = expert_id - start_index

            # Extract individual expert weight and add batch dimension temporarily
            expert_weight = local_grouped_weights[local_expert_index, :, :].unsqueeze(0)

            # Create DTensor and remove batch dimension (experts dimension is removed)
            expert_dtensor = DTensor.from_local(
                expert_weight, device_mesh, new_placements, run_check=False
            ).squeeze(0)

            local_expert_tensors[expert_key] = expert_dtensor

        return local_expert_tensors

    def _concatenate_expert_weights_dtensor(
        self,
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]],
        abstract_key: str,
        layer_num: str,
        device_mesh: "DeviceMesh",
    ) -> torch.Tensor | None:
        """
        Args:
            expert_weights_by_layer: Dictionary tracking expert weights by layer, abstract key, and expert ID.
                Structure: {
                    layer_id: {
                        abstract_key: {
                            expert_id: tensor_weight
                        }
                    }
                }
                Used to collect individual expert weights before concatenating them into GroupedExperts.
            abstract_key: TorchTitan templage key with {} placeholders for layer and expert IDs
            layer_num: Layer identifier
            device_mesh: DeviceMesh for the target GroupedExperts weight DTensor

        Returns:
            Concatenated GroupedExperts weight DTensor if all experts are available, otherwise None
        """
        # If we have all the experts for this abstract_key, concatenate them
        experts = expert_weights_by_layer[layer_num][abstract_key]
        expected_n_experts = (
            self.local_experts_indices[abstract_key][1]
            - self.local_experts_indices[abstract_key][0]
        )
        if len(experts) < expected_n_experts:
            return None

        sorted_expert_ids = sorted(experts.keys())
        sorted_experts = [experts[i] for i in sorted_expert_ids]
        local_tensor = torch.stack(sorted_experts, dim=0)._local_tensor

        assert (
            abstract_key in self.grouped_expert_weight_placements
            and abstract_key in self.grouped_expert_weight_shape
        ), "GroupedExperts weight metadata (placements, shape) can not be None!"

        stacked_dtensor = DTensor.from_local(
            local_tensor,
            device_mesh,
            self.grouped_expert_weight_placements[abstract_key],
            run_check=False,
        )

        del expert_weights_by_layer[layer_num][abstract_key]
        if not expert_weights_by_layer[layer_num]:
            del expert_weights_by_layer[layer_num]

        return stacked_dtensor

    def _split_experts_weights(
        self, weight: torch.Tensor, n_experts: int
    ) -> list[torch.Tensor]:
        """
        Split the weights of the experts into a list of tensors. Used for offline conversion.

        NOTE: If we use this function for online conversion, torch.split() might incur communication
        to gather the weight, which causing OOM.

        """
        split_weight = torch.split(weight, weight.shape[0] // n_experts, dim=0)
        return split_weight

    def _concatenate_expert_weights(
        self,
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]],
        abstract_key: str,
        layer_num: str,
        n_experts: int,
    ) -> torch.Tensor | None:
        """
        Concatenated GroupedExperts weight using torch.stack(). Used for offline conversion.

        Args:
            expert_weights_by_layer: Dictionary tracking expert weights by layer, abstract key, and expert ID.
                Structure: {
                    layer_id: {
                        abstract_key: {
                            expert_id: tensor_weight
                        }
                    }
                }
                Used to collect individual expert weights before concatenating them into GroupedExperts.
            abstract_key: TorchTitan templage key with {} placeholders for layer and expert IDs
            layer_num: Layer identifier
            n_experts: Number of experts in the GroupedExperts module

        Returns:
            Concatenated GroupedExperts weight if all experts are available, otherwise None
        """
        # If we have all the experts for this abstract_key, concatenate them
        experts = expert_weights_by_layer[layer_num][abstract_key]
        if len(experts) < n_experts:
            return None

        sorted_expert_ids = sorted(experts.keys())
        sorted_experts = [experts[i] for i in sorted_expert_ids]
        stacked_tensor = torch.stack(sorted_experts, dim=0)

        del expert_weights_by_layer[layer_num][abstract_key]
        if not expert_weights_by_layer[layer_num]:
            del expert_weights_by_layer[layer_num]

        return stacked_tensor

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Dequantize the weights from float8 to float32.
        """

        scale_inv_keys = []
        for key, weight in state_dict.items():
            if key.endswith(".weight") and key + "_scale_inv" in state_dict:
                scale_inv = state_dict[key + "_scale_inv"]
                dequantized_weight = dequantize_from_fp8(
                    weight, scale_inv, dtype=torch.float32
                )
                # update the weight and remove the scale_inv tensor
                state_dict[key] = dequantized_weight
                scale_inv_keys.append(key + "_scale_inv")

        for key in scale_inv_keys:
            state_dict.pop(key)

        return state_dict

    def _add_quantization_scale_inv_tensors(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Add quantization scale tensors the state_dict.
        """
        non_quantized_keys = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "norm.weight",
            "lm_head.weight",
            "embed_tokens.weight",
            "mlp.gate.weight",
        ]

        weight_scale_inv_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".weight") and not any(
                non_quantized_key in key for non_quantized_key in non_quantized_keys
            ):
                expected_scale_shape = calculate_scale_shape(value)
                # add weight_scale_inv to the state_dict
                weight_scale_inv_state_dict[key + "_scale_inv"] = torch.ones(
                    expected_scale_shape, dtype=torch.float32
                )

        state_dict.update(weight_scale_inv_state_dict)
        return state_dict
