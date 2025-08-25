# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re
from typing import Any, Dict

import torch
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed.tensor import DTensor

from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from torchtitan.tools.logging import logger

from .args import DeepSeekV3ModelArgs
from .quantization import calculate_scale_shape, dequantize_from_fp8


class DeepSeekV3StateDictAdapter(StateDictAdapter):
    """
    StateDictAdapter for DeepSeekV3 model.
    """

    def __init__(
        self,
        model_args: DeepSeekV3ModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)
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
            # Transformer Layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE Module
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Store metadata for GroupedExperts <-> individual experts conversion
        self.grouped_expert_weight_placements = {}  # {titan_abstract_key: placements}
        self.grouped_expert_weight_shape = {}  # {titan_abstract_key: shape}
        self.local_experts_indices = {}  # {titan_abstract_key: (start_idx, end_idx)}

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
                f"Not supported split for strided_shard_dim_degree {strided_shard_dim_degree}, shard_dim_degree {shard_dim_degree}, dim_size_to_split {dim_size_to_split}"
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
        device_mesh: DeviceMesh,
    ):

        mesh_names = []
        dim_i_placements = []

        # Find all the device mesh dimensios that shard on dim-i
        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            print(
                f"In _caculate_indices_from_placements, placement dim = {placement.dim} {type(placement.dim)}, {dim} {type(dim)}"
            )
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

            return start_index, end_index

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

            return start_index, end_index

        elif len(dim_i_placements) == 0:
            # No need to split on this dimension
            return start_index, end_index

        else:
            raise NotImplementedError(
                f"Unsupported DTensor placements for GroupedExperts: {dtensor_placements} {dim_i_placements} {mesh_names}"
            )

    def _get_local_experts_weights(
        self,
        abstract_key: str,
        titan_abstract_key: str,
        layer_id: str,
        grouped_expert_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
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

    def _chunk_local_expert_weights(
        self,
        local_tensor: torch.Tensor,
        dtensor_placements: tuple,
        dtensor_shape: tuple,
        device_mesh: DeviceMesh,
    ):
        """
        Chunk the local individual experts weight, assemble back to GroupedExperts weights DTensor.

        This method is a placeholder for future implementation of expert weight concatenation.

        Args:
            local_tensor: Concatenated local individual expert weights
        """

        # Calculate the index range on dim-i to chunk
        for i in range(1, len(dtensor_placements)):
            dim_size = dtensor_shape[i]
            start_index, end_index = self._caculate_indices_from_placements(
                dim=i,
                dim_size=dim_size,
                dtensor_placements=dtensor_placements,
                device_mesh=device_mesh,
            )
            # No need to chunk on current dimension
            if start_index is None or end_index is None:
                continue

            # Chunk local_tensor on dim-i
            local_tensor = local_tensor.narrow(i, start_index, end_index - start_index)

        # Assemble DTensor
        grouped_expert_weights = DTensor.from_local(
            local_tensor, device_mesh, dtensor_placements, run_check=False
        )

        return grouped_expert_weights

    def _concatenate_local_expert_weights(
        self,
        expert_weights_by_layer: dict[str, Any],
        abstract_key: str,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """
        Concatenate the weights of separate experts into GroupedExperts weights.
        """
        logger.info(f"Concatenating for key {abstract_key} ")
        for layer in expert_weights_by_layer.keys():
            # If we have all the experts for this abstract_key, concatenate them
            experts = expert_weights_by_layer[layer][abstract_key]
            expected_n_experts = (
                self.local_experts_indices[abstract_key][1]
                - self.local_experts_indices[abstract_key][0]
            )
            if len(experts) == expected_n_experts:
                sorted_expert_ids = sorted(experts.keys())
                sorted_experts = [experts[i] for i in sorted_expert_ids]
                local_tensor = torch.stack(sorted_experts, dim=0)

                assert (
                    abstract_key in self.grouped_expert_weight_placements
                    and abstract_key in self.grouped_expert_weight_shape
                ), f"GroupedExperts weight metadata {self.grouped_expert_weight_placements} {self.grouped_expert_weight_shape} can not be None!"

                stacked_dtensor = self._chunk_local_expert_weights(
                    local_tensor,
                    dtensor_placements=self.grouped_expert_weight_placements[
                        abstract_key
                    ],
                    dtensor_shape=self.grouped_expert_weight_shape[abstract_key],
                    device_mesh=device_mesh,
                )

                # Remove these experts from the tracking dict to free memory
                del expert_weights_by_layer[layer][abstract_key]
                if not expert_weights_by_layer[layer]:
                    del expert_weights_by_layer[layer]

                logger.info(f"Concatenated for key {abstract_key} at layer {layer}")

                return stacked_dtensor
            else:
                logger.info("no enough experts to concate")

        return None

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Dequantize the weights from float8 to float32.
        """

        scale_inv_keys = []
        for key, weight in state_dict.items():
            if key.endswith(".weight") and key + "_scale_inv" in state_dict:
                scale_inv = state_dict[key + "_scale_inv"]
                # dequantized_weight = dequantize_from_fp8(
                #     weight, scale_inv, dtype=torch.float32
                # )
                # # update the weight and remove the scale_inv tensor
                # state_dict[key] = dequantized_weight

                state_dict[key] = weight
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

                # Store the GroupedExperts Weight metadata for from_hf()
                self.grouped_expert_weight_placements[abstract_key] = value.placements
                self.grouped_expert_weight_shape[abstract_key] = value.shape

                # Split GroupedExperts weight to local individual expert weights
                local_expert_fqn = self._get_local_experts_weights(
                    new_abstract_key,
                    abstract_key,
                    layer_num,
                    value,
                )

                hf_state_dict.update(local_expert_fqn)

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        # Prepare for dequantization
        hf_state_dict_with_scale_inv = self._add_quantization_scale_inv_tensors(
            hf_state_dict
        )
        return hf_state_dict_with_scale_inv

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concate separate expert's wegiht into GroupedExperts' weight.
        """
        print(
            f"At the beginning of from_hf, the loaded state_dict is {hf_state_dict.keys()}"
        )
        # dequantize the tensor in state_dict and remove the scale_inv tensor

        hf_state_dict = self._dequantize(hf_state_dict)
        state_dict = {}

        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
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

                # try to concat the expert's weight into GroupedExperts' weight.
                # stacked_value = self._concatenate_expert_weights(
                #     expert_weights_by_layer, self.model_args.moe_args.num_experts
                # )
                stacked_value = self._concatenate_local_expert_weights(
                    expert_weights_by_layer, titan_abstract_key, value.device_mesh
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
