# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import raiseExceptions
import re
from typing import Any, Dict

import torch

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import DeepSeekV3ModelArgs
from .quantization import calculate_scale_shape, dequantize_from_fp8

from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Shard,
    Replicate
)

from torch.distributed.tensor import DTensor


class DeepSeekV3StateDictAdapter(StateDictAdapter):
    """
    StateDictAdapter for DeepSeekV3 model.
    """

    def __init__(self, model_args: DeepSeekV3ModelArgs, hf_assets_path: str | None, parallel_dims: ParallelDims):
        super().__init__(model_args, hf_assets_path, parallel_dims)
        self.model_args = model_args
        self.parallel_dims = parallel_dims
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

    def _split_experts_weight(
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

    def _get_local_experts_weights(
        self, abstract_key: str, layer_id: str, grouped_expert_weight: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Spliting the GroupedExperts weight and find the corresponding individual expert's weight in local tensor.

        Potential experts weights shard placements:
        - FSDP + EP when dp_mod_ep * ep <= num_experts: 
            - StridedShard(0)Shard(0)
        - FSDP + EP when dp_mod_ep * ep <= num_experts:
            - Shard(1)Shard(0) 
        - FSDP + ETP + EP when dp_mod_ep * ep <= num_experts:
            - w1/w3: StridedShard(0)Shard(0)Shard(1)
            - w2: StridedShard(0)Shard(0)Shard(2)
        - FSDP + ETP + EP when dp_mod_ep * ep > num_experts:
            - w1/w3: StridedShard(1)Shard(0)Shard(1)
            - w2: Shard(1)Shard(0)Shard(2)
        """
        world_mesh = self.parallel_dims.world_mesh
        num_experts = grouped_expert_weight.shape[0]

        # Matching DTensor sharding placement and device mesh dims,
        # find the dtensor dims that shard on dim-0 (num_experts dim)
        original_placements = grouped_expert_weight.placements
        world_mesh_names = []
        dim_0_placements = []
        for i, name in enumerate(world_mesh.mesh_dim_names):
            placement = original_placements[i]
            if placement.dim == 0:
                world_mesh_names.append(name)
                dim_0_placements.append(placement) 
        
        start_index, end_index = None, None
        # StridedShard(0)Shard(0)
        if len(dim_0_placements) == 2:
            assert isinstance(dim_0_placements[0], _StridedShard)
            strided_shard_mesh = world_mesh[world_mesh_names[0]]
            strided_degree, strided_rank = strided_shard_mesh.size(), strided_shard_mesh.get_local_rank()
            shard_mesh = world_mesh[world_mesh_names[1]]
            shard_degree, shard_rank = shard_mesh.size(), shard_mesh.get_local_rank()
            start_index, end_index = self._get_strided_shard_shard_slice(strided_degree, strided_rank, shard_degree, shard_rank, num_experts)
        # Shard(0)
        elif len(dim_0_placements) == 1:
            assert not isinstance(dim_0_placements[0], _StridedShard)
            shard_mesh = world_mesh[world_mesh_names[0]]
            shard_degree, shard_rank = shard_mesh.size(), shard_mesh.get_local_rank()
            block_size = num_experts // shard_degree
            if block_size * shard_degree != num_experts:
                raise ValueError("Not supported. num_experts can not be evenly divided by Shard(0) dimension degree.")
            
            start_index = block_size * shard_rank
            end_index = start_index + block_size
        else:
            raise NotImplementedError(f"The DTensor placements {original_placements} for GroupedExperts is not supported in StateDictAdapter")

        # Calculate the new placement for individual expert weights
        new_placements = []
        for i, name in enumerate(world_mesh.mesh_dim_names):
            placement = original_placements[i]
            if placement.dim == 0:
                new_placements.append(Replicate())
            elif isinstance(placement, Shard):
                # Individual expert weight has only 2 dimensions
                new_placements.append(Shard(placement.dim-1))
            elif isinstance(placement, _StridedShard):
                new_placements.append(_StridedShard(placement.dim-1, placement.split_factor))
            else:
                raise ValueError("Not supported new placements!")
        print(f"Original placements: {original_placements}, new placements {new_placements}")
       
        assert isinstance(grouped_expert_weight, DTensor), "GroupedExperts weight is not a DTensor"
        local_grouped_weights = grouped_expert_weight._local_tensor
        assert local_grouped_weights.shape[0] == int(end_index - start_index), "Local tensor shape mismatch!"

        # Create new DTensor for each individual expert weights
        local_expert_fqn = {}
        for expert_id in range(start_index, end_index):
            new_key = abstract_key.format(layer_id, expert_id)
            new_value = local_grouped_weights[expert_id - start_index, :, :].squeeze
            local_expert_fqn[new_key] = DTensor.from_local(new_value, world_mesh, new_placements, run_check=False)

        return local_expert_fqn
            
    
    def _get_strided_shard_shard_slice(
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
        if block_size * (strided_shard_dim_degree * shard_dim_degree) != dim_size_to_split:
            raise ValueError(f"Not supported split for strided_shard_dim_degree {strided_shard_dim_degree}, shard_dim_degree {shard_dim_degree}, dim_size_to_split {dim_size_to_split}")

        start_index = block_size * (strided_shard_dim_degree * shard_dim_rank + strided_shard_dim_rank)
        end_index = start_index + block_size

        return start_index, end_index


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

                # # Split expert weights into separate expert weights
                # split_values = self._split_experts_weights(
                #     value, self.model_args.moe_args.num_experts
                # )
                local_expert_fqn = self._get_local_experts_weights(
                    new_abstract_key, layer_num, value
                )
                print(f"groupedWeight placements {value.placements}, local experts keys {local_expert_fqn.keys()}")

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
        print(f"[to_hf] state_dict keys before return: {hf_state_dict_with_scale_inv.keys()}")
        return hf_state_dict_with_scale_inv

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concate separate expert's wegiht into GroupedExperts' weight.
        """
        # dequantize the tensor in state_dict and remove the scale_inv tensor
        hf_state_dict = self._dequantize(hf_state_dict)
        state_dict = {}

        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                new_key = self.from_hf_map[abstract_key]
                new_key = new_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][abstract_key] = {}
                expert_weights_by_layer[layer_num][abstract_key][expert_num] = value

                # try to concat the expert's weight into GroupedExperts' weight.
                stacked_value = self._concatenate_expert_weights(
                    expert_weights_by_layer, self.model_args.moe_args.num_experts
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
