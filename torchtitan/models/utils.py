# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torchtitan.protocols.model import BaseModelArgs
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from torchtitan.tools.logging import logger


class MoEStateDictAdapter(StateDictAdapter):
    """
    StateDictAdapter for MoE models.
    HF MoE models store experts as a module list each with 2D weights. In torchtitan, we
    store experts as a 3D param with the first dimension being num_experts. The functions
    in this class help convert 3D param into list of 2D params so that the checkpoint
    can be loaded without incurring local memory overhead, and then concatenate
    the results back to 3D param.
    """

    def __init__(
        self,
        model_args: BaseModelArgs,
        hf_assets_path: str | None,
    ):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args
        self.hf_assets_path = hf_assets_path
        # Store metadata for GroupedExperts <-> individual experts conversion
        self.grouped_expert_weight_placements = {}  # {titan_abstract_key: placements}
        self.grouped_expert_weight_shape = {}  # {titan_abstract_key: shape}
        self.grouped_expert_weight_mesh = {}  # {titan_abstract_key: device_mesh}
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
        Given a [StridedShard(dim=i), Shard(dim=i)] placement, calculate the start index
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

        Calculate the start_index from inner dimension (Shard(dim=i)) to outer dimension (StridedShard(dim=i)).
        """

        block_size = dim_size_to_split // (strided_shard_dim_degree * shard_dim_degree)

        # Error out if can not evenly divided
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
        device_mesh: DeviceMesh,
    ) -> tuple[int, int]:

        mesh_names = []
        dim_i_placements = []

        # Find all the device mesh dimensios that shard on dim-i
        # pyrefly: ignore [bad-argument-type]
        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if placement.dim == dim:
                mesh_names.append(name)
                dim_i_placements.append(placement)

        # Calculate local expert indices based on sharding strategy
        start_index, end_index = 0, dim_size
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
            # No sharding on this dimension means all elements are local
            pass

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
            Dictionary mapping individual expert keys to their DTensor or plain tensor weights
        """
        # pyrefly: ignore [missing-attribute]
        device_mesh = grouped_expert_weight.device_mesh
        # pyrefly: ignore [missing-attribute]
        dtensor_placements = grouped_expert_weight.placements

        # Step 1: Extract dimension-0 placement information
        num_experts = grouped_expert_weight.shape[0]
        start_index, end_index = self._caculate_indices_from_placements(
            dim=0,
            dim_size=num_experts,
            dtensor_placements=dtensor_placements,
            device_mesh=device_mesh,
        )

        # Step 2: Store indices for potential future use in from_hf()
        self.local_experts_indices[titan_abstract_key] = (start_index, end_index)

        # Step 3: Identify mesh dimensions that shard on dim-0 (expert dimension)
        # exclude expert dimension
        # and build new sub-mesh/placements for individual expert weights
        sub_mesh_names = []
        sub_placements = []

        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if isinstance(placement, Replicate):
                # Replicate (hybrid) doesn't shard any dim, keep in sub-mesh
                sub_mesh_names.append(name)
                sub_placements.append(Replicate())
            elif isinstance(placement, (Shard, _StridedShard)) and placement.dim == 0:
                # Shards on expert dim, exclude from sub-mesh
                pass
            elif isinstance(placement, Shard):
                # Shards on non-expert dim, keep in sub-mesh
                sub_mesh_names.append(name)
                sub_placements.append(Shard(placement.dim))
            elif isinstance(placement, _StridedShard):
                # Strided shard on non-expert dim, keep in sub-mesh
                sub_mesh_names.append(name)
                sub_placements.append(
                    # pyrefly: ignore [unexpected-positional-argument]
                    _StridedShard(placement.dim, placement.split_factor)
                )
            else:
                raise ValueError(f"Unsupported placement type: {type(placement)}")

        # Step 4: Create sub-mesh excluding dim-0 sharding dimensions
        # If all mesh dimensions were sharding on dim-0, sub_mesh will be None (use plain tensors)
        sub_mesh = device_mesh[tuple(sub_mesh_names)] if sub_mesh_names else None

        # Step 5: Create individual expert tensors
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

            if sub_mesh is None:
                # Extract individual expert weight (2D) as plain tensor
                expert_weight = local_grouped_weights[local_expert_index, :, :]
            else:
                # Use index_select to get 3D tensor directly, then create DTensor and squeeze
                expert_weight_3d = torch.index_select(
                    local_grouped_weights,
                    0,
                    torch.tensor(
                        [local_expert_index], device=local_grouped_weights.device
                    ),
                )
                expert_weight = DTensor.from_local(
                    expert_weight_3d,
                    sub_mesh,
                    sub_placements,
                    run_check=False,
                ).squeeze(0)
            local_expert_tensors[expert_key] = expert_weight

        return local_expert_tensors

    def _concatenate_expert_weights_dtensor(
        self,
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]],
        abstract_key: str,
        layer_num: str,
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
        # pyrefly: ignore [missing-attribute]
        local_tensor = torch.stack(sorted_experts, dim=0)._local_tensor

        assert (
            abstract_key in self.grouped_expert_weight_placements
            and abstract_key in self.grouped_expert_weight_shape
            and abstract_key in self.grouped_expert_weight_mesh
        ), "GroupedExperts weight metadata (placements, shape, mesh) can not be None!"

        stacked_dtensor = DTensor.from_local(
            local_tensor,
            self.grouped_expert_weight_mesh[abstract_key],
            self.grouped_expert_weight_placements[abstract_key],
            run_check=False,
        )

        del expert_weights_by_layer[layer_num][abstract_key]
        if not expert_weights_by_layer[layer_num]:
            del expert_weights_by_layer[layer_num]

        return stacked_dtensor

    def _split_experts_weights(
        self, weight: torch.Tensor, n_experts: int
    ) -> tuple[torch.Tensor, ...]:
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


def get_dense_model_nparams_and_flops(
    model_args: BaseModelArgs,
    model: nn.Module,
    head_dims: int,
    seq_len: int,
) -> tuple[int, int]:
    """
    Args:
        model_args: BaseModelArgs object containing model configuration parameters.
        model: nn.Module representing the model.
        head_dims: The sum of qk and v head dimensions.
        seq_len: The sequence length in training configs.

    Returns:
        Tuple of (nparams, num_flops_per_token):
            nparams: Total number of model parameters.
            num_flops_per_token: Estimated number of floating point operations per token.
    """
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )

    # Reasoning behind the factor of 6 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul (attention scores and value aggregation,
    #    combined in head_dims, counted as 1) in the forward and 4 (counted as 2)
    #    in the backward                                                      (3)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    num_flops_per_token = (
        6 * (nparams - nparams_embedding)
        # pyrefly: ignore [missing-attribute]
        + 6 * model_args.n_layers * model_args.n_heads * head_dims * seq_len
    )

    # If weight tying is enabled, subtract embedding parameters from total count
    if hasattr(model_args, "enable_weight_tying") and model_args.enable_weight_tying:
        nparams = nparams - nparams_embedding

    return nparams, num_flops_per_token


def get_moe_model_nparams_and_flops(
    model_args: BaseModelArgs,
    model: nn.Module,
    head_dims: int,
    seq_len: int,
) -> tuple[int, int]:
    """
    Calculate nparams and nflops for MoE models.

    Args:
        model_args: BaseModelArgs object containing model configuration parameters including MoE settings.
        model: nn.Module representing the MoE model.
        head_dims: The sum of qk and v head dimensions.
        seq_len: The sequence length in training configs.

    Returns:
        Tuple of (nparams, num_flops_per_token):
            nparams: Total number of model parameters including all experts.
            num_flops_per_token: Estimated number of floating point operations per token
                                based on active parameters only.
    """
    nparams_embedding = 0
    nparams_moe_router = 0
    nparams_shared_experts = 0
    nparams_experts = 0
    nparams_dense = 0

    for name, p in model.named_parameters():
        if "embedding" in name:
            nparams_embedding += p.numel()
            nparams_dense += p.numel()
        elif "moe.shared_experts" in name:
            nparams_shared_experts += p.numel()
        elif "moe.router" in name:
            nparams_moe_router += p.numel()
        elif "moe.experts" in name:
            nparams_experts += p.numel()
        else:
            nparams_dense += p.numel()

    nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
    nparams = nparams_dense + nparams_sparse
    nparams_sparse_active = (
        nparams_moe_router
        + nparams_shared_experts
        # pyrefly: ignore [missing-attribute]
        + nparams_experts * model_args.moe_args.top_k // model_args.moe_args.num_experts
    )

    logger.info(
        f"Total parameter count: dense {nparams_dense:,}, "
        f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
    )

    num_flops_per_token = (
        6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
        # pyrefly: ignore [missing-attribute]
        + 6 * model_args.n_layers * model_args.n_heads * head_dims * seq_len
    )

    # If weight tying is enabled, subtract embedding parameters from total count
    if hasattr(model_args, "enable_weight_tying") and model_args.enable_weight_tying:
        nparams = nparams - nparams_embedding

    return nparams, num_flops_per_token
