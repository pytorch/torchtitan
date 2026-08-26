# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from fractions import Fraction

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from torchtitan.tools.logging import logger


def validate_converter_order(converters: list) -> None:
    """Validate that quantization/QAT converters precede LoRA.

    Raises ``ValueError`` if a quantization converter appears after a LoRA
    converter in the list.
    """
    from torchtitan.components.lora import LoRAConverter
    from torchtitan.components.quantization import QuantizationConverter

    _BEFORE_LORA = (QuantizationConverter.Config,)

    seen_lora = False
    for converter in converters:
        if isinstance(converter, LoRAConverter.Config):
            seen_lora = True
        elif seen_lora and isinstance(converter, _BEFORE_LORA):
            raise ValueError(
                f"{type(converter).__name__} must be applied before "
                f"LoRAConverter. Reorder the converters list."
            )


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
        model_config: Decoder.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
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
            if isinstance(placement, (Shard, _StridedShard)) and placement.dim == dim:
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
        sub_placements: list[Placement] = []

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
                    _StridedShard(placement.dim, split_factor=placement.split_factor)
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
                # Use slicing and unsqueeze get a 3D tensor, then create DTensor and squeeze
                expert_weight_3d = local_grouped_weights[
                    local_expert_index, :, :
                ].unsqueeze(0)
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

        # Stack experts - result may be DTensor or plain tensor depending on sub_mesh
        local_tensor = torch.stack(sorted_experts, dim=0)
        if isinstance(local_tensor, DTensor):
            local_tensor = local_tensor._local_tensor

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


def quadratic_attention_flops_per_token(
    *,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    seq_len: int,
    sliding_window_size: int | None = None,
) -> int:
    """Training FLOPs per token for quadratic or windowed attention.

    Reasoning behind the factor of 6 for the self-attention part of the formula:
    1. each self-attention has 2 matmul in the forward and 4 (counted as 2)
       in the backward                                                      (3)
       The 2 matmuls per token are:
       a. tmp = q @ K^T: [1, qk_head_dim] @ [qk_head_dim, seq_len]
       b. tmp @ V: [1, seq_len] @ [seq_len, v_head_dim]
       so we get
       seq_len * qk_head_dim + seq_len * v_head_dim = seq_len * (qk_head_dim + v_head_dim)
    2. the flash attention does 1 more matmul recomputation in the backward
       but recomputation should not be counted in calculating MFU           (+0)
    3. each matmul performs 1 multiplication and 1 addition                 (*2)
    4. we follow the convention and do not account for sparsity in causal attention

    ``qk_head_dim`` and ``v_head_dim`` describe the two attention
    contractions. The factor of 6 accounts for multiply-adds in forward and
    backward. As in the existing MFU convention, causal sparsity and backward
    recomputation are not counted.
    """
    attended_tokens = (
        seq_len if sliding_window_size is None else min(seq_len, sliding_window_size)
    )
    return 6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens


def delta_rule_flops_per_token(
    *,
    num_heads: int,
    key_head_dim: int,
    v_head_dim: int,
) -> int:
    """Training FLOPs per token for a recurrent delta-rule state update.

    Omitting batch dimensions,
    ``state``: ``[num_heads, key_head_dim, v_head_dim]``
    ``key`` and ``query``: ``[num_heads, key_head_dim]``
    ``value`` and ``delta``: ``[num_heads, v_head_dim]``

    For each token, the recurrence performs:
    1. Decay the state: ``decayed_state = exp(decay) * state``.
    2. Read the stored value: ``memory = decayed_state.T @ key``.
    3. Form the gated correction: ``delta = beta * (value - memory)``.
    4. Update the state: ``state = decayed_state + key[:, None] * delta[None, :]``.
    5. Read the output: ``output = state.T @ query``.

    Steps 2, 4, and 5 each scale as ``key_head_dim * v_head_dim``, producing the
    factor of 3. The factor of 6 accounts for multiply-adds in forward and
    backward. Gate-producing linear projections are covered by the model's
    ``6 * active_nparams`` term. The elementwise work in steps 1 and 3, output
    gating, normalization, nonlinearities, and backward recomputation are not
    counted.
    """
    return 6 * 3 * num_heads * key_head_dim * v_head_dim


def get_nparams_and_active_nparams(
    model: nn.Module,
    *,
    modules_excluded_from_active_params: Iterable[nn.Module | None] = (),
) -> tuple[int, int]:
    """Count total and matmul-active parameters for a native decoder.

    Routed-expert parameters are weighted by the owning MoE module's active
    expert ratio. Embedding tables are excluded unless their parameter is shared
    with the output head. Explicitly excluded subtrees are also assigned zero
    per-token parameter cost.

    Args:
        model: Built model whose parameters are counted.
        modules_excluded_from_active_params: Module subtrees whose cost does not
            scale per text token, such as a vision encoder.

    Returns:
        Total parameter count and effective parameter count for the conventional
        ``6 * active_parameters`` training FLOP estimate.
    """
    named_parameters = list(model.named_parameters())
    nparams = sum(param.numel() for _, param in named_parameters)
    parameter_weights = {id(param): Fraction(1) for _, param in named_parameters}

    for module in model.modules():
        if isinstance(module, MoE):
            active_expert_ratio = Fraction(
                module.router.top_k, module.router.num_experts
            )
            for param in module.routed_experts.parameters():
                parameter_weights[id(param)] = active_expert_ratio

    lm_head = getattr(model, "lm_head", None)
    lm_head_parameter_ids = (
        {id(param) for param in lm_head.parameters()}
        if isinstance(lm_head, nn.Module)
        else set()
    )
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters(recurse=False):
                if id(param) not in lm_head_parameter_ids:
                    parameter_weights[id(param)] = Fraction(0)

    for module_excluded_from_active_params in modules_excluded_from_active_params:
        if module_excluded_from_active_params is None:
            continue
        for param in module_excluded_from_active_params.parameters():
            parameter_weights[id(param)] = Fraction(0)

    nparams_for_matmul = sum(
        param.numel() * parameter_weights[id(param)] for _, param in named_parameters
    )
    assert nparams_for_matmul.denominator == 1
    active_nparams = nparams_for_matmul.numerator

    logger.info(
        f"Total parameter count: {nparams:,}, active parameters: {active_nparams:,}"
    )
    return nparams, active_nparams
