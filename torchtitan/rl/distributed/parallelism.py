# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelism configuration for RL inference services."""

from dataclasses import dataclass

from torchtitan.config import ParallelismConfig


@dataclass(kw_only=True, slots=True)
class InferenceParallelismConfig:
    """Focused parallelism configuration for vLLM inference."""

    data_parallel_degree: int = 1
    """Generator FSDP degree. One keeps full parameters on each generator."""

    tensor_parallel_degree: int = 1
    """Tensor parallelism degree. One disables tensor parallelism."""

    expert_parallel_degree: int = 1
    """Expert parallelism degree for MoE layers. One disables EP."""

    enable_sequence_parallel: bool = False
    """Enable dense sequence parallelism across the tensor-parallel axis."""

    @property
    def expert_sequence_parallel_size(self) -> int:
        if self.expert_parallel_degree <= 1:
            return 1
        return self.tensor_parallel_degree

    def to_training(self) -> ParallelismConfig:
        """Translate the inference layout for shared TorchTitan utilities."""
        return ParallelismConfig(
            # vLLM's DP ranks occupy the FSDP axis in the shared mesh shape.
            data_parallel_shard_degree=self.data_parallel_degree,
            tensor_parallel_degree=self.tensor_parallel_degree,
            expert_parallel_degree=self.expert_parallel_degree,
            data_parallel_replicate_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            enable_sequence_parallel=self.enable_sequence_parallel,
            # Reuse the unsharded compute representation until weight sync.
            fsdp_reshard_after_forward="never",
        )
