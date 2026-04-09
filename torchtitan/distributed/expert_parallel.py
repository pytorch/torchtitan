# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle


class BaseExpertParallel(ParallelStyle, ABC):
    @abstractmethod
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        ...


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


class ExpertParallel(BaseExpertParallel):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape: torch.Size | None = None
        self.permuted_indices = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # NOTE: Token dispatch/combine is now handled by Experts.token_dispatcher.
        # ExpertParallel only shards expert weights.
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
        )


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ExpertParallel):
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        # w1 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w1",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        mod.register_parameter(
            "w2",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w3",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w3, device_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # NOTE: Token dispatch/combine is now handled by Experts.token_dispatcher.
        # ExpertTensorParallel only shards expert weights with (EP, TP) placement.
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
        )


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()

    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        # shape (batch_size*seq_len, top_k)
        top_scores, selected_experts_indices = inputs
        num_tokens, _ = top_scores.shape

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            if num_tokens % device_mesh.size() != 0:
                raise ValueError(
                    "Uneven split of tokens of is not supported yet. "
                    "Requires EP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        # shape (batch_size * seq_len // ep_degree, top_k)
        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        # shape (batch_size * seq_len * top_k // ep_degree)
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        local_rank = device_mesh.get_local_rank()
        if not hasattr(mod, "top_k"):
            raise ValueError(
                "TokenReorderer class in MoE should always have top_k attribute."
            )
        token_indices_experts_sorted = (
            token_indices_experts_sorted + top_scores.shape[0] // mod.top_k * local_rank
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._prepare_inputput_fn,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._prepare_output_fn,
        )
