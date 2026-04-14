# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle


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


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape: torch.Size | None = None
        self.permuted_indices = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        from torchtitan.models.common.token_dispatcher import (
            AllToAllTokenDispatcher,
            DeepEPTokenDispatcher,
        )

        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)
        # Set ep_group on the token dispatcher for all-to-all communication.
        # device_mesh here is the 1D EP mesh.
        assert hasattr(
            mod, "token_dispatcher"
        ), f"{type(mod)} missing token_dispatcher attribute"
        assert isinstance(
            mod.token_dispatcher,
            (AllToAllTokenDispatcher, DeepEPTokenDispatcher),
        ), f"Expected AllToAllTokenDispatcher or DeepEPTokenDispatcher, got {type(mod.token_dispatcher)}"
        mod.token_dispatcher.ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
        )


# TODO: Remove this class — all TP ranks within the same EP group perform
# redundant all-to-all communication.
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
        # Set ep_group on the token dispatcher for all-to-all communication.
        # device_mesh is the 2D (EP, ETP) mesh; slice the EP dimension.
        from torchtitan.models.common.token_dispatcher import (
            AllToAllTokenDispatcher,
            DeepEPTokenDispatcher,
        )

        assert hasattr(
            mod, "token_dispatcher"
        ), f"{type(mod)} missing token_dispatcher attribute"
        assert isinstance(
            mod.token_dispatcher,
            (AllToAllTokenDispatcher, DeepEPTokenDispatcher),
        ), f"Expected AllToAllTokenDispatcher or DeepEPTokenDispatcher, got {type(mod.token_dispatcher)}"
        mod.token_dispatcher.ep_group = device_mesh["ep"].get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
        )


class ExpertSequenceParallel(ExpertParallel):
    """ExpertParallel + Sequence Parallel for ETP=1.

    When EP borrows from all TP and part of DP, this class combines
    ExpertParallel weight sharding with Sequence Parallel input/output
    hooks (splitting tokens along the sequence dim across ranks),
    all in a single distribute_module call.
    """

    def _prepare_input_fn(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple:
        x, top_scores, selected_experts_indices = inputs
        # x shape (batch_size*seq_len, dim)
        num_tokens = x.shape[0]

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
                    "Uneven split of tokens is not supported yet. "
                    "Requires EP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        x = _split_along_first_dim(x)
        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        # shape (batch_size * seq_len // ep_degree, top_k)
        return x, top_scores, selected_experts_indices

    def _prepare_output_fn(
        self, mod: nn.Module, outputs: tuple, device_mesh: DeviceMesh
    ) -> tuple:
        routed_output, metadata = outputs

        local_rank = device_mesh.get_local_rank()
        if not hasattr(mod.token_dispatcher, "top_k"):
            raise ValueError(
                "Expert's TokenDispatcher class in MoE should always have top_k attribute."
            )
        num_local_tokens = (
            metadata.token_indices_experts_sorted.shape[0] // mod.token_dispatcher.top_k
        )  # pyrefly: ignore [missing-attribute]

        # As we shard routed tokens along bs*slen dim across the TP ranks,
        # the MoE gather and scatter still require global token indices.
        # Offset local token indices to global positions for scatter_add.
        adjusted_metadata = dataclasses.replace(
            metadata,
            token_indices_experts_sorted=(
                metadata.token_indices_experts_sorted + local_rank * num_local_tokens
            ),
        )
        return routed_output, adjusted_metadata

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._prepare_input_fn,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._prepare_output_fn,
        )
