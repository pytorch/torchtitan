# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle

from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
)


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
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:

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

        assert hasattr(
            mod, "token_dispatcher"
        ), f"{type(mod)} missing token_dispatcher attribute"
        assert isinstance(
            mod.token_dispatcher,
            (AllToAllTokenDispatcher, DeepEPTokenDispatcher),
        ), f"Expected AllToAllTokenDispatcher or DeepEPTokenDispatcher, got {type(mod.token_dispatcher)}"
        mod.token_dispatcher.ep_group = device_mesh["ep"].get_group()
