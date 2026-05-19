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
        # device_mesh here is the 1D EP mesh.
        if hasattr(mod, "token_dispatcher"):
            assert isinstance(
                mod.token_dispatcher,
                (AllToAllTokenDispatcher, DeepEPTokenDispatcher),
            ), (
                "Expected AllToAllTokenDispatcher or DeepEPTokenDispatcher, "
                f"got {type(mod.token_dispatcher)}"
            )
            # Pass DeviceMesh (not ProcessGroup) so that CooR precompile
            # can use torch.ops._dtensor.mesh_get_process_group to keep
            # the FX graph rank-agnostic.
            mod.token_dispatcher.ep_mesh = device_mesh
        elif hasattr(mod, "ep_mesh"):
            # pyrefly: ignore[bad-argument-type]
            mod.ep_mesh = device_mesh
        else:
            raise ValueError(
                f"{type(mod)} must expose either token_dispatcher or ep_mesh "
                "for expert parallelism."
            )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
        )
