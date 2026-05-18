# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

from torchtitan.distributed.expert_parallel import TensorParallel


# implementation of Tensor Parallel for the GroupedExperts in MoE
class GptossTensorParallel(TensorParallel):
    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "mlp1_weight",
            nn.Parameter(
                distribute_tensor(module.mlp1_weight, device_mesh, [Shard(1)])
            ),
        )  # Column-wise sharding
        module.register_parameter(
            "mlp1_bias",
            nn.Parameter(distribute_tensor(module.mlp1_bias, device_mesh, [Shard(1)])),
        )  # Column-wise sharding
        module.register_parameter(
            "mlp2_weight",
            nn.Parameter(
                distribute_tensor(module.mlp2_weight, device_mesh, [Shard(2)])
            ),
        )  # Row-wise sharding
        module.register_parameter(
            "mlp2_bias",
            nn.Parameter(
                distribute_tensor(module.mlp2_bias, device_mesh, [Replicate()])
            ),
        )  # Replicate
