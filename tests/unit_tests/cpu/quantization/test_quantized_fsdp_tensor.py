# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp.experimental import AllGatherInput
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)


@dataclass(frozen=True)
class _Operands:
    value: torch.Tensor


class _TestShardedTensor(_ShardedFSDPTensor):
    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _Operands | None = None,
    ) -> _Operands:
        if out is None:
            return _Operands(logical_tensor.clone())
        out.value.copy_(logical_tensor)
        return out


class _Mesh:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _MixedPrecisionPolicy:
    param_dtype = torch.bfloat16


def test_fsdp_gathers_a_stacked_weight_on_its_matrix_row_dim():
    local_2FD = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    sharded = _TestShardedTensor(local_2FD)

    (payload,), metadata = sharded.fsdp_pre_all_gather(
        _Mesh(2),
        torch.Size([2, 6, 4]),
        None,
        None,
        _MixedPrecisionPolicy(),
    )

    # FSDP concatenates the moved shards along dim 0
    assert isinstance(payload, AllGatherInput) and payload.dim == 0
    comm_F2D = payload.tensor
    assert comm_F2D.shape == (3, 2, 4)
    torch.testing.assert_close(comm_F2D, local_2FD.movedim(1, 0).bfloat16())

    gathered_F2D = torch.cat([comm_F2D, comm_F2D + 100], dim=0)
    unsharded, _ = sharded.fsdp_post_all_gather(
        (gathered_F2D,), metadata, torch.bfloat16
    )

    assert isinstance(unsharded, _UnshardedFSDPTensor)
    assert unsharded.shape == (2, 6, 4)
    assert unsharded.stride() == (24, 4, 1)
    flattened = unsharded.flatten(0, -2)
    assert isinstance(flattened, _UnshardedFSDPTensor)
    assert flattened.shape == (12, 4)
    assert flattened.operands is unsharded.operands
    torch.testing.assert_close(
        unsharded.operands.value,
        gathered_F2D.movedim(0, 1),
    )


class _StackedLinear(torch.nn.Module):
    def __init__(self, weight_2FD: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(_TestShardedTensor(weight_2FD))
        self.unsharded_value: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.unsharded_value = self.weight.operands.value.clone()
        return x


class TestFullyShardStackedWeight(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_fully_shard_rebuilds_a_stacked_weight_sharded_on_its_matrix_rows(self):
        # FSDP reassembles plain all-gather inputs along the parameter's shard
        # dim, which once scrambled the moved [F, 2, D] shards of w13.
        weight_2FD = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        module = _StackedLinear(weight_2FD.clone().to(self.device_type))
        fully_shard(
            module,
            mesh=init_device_mesh(self.device_type, (self.world_size,)),
            shard_placement_fn=lambda param: Shard(1),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        module(torch.zeros(1, device=self.device_type))
        torch.testing.assert_close(
            module.unsharded_value.cpu(), weight_2FD.bfloat16(), rtol=0, atol=0
        )
