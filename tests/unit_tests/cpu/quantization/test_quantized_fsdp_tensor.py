# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

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
    local_rank0_2FD = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    local_rank1_2FD = local_rank0_2FD + 100
    sharded = _TestShardedTensor(local_rank0_2FD)

    (comm_2FD,), metadata = sharded.fsdp_pre_all_gather(
        _Mesh(2),
        torch.Size([2, 6, 4]),
        None,
        None,
        _MixedPrecisionPolicy(),
    )

    assert comm_2FD.shape == (2, 3, 4)
    torch.testing.assert_close(comm_2FD, local_rank0_2FD.bfloat16())

    # FSDP reassembles Shard(1) into logical dimension order before the
    # extension API unflattens the storage using its local input shape.
    gathered_2FD = torch.cat([local_rank0_2FD, local_rank1_2FD], dim=1).bfloat16()
    extension_output_2FD = gathered_2FD.view(4, 3, 4)
    unsharded, _ = sharded.fsdp_post_all_gather(
        (extension_output_2FD,), metadata, torch.bfloat16
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
        gathered_2FD,
    )
