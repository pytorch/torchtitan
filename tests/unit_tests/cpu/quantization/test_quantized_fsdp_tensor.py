# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
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


class TestQuantizedFSDPAllGather(DTensorTestBase):
    device_type = "cpu"

    @property
    def world_size(self):
        return 2

    @with_comms
    def test_fsdp_all_gather_restores_the_weight_on_any_shard_dim(self):
        """Real FSDP2 reassembles Shard(0) and Shard(1) weights; the hooks must
        not reorder the rows on either the first gather or a refill after
        reshard. The stacked [2, F, D] FFN w13 is the Shard(1) case."""
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
        # Distinct values, all exact in bf16 (integers below 256, also after the
        # +64 below), so any reordering of the gathered rows shows. 5 rows split
        # 3 and 2, so FSDP pads rank 1's Shard(0) shard.
        stacked_2FD = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        plain_FD = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4) + 100
        module = torch.nn.Module()
        module.stacked = torch.nn.Parameter(_TestShardedTensor(stacked_2FD.clone()))
        module.plain = torch.nn.Parameter(_TestShardedTensor(plain_FD.clone()))
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            # As linear_param_shard_placements shards a stacked Linear's weight:
            # on its matrix rows. Other weights keep FSDP's default Shard(0).
            shard_placement_fn=lambda param: Shard(1) if param.ndim == 3 else None,
        )
        module.unshard()
        for unsharded, expected in (
            (module.stacked, stacked_2FD),
            (module.plain, plain_FD),
        ):
            self.assertIsInstance(unsharded, _UnshardedFSDPTensor)
            torch.testing.assert_close(
                unsharded.operands.value, expected.bfloat16(), rtol=0, atol=0
            )

        # The unsharded stacked weight keeps a contiguous [2, F, D] layout, and
        # flattening it for the matmul shares the same operands.
        self.assertEqual(module.stacked.shape, (2, 6, 4))
        self.assertEqual(module.stacked.stride(), (24, 4, 1))
        flattened = module.stacked.flatten(0, -2)
        self.assertIsInstance(flattened, _UnshardedFSDPTensor)
        self.assertEqual(flattened.shape, (12, 4))
        self.assertIs(flattened.operands, module.stacked.operands)

        first_unsharded = (module.stacked, module.plain)
        first_operands = tuple(param.operands for param in first_unsharded)
        module.reshard()
        with torch.no_grad():
            for param in module.parameters():
                param.to_local()._tensor.add_(64)
        module.unshard()
        for unsharded, original, operands, expected in zip(
            (module.stacked, module.plain),
            first_unsharded,
            first_operands,
            (stacked_2FD, plain_FD),
            strict=True,
        ):
            # The second gather must refill FSDP's existing operands, including
            # the new shard values, after reshard freed their storage.
            self.assertIs(unsharded, original)
            self.assertIs(unsharded.operands, operands)
            updated = (expected + 64).bfloat16()
            torch.testing.assert_close(operands.value, updated, rtol=0, atol=0)


def test_async_checkpoint_saves_the_sharded_parameter_values(tmp_path):
    values = torch.randn(256, 128)
    param = torch.nn.Parameter(_TestShardedTensor(values.clone()))

    stager = DefaultStager(StagingOptions(False, False, False, False))
    staged = stager.stage({"weight": param})
    stager.close()
    assert isinstance(staged["weight"], _TestShardedTensor)
    torch.testing.assert_close(staged["weight"]._tensor, values, rtol=0, atol=0)

    dcp.async_save({"weight": param}, checkpoint_id=tmp_path, no_dist=True).result()
    loaded = {"weight": torch.zeros_like(values)}
    dcp.load(loaded, checkpoint_id=tmp_path, no_dist=True)
    torch.testing.assert_close(loaded["weight"], values, rtol=0, atol=0)
