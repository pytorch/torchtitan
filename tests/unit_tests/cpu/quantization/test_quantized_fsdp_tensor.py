# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Shard

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

    (comm_2FD,), metadata = sharded.fsdp_pre_all_gather(
        _Mesh(2),
        torch.Size([2, 6, 4]),
        None,
        None,
        _MixedPrecisionPolicy(),
    )

    # FSDP reassembles a Shard(1) input along dim 1 itself, so the shard keeps
    # the parameter's layout.
    assert comm_2FD.shape == (2, 3, 4)
    torch.testing.assert_close(comm_2FD, local_2FD.bfloat16())

    # FSDP's output holds the ranks' rows concatenated along dim 1, viewed as
    # (-1, *input.shape[1:]).
    gathered_2FD = torch.cat([comm_2FD, comm_2FD + 100], dim=1)
    unsharded, _ = sharded.fsdp_post_all_gather(
        (gathered_2FD.view(-1, 3, 4),), metadata, torch.bfloat16
    )

    assert isinstance(unsharded, _UnshardedFSDPTensor)
    assert unsharded.shape == (2, 6, 4)
    assert unsharded.stride() == (24, 4, 1)
    flattened = unsharded.flatten(0, -2)
    assert isinstance(flattened, _UnshardedFSDPTensor)
    assert flattened.shape == (12, 4)
    assert flattened.operands is unsharded.operands
    torch.testing.assert_close(unsharded.operands.value, gathered_2FD)


def _gather_with_fsdp(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
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
            assert isinstance(unsharded, _UnshardedFSDPTensor)
            torch.testing.assert_close(
                unsharded.operands.value, expected.bfloat16(), rtol=0, atol=0
            )

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
            assert unsharded is original
            assert unsharded.operands is operands
            updated = (expected + 64).bfloat16()
            torch.testing.assert_close(operands.value, updated, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_fsdp_all_gather_restores_the_weight_on_any_shard_dim(tmp_path):
    """Real FSDP2 reassembles Shard(0) and Shard(1) weights; the hooks must not
    reorder the rows on either the first gather or a refill after reshard.
    The stacked [2, F, D] FFN w13 is the Shard(1) case."""
    # A file rendezvous: a free port picked here could be taken before the
    # spawned ranks bind it.
    mp.spawn(_gather_with_fsdp, args=(str(tmp_path / "init"),), nprocs=2, join=True)


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
