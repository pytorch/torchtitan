# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blackwell TP numerics for the NVFP4 converter's DTensor bridge.

Guards the hand-rolled to_local / from_local backward in NVFP4Linear.forward.
Under tensor parallelism a colwise linear receives a TP-Replicate activation
whose input gradient is Partial (each rank holds only a partial sum over the
output shard, so it must be all-reduced); a rowwise linear receives a
TP-Shard(-1) activation whose input gradient stays Shard(-1). A wrong
grad_placements would still let training loss descend, so an integration smoke
test cannot catch it -- this compares the forward output and both input/weight
gradients against a bf16 reference via SQNR (the metric torchao's own NVFP4
tests use).
"""

import unittest

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

pytest.importorskip("torchao")

from torchtitan.components.quantization import NVFP4Linear  # noqa: E402
from torchtitan.distributed.parallel_dims import ParallelDims  # noqa: E402
from torchtitan.models.common.decoder_sharding import (  # noqa: E402
    colwise_config,
    rowwise_config,
)


def _blackwell_tp2() -> bool:
    return torch.cuda.device_count() >= 2 and all(
        torch.cuda.get_device_capability(i)[0] >= 10 for i in range(2)
    )


@unittest.skipUnless(
    NVFP4Linear is not None and _blackwell_tp2(),
    "NVFP4 TP numerics require two Blackwell (sm_100+) GPUs",
)
class TestNVFP4ConverterTPNumerics(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _parallel_dims(self) -> ParallelDims:
        return ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _parallelized(self, in_features, out_features, sharding_config) -> NVFP4Linear:
        cfg = NVFP4Linear.Config(in_features=in_features, out_features=out_features)
        cfg.sharding_config = sharding_config
        module = cfg.build().to(self.device_type)
        module.parallelize(self._parallel_dims())
        torch.manual_seed(1234)
        module.init_states(buffer_device=torch.device(self.device_type))
        return module

    def _weight(self, shape: tuple[int, int]) -> torch.Tensor:
        gen = torch.Generator(device=self.device_type).manual_seed(7)
        return torch.randn(
            shape, device=self.device_type, dtype=torch.bfloat16, generator=gen
        )

    def _input(self, shape: tuple[int, int, int], scale_dim: int) -> torch.Tensor:
        # Rank-independent input (so a Replicate/Shard distribute is well-defined)
        # with a large dynamic range along one dim to stress NVFP4's block scales.
        gen = torch.Generator(device=self.device_type).manual_seed(42)
        x = torch.randn(
            shape, device=self.device_type, dtype=torch.bfloat16, generator=gen
        )
        scale = torch.ones(
            shape[scale_dim], device=self.device_type, dtype=torch.bfloat16
        )
        scale[: shape[scale_dim] // 2] = 0.25
        scale[shape[scale_dim] // 2 :] = 4.0
        view = [1] * len(shape)
        view[scale_dim] = shape[scale_dim]
        return x * scale.view(view)

    def _assert_shard(self, tensor: DTensor, dim: int) -> None:
        self.assertIsInstance(tensor, DTensor)
        self.assertEqual(len(tensor.placements), 1)
        self.assertIsInstance(tensor.placements[0], Shard)
        actual = tensor.placements[0].dim % tensor.ndim
        self.assertEqual(actual, dim % tensor.ndim)

    def _assert_sqnr(self, actual, expected, threshold: float) -> None:
        from torchao.quantization.utils import compute_error

        sqnr = compute_error(expected.float(), actual.float())
        self.assertGreaterEqual(sqnr.item(), threshold)

    def _check(self, *, out, x, weight, x_full, weight_full, out_shard_dim) -> None:
        self._assert_shard(out, out_shard_dim)
        self._assert_sqnr(
            out.full_tensor(), F.linear(x_full.float(), weight_full.float()), 15.0
        )

        gen = torch.Generator(device=self.device_type).manual_seed(99 + self.rank)
        dy_local = torch.randn(
            out.to_local().shape,
            device=self.device_type,
            dtype=torch.bfloat16,
            generator=gen,
        )
        dy = DTensor.from_local(
            dy_local, out.device_mesh, out.placements, run_check=False
        )
        (out.to_local().float() * dy_local.float()).sum().backward()

        x_ref = x_full.float().detach().requires_grad_(True)
        weight_ref = weight_full.float().detach().requires_grad_(True)
        F.linear(x_ref, weight_ref).backward(dy.full_tensor().float())
        # full_tensor() all-reduces a Partial grad; with a wrong grad_placements
        # the colwise input grad is mislabeled Replicate and this SQNR collapses.
        self._assert_sqnr(x.grad.full_tensor(), x_ref.grad, 14.0)
        self._assert_sqnr(weight.grad.full_tensor(), weight_ref.grad, 14.0)

    @with_comms
    def test_colwise_gradients(self):
        tp_mesh = self._parallel_dims().get_mesh("tp")
        B, L, D, H = 1, 256, 256, 512

        col = self._parallelized(D, H, colwise_config())
        weight_full = self._weight((H, D))
        col.weight = torch.nn.Parameter(
            distribute_tensor(weight_full, tp_mesh, [Shard(0)])
        )
        # Colwise consumes a TP-Replicate activation (stock, no sequence parallel).
        x_full = self._input((B, L, D), scale_dim=1)
        x = distribute_tensor(x_full, tp_mesh, [Replicate()]).requires_grad_()

        self._check(
            out=col(x),
            x=x,
            weight=col.weight,
            x_full=x_full,
            weight_full=weight_full,
            out_shard_dim=-1,
        )

    @with_comms
    def test_rowwise_gradients(self):
        tp_mesh = self._parallel_dims().get_mesh("tp")
        B, L, D, H = 1, 256, 256, 512

        row = self._parallelized(H, D, rowwise_config(output_sp=True))
        weight_full = self._weight((D, H))
        row.weight = torch.nn.Parameter(
            distribute_tensor(weight_full, tp_mesh, [Shard(1)])
        )
        # Rowwise consumes a TP-Shard(-1) activation; the module returns Partial,
        # which the framework reduce-scatters to a sequence Shard(1) output.
        x_full = self._input((B, L, H), scale_dim=2)
        x = distribute_tensor(x_full, tp_mesh, [Shard(-1)]).requires_grad_()

        self._check(
            out=row(x),
            x=x,
            weight=row.weight,
            x_full=x_full,
            weight_full=weight_full,
            out_shard_dim=1,
        )


if __name__ == "__main__":
    unittest.main()
