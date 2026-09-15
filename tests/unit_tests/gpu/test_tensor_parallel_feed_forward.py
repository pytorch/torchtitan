# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.transform import TensorParallelTransform
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_sequence_parallel_placement,
    set_dense_ffn_sharding,
)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestTensorParallelFeedForwardNumerics(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_matches_unsharded_with_and_without_sequence_parallel(self):
        device = self.device_type
        dim, hidden_dim, num_tokens = 64, 128, 16
        init = {"weight": torch.nn.init.zeros_}

        for enable_sp in (False, True):
            with self.subTest(enable_sp=enable_sp):
                torch.manual_seed(0)
                base_config = make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=init,
                    w2w3_param_init=init,
                )
                reference = copy.deepcopy(base_config).build().to(device)
                parallel_config = TensorParallelTransform().transform(
                    copy.deepcopy(base_config)
                )
                set_dense_ffn_sharding(
                    parallel_config,
                    attn_x_layout=(
                        dense_sequence_parallel_placement()
                        if enable_sp
                        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
                    ),
                    enable_sp=enable_sp,
                )
                parallel = parallel_config.build().to(device)

                with torch.no_grad():
                    for reference_weight, parallel_weight in (
                        (reference.w13.weight, parallel.w13.weight),
                        (reference.w2.weight, parallel.w2.weight),
                    ):
                        torch.manual_seed(hash(tuple(reference_weight.shape)) % 2**31)
                        reference_weight.copy_(torch.randn_like(reference_weight) * 0.1)
                        parallel_weight.copy_(reference_weight)

                parallel_dims = ParallelDims(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=1,
                    tp=self.world_size,
                    pp=1,
                    ep=1,
                    world_size=self.world_size,
                )
                with patch("torchtitan.distributed.parallel_dims.device_type", device):
                    parallel_dims.build_mesh()
                parallel.parallelize(parallel_dims)

                torch.manual_seed(1)
                x_full = torch.randn(num_tokens, dim, device=device, requires_grad=True)
                reference_out = reference(x_full)
                reference_out.sum().backward()

                x_local = (
                    x_full.detach().chunk(self.world_size, 0)[self.rank].contiguous()
                    if enable_sp
                    else x_full.detach().clone()
                ).requires_grad_()
                mesh = parallel_dims.spmd_dense_mesh()
                with set_current_spmd_mesh(mesh):
                    parallel_out = parallel(x_local)
                    parallel_out.sum().backward()

                expected_out = (
                    reference_out.chunk(self.world_size, 0)[self.rank]
                    if enable_sp
                    else reference_out
                )
                expected_grad = (
                    x_full.grad.chunk(self.world_size, 0)[self.rank]
                    if enable_sp
                    else x_full.grad
                )
                torch.testing.assert_close(parallel_out, expected_out)
                torch.testing.assert_close(x_local.grad, expected_grad)
                torch.testing.assert_close(
                    parallel.w13.weight.grad,
                    reference.w13.weight.grad.chunk(self.world_size, 0)[self.rank],
                )
                torch.testing.assert_close(
                    parallel.w2.weight.grad,
                    reference.w2.weight.grad.chunk(self.world_size, 1)[self.rank],
                )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
