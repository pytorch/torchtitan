# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.deepseek_v4.attention import CompressedSparseAttention
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss


@unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA devices")
class TestSparseIndexerTP(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_tp1_and_tp2_gradients_and_metrics_match(self):
        torch.manual_seed(42)
        device = torch.device(self.device_type, self.rank)
        q = torch.randn(16, 4, 64, device=device)
        local = torch.randn(16, 64, device=device)
        compressed = torch.randn(4, 64, device=device)
        sink = torch.randn(4, device=device)
        iq = torch.randn(16, 2, 32, device=device)
        ik = torch.randn(4, 32, device=device)
        iw = torch.randn(16, 2, device=device)
        valid = torch.arange(16, device=device) < 12
        mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("tp",))

        for weighted in (False, True):
            cfg = CompressedSparseAttention.Config(
                window_size=4,
                compress_ratio=4,
                softmax_scale=64**-0.5,
                index_topk=3,
                aux_loss=SparseIndexerLoss.Config(
                    coeff=0.01,
                    reduce_mesh="loss",
                    softmax_scale=64**-0.5,
                    num_heads=4,
                    mass_weighted=weighted,
                    chunk_size=3,
                ),
            )
            reference = cfg.build().to(device)
            ref_inputs = [tensor.clone().requires_grad_() for tensor in (iq, ik, iw)]
            ref_out = reference(
                q,
                local,
                compressed,
                *ref_inputs,
                sink,
                aux_loss_denominator=torch.tensor(12.0, device=device),
                padding_mask_T=~valid,
            )
            ref_out.sum().backward()

            sharded = cfg.build().to(device)
            inputs = [tensor.clone().requires_grad_() for tensor in (iq, ik, iw)]
            with set_current_spmd_mesh(mesh):
                out = sharded(
                    q.chunk(2, dim=1)[self.rank],
                    local,
                    compressed,
                    *inputs,
                    sink.chunk(2)[self.rank],
                    aux_loss_denominator=torch.tensor(12.0, device=device),
                    padding_mask_T=~valid,
                )
                out.sum().backward()
            torch.testing.assert_close(out, ref_out.chunk(2, dim=1)[self.rank])
            torch.testing.assert_close(
                sharded.aux_loss.instance_acc,
                reference.aux_loss.instance_acc,
            )
            for tensor, ref_tensor in zip(inputs, ref_inputs):
                # Match the sum performed for TP-replicated indexer parameters.
                dist.all_reduce(tensor.grad)
                torch.testing.assert_close(
                    tensor.grad, ref_tensor.grad, atol=1e-6, rtol=1e-4
                )


if __name__ == "__main__":
    unittest.main()
