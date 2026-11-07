# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss


pytestmark = pytest.mark.multi_gpu


class TestQSAFSDP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_indexer_gradient_participation(self):
        """Short/padded DP batches retain matching FSDP gradient collectives."""
        torch.use_deterministic_algorithms(True)
        mesh = init_device_mesh(self.device_type, (2,), mesh_dim_names=("dp_shard",))
        for use_remat in (False, True):
            for input_kind in ("mixed", "short", "padding", "frozen"):
                with self.subTest(use_remat=use_remat, input_kind=input_kind):
                    torch.manual_seed(42)
                    config = build_model_config(
                        "debugmodel",
                        seq_len=32,
                        parallelism=ParallelismConfig(data_parallel_shard_degree=2),
                        indexer_loss=(
                            None
                            if input_kind == "frozen"
                            else QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss")
                        ),
                    )
                    attention = config.layers[3].attention.build().to(self.device_type)
                    attention.init_states()
                    if use_remat:
                        attention.configure_remat_regions(())
                    fully_shard(attention.indexer, mesh=mesh)
                    fully_shard(attention, mesh=mesh)
                    optimizer = torch.optim.SGD(attention.parameters(), lr=0)
                    forward = remat.checkpoint()(attention) if use_remat else attention
                    for microbatch in range(2):
                        attention.set_requires_gradient_sync(microbatch == 1)
                        attention.set_is_last_backward(microbatch == 1)
                        x = torch.randn(31, 256, device=self.device_type)
                        positions = torch.arange(31, device=self.device_type)
                        if input_kind == "short" or (
                            input_kind in ("mixed", "frozen") and self.rank == 1
                        ):
                            positions = positions.remainder(3)
                        padding = torch.full(
                            (31,), input_kind == "padding", device=self.device_type
                        )
                        metadata = config.layers[
                            3
                        ].attention.inner_attention.build_attention_metadata(
                            positions, padding_mask=padding
                        )
                        output = forward(
                            x,
                            metadata,
                            positions,
                            document_positions_T=positions,
                            padding_mask_T=padding,
                            aux_loss_denominator=torch.tensor(
                                62.0, device=self.device_type
                            ),
                        )
                        # The frozen case also exercises normal backbone training.
                        loss = (
                            output.square().mean()
                            if input_kind == "frozen"
                            else output.sum() * 0
                        )
                        loss.backward()
                    for parameter in attention.indexer.parameters():
                        if input_kind == "frozen":
                            self.assertFalse(parameter.requires_grad)
                            self.assertIsNone(parameter.grad)
                        else:
                            self.assertIsNotNone(parameter.grad)
                            gradient = parameter.grad.full_tensor()
                            self.assertTrue(torch.isfinite(gradient).all())
                            if input_kind == "mixed":
                                self.assertGreater(gradient.norm().item(), 0)
                            else:
                                self.assertEqual(
                                    torch.count_nonzero(gradient).item(), 0
                                )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
