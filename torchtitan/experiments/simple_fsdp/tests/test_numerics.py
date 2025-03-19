# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy

import torch
from torch.distributed._composable.fsdp import fully_shard

from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel


class TestSimpleFSDP(FSDPTest):
    def init_test(self):
        self.optimizer = torch.optim.Adam
        self.loss_fn = cross_entropy_loss
        data_parallel_shard_degree = -1
        if self.mode == "replicate":
            self.dp_mesh_dim_names = ("dp_replicate",)
            data_parallel_replicate_degree = self.world_size
        elif self.mode == "fully_shard":
            self.dp_mesh_dim_names = ("dp_shard_cp",)
            data_parallel_replicate_degree = 1
        elif self.mode == "hybrid_shard":
            self.dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
            data_parallel_replicate_degree = self.world_size // 2
        else:
            raise ValueError(f"Unsupported mode {mode}")

        self.parallel_dims = ParallelDims(
            dp_shard=data_parallel_shard_degree,
            dp_replicate=data_parallel_replicate_degree,
            cp=1,
            tp=1,
            pp=1,
            world_size=self.world_size,
            enable_loss_parallel=True,
        )
        self.device_mesh = self.parallel_dims.build_mesh(device_type="cuda")

    def get_input(self):
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()
        model = torch.nn.Linear(8, 8)
        return model, inputs, labels

    def run_fsdp2(self, model, inputs, labels, epoch=20):
        fully_shard(model, mesh=self.device_mesh[tuple(self.dp_mesh_dim_names)])
        optim = self.optimizer(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(epoch):
            optim.zero_grad()
            out = model(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def run_simple_fsdp(self, model, inputs, labels, epoch=20):
        model = data_parallel(
            model,
            device_mesh=self.device_mesh[tuple(self.dp_mesh_dim_names)],
            mode=self.mode,
        )
        optim = self.optimizer(model.parameters(), lr=1e-4)
        losses = []
        for _ in range(epoch):
            optim.zero_grad()
            out = model(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            optim.step()
            losses.append(loss)
        return losses

    def test_replicate_convergence(self):
        # unit test for replicate mode
        self.mode = "replicate"
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_replicate_losses = self.run_simple_fsdp(
            copy.deepcopy(model), inputs, labels
        )

        for fsdp2_loss, simple_fsdp_replicate_loss in zip(
            fsdp2_losses, simple_fsdp_replicate_losses
        ):
            assert torch.allclose(fsdp2_loss, simple_fsdp_replicate_loss)

    def test_fullyshard_convergence(self):
        # unit test for fully_shard mode
        self.mode = "fully_shard"
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_fullyshard_losses = self.run_simple_fsdp(
            copy.deepcopy(model), inputs, labels
        )

        for fsdp2_loss, simple_fsdp_fullyshard_loss in zip(
            fsdp2_losses, simple_fsdp_fullyshard_losses
        ):
            assert torch.allclose(fsdp2_loss, simple_fsdp_fullyshard_loss)

    def test_hybridshard_convergence(self):
        # unit test for hybrid_shard mode
        self.mode = "hybrid_shard"
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_hybridshard_losses = self.run_simple_fsdp(
            copy.deepcopy(model), inputs, labels
        )

        for fsdp2_loss, simple_fsdp_hybridshard_loss in zip(
            fsdp2_losses, simple_fsdp_hybridshard_losses
        ):
            assert torch.allclose(fsdp2_loss, simple_fsdp_hybridshard_loss)
