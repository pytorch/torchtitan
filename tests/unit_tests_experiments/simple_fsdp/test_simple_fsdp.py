# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import init_device_mesh

from torch.testing._internal.common_fsdp import FSDPTest
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.experiments.simple_fsdp import apply_data_parallel


class TestSimpleFSDP(FSDPTest):
    def init_test(self):
        self.optimizer = torch.optim.Adam
        self.loss_fn = cross_entropy_loss
        self.device_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))

    def get_input(self):
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()
        model = torch.nn.Linear(8, 8)
        return model, inputs, labels

    def run_fsdp2(self, model, inputs, labels, epoch=20):
        fully_shard(model, mesh=self.device_mesh)
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

    def run_simple_fsdp(self, model, inputs, labels, mode, epoch=20):
        model = apply_data_parallel(model, device_mesh=self.device_mesh, mode=mode)
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

    def test_fullyshard_convergence(self):
        # unit test for fully_shard mode
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_fullyshard_losses = self.run_simple_fsdp(
            copy.deepcopy(model), inputs, labels, mode="fully_shard"
        )

        for fsdp2_loss, simple_fsdp_fullyshard_loss in zip(
            fsdp2_losses, simple_fsdp_fullyshard_losses
        ):
            assert torch.allclose(fsdp2_loss, simple_fsdp_fullyshard_loss)

    def test_replicate_convergence(self):
        # unit test for replicate mode
        self.init_test()
        model, inputs, labels = self.get_input()

        fsdp2_losses = self.run_fsdp2(copy.deepcopy(model), inputs, labels)
        simple_fsdp_replicate_losses = self.run_simple_fsdp(
            copy.deepcopy(model), inputs, labels, mode="replicate"
        )

        for fsdp2_loss, simple_fsdp_replicate_loss in zip(
            fsdp2_losses, simple_fsdp_replicate_losses
        ):
            assert torch.allclose(fsdp2_loss, simple_fsdp_replicate_loss)
