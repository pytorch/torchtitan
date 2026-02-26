# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.llama3 import model_registry, parallelize_llama
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec


class FakeModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hidden: int = 8

        def update_from_config(self, *, trainer_config, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len):
            return 0, 0

    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.hidden, config.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)


def fake_post_optimizer_build_fn(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
    optimizer_hook=None,
) -> None:
    if optimizer_hook is not None:
        optimizers.register_step_post_hook(
            partial(optimizer_hook, model_parts=model_parts)
        )


class TestModelSpec:
    def test_model_registry(self):
        spec = model_registry("debugmodel")
        assert isinstance(spec, ModelSpec)
        assert spec.name == "llama3"
        assert spec.flavor == "debugmodel"
        assert spec.model is not None
        assert spec.parallelize_fn == parallelize_llama
        assert spec.build_loss_fn == build_cross_entropy_loss

    def test_model_spec_creation(self):
        fake_config = FakeModel.Config()
        spec = ModelSpec(
            name="fake",
            flavor="test",
            model=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            build_loss_fn=build_cross_entropy_loss,
            post_optimizer_build_fn=None,
            state_dict_adapter=None,
        )
        assert spec.name == "fake"
        assert spec.flavor == "test"
        assert spec.model == fake_config

    def test_optim_hook(self):
        fake_config = FakeModel.Config()

        spec = ModelSpec(
            name="fake",
            flavor="test",
            model=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            build_loss_fn=build_cross_entropy_loss,
            post_optimizer_build_fn=fake_post_optimizer_build_fn,
            state_dict_adapter=None,
        )

        model = FakeModel.Config().build()
        model_parts = [model]

        # Demonstrate how to register a optimizer hook for all model specs
        hook_called = False

        def my_hook(
            optimizer: torch.optim.Optimizer,
            args,
            kwargs,
            model_parts: list[nn.Module],
        ) -> None:
            nonlocal hook_called
            hook_called = True

        # Build optimizers directly and apply post-build hook
        optimizers = OptimizersContainer.Config(
            name="Adam",
            lr=0.1,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            implementation="fused",
        ).build(model_parts=model_parts)
        spec.post_optimizer_build_fn(optimizers, model_parts, None, my_hook)

        assert optimizers.optimizers[0].__class__.__name__ == "Adam"
        batch = torch.randn(8, 8)
        model(batch).sum().backward()
        assert not hook_called
        optimizers.step()
        assert hook_called
