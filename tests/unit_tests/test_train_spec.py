# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import pytest
import torch
import torch.nn as nn
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.llama3 import parallelize_llama
from torchtitan.protocols import BaseModel
from torchtitan.protocols.train_spec import (
    get_train_spec,
    register_train_spec,
    TrainSpec,
)


class FakeModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hidden: int = 8

        def update_from_config(self, *, job_config, **kwargs):
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


class TestTrainSpec:
    def test_register_train_spec(self):
        fake_config = {"fake": FakeModel.Config()}
        spec = TrainSpec(
            model_configs=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            build_dataloader_fn=build_text_dataloader,
            build_tokenizer_fn=build_hf_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
        )
        register_train_spec("fake", spec)
        new_spec = get_train_spec("fake")
        assert new_spec == spec

        with pytest.raises(ValueError):
            new_spec = get_train_spec("fake2")

    def test_optim_hook(self):
        fake_config = {"fake": FakeModel.Config()}

        spec = TrainSpec(
            model_configs=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            build_dataloader_fn=build_text_dataloader,
            build_tokenizer_fn=build_hf_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
            post_optimizer_build_fn=fake_post_optimizer_build_fn,
        )
        register_train_spec("fake2", spec)
        new_spec = get_train_spec("fake2")

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
        optimizer_kwargs = {
            "lr": 0.1,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": True,
            "foreach": False,
        }
        optimizers = OptimizersContainer(
            model_parts=model_parts,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=optimizer_kwargs,
        )
        new_spec.post_optimizer_build_fn(optimizers, model_parts, None, my_hook)

        assert optimizers.optimizers[0].__class__.__name__ == "Adam"
        batch = torch.randn(8, 8)
        model(batch).sum().backward()
        assert not hook_called
        optimizers.step()
        assert hook_called
