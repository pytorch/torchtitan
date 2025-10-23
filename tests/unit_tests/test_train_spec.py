# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
import torch.nn as nn
from torchtitan.components.ft import FTManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.llama3 import parallelize_llama
from torchtitan.protocols import BaseModelArgs, ModelProtocol
from torchtitan.protocols.train_spec import (
    get_train_spec,
    register_train_spec,
    TrainSpec,
)


class FakeModel(nn.Module, ModelProtocol):
    def __init__(self, model_args: BaseModelArgs) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)


def fake_build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager,
) -> OptimizersContainer:
    optimizer_kwargs = {
        "lr": 0.1,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": True,
        "foreach": False,
    }
    return OptimizersContainer(
        model_parts=model_parts,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=optimizer_kwargs,
    )


def fake_build_optimizers_with_hook(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager,
    optimizer_hook,
) -> OptimizersContainer:
    optimizers = fake_build_optimizers(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )
    optimizers.register_step_post_hook(partial(optimizer_hook, model_parts=model_parts))
    return optimizers


class TestTrainSpec:
    def test_register_train_spec(self):
        fake_config = {"fake": BaseModelArgs()}
        spec = TrainSpec(
            model_cls=FakeModel,
            model_args=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            build_optimizers_fn=build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
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
        fake_config = {"fake": BaseModelArgs()}

        spec = TrainSpec(
            model_cls=FakeModel,
            model_args=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=None,
            build_optimizers_fn=fake_build_optimizers_with_hook,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=build_text_dataloader,
            build_tokenizer_fn=build_hf_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
        )
        register_train_spec("fake2", spec)
        new_spec = get_train_spec("fake2")

        model = new_spec.model_cls(BaseModelArgs())
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

        optimizers = new_spec.build_optimizers_fn(
            model_parts, None, None, None, my_hook
        )
        assert optimizers.optimizers[0].__class__.__name__ == "Adam"
        batch = torch.randn(8, 8)
        model(batch).sum().backward()
        assert not hook_called
        optimizers.step()
        assert hook_called
