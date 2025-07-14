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
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.llama3 import parallelize_llama, pipeline_llama
from torchtitan.protocols.train_spec import (
    apply_to_train_specs,
    BaseModelArgs,
    get_train_spec,
    ModelProtocol,
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
    job_config: JobConfig,
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


class TestTrainSpec:
    def test_register_train_spec(self):
        fake_config = {"fake": BaseModelArgs()}
        spec = TrainSpec(
            name="fake",
            model_cls=FakeModel,
            model_args=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=pipeline_llama,
            build_optimizers_fn=build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=build_hf_dataloader,
            build_tokenizer_fn=build_hf_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
        )
        register_train_spec(spec)
        new_spec = get_train_spec("fake")
        assert new_spec == spec

        with pytest.raises(ValueError):
            new_spec = get_train_spec("fake2")

    def test_optim_hook(self):
        fake_config = {"fake": BaseModelArgs()}
        spec = TrainSpec(
            name="fake2",
            model_cls=FakeModel,
            model_args=fake_config,
            parallelize_fn=parallelize_llama,
            pipelining_fn=pipeline_llama,
            build_optimizers_fn=fake_build_optimizers,
            build_lr_schedulers_fn=build_lr_schedulers,
            build_dataloader_fn=build_hf_dataloader,
            build_tokenizer_fn=build_hf_tokenizer,
            build_loss_fn=build_cross_entropy_loss,
        )
        register_train_spec(spec)
        new_spec = get_train_spec("fake2")

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

        def register_optimizer_hook_to_spec(spec: TrainSpec) -> TrainSpec:
            # Create a closure to capture the original spec.build_optimizers_fn
            original_build_optimizers_fn = spec.build_optimizers_fn

            def my_build_optimizer_fn(
                model_parts: list[nn.Module],
                job_config: JobConfig,
                parallel_dims: ParallelDims,
                ft_manager: FTManager,
            ) -> OptimizersContainer:
                optimizers = original_build_optimizers_fn(
                    model_parts, job_config, parallel_dims, ft_manager
                )
                optimizers.register_step_post_hook(
                    partial(my_hook, model_parts=model_parts)
                )
                return optimizers

            spec.build_optimizers_fn = my_build_optimizer_fn
            return spec

        apply_to_train_specs(register_optimizer_hook_to_spec)

        model = new_spec.model_cls(BaseModelArgs())
        model_parts = [model]
        optimizers = new_spec.build_optimizers_fn(model_parts, None, None, None)
        assert optimizers.optimizers[0].__class__.__name__ == "Adam"
        batch = torch.randn(8, 8)
        model(batch).sum().backward()
        assert not hook_called
        optimizers.step()
        assert hook_called
