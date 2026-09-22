# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import Llama3Model, model_registry
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols import BaseModel


class FakeModel(BaseModel):
    hook_called = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hidden: int = 8

        def update_from_config(self, *, config, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len):
            return 0, 0

    def __init__(self, config: Config):
        super().__init__()
        self.linear = Linear.Config(
            in_features=config.hidden,
            out_features=config.hidden,
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def _init_self_parameters(self) -> None:
        nn.init.trunc_normal_(self.linear.weight, std=0.02)

    def _apply_fsdp(self, **kwargs) -> None:
        pass

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallel_dims) -> None:
        def hook(optimizer, args, kwargs):
            cls.hook_called = True

        optimizers.register_step_post_hook(hook)


def test_model_registry_returns_model_config() -> None:
    config = model_registry("debugmodel")
    assert isinstance(config, Llama3Model.Config)
    assert config.max_context_length == 131072
    assert Llama3Model.state_dict_adapter_cls is Llama3StateDictAdapter


def test_optimizer_hook_is_owned_by_model_class() -> None:
    model = FakeModel.Config().build()
    model_parts = [model]
    optimizers = OptimizersContainer.Config(
        implementation="fused",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="Adam",
                optimizer_kwargs={"lr": 0.1},
            )
        ],
    ).build(model_parts=model_parts)

    type(model)._register_optimizer_hooks(optimizers, model_parts, None)
    model(torch.randn(8, 8)).sum().backward()
    optimizers.step()
    assert FakeModel.hook_called
