# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.model_setup import (
    build_model_on_meta,
    materialize_model,
    parallelize_model,
    prepare_model_config,
)
from torchtitan.models.common.linear import Linear
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec


class DummyModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hidden: int = 4
        updated_hidden: int = 0

        def update_from_config(self, *, trainer_config, **kwargs):
            self.updated_hidden = trainer_config.hidden

        def get_nparams_and_flops(self, model, seq_len):
            return (0, 0)

    def __init__(self, config: Config):
        super().__init__()
        self.linear = Linear.Config().build(
            in_features=config.hidden,
            out_features=config.hidden,
        )
        self.saw_init_weights_context = False
        self.init_buffer_device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_weights(self, buffer_device: torch.device | None = None, **kwargs) -> None:
        self.saw_init_weights_context = bool(getattr(self, "_context_active", False))
        self.init_buffer_device = buffer_device


@dataclass(kw_only=True, slots=True)
class DummyTrainerConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hidden: int = 7


class _TrackingContext:
    def __init__(self, model: DummyModel):
        self.model = model
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        self.model._context_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        self.model._context_active = False
        return False


class TestModelSetup:
    def test_prepare_and_build_model_on_meta(self):
        model_spec = ModelSpec(
            name="dummy",
            flavor="test",
            model=DummyModel.Config(),
            build_loss_fn=build_cross_entropy_loss,
            parallelize_fn=lambda model, **kwargs: model,
            pipelining_fn=None,
            post_optimizer_build_fn=None,
            state_dict_adapter=None,
        )
        trainer_config = DummyTrainerConfig()

        model_config = prepare_model_config(
            model_spec=model_spec,
            trainer_config=trainer_config,
        )
        model = build_model_on_meta(
            model_config=model_config,
            training_dtype=trainer_config.training.dtype,
        )

        assert model_config.updated_hidden == trainer_config.hidden
        assert next(model.parameters()).device.type == "meta"

    def test_parallelize_and_materialize_model(self):
        seen_kwargs = {}

        def fake_parallelize_fn(model, **kwargs):
            seen_kwargs.update(kwargs)
            return model

        model_spec = ModelSpec(
            name="dummy",
            flavor="test",
            model=DummyModel.Config(),
            build_loss_fn=build_cross_entropy_loss,
            parallelize_fn=fake_parallelize_fn,
            pipelining_fn=None,
            post_optimizer_build_fn=None,
            state_dict_adapter=None,
        )
        trainer_config = DummyTrainerConfig()
        model_config = prepare_model_config(
            model_spec=model_spec,
            trainer_config=trainer_config,
        )
        model = build_model_on_meta(
            model_config=model_config,
            training_dtype=trainer_config.training.dtype,
        )

        parallel_dims = ParallelDims(
            dp_shard=1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )
        compile_config = CompileConfig()
        activation_checkpoint = ActivationCheckpointConfig()
        parallelism = ParallelismConfig()
        model_converters = ModelConvertersContainer.Config()

        model = parallelize_model(
            model,
            model_spec=model_spec,
            parallel_dims=parallel_dims,
            training=trainer_config.training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=activation_checkpoint,
            dump_folder="/tmp/model_setup_test",
        )
        tracking_context = _TrackingContext(model)
        materialize_model(
            model,
            init_device="cpu",
            buffer_device=torch.device("cpu"),
            init_weights_context=tracking_context,
        )

        assert seen_kwargs["parallel_dims"] is parallel_dims
        assert seen_kwargs["compile_config"] is compile_config
        assert tracking_context.entered
        assert tracking_context.exited
        assert model.saw_init_weights_context
        assert model.init_buffer_device == torch.device("cpu")
        assert next(model.parameters()).device.type == "cpu"
