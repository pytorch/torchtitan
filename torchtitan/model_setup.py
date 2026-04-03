# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, cast

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols import BaseModel, ModelSpec, Module
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools import utils


def prepare_model_config(
    *,
    model_spec: ModelSpec,
    trainer_config: Any,
) -> BaseModel.Config:
    """Update the selected model config from the trainer config."""
    model_config = model_spec.model
    model_config.update_from_config(trainer_config=trainer_config)
    return model_config


def build_model_on_meta(
    *,
    model_config: BaseModel.Config,
    training_dtype: str,
) -> BaseModel:
    """Build a model on the meta device with the configured default dtype."""
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[training_dtype]),
    ):
        return model_config.build()


def parallelize_model(
    model: BaseModel,
    *,
    model_spec: ModelSpec,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
) -> Module:
    """Apply the model spec's non-pipelined parallelization path."""
    return cast(
        Module,
        model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        ),
    )


def materialize_model(
    model: Module,
    *,
    init_device: str | torch.device,
    buffer_device: torch.device | None,
    init_weights_context: contextlib.AbstractContextManager[object] | None = None,
) -> Module:
    """Materialize a meta-initialized model and run init_weights()."""
    model.to_empty(device=init_device)
    with (
        init_weights_context
        if init_weights_context is not None
        else contextlib.nullcontext()
    ):
        with torch.no_grad():
            # TODO: Change this back to init_weights once
            # autoparallel contains the wrap_init_states
            cast(BaseModel, model).init_weights(buffer_device=buffer_device)
    model.train()
    return model
