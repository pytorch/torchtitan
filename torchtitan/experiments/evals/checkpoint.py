# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build the eval model from a training config and load DCP weights into it."""

import dataclasses
import os
import re

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.config import CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP
from torchtitan.config.override import apply_overrides
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model import BaseModel
from torchtitan.tools import utils
from torchtitan.trainer import Trainer

_STEP_DIR = re.compile(r"^step-(\d+)$")


def checkpoint_folder(config: Trainer.Config) -> str:
    """Folder the training job writes ``step-N`` checkpoints into."""
    return os.path.join(config.dump_folder, config.checkpoint.folder)


def list_checkpoint_steps(folder: str) -> list[int]:
    """Steps of the completed checkpoints in ``folder``, ascending.

    A checkpoint counts as complete once DCP has written its ``.metadata``
    file, which happens after all shards are written.
    """
    if not os.path.isdir(folder):
        return []
    steps = []
    for name in os.listdir(folder):
        match = _STEP_DIR.match(name)
        if match and os.path.isfile(os.path.join(folder, name, ".metadata")):
            steps.append(int(match.group(1)))
    return sorted(steps)


def eval_parallelism(
    parallelism: ParallelismConfig, *, world_size: int, tensor_parallel_degree: int
) -> ParallelismConfig:
    """The eval job's layout: tensor parallelism inside data-parallel replicas.

    Every other axis of the training layout is dropped. lm-eval gives each
    data-parallel rank different requests, so any axis whose forward pass
    communicates across data-parallel ranks (FSDP sharding, EP, CP, PP) could
    leave ranks waiting on collectives the others never issue. Replicas do not
    communicate in the forward pass, and TP ranks share a data-parallel rank
    and therefore its requests.
    """
    if world_size % tensor_parallel_degree != 0:
        raise ValueError(
            f"tensor_parallel_degree ({tensor_parallel_degree}) must divide "
            f"the number of GPUs ({world_size})."
        )
    return dataclasses.replace(
        parallelism,
        data_parallel_replicate_degree=world_size // tensor_parallel_degree,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=tensor_parallel_degree,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
    )


def build_model(
    config: Trainer.Config,
    *,
    parallel_dims: ParallelDims,
    device: torch.device,
    dtype: str,
) -> BaseModel:
    """Build and parallelize the model exactly as the trainer does.

    ``config.parallelism`` must already be the eval layout (see
    ``eval_parallelism``). Parameters are held in ``dtype``, and the FSDP
    mixed-precision policy computes in it too.
    """
    model_config = config.model_spec.model
    # Mirror Trainer.__init__: update_from_config, then overrides.
    model_config.update_from_config(config=config)
    if config.override.imports:
        apply_overrides(config.override, config)

    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()
    cast_parameters(model, TORCH_DTYPE_MAP[dtype])

    # The trainer's parallelize path, so parameters are the same DTensors a
    # training job checkpoints. With a shard degree of 1, FSDP only replicates
    # and issues no cross-replica collectives in the forward pass. No
    # activation checkpointing or compile.
    model = config.model_spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=dataclasses.replace(config.training, mixed_precision_param=dtype),
        parallelism=config.parallelism,
        compile_config=CompileConfig(),
        ac_config=None,
        dump_folder=config.dump_folder,
    )
    model.to_empty(device=device)
    with torch.no_grad(), dist_utils.get_spmd_context(parallel_dims=parallel_dims)():
        # Initializes non-persistent buffers (e.g. RoPE caches) that DCP does
        # not restore; parameters are overwritten by load_weights().
        model.init_weights(buffer_device=None)
    model.eval()
    return model


def cast_parameters(model: BaseModel, dtype: torch.dtype) -> None:
    """Cast parameters, not buffers, like FSDP's ``MixedPrecisionPolicy``.

    ``Module.to(dtype)`` also casts complex buffers, which silently drops the
    imaginary part of complex RoPE caches and corrupts every score.
    """
    for param in model.parameters():
        param.data = param.data.to(dtype)


def load_weights(model: BaseModel, checkpoint_path: str) -> None:
    """Load the model weights of a DCP checkpoint into ``model``.

    Works for both full training checkpoints and model-only checkpoints: DCP
    only reads the keys requested, so optimizer and dataloader states are
    skipped.
    """
    wrapper = ModelWrapper(model)
    state_dict = wrapper.state_dict()
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    # Goes through ModelWrapper so state_dict hooks (e.g. fused parameters)
    # write the loaded values back into the module.
    wrapper.load_state_dict(state_dict)
