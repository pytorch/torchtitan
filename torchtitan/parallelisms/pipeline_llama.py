# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from .pipelining import PipelineStage

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import logger
from torchtitan.models.llama.model import ModelArgs
from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.pipelining_utils import (
    build_pipeline_schedule,
    stage_ids_this_rank,
)


DeviceType = Union[int, str, torch.device]


def pipeline_llama(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: ModelArgs,
    loss_fn: Callable[..., torch.Tensor],
):
    stages, models = pipeline_llama_manual_split(
        model, pp_mesh, parallel_dims, job_config, device, model_config
    )

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    logger.info("Applied PP to the model")
    return pp_schedule, models


def _llama_trace_input(job_config: JobConfig, model_config: ModelArgs, device="meta"):
    """Get meta tensors with the right input shapes used for tracing"""
    tokens_shape = (job_config.training.batch_size, job_config.training.seq_len)
    tokens = torch.randint(
        model_config.vocab_size, tokens_shape, dtype=torch.int64, device=device
    )
    return (tokens,)


def _mixed_precision_dtype(
    job_config: JobConfig, parallel_dims, default: torch.dtype = torch.float32
) -> torch.dtype:
    """Get the mixed precision dtype if FSDP is enabled, otherwise return the default"""
    mp_arg = job_config.training.mixed_precision_param
    return TORCH_DTYPE_MAP[mp_arg] if parallel_dims.dp_enabled else default


def pipeline_llama_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: ModelArgs,
):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    microbatches = (
        job_config.experimental.pipeline_parallel_microbatches or parallel_dims.pp
    )
    splits = job_config.experimental.pipeline_parallel_split_points

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.tok_embeddings = None

        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]

        if not is_last:
            model.norm = None
            model.output = None

        # TODO(whc) once ManualPipelineStage supports lazy shape inference, we can leave model on meta device longer and
        # get rid of the input shape hardcoded here. For now, it should not be a big deal since we only materialize the
        # layers of the model that map to this stage, not the whole model.
        mp_dtype = _mixed_precision_dtype(job_config, parallel_dims)
        batch_size = job_config.training.batch_size
        local_seq_len = int(job_config.training.seq_len // parallel_dims.tp)
        layers_io_shape = (batch_size, local_seq_len, model_config.dim)
        output_layer_shape = (
            batch_size,
            job_config.training.seq_len,
            model_config.vocab_size,
        )
        if is_first:
            (input,) = _llama_trace_input(job_config, model_config, device=device)
        else:
            # later layers (assume all start w/ a transformer layer)
            input = torch.rand(layers_io_shape, dtype=mp_dtype, device=device)

        if is_last:
            output = torch.rand(output_layer_shape, dtype=torch.float32, device=device)
        else:
            # earlier layers (assume all end in a transformer layer)
            output = torch.rand(layers_io_shape, dtype=mp_dtype, device=device)

        model.to_empty(device=device)
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            input_args=input.chunk(microbatches)[0],
            output_args=output.chunk(microbatches)[0],
            group=pp_mesh.get_group("pp"),
            # dw_builder=lambda: lambda: print("dummy dw_runner"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []
    loop_style = (
        "v" if job_config.experimental.pipeline_parallel_schedule == "zb_v" else "loop"
    )
    for stage_idx in stage_ids_this_rank(
        pp_rank, pp_size, num_stages, style=loop_style
    ):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models
