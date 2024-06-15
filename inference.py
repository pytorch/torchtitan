# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os
import time

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel

from torchtitan.checkpoint import CheckpointManager
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, create_tokenizer
from torchtitan.float8_linear import build_fp8_linear
from torchtitan.logging_utils import init_logger, logger
from torchtitan.lr_scheduling import get_lr_scheduler
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import (
    models_parallelize_fns,
    models_pipelining_fns,
    ParallelDims,
)
from torchtitan.parallelisms.pipelining_utils import build_pipeline_schedule
from torchtitan.profiling import maybe_enable_profiling
from torchtitan.utils import (
    Color,
    dist_max,
    dist_mean,
    get_metrics_rank,
    get_num_flop_per_token,
    get_num_params,
    get_peak_flops,
    init_distributed,
    NoColor,
    set_pg_timeouts,
)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = Color if job_config.metrics.enable_color_printing else NoColor

    # take control of garbage collection to avoid stragglers
    _gc_freq = job_config.training.gc_freq
    gc.disable()
    gc.collect(1)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=1,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=False,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    init_distributed(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len
    
    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(model, job_config)

    # log model size
    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor()
    # obtain the peak flops of bf16 type for MFU calculation
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)

    if parallel_dims.pp_enabled:
        stage, model = models_pipelining_fns[model_name](
            model, world_mesh, parallel_dims, job_config, device, model_config
        )

    # apply PT-D DP/TP parallelisms and activation checkpointing
    model = models_parallelize_fns[model_name](
        model, world_mesh, parallel_dims, job_config
    )

    init_device = "cuda"
    model.to_empty(device=init_device)

    if parallel_dims.pp_enabled:
        pp_schedule = build_pipeline_schedule(job_config, parallel_dims, stage, loss_fn)
    else:
        # If PP is enabled, we can't rely on init_weights, because some layers are missing.
        # In the future, we may make init_weights handle missing layers, but also have to consider RNG seed propagation.
        # allocate sharded model on GPU and initialize weights via DTensor
        model.init_weights()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    model.eval()

    # Load initial checkpoint
    checkpoint = CheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=data_loader,
        states={"train_state": train_state},
        job_config=job_config,
    )

    checkpoint_loaded = checkpoint.load()

    if parallel_dims.pp_enabled and not checkpoint_loaded:
        raise RuntimeError(
            "Pipeline Parallelism requires meta-initialization and loading seed checkpoint. "
            "Please run `./create_seed_checkpoint.sh` and rerun training with `--checkpoint.enable_checkpoint`"
        )

    gpu_memory_monitor.reset_peak_stats()

    # Single inference pass with model parallel
    logger.info(f"Inference starts")
    with maybe_enable_profiling(
        job_config, global_step=0
    ) as torch_profiler:
        input_ids, labels = batch
        input_ids = input_ids.cuda()

        if parallel_dims.pp_enabled:
            # pipeline parallel forward / backward inside step() call
            is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

            if pp_mesh.get_local_rank() == 0:
                pred = pp_schedule.step(input_ids)
            else:
                pred = pp_schedule.step()

        else:
            # Non-PP forward / backward
            pred = model(input_ids)
            # pred.shape=(bs, seq_len, vocab_size)

        time_delta = timer() - time_last_log

        # tokens per second, abbr. as wps by convention
        wps = ntokens_since_last_log / (
            time_delta * parallel_dims.model_parallel_size
        )

        metrics = {
            "wps": wps,
            "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
            "memory/max_active(%)": gpu_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
            "memory/num_ooms": gpu_mem_stats.num_ooms,
        }
        metric_logger.log(metrics, step=train_state.step)

        logger.info(
            f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}wps: {round(wps):,}  "
        )

        gpu_memory_monitor.reset_peak_stats()


    metric_logger.close()
    logger.info("Inference completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    destroy_process_group()
