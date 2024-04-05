# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import gc
import os

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor.parallel import loss_parallel

from torchtrain.checkpoint import CheckpointManager, IntervalType
from torchtrain.config_manager import JobConfig
from torchtrain.datasets import create_tokenizer, dataloader_fn
from torchtrain.float8_linear import build_fp8_linear
from torchtrain.logging_utils import init_logger, logger
from torchtrain.lr_scheduling import get_lr_scheduler
from torchtrain.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtrain.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtrain.parallelisms import models_parallelize_fns, ParallelDims
from torchtrain.profiling import maybe_run_profiler
from torchtrain.utils import (
    Color,
    dist_max,
    dist_mean,
    get_num_flop_per_token,
    get_num_params,
    get_peak_flops,
    init_distributed,
    set_pg_timeouts,
)

_is_local_logging = True
if "SLURM_JOB_ID" in os.environ:
    _is_local_logging = False


@dataclass
class TrainState:
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


def build_optimizer(model, job_config: JobConfig):
    # build optimizer
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    if name == "Adam":
        # TODO: make the optimizer options configurable by toml/cmd args
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=True
        )
    elif name == "AdamW":
        local_params = [p._local_tensor for p in model.parameters()]
        optimizer = torch.optim.AdamW(
            local_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=True
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    return optimizer


def build_grad_scaler(model):
    # TODO: FSDP2 does not support sharded grad scaler yet.
    # TODO: if enabled, grad scaler's states need saving & loading in checkpointing
    enable_grad_scaling = False
    logger.info("Gradient scaling not enabled")
    return ShardedGradScaler(enabled=enable_grad_scaling)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # take control of garbage collection to avoid stragglers
    _gc_freq = job_config.training.gc_freq
    gc.disable()
    gc.collect(1)

    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.training.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed(job_config)

    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    build_dataloader_fn = dataloader_fn[job_config.training.dataset]
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    data_loader = build_dataloader_fn(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    model_config.vocab_size = tokenizer.n_words

    with torch.device("meta"):
        logger.info(
            f"Building {model_name} {job_config.model.flavor} with {model_config}"
        )
        model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(model, job_config)

    # log model size
    model_param_count = get_num_params(model)
    num_flop_per_token = get_num_flop_per_token(
        model_param_count, model_config, job_config.training.seq_len
    )
    if _is_local_logging:
        logger.info(
            f"{Color.blue}Model {model_name} {job_config.model.flavor} "
            f"{Color.red}size: {model_param_count:,} total parameters{Color.reset}"
        )
    else:
        logger.info(
            f"{model_name} {job_config.model.flavor} size: {model_param_count:,} total parameters"
        )

    # initialize GPU memory monitor before applying parallelisms to the model
    gpu_memory_monitor = build_gpu_memory_monitor()
    # obtain the peak flops of bf16 type for MFU calculation
    gpu_peak_flops = get_peak_flops(gpu_memory_monitor.device_name)

    # apply PT-D parallelisms and activation checkpointing
    model = models_parallelize_fns[model_name](
        model, world_mesh, parallel_dims, job_config
    )
    # allocate sharded model on GPU and initialize weights via DTensor
    model.to_empty(device="cuda")
    model.init_weights()

    # build optimizer after applying parallelisms to the model
    optimizer = build_optimizer(model, job_config)
    scheduler = get_lr_scheduler(optimizer, job_config)

    # build grad scaler which is effective only when mixed precision training
    # is enabled with fp16 param dtype under FSDP
    scaler = build_grad_scaler(model)

    metric_logger = build_metric_logger(job_config)

    # torch.compile model for improved performance
    if job_config.training.compile:
        if (
            job_config.activation_checkpoint.mode == "selective"
            and job_config.activation_checkpoint.selective_ac_option == "op"
        ):
            torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
                True
            )
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    train_state = TrainState()

    # train loop
    model.train()

    checkpoint = CheckpointManager(
        model=model,
        optimizer=optimizer,
        states={"train_state": train_state},
        folder=job_config.checkpoint.folder,
        interval_type=(
            IntervalType.SECONDS
            if job_config.checkpoint.interval_type == "seconds"
            else IntervalType.STEPS
        ),
        interval=job_config.checkpoint.interval,
    )
    checkpoint.load()

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    with maybe_run_profiler(job_config) as torch_profiler:
        checkpoint.reset()

        # variables used to keep info for metrics logging
        losses_since_last_log: List[float] = []
        ntokens_since_last_log = 0
        data_loading_times: List[float] = []
        time_last_log = timer()

        while train_state.step < job_config.training.steps:
            train_state.step += 1
            if train_state.step > 1 and train_state.step % _gc_freq == 0:
                gc.collect(1)

            # get batch
            data_load_start = timer()
            batch = next(data_iterator)
            input_ids, labels = batch
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(timer() - data_load_start)

            input_ids = input_ids.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            # forward
            pred = model(input_ids)

            with (
                loss_parallel()
                if parallel_dims.loss_parallel_enabled
                else contextlib.nullcontext()
            ):
                loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

                # backward on scaled loss to create scaled gradients
                scaler.scale(loss).backward()

            # clip gradients (after unscaling gradients of the optimizer's params)
            # HACK: Forward the local gradient to the local parameter for the
            # local optimizer.
            from torch.distributed._tensor import DTensor
            for param in model.parameters():
                assert isinstance(param, DTensor)
                param._local_tensor.grad = param.grad._local_tensor
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), job_config.training.max_norm, foreach=True
            )

            # optimizer step
            # If gradients don't contain infs/NaNs, optimizer.step() is then called;
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            scheduler.step()

            # updates the scale for next iteration
            scaler.update()

            current_loss = loss.item()
            losses_since_last_log.append(current_loss)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                avg_loss, max_loss = (
                    np.mean(losses_since_last_log),
                    np.max(losses_since_last_log),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh).item(),
                        dist_max(max_loss, dp_mesh).item(),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = timer() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = np.mean(data_loading_times)
                time_data_loading_pct = 100 * np.sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                if _is_local_logging:
                    logger.info(
                        f"{Color.cyan}step: {train_state.step:2}  "
                        f"{Color.green}loss: {global_avg_loss:7.4f}  "
                        f"{Color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                        f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                        f"{Color.blue}wps: {round(wps):,}  "
                        f"{Color.magenta}mfu: {mfu:.2f}%{Color.reset}"
                    )
                else:
                    logger.info(
                        f"step: {train_state.step:2}  "
                        f"loss: {global_avg_loss:7.4f}  "
                        f"memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                        f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                        f"wps: {round(wps):,}  "
                        f"mfu: {mfu:.2f}%"
                    )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = timer()
                gpu_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signals the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()

            # Reduce timeout after first train step for faster signal (assumes lazy init, compile are finished)
            if train_state.step == 1:
                set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    metric_logger.close()


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
