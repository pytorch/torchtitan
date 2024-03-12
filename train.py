# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import os

from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Any, Dict, List

import numpy as np

# torch imports
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor.parallel import loss_parallel

from torchtrain.checkpoint import CheckpointManager, IntervalType
from torchtrain.config_manager import JobConfig

# torchtrain related
from torchtrain.datasets import create_tokenizer, dataloader_fn
from torchtrain.float8_linear import build_fp8_linear
from torchtrain.logging_utils import init_logger, rank0_log
from torchtrain.lr_scheduling import get_lr_scheduler
from torchtrain.meta_init import meta_model_init
from torchtrain.metrics import build_metric_logger, get_num_params, GPUMemoryMonitor

from torchtrain.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtrain.parallelisms import models_parallelize_fns, ParallelDims

from torchtrain.profiling import maybe_run_profiler
from torchtrain.utils import Color, dist_max, dist_mean

_is_local_logging = True
if "SLURM_JOB_ID" in os.environ:
    _is_local_logging = False


@dataclass
class TrainState:
    step: int = 0
    current_loss: float = -1
    losses: List[float] = field(default_factory=list)
    iter_times: List[float] = field(default_factory=list)
    data_load_times: List[float] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "current_loss": torch.tensor(self.current_loss, dtype=torch.float32),
            "losses": torch.tensor(self.current_loss, dtype=torch.float32),
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.current_loss = state_dict["current_loss"].item()
        self.losses = state_dict["losses"].tolist()


def build_optimizer(model, job_config: JobConfig):
    # build optimizer
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    if name == "Adam":
        # TODO: make the optimizer options configurable by toml/cmd args
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
    else:
        raise NotImplementedError(f"optimizer {name} not added")

    return optimizer


def build_grad_scaler(model):
    # apply gradient scaling if mixed precision training is enabled with fp16 param dtype
    if model.mixed_precision.param_dtype == torch.float16:
        enable_grad_scaling = True
        rank0_log("Enabling gradient scaling for mixed precision training.")
    else:
        enable_grad_scaling = False
        rank0_log("Gradient scaling not enabled.")

    return ShardedGradScaler(enabled=enable_grad_scaling)


def main(job_config: JobConfig):
    init_logger()
    # init world mesh
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        sp=job_config.training.sequence_parallel_degree,
        pp=job_config.training.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    rank0_log(f"Starting job: {job_config.job.description}")
    model_name = job_config.model.name
    rank0_log(f"Building {model_name}")
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
    rank0_log(
        f"{Color.green}Built Dataloader for '{job_config.training.dataset}' dataset.{Color.reset}"
    )

    # build model
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    model_config.vocab_size = tokenizer.n_words

    # build model using meta init
    with meta_model_init():
        model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(model, job_config)

    # log model size
    model_param_count = get_num_params(model)

    if _is_local_logging:
        rank0_log(
            f"{Color.blue}Model {model_name} {job_config.model.flavor} {Color.red}size: {model_param_count:,}"
            f" total parameters{Color.reset}"
        )
    else:
        rank0_log(
            f"{model_name} {job_config.model.flavor} size: {model_param_count:,} total parameters"
        )

    gpu_metrics = GPUMemoryMonitor("cuda")
    rank0_log(f"GPU memory usage: {gpu_metrics}")

    # apply PTD parallelisms + AC
    model = models_parallelize_fns[model_name](
        model, world_mesh, parallel_dims, job_config
    )

    # to use FSDP-customized gradient scaler and gradient clipping solutions
    assert isinstance(model, FSDP)

    # build optimizer after apply parallelisms to the model
    optimizer = build_optimizer(model, job_config)
    scheduler = get_lr_scheduler(optimizer, job_config)

    scaler = build_grad_scaler(model)

    metric_logger = build_metric_logger(job_config)

    # torch.compile model for improved performance
    if job_config.training.compile:
        if job_config.training.enable_selective_ac:
            torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = (
                True
            )
        rank0_log(f"Compiling model {model_name} with torch.compile...")
        model = torch.compile(
            model,
        )

    train_state = TrainState()

    # train loop
    model.train()

    if job_config.training.checkpoint_folder:
        folder = f"{job_config.training.checkpoint_folder}_{os.environ['USER']}"
    else:
        folder = ""
    checkpoint = CheckpointManager(
        model=model,
        optimizer=optimizer,
        states={"train_state": train_state},
        folder=folder,
        interval_type=(
            IntervalType.SECONDS
            if job_config.training.checkpoint_interval_type == "seconds"
            else IntervalType.STEPS
        ),
        interval=job_config.training.checkpoint_interval,
        enable_mp=False,
        overlap_with_training=False,
    )
    checkpoint.load()

    data_iterator = iter(data_loader)

    with maybe_run_profiler(job_config) as torch_profiler:
        checkpoint.reset()
        # variables used to keep info for metrics logging
        losses_since_last_log: List[float] = []
        nwords_since_last_log = 0
        time_last_log = timer()
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            # get batch
            data_load_start = timer()
            batch = next(data_iterator)
            input_ids, labels = batch
            input_ids = input_ids.cuda()
            labels = labels.cuda()
            data_load_time = round(timer() - data_load_start, 4)
            train_state.data_load_times.append(data_load_time)
            nwords_since_last_log += labels.numel()

            optimizer.zero_grad()

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            pred = model(input_ids)

            with loss_parallel() if parallel_dims.loss_parallel_enabled else contextlib.nullcontext():
                loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

                # backward on scaled loss to create scaled gradients
                scaler.scale(loss).backward()

            # This is a simple if statement if checkpoint does not happen and doesn't
            # affect the performance.
            checkpoint.wait_staging()

            # clip gradients (after unscaling gradients of the optimizer's params)
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(job_config.training.max_norm)

            # optimizer step
            # If gradients don't contain infs/NaNs, optimizer.step() is then called;
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # updates the scale for next iteration
            scaler.update()

            # training iteration complete
            end_timer.record()
            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)
            train_state.iter_times.append(curr_iter_time)

            # if profiler is active
            if torch_profiler:
                torch_profiler.step()

            train_state.current_loss = loss.item()
            train_state.losses.append(train_state.current_loss)
            losses_since_last_log.append(train_state.current_loss)

            # log metrics
            if (train_state.step - 1) % job_config.metrics.log_freq == 0:
                avg_loss, max_loss = (
                    np.mean(losses_since_last_log),
                    np.max(losses_since_last_log),
                )
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        dist_mean(avg_loss, dp_mesh),
                        dist_max(max_loss, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )

                gpu_mem_stats = gpu_metrics.get_current_stats(return_data=True)

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "wps": wps,
                    "memory_current/active(%)": gpu_mem_stats.active_curr,
                    "memory_current/allocated(%)": gpu_mem_stats.allocated_curr,
                    "memory_current/reserved(%)": gpu_mem_stats.reserved_curr,
                    "memory_peak/active(%)": gpu_mem_stats.active_peak,
                    "memory_peak/allocated(%)": gpu_mem_stats.allocated_peak,
                    "memory_peak/reserved(%)": gpu_mem_stats.reserved_peak,
                }
                metric_logger.log(metrics, step=train_state.step)

                losses_since_last_log.clear()
                nwords_since_last_log = 0
                time_last_log = timer()

            if _is_local_logging:
                rank0_log(
                    f"{Color.cyan}step: {train_state.step:>2}  {Color.green}loss: {round(train_state.current_loss,4):>7}"
                    f"  {Color.reset}iter: {Color.blue}{curr_iter_time:>7}{Color.reset}"
                    f"  data: {Color.blue}{data_load_time:>5}  {Color.reset}"
                    f"lr: {Color.yellow}{round(float(scheduler.get_last_lr()[0]), 8):<6}{Color.reset}"
                )
            else:
                rank0_log(
                    f"step: {train_state.step:>2}  loss: {round(train_state.current_loss,4):>7}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}  "
                    f"lr: {round(float(scheduler.get_last_lr()[0]), 8):<6}"
                )

            scheduler.step()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

    metric_logger.close()
    # calc and show average iter time, disregard first three iterations (warmup)
    if len(train_state.iter_times) > 3:
        avg_iter_time = np.mean(train_state.iter_times[3:])
        rank0_log(f"Average iter time: {avg_iter_time:.4f} seconds")
        avg_data_load_time = np.mean(train_state.data_load_times[3:])
        rank0_log(f"Average data load time: {avg_data_load_time:.4f} seconds")

    rank0_log(f"{gpu_metrics.get_current_stats()}")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
