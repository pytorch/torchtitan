# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 8 train.py
# bash run_training.sh

# this file runs a real training loop with real data, optimizer, metrics, etc.

import os
import time

from typing import Iterable

import torch
import torch.distributed as dist

import torchtitan.components.ft as ft

from torchtitan.components.lr_scheduler import build_lr_schedulers

from torchtitan.components.metrics import build_metrics_processor
from torchtitan.components.optimizer import build_optimizers

from torchtitan.config_manager import ConfigManager, JobConfig

from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.distributed import ParallelDims

from torchtitan.experiments.deepseek_v3.infra.parallelize_deepseek import (
    parallelize_deepseek,
)

# from checkpoint import load_weights_from_hf

from torchtitan.experiments.deepseek_v3.model_config import deepseek_config_registry

from torchtitan.experiments.deepseek_v3.tokenizers.hf_tokenizer import (
    get_hf_tokenizer,
    remove_notset_root_handlers,
)

# from torchtitan.experiments.deepseek_v3.train_configs.custom_args import JobConfig
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import get_device_info

# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


# temp global
_device_type, _device_info = get_device_info()


class Trainer:
    job_config: JobConfig
    device: torch.device

    # states
    step: int


def next_batch(
    data_iterator: Iterable, metrics_processor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

    data_load_start = time.perf_counter()
    batch = next(data_iterator)
    input_dict, labels = batch
    metrics_processor.ntokens_since_last_log += labels.numel()
    metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)

    for k, _ in input_dict.items():
        input_dict[k] = input_dict[k].to(_device_type)
    labels = labels.to(_device_type)
    return input_dict, labels


# Run full model
def run_full_model(
    config: JobConfig,
):

    # setup mesh
    pp_dim = config.parallelism.pipeline_parallel_degree
    ep_dim = config.parallelism.expert_parallel_degree
    if not ep_dim:
        # TODO - the fix for config extension is in PR...need it to land
        # logger.info(f"No EP degree specified, {ep_dim=}")
        ep_dim = 2
        logger.info("Using default EP degree 2")

    fsdp_dim = config.parallelism.data_parallel_shard_degree
    logger.info(f"{pp_dim=}, {ep_dim=}, {fsdp_dim=}, {_device_info=}")

    world_mesh = dist.init_device_mesh(
        "cuda", (pp_dim, ep_dim, fsdp_dim), mesh_dim_names=("pp", "ep", "fsdp")
    )
    logger.info(f"{world_mesh.size()=}")

    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    model_args = deepseek_config_registry.get(model_id, None)
    if model_args is None:
        raise ValueError(f"Model {model_id} not found in registry.")

    # TODO - remove this for full model
    # model_args.num_hidden_layers = 16

    (
        model,
        pp_size,
        pp_rank,
        pp_mesh,
        ep_size,
        ep_rank,
    ) = parallelize_deepseek(world_mesh, device, model_args, rank)

    # build tokenizer
    tokenizer = get_hf_tokenizer(model_id)

    # TODO - ep is not the same as dp really...just a temp shim atm.
    dataloader = build_hf_dataloader(
        dp_world_size=ep_size, dp_rank=ep_rank, tokenizer=tokenizer, job_config=config
    )

    # Synthetic setting
    microbatches = pp_size * 2

    # Use Symmetric Memory for MoE token shuffle.
    # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
    # currently supported for forward only. See `generate.py`.
    # model.setup_symm_mem(torch.bfloat16, device)

    torch.manual_seed(ep_rank)
    bs = config.training.local_batch_size  # * microbatches  # 4
    seqlen = config.training.seq_len  # 128

    # metrics manager
    proxy_parallel_dims = ParallelDims(
        dp_replicate=ep_size,
        dp_shard=fsdp_dim,
        pp=pp_size,
        cp=1,
        tp=1,
        world_size=world_mesh.size(),
        enable_loss_parallel=False,
    )

    metrics_processor = build_metrics_processor(
        config, proxy_parallel_dims, model_args=None
    )
    metrics_processor.num_flops_per_token = 100

    color = metrics_processor.color
    device_memory_monitor = metrics_processor.device_memory_monitor

    # logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    device_module, device_type = utils.device_module, utils.device_type
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{color.yellow}{device_type.upper()} memory usage for model:  {color.reset}"
        f"{color.blue}{device_mem_stats.max_reserved_gib:.2f}GiB {color.reset}"
        f"{color.green}({device_mem_stats.max_reserved_pct:.2f}%){color.reset}"
    )

    # Create loss function
    loss_fn = cross_entropy_loss  # torch.nn.functional.cross_entropy

    ft_manager = ft.init_ft_manager(config)
    optimizer = build_optimizers([model], config, ft_manager)

    lr_scheduler = build_lr_schedulers(optimizer, config)

    # Run forward and backward
    steps = config.training.steps

    loss = float("inf")
    data_iterator = iter(dataloader)

    for step in range(steps):
        optimizer.zero_grad()

        inputs, label = next_batch(data_iterator, metrics_processor)
        x = inputs["input"]

        if pp_size > 1:

            # Create pipeline stage
            stage = PipelineStage(
                model,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
            )

            # Create pipeline schedule
            losses = []
            pp_schedule = Schedule1F1B(stage, microbatches, loss_fn=loss_fn)

            if pp_rank == 0:
                y = pp_schedule.step(x)
            elif pp_rank == pp_size - 1:
                # last rank...run loss function
                y = pp_schedule.step(target=label, losses=losses)
                loss = torch.mean(torch.stack(losses))
            else:
                pp_schedule.step()
        else:
            y = model(x)
            loss = loss_fn(y, label)
            loss.backward()

        if pp_rank == pp_size - 1:

            global_avg_loss = global_max_loss = loss  # .detach().item()

            metrics_processor.log(step, global_avg_loss, global_max_loss)

        optimizer.step()
        lr_scheduler.step()

    metrics_processor.close()
    logger.info("Training completed")


if __name__ == "__main__":
    # set device before init_device mesh, otherwise ep will have duplicate device mapping
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    init_logger()
    # we do this to remove a root logger that is added by HF
    # otherwise we get duplicate logs
    remove_notset_root_handlers()

    config_manager = ConfigManager()
    config = config_manager.parse_args()

    run_full_model(config)

    dist.destroy_process_group()
