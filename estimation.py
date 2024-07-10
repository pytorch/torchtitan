# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os

import torch
import torch.nn.functional as F
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed import destroy_process_group
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.distributed.tensor.parallel import loss_parallel
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import create_tokenizer
from torchtitan.float8_linear import build_fp8_linear
from torchtitan.logging_utils import init_logger, logger
from torchtitan.lr_scheduling import get_lr_schedulers
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims
from train import build_optimizers


def estimate_memory(job_config: JobConfig):
    init_logger()
    logger.info("Estimating memory usage...")
    gc.disable()
    gc.collect(1)

    # Get the world size
    world_size = int(os.environ["WORLD_SIZE"])

    # if tp > or pp > 1, we exit
    if (
        job_config.training.tensor_parallel_degree > 1
        or job_config.experimental.pipeline_parallel_degree > 1
    ):
        logger.info(
            "Tensor parallelism and pipeline parallelism are not supported yet."
        )
        return

    # fake tensor doesn't work with fused rmsnorm
    if (
        job_config.model.norm_type == "fused_rmsnorm"
        and not job_config.memory_estimation.disable_fake_mode
    ):
        logger.info(
            "Fused RMSNorm is not supported yet under fake estimation mode. "
            "Switching to rmsnorm."
        )
        job_config.model.norm_type = "rmsnorm"

    if job_config.training.compile:
        logger.info("Compile mode is not supported yet. " "Switching to Eager mode.")
        job_config.training.compile = False

    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    # init fake pg
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=int(os.environ["LOCAL_RANK"]), world_size=world_size, store=store
    )

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    if not parallel_dims.dp_enabled:
        logger.info("Data parallelism is not enabled. Skipping memory estimation.")
        return

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # loss_parallel enables dispatching to efficient loss operators
    loss_parallel_ctx = (
        loss_parallel if parallel_dims.loss_parallel_enabled else contextlib.nullcontext
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))

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

    with FakeTensorMode() if not job_config.memory_estimation.disable_fake_mode else contextlib.nullcontext():

        logger.info(
            f"Building {model_name} {job_config.model.flavor} with {model_config}"
        )
        with torch.device("meta"):
            whole_model = model_cls.from_model_args(model_config)

        # apply fp8 linear module swap
        if job_config.training.fp8_linear:
            build_fp8_linear(whole_model, job_config)

        # apply PT-D DP/TP parallelisms and activation checkpointing
        model_parts = [whole_model]
        model_parts = [
            models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
            for m in model_parts
        ]

        init_device = "cuda"
        for model in model_parts:
            model.to_empty(device=init_device)

        if not active_fake_mode():
            whole_model.init_weights()

        # build optimizer after applying parallelisms to the model
        optimizers = build_optimizers(model_parts, job_config)
        lr_schedulers = get_lr_schedulers(optimizers.optimizers, job_config)

        for model in model_parts:
            model.train()
        logger.info(f"Vocab size: {model_config.vocab_size}")
        # Create a dummy batch instead of loading from a dataset
        batch = (
            torch.randint(
                0,
                model_config.vocab_size,
                (job_config.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
            torch.randint(
                0,
                model_config.vocab_size,
                (job_config.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
        )
        fsdp_memtracker = FSDPMemTracker(mod=whole_model, optm=optimizers.optimizers[0])
        fsdp_memtracker.track_inputs(batch)

        with fsdp_memtracker:
            for iter_idx in range(2):
                input_ids, labels = batch
                # train step
                with loss_parallel_ctx():
                    pred = whole_model(input_ids)
                    loss = loss_fn(pred, labels)
                    del pred
                    loss.backward()

                # clip gradients
                for model in model_parts:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), job_config.training.max_norm, foreach=True
                    )
                # optimizer step
                optimizers.step()
                lr_schedulers.step()
                optimizers.zero_grad()
                print(f"Peak Memory at iter: {iter_idx}")
                fsdp_memtracker.display_snapshot("peak", units="MiB", tabulate=True)
                if iter_idx == 0:
                    fsdp_memtracker.reset_mod_stats()  # iter 0 does not have optimizer state
                gc.collect(1)

        fsdp_memtracker.display_modulewise_snapshots(
            depth=3, units="MiB", tabulate=True
        )
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        num_retries = mem_stats["num_alloc_retries"]
        dev = torch.device(torch.cuda.current_device())
        tracker_peak = fsdp_memtracker.get_tracker_snapshot("peak")[dev]["Total"]
        gib = 1024**3
        print(
            f"peak active: {peak_active / gib} GiB | peak reserved:"
            f" {peak_reserved / gib} GiB | num_retries: {num_retries}"
        )
        print(f"Tracker Max: {tracker_peak / gib} GiB")
        if job_config.memory_estimation.disable_fake_mode and peak_active > 0:
            print(f"Tracker Accuracy: {tracker_peak/peak_active}")
        gc.enable()


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    try:
        estimate_memory(config)
    finally:
        destroy_process_group()
