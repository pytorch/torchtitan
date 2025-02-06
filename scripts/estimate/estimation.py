# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import os

import torch
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan import utils
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims


def estimate_memory(job_config: JobConfig):
    init_logger()
    logger.info("Estimating memory usage...")
    gc.disable()
    gc.collect(1)

    # Get the world size
    world_size = int(os.environ["WORLD_SIZE"])

    if job_config.model.norm_type == "compiled_rmsnorm":
        logger.info("Compiled RMSNorm is not supported yet. Switching to RMSNorm.")
        job_config.model.norm_type = "rmsnorm"

    if job_config.training.compile or job_config.experimental.enable_compiled_autograd:
        logger.info("Compile mode is not supported yet. Switching to eager mode.")
        job_config.training.compile = False
        job_config.experimental.enable_compiled_autograd = False

    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )

    # only FSDP and HSDP are supported
    if (
        (parallel_dims.dp_replicate_enabled and not parallel_dims.dp_shard_enabled)
        or parallel_dims.tp_enabled
        or parallel_dims.pp_enabled
        or parallel_dims.cp_enabled
    ):
        logger.warning("DDP, TP, PP, CP are not supported yet.")
        return
    if not parallel_dims.dp_shard_enabled:
        logger.warning("FSDP or HSDP is not enabled. Skipping memory estimation.")
        return

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    # init fake pg
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=int(os.environ["LOCAL_RANK"]), world_size=world_size, store=store
    )

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

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

    with (
        FakeTensorMode()
        if not job_config.memory_estimation.disable_fake_mode
        else contextlib.nullcontext()
    ):

        logger.info(
            f"Building {model_name} {job_config.model.flavor} with {model_config}"
        )
        with torch.device("meta"):
            model = model_cls.from_model_args(model_config)

        # a no-op hander if float8 is not enabled
        float8_handler = Float8Handler(job_config, parallel_dims)
        # swap to Float8Linear based on float8 configs
        float8_handler.convert_to_float8_training(model)

        # apply PT-D DP/TP parallelisms and activation checkpointing
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)

        model.to_empty(device="cuda")
        if not active_fake_mode():
            model.init_weights()
        model.train()

        # build optimizer after applying parallelisms to the model
        optimizers = build_optimizers([model], job_config)
        lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

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
        fsdp_memtracker = FSDPMemTracker(mod=model, optm=optimizers.optimizers[0])
        fsdp_memtracker.track_inputs(batch)

        with fsdp_memtracker:
            for iter_idx in range(2):
                input_ids, labels = batch
                # train step
                with train_context():
                    pred = model(input_ids)
                    loss = loss_fn(pred, labels)
                    del pred
                    loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), job_config.training.max_norm, foreach=True
                )
                # optimizer step
                optimizers.step()
                lr_schedulers.step()
                # calculate float8 dynamic amax/scale for all-parameter for FSDP2
                # it issues a single all-reduce for all parameters at once for better performance
                float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)
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
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
