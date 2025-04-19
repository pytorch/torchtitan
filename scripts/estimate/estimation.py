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

from torchtitan.components.ft import init_ft_manager
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import init_logger, logger


def estimate_memory(job_config: JobConfig):
    init_logger()
    logger.info("Estimating memory usage...")
    gc.disable()
    gc.collect(1)

    # Get the world size
    world_size = int(os.environ["WORLD_SIZE"])

    if job_config.training.compile or job_config.parallelism.enable_compiled_autograd:
        logger.info("Compile mode is not supported yet. Switching to eager mode.")
        job_config.training.compile = False
        job_config.parallelism.enable_compiled_autograd = False

    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not parallelism_config.disable_loss_parallel,
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

    train_spec = get_train_spec(job_config.model.name)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")

    # build tokenizer
    tokenizer = train_spec.build_tokenizer_fn(job_config)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.parallelism.enable_compiled_autograd,
    )

    # build model (using meta init)
    model_cls = train_spec.cls
    model_args = train_spec.config[job_config.model.flavor]
    model_args.update_from_config(job_config, tokenizer)

    with (
        FakeTensorMode()
        if not job_config.memory_estimation.disable_fake_mode
        else contextlib.nullcontext()
    ):
        logger.info(
            f"Building {train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = model_cls.from_model_args(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # apply PT-D DP/TP parallelisms and activation checkpointing
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)

        model.to_empty(device="cuda")
        if not active_fake_mode():
            model.init_weights()
        model.train()

        # build optimizer after applying parallelisms to the model
        ft_manager = init_ft_manager(job_config)
        optimizers = build_optimizers([model], job_config, ft_manager)
        lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(model)
        )

        logger.info(f"Vocab size: {model_args.vocab_size}")
        # Create a dummy batch instead of loading from a dataset
        batch = (
            torch.randint(
                0,
                model_args.vocab_size,
                (job_config.training.batch_size, model_args.max_seq_len),
                device="cuda",
            ),
            torch.randint(
                0,
                model_args.vocab_size,
                (job_config.training.batch_size, model_args.max_seq_len),
                device="cuda",
            ),
        )
        fsdp_memtracker = FSDPMemTracker(mod=model, optm=optimizers.optimizers[0])
        fsdp_memtracker.track_inputs(batch)

        loss_fn = train_spec.build_loss_fn(job_config)
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
            print(f"Tracker Accuracy: {tracker_peak / peak_active}")
        gc.enable()


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    try:
        estimate_memory(config)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
