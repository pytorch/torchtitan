# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable entrypoints for matched four- and 256-GPU MTP1 runs."""

import os

from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from . import recipes


def _configure(config: Trainer.Config, *, num_gpus: int) -> Trainer.Config:
    mode = os.environ.get("GT_RUN_MODE", "benchmark")
    if mode not in {"benchmark", "verify"}:
        raise ValueError("GT_RUN_MODE must be benchmark or verify")
    config.training.steps = int(
        os.environ.get("GT_RUN_STEPS", "40" if num_gpus == 4 else "60")
    )
    if config.training.steps < 10:
        raise ValueError("run recipes require at least ten training steps")
    config.training.disable_cuda_graphs = False
    config.parallelism.fsdp_symm_mem_scope = None
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.model.local_compile_regions = []
    if "GT_RUN_TOKENIZER_PATH" in os.environ:
        config.hf_assets_path = os.environ["GT_RUN_TOKENIZER_PATH"]
    if mode == "benchmark":
        profile_step = min(41, config.training.steps - 5)
        config.profiler = Profiler.Config(
            enable_profiling=True,
            profile_freq=profile_step,
            profiler_warmup=0,
            profiler_active=1,
            profiler_repeat=1,
            enable_memory_snapshot=True,
            memory_snapshot_freq=profile_step,
        )
    else:
        config.profiler = Profiler.Config(enable_profiling=False)
    return config


def spmd_256gpu_eager() -> Trainer.Config:
    return _configure(
        recipes.deepseek_v3_671b_dist_moe_mxfp8_spmd_mtp1_256gpu_numerics(),
        num_gpus=256,
    )


def spmd_256gpu_graph_trainer() -> Trainer.Config:
    return _configure(
        recipes.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_spmd_mtp1_256gpu_numerics(),
        num_gpus=256,
    )


def pp_256gpu_eager() -> Trainer.Config:
    return _configure(
        recipes.deepseek_v3_671b_dist_moe_mxfp8_pp_mtp1_256gpu_numerics(), num_gpus=256
    )


def pp_256gpu_graph_trainer() -> Trainer.Config:
    return _configure(
        recipes.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_pp_mtp1_256gpu_numerics(),
        num_gpus=256,
    )


def spmd_4gpu_graph_trainer() -> Trainer.Config:
    return _configure(
        recipes.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mtp1_spmd_verification_4gpu(),
        num_gpus=4,
    )


def spmd_4gpu_eager() -> Trainer.Config:
    from .paired_numerics_configs import _eager_spmd_base

    config = _eager_spmd_base()
    config.parallelism.expert_parallel_degree = 4
    recipes._configure_dist_moe_runtime(
        config, inplace_wgrad_accum=True, activation_slot_capacity_factor=0.5
    )
    return _configure(config, num_gpus=4)


def pp_4gpu_eager() -> Trainer.Config:
    return _configure(
        recipes.deepseek_v3_16b_dist_moe_mxfp8_pp_mtp1_verification_4gpu(), num_gpus=4
    )


def pp_4gpu_graph_trainer() -> Trainer.Config:
    return _configure(
        recipes.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_pp_mtp1_verification_4gpu(),
        num_gpus=4,
    )
