# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public 256-GPU SPMD, PP2 and PP4 performance recipes."""

import copy
import os

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
)
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    SPMDGradientAccumulationConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer
from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b_dist_moe_mxfp8

from . import recipes
from .configs import _configure


def _build(*, pp_degree: int, graph_trainer: bool, balanced: bool) -> Trainer.Config:
    recipes._prepare_outer_cuda_graph_capture()
    config = deepseek_v3_671b_dist_moe_mxfp8(seq_len=4096)
    config.parallelism.enable_sequence_parallel = False
    recipes._configure_mtp1(config)
    recipes._configure_dist_moe_runtime(config, inplace_wgrad_accum=not graph_trainer)
    runtime = config.dist_moe_runtime
    assert isinstance(runtime, DistMoeRuntime.Config)
    runtime.scratch_capacity_factor = 1.0 if balanced else 8.0
    runtime.pp_activation_slot_policy = "stage_microbatch"
    runtime.vmm_capacity_factor = None
    if balanced:
        recipes._configure_round_robin_routing(config)

    parallelism = config.parallelism
    parallelism.data_parallel_replicate_degree = 1
    parallelism.data_parallel_shard_degree = 256 // pp_degree
    parallelism.expert_parallel_degree = 64
    parallelism.tensor_parallel_degree = 1
    parallelism.context_parallel_degree = 1
    parallelism.pipeline_parallel_degree = pp_degree
    parallelism.num_pp_microbatches = 15_360 // parallelism.data_parallel_shard_degree
    parallelism.pipeline_parallel_layers_per_stage = None
    parallelism.pipeline_parallel_module_fqns_per_model_part = (
        recipes._pp_mtp1_stage_modules() if pp_degree > 1 else None
    )
    parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    parallelism.pipeline_parallel_max_outstanding_sends = 16
    parallelism.pp_max_unsharded_active_stages = 16 // pp_degree
    parallelism.pp_num_unshard_lookahead_factor = "auto"
    parallelism.fsdp_reshard_after_forward = "never"
    parallelism.fsdp_defer_gradient_reduction = not graph_trainer
    parallelism.fsdp_symm_mem_scope = None
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 15_360 * 4096
    config.training.disable_cuda_graphs = False
    config.activation_remat = None

    dataset = copy.deepcopy(DATASETS["c4"])
    source = dataset.source
    assert isinstance(source, HuggingFaceStreamingSource.Config)
    source.path = os.environ.get("GT_RUN_C4_SOURCE", source.path)
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=dataset),
        max_num_documents=1,
        num_mtp_layers=1,
        shuffle=False,
        repeat=True,
        num_prefetch_microbatches=64,
    )
    if graph_trainer:
        config = to_graph_trainer_config(config, GraphTrainerDeepSeekV3Model.Config)
        config.compile = GraphTrainerCompileConfig(
            spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
                fuse_wgrad_accumulation="enabled",
            ),
            memory_policy="none",
            inductor_compilation="regional",
            numerics_changing_optim=False,
        )
    config = _configure(config, num_gpus=256)
    mode = os.environ.get("GT_RUN_MODE", "benchmark")
    config.debug.seed = 42
    config.debug.deterministic = mode == "verify"
    config.optim.lr_scheduler.total_steps = config.training.steps
    config.optim.lr_scheduler.warmup_steps = 4

    profile = os.environ.get("GT_RUN_PROFILE", "0")
    memory_profile = os.environ.get("GT_RUN_MEMORY_PROFILE", "0")
    if profile not in {"0", "1"} or memory_profile not in {"0", "1"}:
        raise ValueError("GT_RUN_PROFILE and GT_RUN_MEMORY_PROFILE must be 0 or 1")
    selected_rank = int(os.environ.get("RANK", "0")) in {0, 128}
    enabled = mode == "benchmark" and selected_rank
    profile_step = min(39, config.training.steps - 1)
    config.profiler = Profiler.Config(
        enable_profiling=enabled and profile == "1",
        profile_freq=profile_step,
        profiler_warmup=3,
        profiler_active=2,
        profiler_repeat=1,
        enable_memory_snapshot=enabled and memory_profile == "1",
        memory_snapshot_freq=profile_step,
    )
    return config


def spmd_256gpu_graph_trainer_natural() -> Trainer.Config:
    return _build(pp_degree=1, graph_trainer=True, balanced=False)


def spmd_256gpu_eager_natural() -> Trainer.Config:
    return _build(pp_degree=1, graph_trainer=False, balanced=False)


def spmd_256gpu_graph_trainer_force_balanced() -> Trainer.Config:
    return _build(pp_degree=1, graph_trainer=True, balanced=True)


def spmd_256gpu_eager_force_balanced() -> Trainer.Config:
    return _build(pp_degree=1, graph_trainer=False, balanced=True)


def pp2_256gpu_graph_trainer_natural() -> Trainer.Config:
    return _build(pp_degree=2, graph_trainer=True, balanced=False)


def pp2_256gpu_eager_natural() -> Trainer.Config:
    return _build(pp_degree=2, graph_trainer=False, balanced=False)


def pp2_256gpu_graph_trainer_force_balanced() -> Trainer.Config:
    return _build(pp_degree=2, graph_trainer=True, balanced=True)


def pp2_256gpu_eager_force_balanced() -> Trainer.Config:
    return _build(pp_degree=2, graph_trainer=False, balanced=True)


def pp4_256gpu_graph_trainer_natural() -> Trainer.Config:
    return _build(pp_degree=4, graph_trainer=True, balanced=False)


def pp4_256gpu_eager_natural() -> Trainer.Config:
    return _build(pp_degree=4, graph_trainer=False, balanced=False)


def pp4_256gpu_graph_trainer_force_balanced() -> Trainer.Config:
    return _build(pp_degree=4, graph_trainer=True, balanced=True)


def pp4_256gpu_eager_force_balanced() -> Trainer.Config:
    return _build(pp_degree=4, graph_trainer=False, balanced=True)
