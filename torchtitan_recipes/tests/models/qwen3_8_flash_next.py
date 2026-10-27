# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next feature fixtures, combined parallelism, and deterministic numerics."""

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel
from torchtitan_recipes.tests.models.qwen3_8 import qwen38_debugmodel


def qwen38_flash_next_debugmodel(
    *,
    seq_len: int = 64,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    parallelism = parallelism or ParallelismConfig()
    config = llama3_debugmodel(seq_len=seq_len, parallelism=parallelism)
    config.model = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=parallelism
    )
    config.model.vision_encoder = None
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=config.model.vocab_size),
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * seq_len
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    config.model.local_compile_regions = []
    config.activation_remat = None
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def qwen38_flash_next_debugmodel_all_parallelism() -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
        pipeline_parallel_degree=2,
        num_pp_microbatches=2,
        pipeline_parallel_schedule="1F1B",
        enable_sequence_parallel=True,
        # Place PLE on the second stage to exercise the original-ID payload.
        pipeline_parallel_first_stage_less_layers=7,
    )
    return qwen38_flash_next_debugmodel(parallelism=parallelism)


def qwen38_flash_next_debugmodel_tp2() -> Trainer.Config:
    config = qwen38_flash_next_debugmodel(
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
            enable_sequence_parallel=True,
        )
    )
    config.debug.spmd_typechecking = True
    return config


def qwen38_flash_next_debugmodel_ac() -> Trainer.Config:
    config = qwen38_flash_next_debugmodel()
    config.activation_remat = FullActivationRematConfig()
    return config


def qwen38_flash_next_debugmodel_pp2() -> Trainer.Config:
    return qwen38_flash_next_debugmodel(
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=2,
            pipeline_parallel_first_stage_less_layers=7,
        )
    )


def qwen38_flash_next_debugmodel_mm() -> Trainer.Config:
    config = qwen38_debugmodel(seq_len=1024)
    config.model = build_model_config(
        "debugmodel", seq_len=1024, parallelism=config.parallelism
    )
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=config.model.vocab_size),
    )
    config.activation_remat = None
    config.training.disable_cuda_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    return config
