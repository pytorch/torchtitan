# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next feature fixtures, combined parallelism, and deterministic numerics."""

from dataclasses import replace

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def qwen38_flash_next_debugmodel(
    *,
    seq_len: int = 64,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    parallelism = parallelism or ParallelismConfig()
    config = llama3_debugmodel(seq_len=seq_len, parallelism=parallelism)
    config.optim.optimizer.optimizers = [
        replace(
            config.optim.optimizer.optimizers[0],
            pattern=r"\.ngram_embedding\.weight$",
        ),
        config.optim.optimizer.optimizers[0],
    ]
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        parallelism=parallelism,
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=config.model.vocab_size),
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * seq_len
    config.training.steps = 10
    config.training.disable_accelerator_graphs = True
    config.model.local_compile = LocalCompileConfig(regions=[])
    config.activation_remat = None
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.loggers = [TensorBoardLogger.Config()]
    return config


def qwen38_flash_next_debugmodel_ep2() -> Trainer.Config:
    config = qwen38_flash_next_debugmodel(
        parallelism=ParallelismConfig(expert_parallel_degree=2)
    )
    config.debug.spmd_typechecking = True
    return config


def qwen38_flash_next_debugmodel_all_parallelism() -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
        pipeline_parallel_degree=2,
        num_pp_microbatches=4,
        pipeline_parallel_schedule="Interleaved1F1B",
        enable_sequence_parallel=True,
        # PLE belongs to the second local part of PP rank 0.
        pipeline_parallel_module_fqns_per_model_part=[
            ["tok_embeddings", "vision_encoder"],
            ["layers.0"],
            ["layers.1", "layers.2", "layers.3"],
            [
                "layers.4",
                "layers.5",
                "layers.6",
                "layers.7",
                "norm",
                "lm_head",
                "hyper_connection_mixer",
            ],
        ],
    )
    config = qwen38_flash_next_debugmodel(parallelism=parallelism)
    config.activation_remat = FullActivationRematConfig()
    # The shared engine rejects SPMD typechecking under PP. The TP/EP recipe
    # validates types independently; this recipe checks real pipeline traffic.
    config.debug.spmd_typechecking = False
    return config


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


def qwen38_flash_next_debugmodel_pp2() -> Trainer.Config:
    config = qwen38_flash_next_debugmodel(
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=2,
            pipeline_parallel_first_stage_less_layers=7,
        )
    )
    config.debug.spmd_typechecking = False
    return config


def qwen38_flash_next_debugmodel_mm() -> Trainer.Config:
    from torchtitan_recipes.tests.models.qwen3_8 import qwen38_debugmodel

    config = qwen38_debugmodel(seq_len=1024)
    config.optim.optimizer.optimizers = [
        replace(
            config.optim.optimizer.optimizers[0],
            pattern=r"\.ngram_embedding\.weight$",
        ),
        config.optim.optimizer.optimizers[0],
    ]
    config.model = build_model_config(
        "debugmodel_mm",
        seq_len=1024,
        parallelism=config.parallelism,
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(global_vocab_size=config.model.vocab_size),
    )
    config.activation_remat = None
    config.training.disable_accelerator_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.loggers = [TensorBoardLogger.Config()]
    return config
