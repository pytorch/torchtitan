# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Llama 3 model recipes."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import MXFP8LinearTransform
from torchtitan.distributed.activation_remat import (
    BlockRematOverride,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config, Llama3Model
from torchtitan.observability.metrics import MetricsProcessor, TensorBoardLogger
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity


def llama3_mxfp8_linear_transform() -> MXFP8LinearTransform:
    """Build the MXFP8 policy shared by eager and GraphTrainer recipes."""
    return MXFP8LinearTransform(
        linears_saving_inputs_for_backward_in_mxfp8=[
            "attention.qkv_linear.wqkv",
            "feed_forward.w2",
        ],
    )


def llama3_8b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config("8B", seq_len=seq_len, parallelism=parallelism)
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        metrics=MetricsProcessor.Config(loggers=[TensorBoardLogger.Config()]),
        model=model_config,
        parallelism=parallelism,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        validator=None,
    )


def llama3_8b_activation_offload(seq_len: int | None = None) -> Trainer.Config:
    """Llama 3 8B with selective remat, offloading each block's input and what
    the attention regions retain to CPU.

    Measured on 4 x GB300 with FSDP, 8K tokens per rank, a whole-step CUDA
    graph and NUMA binding: 8.4 GiB less peak memory at unchanged throughput.
    Without CUDA graphs or NUMA binding the copies compete with the CPU that
    launches kernels (2 x GB300: 8.5 GiB less, 5% lower throughput); see
    torchtitan/distributed/ACTIVATION_OFFLOAD.md.
    The last block stays on GPU: only the loss runs between its forward and its
    backward, so its copies could not overlap compute.
    """
    config = llama3_8b(seq_len)
    assert isinstance(config.model, Llama3Model.Config)
    last_block = f"layers.{len(config.model.layers) - 1}"
    config.activation_remat = DefaultActivationRematConfig(
        offload_block_input=True,
        offload_regions=["attention.*"],
        block_overrides=[
            BlockRematOverride(
                blocks=[last_block], offload_block_input=False, offload_regions=[]
            ),
        ],
    )
    return config


def llama3_70b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig(tensor_parallel_degree=8)
    model_config = build_model_config("70B", seq_len=seq_len, parallelism=parallelism)
    num_tokens_per_microbatch_per_dp_rank = 8 * model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiler=Profiler.Config(enable_profiling=True, profile_freq=100),
        metrics=MetricsProcessor.Config(loggers=[TensorBoardLogger.Config()]),
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1.5e-4)]
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
        validator=None,
    )
