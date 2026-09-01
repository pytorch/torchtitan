# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.quantization import (
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def _gpt_oss_debugmodel(attn_backend: str = "varlen") -> Trainer.Config:
    model_spec = model_registry("debugmodel", attn_backend=attn_backend)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 2048,
            max_context_length=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
        validator=Validator.Config(
            freq=5,
            steps=10,
        ),
    )


def gpt_oss_debugmodel() -> Trainer.Config:
    return _gpt_oss_debugmodel()


def gpt_oss_debugmodel_flex() -> Trainer.Config:
    return _gpt_oss_debugmodel(attn_backend="flex")


def gpt_oss_mxfp8_linear_converter_config(
    *, model_compile_enabled: bool
) -> MXFP8LinearConverter.Config:
    """Build the dense MXFP8 policy for the debug model.

    ``fqns`` is an include-list, so the router gate and lm_head stay in BF16.

    The fused QKV projection has a single-consumer input that nothing else
    saves for backward, so its columnwise MXFP8 representation replaces BF16
    storage. The output projection keeps the conservative BF16 format because
    attention already retains its input.
    """
    return MXFP8LinearConverter.Config(
        model_compile_enabled=model_compile_enabled,
        fqns=["attention"],
        linears_saving_inputs_for_backward_in_mxfp8=["attention.qkv_linear.wqkv"],
    )


def gpt_oss_debugmodel_mxfp8() -> Trainer.Config:
    """Debug model with MXFP8 expert grouped GEMMs and dense linears.

    The experts carry per-expert biases and a SwiGLU clamp, which the grouped
    converter preserves by subclassing ``GptOssGroupedExperts``: only the
    grouped GEMM seam is replaced, and the biases stay BF16 parameters.
    """
    config = _gpt_oss_debugmodel()
    # The grouped converter swaps in a padding-capable token dispatcher, and
    # TorchAOTokenDispatcher needs a CPU sync, which CUDA graphs reject. Run
    # eager so this config works at any expert-parallel degree; the
    # CUDA-graph path for MXFP8 experts needs HybridEP with a
    # non_blocking_capacity_factor.
    config.training.disable_cuda_graphs = True
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            gpt_oss_mxfp8_linear_converter_config(
                model_compile_enabled=model_compile_enabled,
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def gpt_oss_20b() -> Trainer.Config:
    model_spec = model_registry("20b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-20b",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 8192,
            max_context_length=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def gpt_oss_120b() -> Trainer.Config:
    model_spec = model_registry("120b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/gpt-oss-120b",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 8192,
            max_context_length=8192,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )
