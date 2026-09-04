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
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.trainer import Trainer

from . import model_registry


def deepseek_v3_mxfp8_linear_converter_config(
    *, model_compile_enabled: bool
) -> MXFP8LinearConverter.Config:
    """Build the dense MXFP8 policy shared by eager and GraphTrainer configs.

    The KV up projection and FFN down projections have single-consumer inputs
    that are not saved elsewhere for backward, so their columnwise MXFP8
    representations replace BF16 storage. Shared-input and attention output
    projections use the conservative BF16 save format. This selection is based
    on activation ownership, not the activation-checkpointing policy.
    Checkpointing changes when the selected representation is recreated and how
    long it remains live.
    """
    return MXFP8LinearConverter.Config(
        model_compile_enabled=model_compile_enabled,
        fqns=["attention", "shared_experts", "feed_forward"],
        linears_saving_inputs_for_backward_in_mxfp8=[
            "attention.wkv_b",
            "feed_forward.w2",
            "shared_experts.w2",
        ],
    )


def enable_fused_swiglu(config: Trainer.Config) -> None:
    # Activate the stock dense-FFN and MoE grouped-expert overrides. The separate
    # dist-GEMM FFN override is not needed by these configs.
    for override in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ):
        assert override not in config.override.imports
        config.override.imports.append(override)


def deepseek_v3_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
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
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_debugmodel_mtp(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model_spec = model_registry("debugmodel", seq_len=seq_len, num_mtp_layers=1)
    config.loss = MTPLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec),
    )
    return config


def deepseek_v3_debugmodel_mxfp8(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    # Quantize the MoE expert grouped GEMMs to MXFP8, plus the dense Linear
    # layers in attention, the shared experts, and the dense-layer feed-forward.
    # fqns is an include-list (substring match), so the MoE router gate
    # (moe.router.gate) and lm_head are left in bf16.
    # pad_multiple=128 is required by the CuTeDSL quantization kernel
    # on sm_100 (e.g. B200)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        seq_len=seq_len,
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(
                model_compile_enabled=model_compile_enabled,
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_hybridep(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model_spec = model_registry(
        "debugmodel",
        seq_len=seq_len,
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_debugmodel_minimal_async_ep(
    seq_len: int | None = None,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model_spec = model_registry(
        "debugmodel",
        seq_len=seq_len,
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_16b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("16B", seq_len=seq_len, attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_16b_hybridep(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_16b(seq_len=seq_len)
    config.model_spec = model_registry(
        "16B",
        seq_len=seq_len,
        attn_backend="flex",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    config.training.disable_cuda_graphs = False
    return config


def deepseek_v3_16b_minimal_async_ep(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_16b(seq_len=seq_len)
    config.model_spec = model_registry(
        "16B",
        seq_len=seq_len,
        attn_backend="flex",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    config.training.disable_cuda_graphs = False
    return config


def deepseek_v3_671b(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry(
        "671B",
        seq_len=seq_len,
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b_float8(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_671b(seq_len=seq_len)
    # Quantize the dense Linear layers and the MoE expert grouped GEMMs to
    # float8 (fp8). This requires torchao and is only supported on NVIDIA SM89+
    # or AMD MI300+; on other backends (e.g. Intel XPU) the converter raises at
    # build time, so use the plain deepseek_v3_671b config there.
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "671B",
        seq_len=seq_len,
        attn_backend="flex",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=model_compile_enabled,
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled
            ),
        ],
    )
    return config
