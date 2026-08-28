# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.quantization import (
    Float8LinearConverter,
    MXFP8LinearConverter,
    NVFP4LinearConverter,
)
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import ChatProcessor, DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry
from .model import Llama3Model


def llama3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    packed = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
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
        dataloader=GrainDataLoader.Config(
            dataset=packed,
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=5,
            steps=10,
            dataloader=GrainDataLoader.Config(
                dataset=packed,
                shuffle=False,
            ),
        ),
    )


def llama3_debugmodel_varlen_attn() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_dist_gemm() -> Trainer.Config:
    """Async-TP: the attention TP collectives are folded into their GEMMs.

    Needs tensor_parallel_degree > 1 and CUDA. With TP off the fused modules
    fall back to the stock projections, so this stays runnable on one rank.

    ``spmd_backend`` is pinned to spmd_types: the fused modules take and return
    plain local tensors, which is that backend's contract. The DTensor backends
    are being deprecated and are not supported here.
    """
    config = llama3_debugmodel()
    config.model_spec = model_registry("debugmodel", tp_gemm_backend="dist_gemm")
    config.parallelism.spmd_backend = "spmd_types"
    return config


def llama3_debugmodel_float8(
    model_compile_enabled: bool | None = None,
) -> Trainer.Config:
    config = llama3_debugmodel()
    if model_compile_enabled is None:
        model_compile_enabled = (
            config.compile.enable and "model" in config.compile.components
        )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            Float8LinearConverter.Config(model_compile_enabled=model_compile_enabled),
        ],
    )
    return config


def llama3_debugmodel_nvfp4() -> Trainer.Config:
    config = llama3_debugmodel()
    config.parallelism.spmd_backend = "spmd_types"
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # fqns=["layers"] converts every in-layer Linear (attention + feed_forward)
    # while leaving the lm_head stock: NVFP4 requires each GEMM dim divisible by
    # 128, which the vocab projection does not satisfy.
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=["layers"],
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def llama3_debugmodel_first_85_pct_layers_nvfp4() -> Trainer.Config:
    config = llama3_debugmodel()
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Mixed precision: convert the leading decoder layers to NVFP4 and keep the
    # last _NVFP4_BF16_TAIL_FRACTION of layers (plus the lm_head) in bf16.
    n_layers = len(cast(Llama3Model.Config, config.model_spec.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def llama3_debugmodel_float8_emulate_lora() -> Trainer.Config:
    from torchtitan.components.lora import LoRAConverter

    config = llama3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            Float8LinearConverter.Config(
                emulate=True,
                model_compile_enabled=False,
            ),
            LoRAConverter.Config(rank=8, alpha=16.0, target_modules=["wqkv", "wo"]),
        ],
    )
    return config


def llama3_debugmodel_ce_loss() -> Trainer.Config:
    """Debug model with standard (non-chunked) CrossEntropyLoss."""
    config = llama3_debugmodel()
    assert config.model_spec is not None
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec),
    )
    return config


def llama3_8b() -> Trainer.Config:
    model_spec = model_registry("8B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 8192,
            max_context_length=8192,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_8b_first_85_pct_layers_nvfp4() -> Trainer.Config:
    config = llama3_8b()
    config.parallelism.spmd_backend = "spmd_types"
    assert config.model_spec is not None
    # Enable compile so NVFP4's dynamic quantization runs at competitive perf.
    config.compile = CompileConfig(enable=True, components=["model"])
    # Mixed precision: convert the leading decoder layers to NVFP4 and keep the
    # last _NVFP4_BF16_TAIL_FRACTION of layers (plus the lm_head) in bf16.
    n_layers = len(cast(Llama3Model.Config, config.model_spec.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    config.model_spec = model_registry(
        "8B",
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
                model_compile_enabled=True,
            ),
        ],
    )
    return config


def llama3_8b_mxfp8() -> Trainer.Config:
    config = llama3_8b()
    # Swap dense Linear layers for MXFP8Linear. compile is enabled so the
    # converter's compile requirement is satisfied. This is the regular-Trainer
    # (torch.compile) baseline counterpart to graph_trainer_llama3_8b_mxfp8.
    config.compile = CompileConfig(enable=True, components=["model"])
    config.model_spec = model_registry(
        "8B",
        converters=[
            MXFP8LinearConverter.Config(model_compile_enabled=True),
        ],
    )
    return config


def llama3_70b() -> Trainer.Config:
    model_spec = model_registry("70B")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-70B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=1.5e-4),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 8192,
            max_context_length=8192,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def llama3_405b() -> Trainer.Config:
    compile_config = CompileConfig(
        enable=True,
        enable_async_tensor_parallel=True,
    )
    model_spec = model_registry(
        "405B",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head"],
                model_compile_enabled=(
                    compile_config.enable and "model" in compile_config.components
                ),
            ),
        ],
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Llama-3.1-405B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * 8192,
            max_context_length=8192,
            steps=3000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        compile=compile_config,
        validator=Validator.Config(
            freq=500,
            steps=1200,
        ),
    )


def sft_debugmodel() -> Trainer.Config:
    """SFT debug config with Llama3 debugmodel and local test data."""

    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    model_spec = model_registry("debugmodel", attn_backend="flex")

    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
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
        dataloader=GrainDataLoader.Config(
            dataset=FirstFitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={
                            "data_files": "tests/assets/sft_test/data.json",
                        },
                    ),
                    processor=ChatProcessor.Config(messages_fn=process_sample),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
