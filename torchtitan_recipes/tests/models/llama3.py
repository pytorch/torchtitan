# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama 3 model configurations used by tests."""

from typing import cast

from renderers import Message

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    AsyncTensorParallelTransform,
    NVFP4LinearConverter,
)
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import ChatProcessor, DATASETS
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)

from torchtitan.models.llama3 import build_model_config
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.quantization.nvfp4 import nvfp4_bf16_tail_fqns
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.llama3 import llama3_mxfp8_linear_converter_config


def llama3_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_config = build_model_config("debugmodel", seq_len=seq_len)
    packed = ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=packed,
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B"),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
        validator=None,
    )


def llama3_debugmodel_varlen_attn(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = llama3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.max_num_documents = 64
    return config


def llama3_debugmodel_dist_gemm(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    """Async-TP: attention and FFN collectives are folded into their GEMMs.

    Needs tensor_parallel_degree > 1 and CUDA. With TP off the fused modules
    fall back to the stock projections, so this stays runnable on one rank.

    The fused modules take and return plain local tensors.
    """
    config = llama3_debugmodel(seq_len=seq_len)
    config = apply_transforms(
        config,
        [
            AsyncTensorParallelTransform(
                enable_sequence_parallel=config.parallelism.enable_sequence_parallel
            )
        ],
    )
    return config


def llama3_debugmodel_mxfp8(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = llama3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel_mxfp8",
        seq_len=seq_len,
        converters=[
            llama3_mxfp8_linear_converter_config(),
        ],
    )
    return config


def llama3_debugmodel_nvfp4(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = llama3_debugmodel(seq_len=seq_len)
    # fqns=["layers"] converts every in-layer Linear (attention + feed_forward)
    # while leaving the lm_head stock: NVFP4 requires each GEMM dim divisible by
    # 128, which the vocab projection does not satisfy.
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        converters=[
            NVFP4LinearConverter.Config(
                fqns=["layers"],
            ),
        ],
    )
    return config


def llama3_debugmodel_first_85_pct_layers_nvfp4(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = llama3_debugmodel(seq_len=seq_len)
    # Mixed precision: convert the leading decoder layers to NVFP4 and keep the
    # last _NVFP4_BF16_TAIL_FRACTION of layers (plus the lm_head) in bf16.
    n_layers = len(cast(Llama3Model.Config, config.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
            ),
        ],
    )
    return config


def llama3_debugmodel_ce_loss(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    """Debug model with standard (non-chunked) CrossEntropyLoss."""
    config = llama3_debugmodel(seq_len=seq_len)
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model),
    )
    return config


def llama3_8b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config("8B", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
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
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            )
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
        validator=None,
    )


def llama3_8b_first_85_pct_layers_nvfp4(seq_len: int | None = None) -> Trainer.Config:
    config = llama3_8b(seq_len=seq_len)
    # Mixed precision: convert the leading decoder layers to NVFP4 and keep the
    # last _NVFP4_BF16_TAIL_FRACTION of layers (plus the lm_head) in bf16.
    n_layers = len(cast(Llama3Model.Config, config.model).layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    config.model = build_model_config(
        "8B",
        seq_len=seq_len,
        converters=[
            NVFP4LinearConverter.Config(
                fqns=fqns,
            ),
        ],
    )
    return config


def llama3_8b_mxfp8(seq_len: int | None = None) -> Trainer.Config:
    config = llama3_8b(seq_len=seq_len)
    config.model = build_model_config(
        "8B",
        seq_len=seq_len,
        converters=[
            llama3_mxfp8_linear_converter_config(),
        ],
    )
    return config


def llama3_70b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config("70B", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
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
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1.5e-4)]
            )
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=8,
        ),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
        validator=None,
    )


def sft_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    """SFT debug config with Llama3 debugmodel and local test data."""

    def process_sample(sample) -> list[Message]:
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, attn_backend="flex"
    )

    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
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
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )
