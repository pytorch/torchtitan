# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Muse Glimmer model recipes."""

from collections.abc import Mapping

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    HeadTailCPLoadBalancer,
)
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    KVAllGatherCPSlidingWindowFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
    UlyssesCPSlidingWindowFlexInnerAttention,
    UlyssesCPVarlenInnerAttention,
)
from torchtitan.models.muse_glimmer import build_model_config
from torchtitan.trainer import Trainer


def muse_glimmer_30b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config("30B", seq_len=seq_len, attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Muse-Glimmer-30B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        parallelism=ParallelismConfig(),
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
    )


def muse_glimmer_30b_bf16_optimizer_states() -> Trainer.Config:
    """Use BF16 optimizer states with FP32 parameters and reduction."""
    config = muse_glimmer_30b()
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "float32"
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", moment_dtype="bfloat16")]
    )
    return config


def _muse_glimmer_30b_cp(
    *,
    inner_attention_map: Mapping[type[InnerAttention], type[InnerAttention]],
    cp_degree: int,
    load_balancer: ContextParallelLoadBalancer.Config | None,
) -> Trainer.Config:
    config = muse_glimmer_30b()
    config.parallelism.context_parallel_degree = cp_degree
    config.parallelism.context_parallel_load_balancer = load_balancer
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention_map=inner_attention_map)],
    )


def muse_glimmer_30b_allgather_cp8() -> Trainer.Config:
    """Muse Glimmer 30B with all-gather CP degree 8."""
    return _muse_glimmer_30b_cp(
        inner_attention_map={
            FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
            SlidingWindowFlexInnerAttention: (
                KVAllGatherCPSlidingWindowFlexInnerAttention
            ),
        },
        cp_degree=8,
        load_balancer=HeadTailCPLoadBalancer.Config(),
    )


def muse_glimmer_30b_ulysses_cp2() -> Trainer.Config:
    """Muse Glimmer 30B with Ulysses CP degree 2.

    The model has two KV heads, which limits Ulysses CP to degree 2.
    """
    return _muse_glimmer_30b_cp(
        inner_attention_map={
            FlexInnerAttention: UlyssesCPFlexInnerAttention,
            SlidingWindowFlexInnerAttention: (UlyssesCPSlidingWindowFlexInnerAttention),
        },
        cp_degree=2,
        # Ulysses does not support token reordering.
        load_balancer=None,
    )


def muse_glimmer_30b_ulysses_varlen_cp2() -> Trainer.Config:
    """Muse Glimmer 30B with varlen Ulysses CP degree 2.

    The model has two KV heads, which limits Ulysses CP to degree 2.
    """
    config = muse_glimmer_30b()
    config.model = build_model_config("30B", attn_backend="varlen")
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.max_num_documents = 64
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    VarlenInnerAttention: UlyssesCPVarlenInnerAttention
                }
            )
        ],
    )
