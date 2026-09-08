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
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry
from .mtp import MTPLoss


def deepseek_v4_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
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
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_mtp_debugmodel(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len, n_mtp_layers=1)
    return Trainer.Config(
        loss=MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
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
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_flash(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("deepseek_v4_flash", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_pro(seq_len: int | None = None) -> Trainer.Config:
    model_spec = model_registry("deepseek_v4_pro", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


_GB300_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_stages": 1,
    "num_warps": 4,
}


def deepseek_v4_pro_64xgb300(seq_len: int | None = None) -> Trainer.Config:
    """`deepseek_v4_pro` made to fit on 64x GB300 (16 nodes x 4 GPUs).

    Derived from the stock ``deepseek_v4_pro`` so the delta stays auditable;
    the model itself is untouched. Three changes, each forced by a measured
    constraint rather than by taste:

    1. ``training.dtype = "bfloat16"``. This is the change that makes the model
       fit at all, and no amount of parallelism substitutes for it. Sharding
       divides the model state, it does not shrink it: `pro` is 1.573 T
       parameters, and torchtitan's default ``training.dtype="float32"`` costs
       16 B/param (fp32 shard + fp32 grad + two fp32 AdamW moments) = 25.17 TB.
       Spread perfectly over all 64 GPUs that is 393 GB/GPU against 298 GB of
       HBM -- 132 % of the machine before a single activation. Full bf16 is
       8 B/param = 12.58 TB = 196.6 GB/GPU, leaving ~101 GB/GPU of headroom.
       ``mixed_precision_param`` is already bfloat16, so this only drops the
       extra fp32 master copy, exactly as documented on the field.

    2. ``expert_parallel_degree = 64``. Stock is 1, which leaves all 384
       experts of a layer to FSDP: one all-gather would materialize
       384 x 3 x 7168 x 3072 x 2 B = 50.7 GB for a single layer, more with
       prefetch, which does not survive a 101 GB budget. At EP=64 each rank
       owns 384/64 = 6 whole experts, so the expert stack is never all-gathered
       and only token dispatch crosses the wire. EP must divide
       ``dp_shard * cp * tp`` (= 64 here), and 64 | 384, so the mesh is legal.

    3. ``activation_checkpoint = FullAC``. Stock is ``None``. 61 blocks of
       dim 7168 with 128 heads x 512 head_dim cannot keep their forward
       activations in what is left after the weights. FullAC (rather than the
       SelectiveAC that upstream's ``deepseek_v3_671b`` uses) because the
       headroom here is far tighter than that recipe assumes.

    ``disable_cuda_graphs`` follows upstream's own ``deepseek_v3_671b`` recipe.
    Everything else -- optimizer, LR schedule, loss, dataloader, compile
    settings, batch shape -- is inherited from stock ``deepseek_v4_pro``.

    Note: bf16 AdamW moments are fine for a throughput baseline but are not a
    convergence-grade choice for a real 1.573 T pretrain.
    """
    config = deepseek_v4_pro(seq_len=seq_len)
    config.training.dtype = "bfloat16"
    config.training.disable_cuda_graphs = True
    config.parallelism = ParallelismConfig(expert_parallel_degree=64)
    config.activation_checkpoint = FullAC.Config()

    # Pin the FlexAttention Triton tile. `pro` has head_dim=512, and on GB300
    # (232448 B of shared memory per block) Inductor finds no valid Triton
    # config for the default autotune sweep -- the first forward dies with
    # "No valid triton configs. OutOfMemoryError: out of resource:
    # triton_flex_attention Required: 294912 Hardware limit: 232448".
    #
    # Measured on one GB300 at D=512 with a causal block mask: every larger
    # tile fails (64x64, 64x32 and 128x32 all raise NoValidChoicesError or a
    # launch failure, with or without num_stages=1), and 32x32 fails unless
    # num_stages and num_warps are pinned too. This is the only tile that runs.
    # torchtitan's FlexAttention docstring names this the intended workflow:
    # autotune once, then set kernel_options explicitly.
    #
    # This is a correctness requirement, not a tuning choice -- without it the
    # model cannot execute a forward pass on this hardware. It is also small
    # enough to cost attention throughput, so revisit if a future PyTorch
    # lowers the shared-memory demand at head_dim=512.
    for layer in config.model_spec.model.layers:
        attention = getattr(layer, "attention", None)
        inner = getattr(attention, "inner_attention", None)
        if isinstance(inner, FlexAttention.Config):
            inner.kernel_options = dict(_GB300_FLEX_KERNEL_OPTIONS)
    return config
