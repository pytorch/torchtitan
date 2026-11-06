# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Muse Glimmer model recipes."""

import os
from collections.abc import Mapping
from dataclasses import replace
from typing import cast, Literal

from renderers import CustomRendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    ContextParallelTransform,
    LoRATransform,
)
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    HeadTailCPLoadBalancer,
)
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
    AllGatherKVSlidingWindowFlexInnerAttention,
    UlyssesFlexInnerAttention,
    UlyssesSlidingWindowFlexInnerAttention,
    UlyssesSlidingWindowVarlenInnerAttention,
    UlyssesVarlenInnerAttention,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_glimmer import build_model_config
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import (
    c4_document_capacity,
    slake_dataset,
    SLAKE_DATASET_PATH,
    use_slake_sft_data,
)


def muse_glimmer_30b(
    seq_len: int | None = None, *, attn_backend: str = "varlen"
) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "30B", seq_len=seq_len, attn_backend=attn_backend, parallelism=parallelism
    )
    num_tokens_per_microbatch_per_dp_rank = model_config.max_context_length
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
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=3e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            steps=1000,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )


def muse_glimmer_30b_bf16_optimizer_states() -> Trainer.Config:
    """Use BF16 optimizer states with FP32 parameters and reduction."""
    config = muse_glimmer_30b()
    config.precision.dtype = "float32"
    config.precision.param_dtype = "bfloat16"
    config.precision.reduce_dtype = "float32"
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
    # These CP recipes use Flex-specific kernels and load balancers.
    config = muse_glimmer_30b(attn_backend="flex")
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
            FlexInnerAttention: AllGatherKVFlexInnerAttention,
            SlidingWindowFlexInnerAttention: (
                AllGatherKVSlidingWindowFlexInnerAttention
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
            FlexInnerAttention: UlyssesFlexInnerAttention,
            SlidingWindowFlexInnerAttention: (UlyssesSlidingWindowFlexInnerAttention),
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
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    config.model = build_model_config(
        "30B", attn_backend="varlen", parallelism=config.parallelism
    )
    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    VarlenInnerAttention: UlyssesVarlenInnerAttention,
                    SlidingWindowVarlenInnerAttention: (
                        UlyssesSlidingWindowVarlenInnerAttention
                    ),
                }
            )
        ],
    )


# Multimodal special tokens of the released Muse Glimmer 30B tokenizer.
MUSE_GLIMMER_30B_SPECIAL_TOKENS = {
    "image_token": "<|patch|>",
    "video_token": "<|video|>",
    "vision_start_token": "<|image_start|>",
    "vision_end_token": "<|image_end|>",
    "pad_token": "<|finetune_right_pad|>",
}


# SLAKE medical VQA SFT recipes.

# The shortest English SLAKE sample is 142 tokens (64 image tokens).
_SLAKE_MIN_SAMPLE_TOKENS = 142
# The English validation split packs into 196 microbatches of 4096 tokens at DP 4.
_SLAKE_VALIDATION_MICROBATCHES = 196
_SLAKE_LORA_DECODER_TARGETS = ["wqkv", "wo", "o_gate", "w13", "w2"]
# The vision tower is the Muse Spark NaViT. conv1_linear (the patch embedding)
# stays frozen.
_SLAKE_LORA_VISION_TARGETS = [
    "vision_encoder.block.attn.wq",
    "vision_encoder.block.attn.wk",
    "vision_encoder.block.attn.wv",
    "vision_encoder.block.attn.proj",
    "vision_encoder.block.mlp.fc1",
    "vision_encoder.block.mlp.fc2",
    "vision_adapter.c_fc",
    "vision_adapter.c_proj",
    "vision_projection",
]
_SlakeVariant = Literal["lora_decoder", "lora_vision_decoder"]


def _slake_lora_transform(variant: _SlakeVariant) -> LoRATransform:
    targets = list(_SLAKE_LORA_DECODER_TARGETS)
    if variant == "lora_vision_decoder":
        targets += _SLAKE_LORA_VISION_TARGETS
    return LoRATransform(rank=8, alpha=16.0, target_modules=targets)


def _muse_glimmer_30b_slake_sft(variant: _SlakeVariant) -> Trainer.Config:
    """Configure English SLAKE LoRA SFT from the released Muse Glimmer 30B weights.

    Stage the HF checkpoint at ``./assets/hf/Muse-Glimmer-30B``, or set
    ``MUSE_GLIMMER_HF_ASSETS_PATH``. ``MUSE_GLIMMER_SLAKE_PATH`` overrides the SLAKE
    root. See ``use_slake_sft_data`` for packing and validation. Learning rates and
    step counts are untuned starting points.
    """
    import torchvision.transforms.v2.functional as TVF

    from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
    from torchtitan.components.data.multimodal.utils.image import (
        resize_to_muse_glimmer_patch_grid,
    )

    seq_len = 4096  # The longest English SLAKE sample is 1476 tokens.
    config = muse_glimmer_30b(seq_len)
    # LoRA on the dense 30B model runs with FSDP on one node of 4 GB300 GPUs.
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=4,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    config.model = build_model_config(
        "30B_mm", seq_len=seq_len, parallelism=config.parallelism
    )
    vision_encoder = config.model.vision_encoder
    assert vision_encoder is not None
    config.hf_assets_path = os.environ.get(
        "MUSE_GLIMMER_HF_ASSETS_PATH", config.hf_assets_path
    )
    config.tokenizer = MultiModalTokenizer.Config(**MUSE_GLIMMER_30B_SPECIAL_TOKENS)
    # No remat: peak 116 GiB of 276.6 GiB per GPU on 4 GB300, about 28% faster
    # than full remat.
    config.activation_remat = None
    # A step is 32 x 4096 tokens, as in the Muse Spark SLAKE recipes. One pass over
    # the English train split is 26.8 steps (859 microbatches at DP 4), so 90 steps
    # is about 3.4 passes. In a 90-step run validation loss went from 2.73 at step
    # 1 to 0.169 at step 90; a 150-step run plateaued after step 70 and did not
    # improve.
    config.training.num_tokens_per_train_step = 32 * seq_len
    config.training.steps = 90
    config.training.disable_accelerator_graphs = True
    config.metrics.log_freq = 1
    config.metrics.loggers = [TensorBoardLogger.Config()]
    config.optim = Optim.Config(
        optimizer=OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=1e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
                )
            ]
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.1,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
    )
    config.checkpointer = cast(
        CheckpointManager.Config,
        TorchCheckpointingManager.Config(
            folder=f"muse_glimmer_slake_{variant}",
            interval=2000,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=False,
            keep_latest_k=2,
        ),
    )
    root = os.environ.get("MUSE_GLIMMER_SLAKE_PATH", SLAKE_DATASET_PATH)
    dataset = slake_dataset(
        root,
        renderer=from_renderers(
            CustomRendererConfig(
                import_path=(
                    "torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer"
                ),
                reasoning_strength="low",
                # The chat template always renders a date line; pin it so the
                # prompt does not change from day to day.
                current_date="2026-01-04",
            )
        ),
    )
    patch_size = vision_encoder.patch_size
    temporal_patch_size = vision_encoder.patch_temporal
    spatial_merge_size = vision_encoder.downsample_factor
    # The released image processor: closest-aspect patch grid, LANCZOS.
    processor = replace(
        dataset.processor,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        resize_fn=resize_to_muse_glimmer_patch_grid,
        image_interpolation_mode=TVF.InterpolationMode.LANCZOS,
    )
    loader = GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )
    use_slake_sft_data(
        config,
        loader,
        root=root,
        min_sample_tokens=_SLAKE_MIN_SAMPLE_TOKENS,
        num_validation_microbatches=_SLAKE_VALIDATION_MICROBATCHES,
    )
    return apply_transforms(config, [_slake_lora_transform(variant)])


def muse_glimmer_30b_slake_lora_decoder_sft() -> Trainer.Config:
    return _muse_glimmer_30b_slake_sft("lora_decoder")


def muse_glimmer_30b_slake_lora_vision_decoder_sft() -> Trainer.Config:
    return _muse_glimmer_30b_slake_sft("lora_vision_decoder")
