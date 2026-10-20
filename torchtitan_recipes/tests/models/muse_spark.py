# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark model configurations used by tests."""

from dataclasses import replace
from functools import partial

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetMixConfig,
    GrainDataLoader,
    WeightedDataset,
)
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    ContextParallelTransform,
    MXFP8LinearConverter,
)
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_spark import build_model_config
from torchtitan.models.muse_spark.flavors import _MUSE_SPARK_816B_VOCAB_SIZE
from torchtitan.models.muse_spark.model import MuseSparkModel
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.muse_spark import (
    _muse_spark_context_parallel_transform,
    _MUSE_SPARK_FUSED_SWIGLU_OVERRIDE,
    _MUSE_SPARK_HF_ASSETS_PATH,
    _muse_spark_mm_dataloader,
    _ultradata_lora_sft_optim,
    _ultradata_lora_sft_transform,
    _ultradata_sft_dataloader,
    _with_dist_moe,
    MUSE_SPARK_SPECIAL_TOKENS,
)


_MUSE_SPARK_MXFP8_LINEAR_FQNS = [
    "attention",
    "feed_forward.pooler",
    "feed_forward.unpooler",
]


# TODO: Add the Muse Spark Open tokens to tests/assets/tokenizer and remove
# this debug-only mapping.
MUSE_SPARK_DEBUG_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "audio_token": "<|media_pad|>",
    "audio_start_token": "<|media_begin|>",
    "audio_end_token": "<|media_end|>",
    "pad_token": "<|endoftext|>",
}


def _muse_spark_mxfp8_linear_converter_config() -> MXFP8LinearConverter.Config:
    """Quantize debug-model attention and compressed-MoE projections."""
    return MXFP8LinearConverter.Config(
        fqns=list(_MUSE_SPARK_MXFP8_LINEAR_FQNS),
    )


def muse_spark_debugmodel(
    seq_len: int | None = None,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
            enable_sequence_parallel=True,
        )
    model_config = build_model_config(
        "debugmodel",
        parallelism=parallelism,
        seq_len=seq_len,
        moe_comm_backend="standard",
        attn_backend="flex",
    )
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            num_chunks=8,
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
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
        metrics=MetricsProcessor.Config(log_freq=1),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        override=OverrideConfig(
            imports=[_MUSE_SPARK_FUSED_SWIGLU_OVERRIDE],
        ),
    )
    return config


def muse_spark_debugmodel_with_cp(
    seq_len: int | None = None,
) -> Trainer.Config:
    config = muse_spark_debugmodel(seq_len)
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [_muse_spark_context_parallel_transform()],
    )


def muse_spark_debugmodel_dist_moe_bf16(
    seq_len: int | None = None,
) -> Trainer.Config:
    """Configure the Muse debug model with BF16 Dist-MoE experts."""
    return _with_dist_moe(
        muse_spark_debugmodel(seq_len),
        expert_precision="bf16",
        scratch_capacity_factor=2.0,
    )


def muse_spark_debugmodel_dist_moe_mxfp8(
    seq_len: int | None = None,
) -> Trainer.Config:
    """Configure the Muse debug model with asynchronous MXFP8 Dist-MoE experts."""
    config = muse_spark_debugmodel(seq_len)
    config.model = build_model_config(
        "debugmodel",
        parallelism=config.parallelism,
        seq_len=seq_len,
        moe_comm_backend="standard",
        attn_backend="flex",
        converters=[_muse_spark_mxfp8_linear_converter_config()],
    )
    return _with_dist_moe(
        config,
        expert_precision="mxfp8",
        scratch_capacity_factor=2.0,
    )


def muse_spark_debugmodel_mm(
    seq_len: int | None = None,
    *,
    parallelism: ParallelismConfig | None = None,
    context_parallel_transform: ContextParallelTransform | None = None,
) -> Trainer.Config:
    """Pack mixed CC12M test images and synthetic audio/video at equal weights."""
    from torchtitan.components.data.multimodal.mm_datasets import (
        AudioProcessor,
        MM_DATASETS,
        MMSamplePackingConfig,
    )

    from torchtitan_recipes.tests.datasets.synthetic_multimodal import (
        _audio_token_count,
        SYNTHETIC_MM_DATASETS,
    )

    config = muse_spark_debugmodel(seq_len, parallelism=parallelism)
    config.parallelism.context_parallel_degree = 2
    model_config = build_model_config(
        "debugmodel_mm",
        seq_len=seq_len,
        parallelism=config.parallelism,
        moe_comm_backend="standard",
        attn_backend="flex",
    )
    config.model = model_config
    config.tokenizer = MuseSparkTokenizer.Config(**MUSE_SPARK_DEBUG_SPECIAL_TOKENS)
    image_dataset = _muse_spark_mm_dataloader(
        model_config, MM_DATASETS["cc12m-test"]
    ).dataset
    video_loader = _muse_spark_mm_dataloader(
        model_config, SYNTHETIC_MM_DATASETS["synthetic-video-text"]
    )
    audio_encoder = model_config.audio_encoder
    assert audio_encoder is not None
    audio_dataset = SYNTHETIC_MM_DATASETS["synthetic-audio-text"]
    assert isinstance(audio_dataset.processor, AudioProcessor.Config)
    audio_dataset = replace(
        audio_dataset,
        processor=replace(
            audio_dataset.processor,
            expected_num_channels=audio_encoder.input_channels,
            target_sample_rate=24_000,
            max_audio_samples=audio_encoder.patch_size
            * config.training.max_context_length
            // 2,
            audio_token_count_fn=partial(
                _audio_token_count, samples_per_audio_token=audio_encoder.patch_size
            ),
        ),
    )
    config.dataloader = replace(
        video_loader,
        dataset=MMSamplePackingConfig(
            dataset=DatasetMixConfig(
                datasets=(
                    WeightedDataset(dataset=audio_dataset),
                    WeightedDataset(dataset=video_loader.dataset),
                    WeightedDataset(dataset=image_dataset),
                )
            )
        ),
        collator=replace(video_loader.collator, max_images_per_microbatch=128),
    )
    if context_parallel_transform is None:
        context_parallel_transform = _muse_spark_context_parallel_transform()
    return apply_transforms(
        config,
        [context_parallel_transform],
    )


def _use_released_tokenizer_with_debug_model(config: Trainer.Config) -> None:
    model_config = config.model
    loss_config = config.loss
    assert isinstance(model_config, MuseSparkModel.Config)
    assert isinstance(loss_config, ChunkedLossWrapper.Config)
    assert isinstance(loss_config.loss_fn, CrossEntropyLoss.Config)

    config.hf_assets_path = _MUSE_SPARK_HF_ASSETS_PATH
    config.tokenizer = MuseSparkTokenizer.Config(**MUSE_SPARK_SPECIAL_TOKENS)
    model_config.vocab_size = _MUSE_SPARK_816B_VOCAB_SIZE
    model_config.tok_embeddings.embedding.num_embeddings = model_config.vocab_size
    model_config.lm_head.out_features = model_config.vocab_size
    loss_config.loss_fn.global_vocab_size = model_config.vocab_size


def muse_spark_debugmodel_ultradata_lora_sft() -> Trainer.Config:
    """Configure ten-step UltraData LoRA SFT with a debug model on 8 GPUs."""
    config = muse_spark_debugmodel_with_cp(seq_len=32_768)
    _use_released_tokenizer_with_debug_model(config)
    config.dataloader = _ultradata_sft_dataloader()
    config.optim = _ultradata_lora_sft_optim(warmup_steps=2)
    config.training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=32_768,
        num_tokens_per_train_step=65_536,
        max_context_length=32_768,
        steps=10,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        disable_cuda_graphs=True,
    )
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        context_parallel_degree=2,
        expert_parallel_degree=8,
        pipeline_parallel_degree=1,
        enable_sequence_parallel=True,
        context_parallel_load_balancer=None,
    )
    config.comm.train_timeout_seconds = 1800
    config.activation_remat = FullActivationRematConfig()
    config.model.local_compile.regions = ["loss"]
    config.debug.seed = 42
    return apply_transforms(config, [_ultradata_lora_sft_transform()])
