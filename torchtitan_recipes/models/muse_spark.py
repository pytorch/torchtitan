# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Spark Open training recipes."""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast, Literal

from renderers import Message, ToolSpec

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor, DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    AsyncTensorParallelTransform,
    ContextParallelTransform,
    LoRATransform,
    transform_model_config_,
)
from torchtitan.config.transform.base import convert_config_type
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
)
from torchtitan.models.common.activation import ClampedSwiGLU
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import (
    AllGatherKVFlexInnerAttention,
    AllGatherKVSlidingWindowFlexInnerAttention,
    UlyssesFlexInnerAttention,
    UlyssesSlidingWindowFlexInnerAttention,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.moe import (
    RoundRobinTokenChoiceTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)

from torchtitan.models.muse_spark import build_model_config
from torchtitan.models.muse_spark.model import MuseSparkModel
from torchtitan.models.muse_spark.renderer import MuseSparkRendererConfig
from torchtitan.models.muse_spark.tokenizer import MuseSparkTokenizer
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


_TRAINING_LR = 5e-5
_ROUTER_LR_MULTIPLIER = 0.125
_MUSE_SPARK_HF_ASSETS_PATH = "./assets/hf/Muse-Spark-Open-816B-A42B"
_MUSE_GLIMMER_MAX_IMAGE_TOKENS = 4096
_MUSE_SPARK_FUSED_SWIGLU_OVERRIDE = (
    "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"
)
_MUSE_SPARK_DIST_MOE_SAVE_REGIONS = [
    "attention.inner_attention",
    "feed_forward.routed_experts.dist_moe",
]


def _muse_spark_context_parallel_transform() -> ContextParallelTransform:
    return ContextParallelTransform(
        inner_attention_map={
            FlexInnerAttention: UlyssesFlexInnerAttention,
            SlidingWindowFlexInnerAttention: (UlyssesSlidingWindowFlexInnerAttention),
        },
    )


def _muse_spark_kv_all_gather_context_parallel_transform() -> ContextParallelTransform:
    return ContextParallelTransform(
        inner_attention_map={
            FlexInnerAttention: AllGatherKVFlexInnerAttention,
            SlidingWindowFlexInnerAttention: (
                AllGatherKVSlidingWindowFlexInnerAttention
            ),
        },
    )


# TODO: Add production Muse Spark Open video and audio datasets. Muse Spark Open encodes images as
# <|image_start|> + repeated <|patch|> + <|image_end|>; videos as <|vid_start|>
# + timestamped runs of <|video|>, separated by <|vid_frame_separator|>, then
# <|vid_end|>; and audio as <|soft_audio_start|> + repeated <|audio|> +
# <|soft_audio_end|>.
MUSE_SPARK_SPECIAL_TOKENS = {
    "image_token": "<|patch|>",
    "video_token": "<|video|>",
    "vision_start_token": "<|image_start|>",
    "vision_end_token": "<|image_end|>",
    "audio_token": "<|audio|>",
    "audio_start_token": "<|soft_audio_start|>",
    "audio_end_token": "<|soft_audio_end|>",
    "pad_token": "<|finetune_right_pad|>",
}


def _with_dist_moe(
    config: Trainer.Config,
    *,
    expert_precision: Literal["bf16", "mxfp8"],
    activation_slot_capacity_factor: float = 1.0,
    scratch_capacity_factor: float,
    vmm_capacity_factor: float | None = None,
    expert_activation: ClampedSwiGLU.Config | None = None,
    remat_save_regions: Sequence[str] = _MUSE_SPARK_DIST_MOE_SAVE_REGIONS,
) -> Trainer.Config:
    """Configure and replace Muse routed experts with Dist-MoE."""
    try:
        from torchtitan.config.transform.dist_moe import DistMoeTransform
        from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
    except ModuleNotFoundError as error:
        if error.name != "dist_moe":
            raise
        raise ModuleNotFoundError(
            "This recipe requires the optional dist_moe package.",
            name="dist_moe",
        ) from error

    config.training.disable_cuda_graphs = False
    # Activation remat retains the expensive attention and effectful Dist-MoE
    # regions. Projections, routing preparation, norms, and pool/unpool boundaries
    # are cheaper to replay and provide the activation-memory reduction.
    config.activation_remat = ActivationRematConfig(
        save_regions=list(remat_save_regions),
    )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=activation_slot_capacity_factor,
        scratch_capacity_factor=scratch_capacity_factor,
        vmm_capacity_factor=vmm_capacity_factor,
    )
    if expert_activation is not None:
        for _fqn, routed_experts, _parent, _attr in config.model.traverse(
            RoutedExperts.Config
        ):
            routed_experts.activation_fn = replace(expert_activation)
    config = apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision=expert_precision,
                inplace_wgrad_accum=True,
            )
        ],
    )
    # The standard Muse override fuses SwiGLU inside stock routed experts.
    # Dist-MoE replaces every debug and 816B expert and owns that fusion, so
    # retaining the override would leave no legitimate target. RoPE is compiled
    # with the q/k norms in the muse_spark_attention_qk region.
    config.override.imports = []
    return config


def _use_forced_balanced_routing(model_config: MuseSparkModel.Config) -> None:
    """Replace token-choice routers with deterministic round-robin routing."""
    for _fqn, router, parent, attr in list(
        model_config.traverse(TokenChoiceTopKRouter.Config)
    ):
        replacement = convert_config_type(
            router,
            RoundRobinTokenChoiceTopKRouter,
        )
        assert parent is not None and isinstance(attr, str)
        setattr(parent, attr, replacement)


def _use_flex_flash_attention(model_config: MuseSparkModel.Config) -> None:
    """Select FA4 and layer-appropriate block-mask geometry under CP."""
    for layer in model_config.layers:
        attention = layer.attention
        inner_attention = attention.inner_attention
        if not isinstance(
            inner_attention,
            (
                FlexInnerAttention.Config,
                SlidingWindowFlexInnerAttention.Config,
                AllGatherKVFlexInnerAttention.Config,
                AllGatherKVSlidingWindowFlexInnerAttention.Config,
            ),
        ):
            raise TypeError("Muse Spark scale recipes require Flex Attention")
        inner_attention = replace(
            inner_attention,
            block_size=((256, 256) if attention.window_size is None else (256, 128)),
            kernel_options={**inner_attention.kernel_options, "BACKEND": "FLASH"},
        )
        attention.inner_attention = inner_attention


def _resize_to_muse_glimmer_patch_grid(
    height: int,
    width: int,
    *,
    patch_size: int,
    merge_size: int,
    max_image_tokens: int = _MUSE_GLIMMER_MAX_IMAGE_TOKENS,
    **_: object,
) -> tuple[int, int, int, int]:
    """Select the closest-aspect-ratio merged-patch grid under the token cap."""
    merged_patch_size = patch_size * merge_size
    ideal_grid_h = height / merged_patch_size
    ideal_grid_w = width / merged_patch_size
    aspect_ratio = ideal_grid_w / ideal_grid_h if ideal_grid_h > 0 else 1.0
    if ideal_grid_h * ideal_grid_w > max_image_tokens:
        ideal_grid_h = math.sqrt(max_image_tokens / aspect_ratio)
        ideal_grid_w = ideal_grid_h * aspect_ratio

    candidates = list(
        set(
            itertools.product(
                [math.floor(ideal_grid_h), math.ceil(ideal_grid_h)],
                [math.floor(ideal_grid_w), math.ceil(ideal_grid_w)],
            )
        )
    )
    candidates = [
        (grid_h, grid_w)
        for grid_h, grid_w in candidates
        if grid_h >= 1 and grid_w >= 1 and grid_h * grid_w <= max_image_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(ideal_grid_h)), max(1, round(ideal_grid_w)))]

    grid_h, grid_w = min(
        candidates,
        key=lambda grid: abs(grid[0] / grid[1] - height / width),
    )
    return (
        grid_h * merged_patch_size,
        grid_w * merged_patch_size,
        0,
        0,
    )


def _muse_spark_mm_dataloader(
    model_config: MuseSparkModel.Config,
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
    from torchtitan.components.data.multimodal.mm_datasets import VisionProcessor

    vision_encoder = model_config.vision_encoder
    if vision_encoder is None:
        raise ValueError("Multimodal Muse Spark must own a vision encoder")

    base_processor = dataset.processor
    if not isinstance(base_processor, VisionProcessor.Config):
        raise ValueError("Multimodal vision datasets must use VisionProcessor.Config")
    processor = replace(
        base_processor,
        patch_size=vision_encoder.patch_size,
        temporal_patch_size=vision_encoder.patch_temporal,
        spatial_merge_size=vision_encoder.downsample_factor,
        resize_fn=_resize_to_muse_glimmer_patch_grid,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    audio_encoder = model_config.audio_encoder
    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            max_images_per_microbatch=8,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
            waveform_pad_multiple=(
                None if audio_encoder is None else audio_encoder.patch_size
            ),
            audio_input_channels=(
                None if audio_encoder is None else audio_encoder.input_channels
            ),
        ),
    )


def _muse_spark_816b_a42b(
    seq_len: int | None = None,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Configure full-parameter text continuation from the Muse Spark Open checkpoint.

    Stage ``meta-models/Muse-Spark-Open-816B-A42B`` locally at ``hf_assets_path``.
    To load weights from a separate directory, set
    ``checkpointer.initial_load_path`` to an absolute path in the recipe.
    Use a fresh checkpoint folder; native DCP checkpoints cannot be resumed
    by this backend.
    """
    if parallelism is None:
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=16,
            tensor_parallel_degree=8,
            expert_parallel_degree=32,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            enable_sequence_parallel=True,
        )
    model_config = build_model_config(
        "816B-A42B",
        parallelism=parallelism,
        seq_len=seq_len,
        moe_comm_backend="standard",
        attn_backend="flex",
    )
    if parallelism.context_parallel_degree > 1:
        transform_model_config_(
            model_config, [_muse_spark_context_parallel_transform()]
        )
    common_optimizer_kwargs = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            num_chunks=8,
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=_MUSE_SPARK_HF_ASSETS_PATH,
        tokenizer=MuseSparkTokenizer.Config(**MUSE_SPARK_SPECIAL_TOKENS),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
            shuffle=True,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r"\.router\.gate\.",
                        lr=_TRAINING_LR * _ROUTER_LR_MULTIPLIER,
                        **common_optimizer_kwargs,
                    ),
                    AdamW.Config(
                        pattern=r".*",
                        lr=_TRAINING_LR,
                        **common_optimizer_kwargs,
                    ),
                ],
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=10,
                decay_ratio=None,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            num_tokens_per_train_step=32 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=120,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=cast(
            CheckpointManager.Config,
            TorchCheckpointingManager.Config(
                folder="muse_spark_816b_a42b",
                interval=2000,
                initial_load_model_only=True,
                initial_load_in_hf=True,
                last_save_model_only=False,
                keep_latest_k=2,
            ),
        ),
        activation_remat=DefaultActivationRematConfig(),
        override=OverrideConfig(
            imports=[_MUSE_SPARK_FUSED_SWIGLU_OVERRIDE],
        ),
    )


def _muse_spark_816b_dist_moe_bf16_spmd(
    *,
    seq_len: int,
    global_batch_size: int,
    data_parallel_shard_degree: int,
    tensor_parallel_degree: int,
    context_parallel_degree: int,
    expert_parallel_degree: int,
    fake_backend: bool,
    remat_save_regions: Sequence[str] = _MUSE_SPARK_DIST_MOE_SAVE_REGIONS,
) -> Trainer.Config:
    """Build a long-context BF16 Dist-MoE SPMD recipe.

    The caller owns workload geometry and the activation-rematerialization
    policy. This helper owns the common Muse Spark Open BF16 Dist-MoE, Flex-Flash,
    FSDP, and production-versus-FakePG integration.

    Args:
        seq_len: Tokens in each data-parallel microbatch before TP and CP
            sequence sharding.
        global_batch_size: Number of sequences in one optimizer step.
        data_parallel_shard_degree: FSDP shard degree and the number of
            independent sequences processed per accumulation step.
        tensor_parallel_degree: Tensor-parallel degree for model projections.
        context_parallel_degree: Context-parallel degree for the sequence.
        expert_parallel_degree: Expert-parallel degree for routed experts.
        fake_backend: Whether to replace SPMD communication with FakePG for
            local geometry and memory validation.
        remat_save_regions: Named remat regions retained by activation remat.
            All other registered regions are recomputed.
    """
    degrees = {
        "data_parallel_shard_degree": data_parallel_shard_degree,
        "tensor_parallel_degree": tensor_parallel_degree,
        "context_parallel_degree": context_parallel_degree,
        "expert_parallel_degree": expert_parallel_degree,
    }
    if invalid_degrees := [name for name, degree in degrees.items() if degree < 1]:
        raise ValueError(f"parallel degrees must be positive: {invalid_degrees}")
    if seq_len < 1 or global_batch_size < 1:
        raise ValueError("seq_len and global_batch_size must be positive")
    if global_batch_size % data_parallel_shard_degree != 0:
        raise ValueError(
            "global_batch_size must be divisible by data_parallel_shard_degree"
        )
    sequence_shard_degree = tensor_parallel_degree * context_parallel_degree
    if seq_len % sequence_shard_degree != 0:
        raise ValueError("seq_len must be divisible by TP * CP")
    logical_world_size = (
        data_parallel_shard_degree * tensor_parallel_degree * context_parallel_degree
    )
    if logical_world_size % expert_parallel_degree != 0:
        raise ValueError("expert_parallel_degree must divide the logical world size")

    # These are BF16-only scale recipes. Muse inner attention and the LM head
    # remain BF16, and TorchTitan has no asynchronous MXFP8 TP path, so an
    # MXFP8 long-context recipe would quantize only a minority of the critical
    # path while exposing TP communication. Keep MXFP8 coverage in the debug
    # recipe until those two execution paths exist.
    config = _with_dist_moe(
        _muse_spark_816b_a42b(seq_len=seq_len),
        expert_precision="bf16",
        activation_slot_capacity_factor=1.0,
        scratch_capacity_factor=32.0,
        vmm_capacity_factor=None,
        expert_activation=ClampedSwiGLU.Config(
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
        ),
        remat_save_regions=remat_save_regions,
    )
    config.loss = ChunkedLossWrapper.Config(
        num_chunks=8,
        loss_fn=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(config.model),
        ),
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.num_tokens_per_train_step = global_batch_size * seq_len
    config.training.mixed_precision_reduce = "float32"
    # Non-deferred reduction makes every outer-accumulation group identical.
    # Capture one group and replay it instead of unrolling all eight groups into
    # one graph; persistent sharded gradients accumulate across those replays.
    config.training.cuda_graph_per_microbatch_group = True

    p = config.parallelism
    p.data_parallel_replicate_degree = 1
    p.data_parallel_shard_degree = data_parallel_shard_degree
    p.tensor_parallel_degree = tensor_parallel_degree
    p.context_parallel_degree = context_parallel_degree
    p.context_parallel_load_balancer = None
    p.expert_parallel_degree = expert_parallel_degree
    p.enable_sequence_parallel = tensor_parallel_degree > 1
    p.fsdp_defer_gradient_reduction = False
    p.pipeline_parallel_degree = 1
    p.pipeline_parallel_module_fqns_per_model_part = None
    p.num_pp_microbatches = 1
    p.fsdp_reshard_after_forward = "always"

    if fake_backend:
        config.hf_assets_path = "./tests/assets/tokenizer"
        config.tokenizer = HuggingFaceTokenizer.Config()
        config.dataloader = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        )
        config.training.steps = 10
        config.checkpointer = None
        _use_forced_balanced_routing(cast(MuseSparkModel.Config, config.model))
        config.comm.backend = "fake"
    else:
        config.checkpointer = replace(
            config.checkpointer,
            folder=("muse_spark_816b_a42b_dist_moe_bf16_spmd_" f"{logical_world_size}"),
        )
    if context_parallel_degree > 1:
        config = apply_transforms(
            config,
            [_muse_spark_kv_all_gather_context_parallel_transform()],
        )
    # KV all-gather keeps queries sequence-sharded while making the complete KV
    # sequence available to Flex-Flash. This preserves Muse global/sliding
    # attention semantics without a query all-to-all; the global and sliding
    # layers use independently tuned block-mask geometry.
    _use_flex_flash_attention(cast(MuseSparkModel.Config, config.model))
    if not fake_backend and tensor_parallel_degree > 1:
        config = apply_transforms(
            config,
            [
                AsyncTensorParallelTransform(
                    enable_sequence_parallel=True,
                    disable_inplace_wgrad_accum=True,
                )
            ],
        )
    # TODO: Enable async TP for FakePG after pytorch/pytorch#200287 lands. Until
    # then FakePG validates geometry and memory with synchronous TP; only
    # production recipes enable communication overlap.
    return config


def muse_spark_816b_a42b_dist_moe_bf16_spmd_256() -> Trainer.Config:
    """Configure 96K BF16 Dist-MoE SPMD on 256 GPUs."""
    return _muse_spark_816b_dist_moe_bf16_spmd(
        seq_len=96 * 1024,
        global_batch_size=128,
        data_parallel_shard_degree=16,
        tensor_parallel_degree=4,
        context_parallel_degree=4,
        expert_parallel_degree=64,
        fake_backend=False,
    )


def muse_spark_816b_a42b_dist_moe_bf16_spmd_256_fake() -> Trainer.Config:
    """Validate the logical-256 BF16 SPMD recipe with FakePG."""
    return _muse_spark_816b_dist_moe_bf16_spmd(
        seq_len=96 * 1024,
        global_batch_size=128,
        data_parallel_shard_degree=16,
        tensor_parallel_degree=4,
        context_parallel_degree=4,
        expert_parallel_degree=64,
        fake_backend=True,
    )


def muse_spark_816b_a42b_dist_moe_bf16_spmd_512() -> Trainer.Config:
    """Configure 192K BF16 Dist-MoE SPMD on 512 GPUs."""
    return _muse_spark_816b_dist_moe_bf16_spmd(
        seq_len=192 * 1024,
        global_batch_size=128,
        data_parallel_shard_degree=16,
        tensor_parallel_degree=4,
        context_parallel_degree=8,
        expert_parallel_degree=64,
        fake_backend=False,
    )


def muse_spark_816b_a42b_dist_moe_bf16_spmd_512_fake() -> Trainer.Config:
    """Validate the logical-512 BF16 SPMD recipe with FakePG."""
    return _muse_spark_816b_dist_moe_bf16_spmd(
        seq_len=192 * 1024,
        global_batch_size=128,
        data_parallel_shard_degree=16,
        tensor_parallel_degree=4,
        context_parallel_degree=8,
        expert_parallel_degree=64,
        fake_backend=True,
    )


# UltraData LoRA SFT recipes.

_ULTRADATA_DATASET_PATH = "openbmb/UltraData-SFT-Agent-2609"
_ULTRADATA_DATASET_NAME = "General-Agent"
_ULTRADATA_LOCAL_JSONL_PATTERN = (
    "./assets/datasets/muse_spark_ultradata/General_Agent_part-*.jsonl"
)


def _tool_call_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or ():
        function = tool_call.get("function") or {}
        name = function.get("name")
        if not name:
            raise ValueError("assistant tool call is missing function.name")
        names.append(name)
    return names


def ultradata_messages(row: dict[str, Any]) -> list[Message]:
    """Return messages with positional names added to unnamed tool results."""
    normalized: list[dict[str, Any]] = []
    pending_names: list[str] = []
    result_index = 0
    for original in row["messages"]:
        message = dict(original)
        role = message.get("role")
        if role == "assistant":
            pending_names = _tool_call_names(message)
            result_index = 0
        elif role == "tool":
            if not message.get("name") and not message.get("tool_call_id"):
                if not pending_names:
                    raise ValueError(
                        "Cannot normalize unnamed tool result without a preceding "
                        "assistant tool call."
                    )
                position = min(result_index, len(pending_names) - 1)
                message["name"] = pending_names[position]
            result_index += 1
        else:
            pending_names = []
            result_index = 0
        normalized.append(message)
    return cast(list[Message], normalized)


def ultradata_tools(row: dict[str, Any]) -> list[ToolSpec] | None:
    return cast(list[ToolSpec] | None, row.get("tools"))


def _ultradata_sft_dataloader(
    source: HuggingFaceStreamingSource.Config | IndexedJsonlSource.Config | None = None,
) -> GrainDataLoader.Config:
    if source is None:
        source = HuggingFaceStreamingSource.Config(
            path=_ULTRADATA_DATASET_PATH,
            name=_ULTRADATA_DATASET_NAME,
            split="train",
        )
    return GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=SingleDatasetConfig(
                source=source,
                processor=ChatProcessor.Config(
                    messages_fn=ultradata_messages,
                    tools_fn=ultradata_tools,
                    renderer=MuseSparkRendererConfig(current_date="2026-10-02"),
                ),
                post_filters=(lambda sample: sample is not None,),
            ),
            num_packing_bins=2,
        ),
        seed=42,
        shuffle=True,
        repeat=True,
        num_prefetch_microbatches=1,
        max_num_documents=8,
    )


def _ultradata_lora_sft_optim(*, warmup_steps: int) -> Optim.Config:
    return Optim.Config(
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
            warmup_steps=warmup_steps,
            decay_ratio=0.1,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
    )


def _ultradata_lora_sft_transform() -> LoRATransform:
    return LoRATransform(
        rank=8,
        alpha=16.0,
        target_modules=["wqkv", "wo", "output_gate", "w13", "w2"],
    )


def muse_spark_816b_a42b_ultradata_lora_sft_cp16_128k() -> Trainer.Config:
    """Configure 128K UltraData LoRA SFT from released Muse Spark Open weights.

    This example targets 16 GB300 GPUs with BF16 Dist-MoE and FA4. Install
    the optional ``dist_moe`` package. The activation and scratch budgets
    target GB300; review them when adapting the recipe to other hardware.

    Stage ``meta-models/Muse-Spark-Open-816B-A42B`` weights and tokenizer at
    ``./assets/hf/Muse-Spark-Open-816B-A42B``, or set
    ``MUSE_SPARK_HF_ASSETS_PATH`` to another local directory.
    Supply General-Agent JSONL files from ``openbmb/UltraData-SFT-Agent-2609``
    at ``./assets/datasets/muse_spark_ultradata/General_Agent_part-*.jsonl``,
    or override the pattern with ``MUSE_SPARK_ULTRADATA_JSONL_PATTERN``.

    Native torch_checkpointing checkpoints retain LoRA adapters and training state;
    HF exports omit the adapters. The default checkpoint folder is
    ``muse_spark_ultradata_lora_cp16_128k`` under ``--output-dir`` (default:
    ``outputs``). ``MUSE_SPARK_CHECKPOINT_FOLDER`` overrides it and accepts
    an absolute path. Existing native checkpoints take precedence over
    initial HF loading, so use a fresh folder to start from released weights.
    Native DCP checkpoints cannot be resumed here; use a separate folder.
    Native checkpoints with ``layers.*.moe`` keys cannot be resumed here.
    The 1,000-step target includes restored steps. Intermediate saves are
    asynchronous; the final save is synchronous.

    Use the environment overrides above for paths; edit or derive this recipe
    for other configuration changes.
    """
    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    seq_len = 131_072

    config = _muse_spark_816b_a42b(
        seq_len=seq_len,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            context_parallel_degree=16,
            expert_parallel_degree=16,
            enable_sequence_parallel=False,
            context_parallel_load_balancer=None,
        ),
    )
    config.hf_assets_path = os.environ.get(
        "MUSE_SPARK_HF_ASSETS_PATH", _MUSE_SPARK_HF_ASSETS_PATH
    )
    jsonl_pattern = os.environ.get(
        "MUSE_SPARK_ULTRADATA_JSONL_PATTERN", _ULTRADATA_LOCAL_JSONL_PATTERN
    )
    config.dataloader = _ultradata_sft_dataloader(
        IndexedJsonlSource.Config(patterns=(jsonl_pattern,))
    )
    config.optim = _ultradata_lora_sft_optim(warmup_steps=100)
    config.optim.enable_cuda_graph = False
    config.optim.lr_scheduler.total_steps = 1000
    config.training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=seq_len,
        num_tokens_per_train_step=seq_len,
        max_context_length=seq_len,
        steps=1000,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        disable_cuda_graphs=False,
    )
    config.comm.init_timeout_seconds = 1800
    config.comm.train_timeout_seconds = 1800
    config.model.local_compile_regions = ["loss"]
    config.debug.seed = 42
    config.activation_remat = ActivationRematConfig(
        save_regions=list(_MUSE_SPARK_DIST_MOE_SAVE_REGIONS),
    )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=0.25,
        scratch_capacity_factor=16.0,
        vmm_capacity_factor=None,
    )
    config.override.imports = []
    config.checkpointer = cast(
        CheckpointManager.Config,
        TorchCheckpointingManager.Config(
            folder=os.environ.get(
                "MUSE_SPARK_CHECKPOINT_FOLDER",
                "muse_spark_ultradata_lora_cp16_128k",
            ),
            initial_load_path=os.path.abspath(config.hf_assets_path),
            initial_load_model_only=True,
            initial_load_in_hf=True,
            interval=100,
            load_only=False,
            keep_latest_k=2,
            last_save_model_only=False,
            last_save_in_hf=False,
            export_dtype="bfloat16",
        ),
    )
    config = apply_transforms(
        config,
        [
            DistMoeTransform(expert_precision="bf16", inplace_wgrad_accum=False),
            _ultradata_lora_sft_transform(),
        ],
    )
    for _, inner_attention, _, _ in config.model.traverse(FlexInnerAttention.Config):
        inner_attention.kernel_options = {"BACKEND": "FLASH"}
        inner_attention.block_size = 256
    return config
