# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MultiModalProcessor,
)
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_navit_patch_grid
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, model_registry


def _kimi_k3_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    processor = dataset.processor
    if not isinstance(processor, MultiModalProcessor.Config):
        raise ValueError("Kimi K3 multimodal data requires MultiModalProcessor.Config")

    processor = MultiModalProcessor.Config(
        sample_processor=processor.sample_processor,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        resize_fn=resize_to_navit_patch_grid,
        min_pixels=56 * 56,
        max_pixels=224 * 224,
        max_patches=256,
        max_patches_per_side=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            max_images_per_batch=8,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def kimi_k3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=256,
            max_context_length=256,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def kimi_k3_debugmodel_lora() -> Trainer.Config:
    """The multimodal debug model with LoRA adapters on the attention output.

    Uses core's LoRAConverter rather than a model-local implementation. The
    Targets are matched on the last segment of the FQN. The set mirrors the
    reference tree's DEFAULT_LORA_TARGETS: the MLA projections, and -- the part
    that matters structurally -- the dense FFN and latent-MoE projections.
    Every decoder layer carries an FFN or MoE, while only one layer in four is
    MLA (K3 is 3 KDA : 1 MLA), so an MLA-only target set leaves an all-KDA
    pipeline stage with zero trainable parameters and the optimizer then raises
    "param_groups pattern matched no parameters". That is what pp8 hit.

    Not covered: the reference also adapts the MLA output gate. Here that module
    is named ``gate``, which is also the router's gate in every MoE layer, and
    last-segment matching cannot separate them -- adding it would silently adapt
    the routers too. Left out rather than guessed.
    """
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel", converters=[_kimi_k3_lora_converter()]
    )
    return config


def _kimi_k3_lora_converter(
    *, quantize_base: str | None = None, quantize_experts: str | None = None
):
    from torchtitan.components.lora import LoRAConverter

    return LoRAConverter.Config(
        rank=8,
        alpha=16.0,
        target_modules=[
            # MLA
            "wq_a",
            "wq_b",
            "wkv_a",
            "wkv_b",
            "wo",
            # dense FFN and shared experts
            "w1",
            "w2",
            "w3",
            # latent MoE down/up projections
            "routed_down",
            "routed_up",
        ],
        quantize_base=quantize_base,
        quantize_experts=quantize_experts,
    )


def kimi_k3_debugmodel_qlora_mxfp4_linear() -> Trainer.Config:
    """QLoRA with only the base LINEARS packed (experts stay bf16).

    The packed-TP forward covers colwise/rowwise linears; packed experts
    under expert-TP need a shape-preserving layout and refuse -- this
    flavor is the TP-composable subset.
    """
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[_kimi_k3_lora_converter(quantize_base="mxfp4")],
    )
    return config


def kimi_k3_debugmodel_mx_qat() -> Trainer.Config:
    """The debug model under MXFP4-weight / MXFP8-activation fake-quant QAT.

    K3's official quantization scope: the routed experts only, bf16 masters
    training underneath (full-param QAT, no LoRA). Fake-quant is bf16
    compute, so this runs on any GPU.
    """
    from torchtitan.components.quantization.mx_qat import MXFP4QATConverter

    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel", converters=[MXFP4QATConverter.Config()]
    )
    return config


def kimi_k3_debugmodel_qlora_mxfp4() -> Trainer.Config:
    """The LoRA debug model with MXFP4-packed bases (QLoRA, K3's native
    weight format).

    The packing swaps the base for split storage AT BUILD, before
    parallelize, so FSDP2 shards the packed bytes -- this is the
    pack-then-shard order the nf4 path cannot reach, and the flavor trains
    under the normal sharded flow.
    """
    config = kimi_k3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            _kimi_k3_lora_converter(quantize_base="mxfp4", quantize_experts="mxfp4")
        ],
    )
    return config
