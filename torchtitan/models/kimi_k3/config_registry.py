# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.transform import apply_transforms, MXQATTransform
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MultiModalProcessor,
)
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_navit_patch_grid
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy
from torchtitan.quantization.mx_qat.experts import MXFakeQuantizeConfig
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, model_registry
from .quantization import MXFP4_QUANTIZATION_CONFIG
from .state_dict_adapter import KimiK3StateDictAdapter


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
        max_patches=256,
        max_patches_per_side=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def kimi_k3_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_config = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def kimi_k3_debugmodel_mx_qat(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    checkpoint_path: str | None = None,
    weight_fake_quant_config: MXFakeQuantizeConfig | None = None,
    activation_fake_quant_config: MXFakeQuantizeConfig | None = None,
) -> Trainer.Config:
    """Kimi QAT using the released policy and optional packed HF initialization.

    Pass an absolute checkpoint_path to load the packed debug fixture. Without
    it, the recipe uses random initialization and remains valid before overrides.
    Optional TorchAO configs control fake quantization and kernel_preference;
    model-specific parameter selection stays inside the recipe.
    """
    config = kimi_k3_debugmodel(seq_len=seq_len)
    adapter = KimiK3StateDictAdapter(config.model_spec.model, hf_assets_path=None)
    mapping = adapter.hf_linear_weight_mapping()
    policy = MXFP4CheckpointPolicy.from_config(MXFP4_QUANTIZATION_CONFIG, mapping)
    weights = {mapping[key] for key in policy.weight_fqns if mapping[key] is not None}
    transform = MXQATTransform.from_weight_fqns(config.model_spec.model, weights)
    if weight_fake_quant_config is not None:
        transform.weight_fake_quant_config = weight_fake_quant_config
    if activation_fake_quant_config is not None:
        transform.activation_fake_quant_config = activation_fake_quant_config
    config.checkpointer = CheckpointManager.Config(
        interval=5,
        initial_load_path=checkpoint_path,
        initial_load_in_hf=checkpoint_path is not None,
        initial_load_in_hf_quantized=checkpoint_path is not None,
        last_save_model_only=False,
    )
    return apply_transforms(config, [transform])
