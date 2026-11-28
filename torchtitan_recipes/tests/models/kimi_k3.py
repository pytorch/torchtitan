# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 model configurations used by tests."""

from dataclasses import dataclass, replace

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.data.multimodal.vision_processor import (
    NaViTPatchGridResize,
    VisionProcessor,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import DistMuon, LRSchedulersContainer, Optim
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.kimi_k3 import (
    build_model_config,
    KIMI_K3_SPECIAL_TOKENS,
    KIMI_K3_VISION_PROCESSOR_KWARGS,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.dist_moe import dist_moe_runtime_config, require_dist_moe
from torchtitan_recipes.models.kimi_k3 import dist_muon_optimizer
from torchtitan_recipes.tests.models.kimi_k2_7 import (
    _align_dist_muon_expert_compute_layouts,
)


def _kimi_k3_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(
            build_mrope_positions=False,
        ),
    )


def kimi_k3_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    attn_backend: str = "varlen",
) -> Trainer.Config:
    """Debugmodel with per-head Muon for all logical 2D matrices."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel_mm",
        seq_len=seq_len,
        attn_backend=attn_backend,
        parallelism=parallelism,
    )
    return _KimiK3TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        # Small images keep the debug vision encoder's grids short.
        vision_processor=replace(
            VisionProcessor.Config(**KIMI_K3_VISION_PROCESSOR_KWARGS),
            resize=NaViTPatchGridResize.Config(
                max_patches=256, max_patches_per_side=16
            ),
        ),
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optim=Optim.Config(
            optimizer=dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        precision=PrecisionConfig(dtype="bfloat16"),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def kimi_k3_debugmodel_dist_moe_mxfp8(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    scratch_capacity_factor: float = 1.0,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build the Kimi K3 debug recipe with MXFP8 Dist-MoE experts."""
    dist_moe = require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = kimi_k3_debugmodel(seq_len=seq_len, parallelism=parallelism)
    config.dist_moe_runtime = dist_moe_runtime_config(
        scratch_capacity_factor=scratch_capacity_factor
    )
    return apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )


@dataclass(kw_only=True, slots=True)
class _KimiK3TrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optim.optimizer = _align_dist_muon_expert_compute_layouts(
            self.optim.optimizer,
            parallelism=self.parallelism,
        )
        # TODO(#3353): Support TP-produced _StridedShard layouts in DistMuon.
        uses_dist_muon = any(
            isinstance(config, DistMuon.Config)
            for config in self.optim.optimizer.optimizers
        )
        if uses_dist_muon and self.parallelism.tensor_parallel_degree > 1:
            # Fail during config parsing, before TP/FSDP creates _StridedShard
            # storage. Recipes that need TP replace the optimizer.
            raise ValueError(
                "Kimi K3 DistMuon currently requires tensor_parallel_degree=1: "
                "tensor parallelism can produce unsupported _StridedShard "
                "parameter layouts."
            )
