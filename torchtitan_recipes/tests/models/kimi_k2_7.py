# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K2.5 model configurations used by tests."""

from dataclasses import dataclass, replace
from typing import cast

from torch.distributed.tensor import Shard

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS, VisionProcessor
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_navit_patch_grid
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekV3Attention

from torchtitan.models.kimi_k2_7 import (
    build_model_config,
    KIMI_K2_5_SPECIAL_TOKENS,
    KimiK25Model,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def _kimi_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    processor = dataset.processor
    if not isinstance(processor, VisionProcessor.Config):
        raise ValueError("Kimi multimodal data requires VisionProcessor.Config")

    processor = VisionProcessor.Config(
        sample_processor=processor.sample_processor,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        min_pixels=65_536,
        max_pixels=16_777_216,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize_fn=resize_to_navit_patch_grid,
        max_patches=16_384,
        max_patches_per_side=512,
    )

    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            max_images_per_microbatch=128,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def kimi_k2_5_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model_config = build_model_config("debugmodel", seq_len=seq_len)
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K2_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=_kimi_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
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
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def moonlight_16b_a3b(seq_len: int | None = None) -> Trainer.Config:
    """Moonlight 16B-A3B: the text-only DeepSeekV3 sibling (no vision tower)."""
    model_config = build_model_config(
        "moonlight-16B-A3B", seq_len=seq_len, attn_backend="flex"
    )
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
    )
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Moonlight-16B-A3B",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=3e-4,
                adamw_lr=3e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
    )


def kimi_vl_a3b(seq_len: int | None = None) -> Trainer.Config:
    """Kimi-VL A3B: Moonlight text tower + 2D MoonViT vision (image-text)."""
    model_config = build_model_config(
        "Kimi-VL-A3B", seq_len=seq_len, attn_backend="flex"
    )
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
    )
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-VL-A3B",
        # Kimi-VL-A3B names the vision-start token <|media_start|>, whereas the
        # K2.5 family uses <|media_begin|>; override just that one entry.
        tokenizer=MultiModalTokenizer.Config(
            **{**KIMI_K2_5_SPECIAL_TOKENS, "vision_start_token": "<|media_start|>"}
        ),
        model=model_config,
        # Kimi-VL is a compatibility flavor; resizing intentionally follows
        # Kimi-K2.5 per-side scaling instead of legacy Kimi-VL's side rejection.
        dataloader=_kimi_multimodal_dataloader(MM_DATASETS["cc12m"]),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=3e-4,
                adamw_lr=3e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
    )


def kimi_k2_5(seq_len: int | None = None) -> Trainer.Config:
    """Full Kimi K2.5 (~1T-total / ~32B-active)."""
    # The report uses BF16 compute; its FP8 path only compresses saved activations.
    model_config = build_model_config("Kimi-K2.5", seq_len=seq_len, attn_backend="flex")
    parallelism = ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
        expert_parallel_degree=8,
    )
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-K2.5",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=2.2e-4,
                adamw_lr=2.2e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=FullAC.Config(),
    )


def _per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: Shard(0),
            },
        )

    # Preserve exact EP-first DTensor ownership. If an EP-local expert count is
    # smaller than the edp_shard size, add balanced rank assignment only after
    # benchmarks show that the fixed nonempty edp_shard coordinates are a hotspot.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        # EP splits the expert dimension first, then edp_shard repartitions each
        # EP-local expert domain, which reverses the storage-mesh axis order.
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
    )


def _dist_muon_optimizer(
    model_config: KimiK25Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    attention = cast(DeepSeekV3Attention.Config, model_config.first_base_attention)
    owned = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
        },
    )
    # Kimi runs Newton-Schulz per logical projection within each MLA head.
    per_query_head = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                # per head: [q_nope_h; q_rope_h]
                block_sizes=(attention.qk_nope_head_dim, attention.qk_rope_head_dim),
            )
        },
    )
    kv_latent_and_rope = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                # Shared projections: [kv_latent; k_rope]
                block_sizes=(attention.kv_lora_rank, attention.qk_rope_head_dim),
            )
        },
    )
    per_key_value_head = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                # per head: [k_nope_h; v_h]
                block_sizes=(attention.qk_nope_head_dim, attention.v_head_dim),
            )
        },
    )
    per_expert = _per_expert_compute_layout(parallelism)
    query_shardings: dict[str, ComputeLayout] = (
        {
            "wq_a": owned,
            "wq_b": per_query_head,
        }
        if attention.q_lora_rank
        else {"wq": per_query_head}
    )
    attention_shardings = {
        **query_shardings,
        "wkv_a": kv_latent_and_rope,
        "wkv_b": per_key_value_head,
        "wo": owned,
    }
    feed_forward_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
        ),
        "w2": owned,
    }
    num_layers = len(model_config.layers)
    adamw_kwargs = {
        "lr": adamw_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    expert_projections = ("w13.weight", "w2.weight")

    def compute_shardings_for_layer(
        layer_id: int,
    ) -> dict[str, ComputeLayout]:
        prefix = f"layers.{layer_id}"
        shardings = {
            f"{prefix}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in attention_shardings.items()
        }
        if not layer_id:
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": compute_sharding
                    for projection, compute_sharding in feed_forward_shardings.items()
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.moe.routed_experts.{projection}": per_expert
                    for projection in expert_projections
                }
            )
            shardings[f"{prefix}.moe.router.gate.weight"] = owned
            shardings.update(
                {
                    f"{prefix}.moe.shared_experts.{projection}.weight": compute_sharding
                    for projection, compute_sharding in feed_forward_shardings.items()
                }
            )
        return shardings

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    layer_fqns = []
    for layer_id in range(num_layers):
        layer_shardings = compute_shardings_for_layer(layer_id)
        compute_sharding_by_fqn.update(layer_shardings)
        layer_fqns.append(tuple(layer_shardings))
    # Layer 0 has a much larger dense MLP, so keep it separate while amortizing
    # collective launch overhead across pairs of MoE layers.
    bucket_layer_ids = [(0,)] + [
        tuple(range(first_layer_id, min(first_layer_id + 2, num_layers)))
        for first_layer_id in range(1, num_layers, 2)
    ]
    bucket_configs = []
    for layer_ids in bucket_layer_ids:
        name = "layers." + "-".join(map(str, layer_ids))
        fqns = [fqn for layer_id in layer_ids for fqn in layer_fqns[layer_id]]
        routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is per_expert
        )
        non_routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is not per_expert
        )
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed_fqns))
        if routed_fqns:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed_fqns)
            )
    # Muon is designed for matrix parameters; Moonlight uses AdamW for
    # non-matrix parameters such as RMSNorm, LM head, and embeddings. Expert
    # tensors below are batch-first stacks of matrices. See Sec. 2.2:
    # https://arxiv.org/abs/2502.16982
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)})\.weight|"
        rf"routed_experts\.(?:{'|'.join(expert_projections)})|"
        r"feed_forward\.(?:w13|w2)\.weight|"
        # Keep the 2D router gate on Muon: Moonlight Figure 4 reports its
        # SVD-entropy gain over AdamW is larger than for other matrix groups.
        r"moe\.router\.gate\.weight|"
        r"moe\.shared_experts\.(?:w13|w2)\.weight"
        r")$"
    )
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=muon_pattern,
                bucket_configs=tuple(bucket_configs),
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                lr=muon_lr,
                weight_decay=0.1,
                # Kimi K2 uses 0.2 * sqrt(max(rows, columns)) for
                # shape-consistent AdamW-scale updates instead of Muon's
                # original aspect-ratio scaling.
                adjust_lr_fn="match_rms_adamw",
            ),
            # The remaining parameters are embeddings, norms, biases, LM head,
            # and the vision tower.
            AdamW.Config(
                pattern=r".*",
                foreach=True,
                fused=False,
                **adamw_kwargs,
            ),
        ],
    )


def _align_dist_muon_expert_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Align routed-expert layouts with the final parallelism config.

    Config variants can replace the base recipe's parallelism before invoking
    ``__post_init__``. That parallelism decides whether routed experts use the
    1D ``dp_shard`` layout or the 2D ep/edp_shard layout, so their layouts have
    to be rebuilt here.
    """
    dist_muon_index = next(
        (
            index
            for index, config in enumerate(optimizer_config.optimizers)
            if isinstance(config, DistMuon.Config)
        ),
        None,
    )
    if dist_muon_index is None:
        return optimizer_config
    dist_muon_config = cast(
        DistMuon.Config,
        optimizer_config.optimizers[dist_muon_index],
    )
    compute_sharding_by_fqn = cast(
        dict[str, ComputeLayout],
        dist_muon_config.compute_sharding_by_fqn,
    )
    per_expert = _per_expert_compute_layout(parallelism)
    aligned_shardings = {}
    changed = False
    for fqn, compute_layout in compute_sharding_by_fqn.items():
        if ".moe.routed_experts." in fqn and compute_layout != per_expert:
            aligned_shardings[fqn] = per_expert
            changed = True
        else:
            aligned_shardings[fqn] = compute_layout
    if not changed:
        return optimizer_config

    optimizers = list(optimizer_config.optimizers)
    optimizers[dist_muon_index] = replace(
        dist_muon_config,
        compute_sharding_by_fqn=aligned_shardings,
    )
    return replace(
        optimizer_config,
        optimizers=optimizers,
    )


@dataclass(kw_only=True, slots=True)
class _KimiTrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optim.optimizer = _align_dist_muon_expert_compute_layouts(
            self.optim.optimizer,
            parallelism=self.parallelism,
        )
        # TODO(#3353): Support TP-produced _StridedShard layouts in DistMuon.
        if self.parallelism.tensor_parallel_degree > 1:
            # Fail during config parsing, before TP/FSDP creates _StridedShard
            # storage.
            raise ValueError(
                "Kimi DistMuon currently requires "
                "tensor_parallel_degree=1: tensor parallelism can produce "
                "unsupported _StridedShard parameter layouts."
            )
        # No PP gate: DistMuon is PP-safe. The one precondition -- every stage
        # must own at least one transformer layer, or its Muon pattern claims
        # nothing and OptimizersContainer rejects the empty param group -- needs
        # the stage count to approach the layer count. See README, Supported
        # Parallelisms.
