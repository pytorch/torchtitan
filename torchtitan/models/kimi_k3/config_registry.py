# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from typing import cast

from torch.distributed.tensor import Shard

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
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
from torchtitan.models.kimi_k2_7.config_registry import (
    _align_dist_muon_expert_compute_layouts,
    _per_expert_compute_layout,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, KimiK3Model, model_registry
from .model import KimiMLAAttention


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
    """Debugmodel with per-head Muon for all logical 2D matrices."""
    model_config = model_registry("debugmodel", seq_len=seq_len)
    parallelism = ParallelismConfig()
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
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
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
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def _dist_muon_optimizer(
    model_config: KimiK3Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    attention = cast(KimiMLAAttention.Config, model_config.first_attention)
    delta_attention = next(
        layer.delta_attention
        for layer in model_config.layers
        if layer.delta_attention is not None
    )
    dp_shard = MeshAxisName.DP_SHARD.value
    owned = ComputeLayout(shardings_by_mesh_axis={dp_shard: Owned()})

    def blocks_of(*num_rows: int) -> ComputeLayout:
        """A repeating BlockShard pattern with one Muon matrix per entry."""
        return ComputeLayout(
            shardings_by_mesh_axis={dp_shard: BlockShard(dim=0, block_sizes=num_rows)},
        )

    # MLA fuses several projections into one parameter; Kimi runs Newton-Schulz
    # per logical projection, so each fused block is split (see #4692).
    attention_shardings = {
        "wq_a": owned,
        # per head: [q_nope_h; q_rope_h]
        "wq_b": blocks_of(attention.qk_nope_head_dim, attention.qk_rope_head_dim),
        # one block: [kv_latent; k_rope]
        "wkv_a": blocks_of(attention.kv_lora_rank, attention.qk_rope_head_dim),
        # per head: [k_nope_h; v_h]
        "wkv_b": blocks_of(attention.qk_nope_head_dim, attention.v_head_dim),
        "gate": blocks_of(attention.v_head_dim),
        "wo": owned,
    }
    per_kda_head = blocks_of(delta_attention.head_dim)
    # forget_a is the shared low-rank down projection and output_proj mixes
    # heads back into the model dimension, so both are whole-matrix compute.
    delta_attention_shardings = {
        "q_proj": per_kda_head,
        "k_proj": per_kda_head,
        "v_proj": per_kda_head,
        "forget_a": owned,
        "forget_b": per_kda_head,
        "output_gate": per_kda_head,
        "output_proj": owned,
    }
    per_expert = _per_expert_compute_layout(parallelism)
    feed_forward_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={dp_shard: Shard(0)},
        ),
        "w2": owned,
    }
    expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")

    def compute_shardings_for_layer(layer_id: int) -> dict[str, ComputeLayout]:
        layer = model_config.layers[layer_id]
        prefix = f"layers.{layer_id}"
        shardings = {}
        if layer.attention is not None:
            shardings.update(
                {
                    f"{prefix}.attention.{projection}.weight": compute_sharding
                    for projection, compute_sharding in attention_shardings.items()
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.delta_attention.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (
                        delta_attention_shardings.items()
                    )
                }
            )
        if layer.feed_forward is not None:
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (feed_forward_shardings.items())
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.moe.routed_experts.inner_experts.{projection}": per_expert
                    for projection in expert_projections
                }
            )
            shardings[f"{prefix}.moe.router.gate.weight"] = owned
            shardings.update(
                {
                    f"{prefix}.moe.shared_experts.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (feed_forward_shardings.items())
                }
            )
            shardings.update(
                {
                    f"{prefix}.moe.{projection}.weight": owned
                    for projection in ("routed_down", "routed_up")
                }
            )
        return shardings

    num_layers = len(model_config.layers)
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
    muon_kwargs = {
        "lr": muon_lr,
        "weight_decay": 0.1,
        "foreach": False,
        "adjust_lr_fn": "match_rms_adamw",
    }
    adamw_kwargs = {
        "lr": adamw_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    # Muon is designed for matrix parameters; AdamW handles non-matrix
    # parameters (RMSNorm, embeddings, LM head, biases, A_log, dt_bias, the
    # depthwise convolutions) and the degenerate matrices: beta's rows are
    # per-head scalar matrices and the residual projections are
    # single-row [1, D], so Newton-Schulz would only rescale them.
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)})\.weight|"
        rf"delta_attention\.(?:{'|'.join(delta_attention_shardings)})\.weight|"
        rf"routed_experts\.inner_experts\.(?:{'|'.join(expert_projections)})|"
        r"feed_forward\.w(?:13|2)\.weight|"
        r"moe\.router\.gate\.weight|"
        r"moe\.shared_experts\.w(?:13|2)\.weight|"
        r"moe\.routed_(?:down|up)\.weight"
        r")$"
    )
    return OptimizersContainer.Config(
        implementation="foreach",
        param_groups=[
            ParamGroupConfig(
                pattern=muon_pattern,
                optimizer_name="DistMuon",
                optimizer_kwargs=muon_kwargs,
            ),
            # The remaining parameters are embeddings, norms, biases, the LM
            # head, the KDA scalars/convolutions, and the vision tower.
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs=adamw_kwargs,
            ),
        ],
        optimizer_factory_kwargs_by_name={
            "DistMuon": {
                "bucket_configs": tuple(bucket_configs),
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
            }
        },
    )


@dataclass(kw_only=True, slots=True)
class _KimiK3TrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optimizer = _align_dist_muon_expert_compute_layouts(
            self.optimizer,
            parallelism=self.parallelism,
        )
        # TODO(#3353): Support TP-produced _StridedShard layouts in DistMuon.
        uses_dist_muon = any(
            group.optimizer_name == "DistMuon" for group in self.optimizer.param_groups
        )
        if uses_dist_muon and self.parallelism.tensor_parallel_degree > 1:
            # Fail during config parsing, before TP/FSDP creates _StridedShard
            # storage. Recipes that need TP replace the optimizer.
            raise ValueError(
                "Kimi K3 DistMuon currently requires tensor_parallel_degree=1: "
                "tensor parallelism can produce unsupported _StridedShard "
                "parameter layouts."
            )
