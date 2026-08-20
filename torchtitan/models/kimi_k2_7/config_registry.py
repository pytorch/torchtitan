# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from typing import cast

from torch.distributed.tensor import Shard

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
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
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekV3Attention
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from . import KIMI_K2_5_SPECIAL_TOKENS, KimiK25Model, model_registry


def _kimi_multimodal_dataloader(
    dataset: SingleDatasetConfig,
) -> GrainDataLoader.Config:
    processor = dataset.processor
    if not isinstance(processor, MultiModalProcessor.Config):
        raise ValueError("Kimi multimodal data requires MultiModalProcessor.Config")

    processor = MultiModalProcessor.Config(
        sample_processor=processor.sample_processor,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        min_pixels=65_536,
        max_pixels=16_777_216,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize_fn=resize_to_patch_budget,
        max_patches=16_384,
        max_patches_per_side=512,
        video_dir="",
        video_fps=2.0,
        video_min_frames=4,
        video_max_frames=768,
    )

    return GrainDataLoader.Config(
        dataset=replace(dataset, processor=processor),
        collator=MultiModalCollator.Config(
            max_images_per_batch=128,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
            patch_order="raster",
            build_mrope_positions=False,
        ),
    )


def kimi_k2_5_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    parallelism = ParallelismConfig(spmd_backend="spmd_types")
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K2_5_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=_kimi_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optimizer=_dist_muon_optimizer(
            model_spec,
            lr=8e-4,
            parallelism=parallelism,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def moonlight_16b_a3b() -> Trainer.Config:
    """Moonlight 16B-A3B: the text-only DeepSeekV3 sibling (no vision tower)."""
    model_spec = model_registry("moonlight-16B-A3B", attn_backend="flex")
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
        spmd_backend="spmd_types",
    )
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Moonlight-16B-A3B",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=_dist_muon_optimizer(
            model_spec,
            lr=3e-4,
            parallelism=parallelism,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def kimi_vl_a3b() -> Trainer.Config:
    """Kimi-VL A3B: Moonlight text tower + 2D MoonViT vision (image-text)."""
    model_spec = model_registry("Kimi-VL-A3B", attn_backend="flex")
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
        spmd_backend="spmd_types",
    )
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-VL-A3B",
        # Kimi-VL-A3B names the vision-start token <|media_start|>, whereas the
        # K2.5 family uses <|media_begin|>; override just that one entry.
        tokenizer=MultiModalTokenizer.Config(
            **{**KIMI_K2_5_SPECIAL_TOKENS, "vision_start_token": "<|media_start|>"}
        ),
        model_spec=model_spec,
        # Kimi-VL is a compatibility flavor; resizing intentionally follows
        # Kimi-K2.5 per-side scaling instead of legacy Kimi-VL's side rejection.
        dataloader=_kimi_multimodal_dataloader(MM_DATASETS["cc12m"]),
        optimizer=_dist_muon_optimizer(
            model_spec,
            lr=3e-4,
            parallelism=parallelism,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def kimi_k2_5() -> Trainer.Config:
    """Full Kimi K2.5 (~1T-total / ~32B-active)."""
    compile_config = CompileConfig(enable=True, components=["loss"])
    # The report uses BF16 compute; its FP8 path only compresses saved activations.
    model_spec = model_registry("Kimi-K2.5", attn_backend="flex")
    parallelism = ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
        expert_parallel_degree=8,
        spmd_backend="spmd_types",
    )
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-K2.5",
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optimizer=_dist_muon_optimizer(
            model_spec,
            lr=2.2e-4,
            parallelism=parallelism,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        compile=compile_config,
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
    # smaller than the EFSDP size, add balanced rank assignment only after
    # benchmarks show that the fixed nonempty EFSDP coordinates are a hotspot.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EFSDP.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        # EP splits the expert dimension first, then EFSDP repartitions each
        # EP-local expert domain, which reverses the storage-mesh axis order.
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EFSDP.value),
        },
    )


def _dist_muon_optimizer(
    model_spec: ModelSpec,
    *,
    lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    model_config = cast(KimiK25Model.Config, model_spec.model)
    attention = cast(DeepSeekV3Attention.Config, model_config.first_attention)
    owned = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
        },
    )
    per_query_head = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                block_size=(attention.qk_nope_head_dim + attention.qk_rope_head_dim),
            )
        },
    )
    per_key_value_head = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                block_size=attention.qk_nope_head_dim + attention.v_head_dim,
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
        "wkv_a": owned,
        "wkv_b": per_key_value_head,
        "wo": owned,
    }
    num_layers = len(model_config.layers)
    muon_kwargs = {
        "lr": lr,
        "weight_decay": 0.1,
        "foreach": False,
        # Kimi K2's MuonClip recipe uses 0.2 * sqrt(max(rows, columns))
        # for shape-consistent AdamW-scale updates instead of Muon's original
        # aspect-ratio scaling.
        "adjust_lr_fn": "match_rms_adamw",
    }
    adamw_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")

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
                    f"{prefix}.feed_forward.{projection}.weight": owned
                    for projection in ("w1", "w2", "w3")
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
                    f"{prefix}.moe.shared_experts.{projection}.weight": owned
                    for projection in ("w1", "w2", "w3")
                }
            )
        return shardings

    compute_sharding_by_fqn_per_layer = tuple(
        compute_shardings_for_layer(layer_id) for layer_id in range(num_layers)
    )
    compute_sharding_by_fqn = {
        fqn: compute_sharding
        for layer_compute_sharding_by_fqn in compute_sharding_by_fqn_per_layer
        for fqn, compute_sharding in layer_compute_sharding_by_fqn.items()
    }
    layer_bucket_fqns = tuple(
        tuple(layer_compute_sharding_by_fqn)
        for layer_compute_sharding_by_fqn in compute_sharding_by_fqn_per_layer
    )
    # Layer 0 has a much larger dense MLP, so keep it separate while amortizing
    # collective launch overhead across pairs of MoE layers.
    bucket_layer_ids = ((0,),) + tuple(
        tuple(range(first_layer_id, min(first_layer_id + 2, num_layers)))
        for first_layer_id in range(1, num_layers, 2)
    )
    bucket_fqns = tuple(
        tuple(fqn for layer_id in layer_ids for fqn in layer_bucket_fqns[layer_id])
        for layer_ids in bucket_layer_ids
    )
    bucket_configs_list = []
    for layer_ids, fqns in zip(bucket_layer_ids, bucket_fqns, strict=True):
        name = "layers." + "-".join(map(str, layer_ids))
        routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is per_expert
        )
        non_routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is not per_expert
        )
        bucket_configs_list.append(
            BucketConfig(
                name=name,
                patterns=non_routed_fqns,
            )
        )
        if routed_fqns:
            bucket_configs_list.append(
                BucketConfig(
                    name=f"{name}.routed-experts",
                    patterns=routed_fqns,
                )
            )
    bucket_configs = tuple(bucket_configs_list)
    # Muon is designed for matrix parameters; Moonlight uses AdamW for
    # non-matrix parameters such as RMSNorm, LM head, and embeddings. Expert
    # tensors below are batch-first stacks of matrices. See Sec. 2.2:
    # https://arxiv.org/abs/2502.16982
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)})\.weight|"
        rf"routed_experts\.inner_experts\.(?:{'|'.join(expert_projections)})|"
        r"feed_forward\.w[123]\.weight|"
        # Keep the 2D router gate on Muon: Moonlight Figure 4 reports its
        # SVD-entropy gain over AdamW is larger than for other matrix groups.
        r"moe\.router\.gate\.weight|"
        r"moe\.shared_experts\.w[123]\.weight"
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
            # The remaining parameters are embeddings, norms, biases, LM head,
            # and the vision tower.
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs=adamw_kwargs,
            ),
        ],
        optimizer_factory_kwargs_by_name={
            "DistMuon": {
                "bucket_configs": bucket_configs,
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
            }
        },
    )


def _align_dist_muon_expert_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Align routed-expert layouts with the final parallelism config.

    The registry builds compute layouts from the recipe's declared parallelism,
    but the CLI can still override ``expert_parallel_degree`` afterwards. That
    override decides whether routed experts use the 1D ``dp_shard`` layout or
    the 2D EP/EFSDP layout, so their layouts have to be rebuilt here.
    """
    # TODO: Remove this function once parallelism can no longer be overridden
    # from the CLI; the registry layouts are then already final.
    factory_kwargs_by_name = {
        name: dict(factory_kwargs)
        for name, factory_kwargs in (
            optimizer_config.optimizer_factory_kwargs_by_name.items()
        )
    }
    dist_muon_kwargs = factory_kwargs_by_name.get("DistMuon")
    if dist_muon_kwargs is None:
        return optimizer_config
    compute_sharding_by_fqn = cast(
        dict[str, ComputeLayout],
        dist_muon_kwargs["compute_sharding_by_fqn"],
    )
    per_expert = _per_expert_compute_layout(parallelism)
    aligned_shardings = {}
    changed = False
    for fqn, compute_layout in compute_sharding_by_fqn.items():
        if ".moe.routed_experts.inner_experts." in fqn and compute_layout != per_expert:
            aligned_shardings[fqn] = per_expert
            changed = True
        else:
            aligned_shardings[fqn] = compute_layout
    if not changed:
        return optimizer_config

    dist_muon_kwargs["compute_sharding_by_fqn"] = aligned_shardings
    return replace(
        optimizer_config,
        optimizer_factory_kwargs_by_name=factory_kwargs_by_name,
    )


@dataclass(kw_only=True, slots=True)
class _KimiTrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optimizer = _align_dist_muon_expert_compute_layouts(
            self.optimizer,
            parallelism=self.parallelism,
        )
        # TODO(#3353): Support TP-produced _StridedShard layouts in DistMuon.
        # TODO(#4102): Build DistMuon from PP stage-local parameter groups.
        if (
            self.parallelism.tensor_parallel_degree > 1
            or self.parallelism.pipeline_parallel_degree > 1
        ):
            # Fail during config parsing, before TP/FSDP creates _StridedShard
            # storage or PP constructs optimizers from stage-local parameters.
            raise ValueError(
                "Kimi DistMuon currently requires "
                "tensor_parallel_degree=1 and pipeline_parallel_degree=1: "
                "tensor parallelism can produce unsupported _StridedShard "
                "parameter layouts, and pipeline parallelism gives each stage "
                "only a subset of the optimizer's parameter-group patterns."
            )
