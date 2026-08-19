# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import cast

from torch.distributed.tensor import Shard

from torchtitan.components.checkpoint import CheckpointManager
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
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekV3Attention
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from . import KIMI_K2_5_SPECIAL_TOKENS, KimiK25Model, model_registry


def _mm_dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=128,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        patch_order="raster",
        resize_fn=resize_to_patch_budget,
        max_patches=16384,
        max_patches_per_side=512,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    )


def kimi_k2_5_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
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
        dataloader=_mm_dataloader("cc12m-test"),
        optimizer=_dist_muon_optimizer(model_spec, lr=8e-4),
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
        parallelism=ParallelismConfig(spmd_backend="spmd_types"),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def moonlight_16b_a3b() -> Trainer.Config:
    """Moonlight 16B-A3B: the text-only DeepSeekV3 sibling (no vision tower)."""
    model_spec = model_registry("moonlight-16B-A3B", attn_backend="flex")
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Moonlight-16B-A3B",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=_dist_muon_optimizer(model_spec, lr=3e-4),
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=8,
            spmd_backend="spmd_types",
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def kimi_vl_a3b() -> Trainer.Config:
    """Kimi-VL A3B: Moonlight text tower + 2D MoonViT vision (image-text)."""
    model_spec = model_registry("Kimi-VL-A3B", attn_backend="flex")
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
        dataloader=_mm_dataloader("cc12m"),
        optimizer=_dist_muon_optimizer(model_spec, lr=3e-4),
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=8,
            spmd_backend="spmd_types",
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
    )


def kimi_k2_5() -> Trainer.Config:
    """Full Kimi K2.5 (~1T-total / ~32B-active)."""
    compile_config = CompileConfig(enable=True, components=["loss"])
    # The report uses BF16 compute; its FP8 path only compresses saved activations.
    model_spec = model_registry("Kimi-K2.5", attn_backend="flex")
    return _KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Kimi-K2.5",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=_dist_muon_optimizer(model_spec, lr=2.2e-4),
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
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
            spmd_backend="spmd_types",
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=FullAC.Config(),
        compile=compile_config,
    )


def _dist_muon_optimizer(
    model_spec: ModelSpec,
    *,
    lr: float,
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
    per_expert = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Shard(0),
            MeshAxisName.EFSDP.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
    )
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
    bucket_configs = tuple(
        BucketConfig(
            name="layers." + "-".join(map(str, layer_ids)),
            patterns=fqns,
        )
        for layer_ids, fqns in zip(
            bucket_layer_ids,
            bucket_fqns,
            strict=True,
        )
    )
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


@dataclass(kw_only=True, slots=True)
class _KimiTrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
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
