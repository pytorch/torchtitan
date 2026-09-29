# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from typing import Any, cast

from torch.distributed.tensor import Shard

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from . import DeepSeekV4Model, model_registry
from .mtp import MTPLoss

_NSCoefficients = tuple[float, float, float]
_NSCoefficientSchedule = _NSCoefficients | tuple[_NSCoefficients, ...]
_FAST_NS: _NSCoefficients = (3.4445, -4.7750, 2.0315)
_STABLE_NS: _NSCoefficients = (2.0, -1.5, 0.5)
_DEFAULT_NS_COEFFICIENTS = (_FAST_NS,) * 8 + (_STABLE_NS,) * 2


def _per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        return ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
        )
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
    )


def _dist_muon_optimizer(
    model_config: DeepSeekV4Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
    muon_ns_steps: int,
    muon_ns_coefficients: _NSCoefficientSchedule,
) -> OptimizersContainer.Config:
    owned = ComputeLayout(
        shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()},
    )
    shared_expert_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
        ),
        "w2": owned,
    }
    attention_shardings = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_per_head_projections = ("wq_b", "wo_a")
    attention_projections = tuple(attention_shardings) + attention_per_head_projections
    expert_projections = ("w13", "w2")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    mtp_projections = ("e_proj", "h_proj")
    per_expert = _per_expert_compute_layout(parallelism)

    def compute_shardings_for_layer(
        prefix: str,
        layer_config: Any,
        *,
        include_mtp_projections: bool,
    ) -> dict[str, ComputeLayout]:
        attention = layer_config.attention
        shardings = {
            f"{prefix}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in attention_shardings.items()
        }
        shardings.update(
            {
                f"{prefix}.moe.shared_experts.{projection}.weight": compute_sharding
                for projection, compute_sharding in shared_expert_shardings.items()
            }
        )
        shardings[f"{prefix}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: BlockShard(
                    dim=0,
                    block_sizes=(attention.head_dim,),
                )
            },
        )
        shardings[f"{prefix}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: BlockShard(
                    dim=0,
                    block_sizes=(attention.wo_a.out_features,),
                )
            },
        )
        for projection in expert_projections:
            shardings[f"{prefix}.moe.routed_experts.{projection}.weight"] = per_expert
        shardings[f"{prefix}.moe.router.gate.weight"] = owned
        for module in hc_pre_modules:
            shardings[f"{prefix}.{module}.hc_fn"] = owned
        if attention.compressor is not None:
            for projection in compressor_projections:
                shardings[f"{prefix}.attention.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.compressor.ape"] = owned
        if include_mtp_projections:
            for projection in mtp_projections:
                shardings[f"{prefix}.{projection}.weight"] = owned
            shardings[f"{prefix}.hc_head.hc_fn"] = owned
        return shardings

    main_layer_shardings = tuple(
        compute_shardings_for_layer(
            f"layers.{layer_id}",
            layer_config,
            include_mtp_projections=False,
        )
        for layer_id, layer_config in enumerate(model_config.layers)
    )
    mtp_layer_shardings = tuple(
        compute_shardings_for_layer(
            f"mtp_layers.{layer_id}",
            layer_config,
            include_mtp_projections=True,
        )
        for layer_id, layer_config in enumerate(model_config.mtp_layers or ())
    )
    layer_shardings = main_layer_shardings + mtp_layer_shardings
    compute_sharding_by_fqn = {
        fqn: compute_sharding
        for shardings in layer_shardings
        for fqn, compute_sharding in shardings.items()
    }
    compute_sharding_by_fqn["hc_head.hc_fn"] = owned

    bucket_configs = []
    for prefix, shardings in (
        *(
            (f"layers.{layer_id}", shardings)
            for layer_id, shardings in enumerate(main_layer_shardings)
        ),
        *(
            (f"mtp_layers.{layer_id}", shardings)
            for layer_id, shardings in enumerate(mtp_layer_shardings)
        ),
    ):
        routed_fqns = tuple(
            fqn
            for fqn, compute_sharding in shardings.items()
            if compute_sharding is per_expert
        )
        non_routed_fqns = tuple(
            fqn
            for fqn, compute_sharding in shardings.items()
            if compute_sharding is not per_expert
        )
        bucket_configs.append(BucketConfig(name=prefix, patterns=non_routed_fqns))
        if routed_fqns:
            bucket_configs.append(
                BucketConfig(name=f"{prefix}.routed-experts", patterns=routed_fqns)
            )
    bucket_configs.append(BucketConfig(name="hc_head", patterns=("hc_head.hc_fn",)))

    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        r"attention\.compressor\."
        rf"(?:{'|'.join(compressor_projections)})\.weight|"
        r"attention\.compressor\.ape|"
        rf"moe\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"moe\.routed_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        r"moe\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn|"
        rf"(?:{'|'.join(mtp_projections)})\.weight|"
        r"hc_head\.hc_fn"
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
                adjust_lr_fn="match_rms_adamw",
                ns_steps=muon_ns_steps,
                ns_coefficients=muon_ns_coefficients,
            ),
            AdamW.Config(
                pattern=r".*",
                lr=adamw_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                foreach=True,
                fused=False,
            ),
        ],
    )


def _align_dist_muon_expert_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
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
class _DeepSeekV4TrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optimizer = _align_dist_muon_expert_compute_layouts(
            self.optimizer,
            parallelism=self.parallelism,
        )
        if self.parallelism.tensor_parallel_degree > 1:
            raise ValueError(
                "DeepSeek V4 DistMuon requires tensor_parallel_degree=1: "
                "tensor parallelism can produce unsupported _StridedShard "
                "parameter layouts."
            )


# TODO: Restore packed-document recipes once compression and index selection
# respect document boundaries (#4801). Remove max_num_documents=1 from all four
# recipes and restore num_tokens_per_microbatch_per_dp_rank to
# 8 * model_config.max_context_length in both debug recipes.


def deepseek_v4_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    model_config = model_registry("debugmodel", enable_sp=True, seq_len=seq_len)
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    return _DeepSeekV4TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=1,
        ),
        optimizer=_dist_muon_optimizer(
            model_config,
            muon_lr=8e-4,
            adamw_lr=8e-4,
            parallelism=parallelism,
            muon_ns_steps=muon_ns_steps,
            muon_ns_coefficients=muon_ns_coefficients,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_checkpoint=None,
        compile=None,
        checkpointer=None,
    )


def deepseek_v4_mtp_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    model_config = model_registry(
        "debugmodel", enable_sp=True, seq_len=seq_len, n_mtp_layers=1
    )
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    return _DeepSeekV4TrainerConfig(
        loss=MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=1,
        ),
        optimizer=_dist_muon_optimizer(
            model_config,
            muon_lr=8e-4,
            adamw_lr=8e-4,
            parallelism=parallelism,
            muon_ns_steps=muon_ns_steps,
            muon_ns_coefficients=muon_ns_coefficients,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_checkpoint=None,
        compile=None,
        checkpointer=None,
    )


def deepseek_v4_flash(
    seq_len: int | None = None,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    model_config = model_registry("deepseek_v4_flash", enable_sp=True, seq_len=seq_len)
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    return _DeepSeekV4TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=1,
        ),
        optimizer=_dist_muon_optimizer(
            model_config,
            muon_lr=8e-4,
            adamw_lr=8e-4,
            parallelism=parallelism,
            muon_ns_steps=muon_ns_steps,
            muon_ns_coefficients=muon_ns_coefficients,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_checkpoint=None,
        compile=None,
        checkpointer=None,
    )


def deepseek_v4_pro(
    seq_len: int | None = None,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    model_config = model_registry("deepseek_v4_pro", enable_sp=True, seq_len=seq_len)
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    return _DeepSeekV4TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=1,
        ),
        optimizer=_dist_muon_optimizer(
            model_config,
            muon_lr=8e-4,
            adamw_lr=8e-4,
            parallelism=parallelism,
            muon_ns_steps=muon_ns_steps,
            muon_ns_coefficients=muon_ns_coefficients,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_checkpoint=None,
        compile=None,
        checkpointer=None,
    )
