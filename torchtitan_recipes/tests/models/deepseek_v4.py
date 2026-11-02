# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4 model configurations used by tests."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    Optim,
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
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.deepseek_v4 import build_model_config, DeepSeekV4Model
from torchtitan.models.deepseek_v4.mtp import MTPCrossEntropyLoss
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

_NSCoefficients = tuple[float, float, float]
_NSCoefficientSchedule = _NSCoefficients | tuple[_NSCoefficients, ...]
# DeepSeek-V4 report, Section 2.4: hybrid Newton-Schulz iterations.
# https://arxiv.org/html/2606.19348v1#S2.SS4
_FAST_NS: _NSCoefficients = (3.4445, -4.7750, 2.0315)
_STABLE_NS: _NSCoefficients = (2.0, -1.5, 0.5)
_DEFAULT_NS_COEFFICIENTS = (_FAST_NS,) * 8 + (_STABLE_NS,) * 2


def _dist_muon_optimizer(
    model_config: DeepSeekV4Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    muon_ns_steps: int,
    muon_ns_coefficients: _NSCoefficientSchedule,
) -> OptimizersContainer.Config:
    dp_axis = MeshAxisName.DP_SHARD.value
    owned = ComputeLayout(
        shardings_by_mesh_axis={dp_axis: Owned()},
        compute_dtype=torch.bfloat16,
    )

    def blocks(*sizes: int) -> ComputeLayout:
        return ComputeLayout(
            shardings_by_mesh_axis={dp_axis: BlockShard(dim=0, block_sizes=sizes)},
            compute_dtype=torch.bfloat16,
        )

    shared_expert_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={dp_axis: Shard(0)},
            compute_dtype=torch.bfloat16,
        ),
        "w2": owned,
    }
    attention_shardings = {"wq_a": owned, "wo_b": owned}
    attention_per_head_projections = ("wq_b", "wkv", "wo_a")
    attention_projections = tuple(attention_shardings) + attention_per_head_projections
    expert_projections = ("w13", "w2")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    mtp_projections = ("e_proj", "h_proj")
    per_expert = ComputeLayout(
        shardings_by_mesh_axis={
            "dp_shard": Shard(0),
            "dp_shard_cp": Shard(0),
            "ep": Shard(0),
            "edp_shard": Shard(0),
        },
        shard_order_by_tensor_dim={0: ("ep", "edp_shard")},
        compute_dtype=torch.bfloat16,
    )

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
                f"{prefix}.feed_forward.shared_experts.{projection}.weight": compute_sharding
                for projection, compute_sharding in shared_expert_shardings.items()
            }
        )
        nope_dim = attention.head_dim - attention.rope_head_dim
        shardings[f"{prefix}.attention.wq_b.weight"] = blocks(
            nope_dim, attention.rope_head_dim
        )
        shardings[f"{prefix}.attention.wkv.weight"] = blocks(
            nope_dim, attention.rope_head_dim
        )
        shardings[f"{prefix}.attention.wo_a.weight"] = blocks(attention.o_lora_rank)
        for projection in expert_projections:
            shardings[
                f"{prefix}.feed_forward.routed_experts.{projection}.weight"
            ] = per_expert
        shardings[f"{prefix}.feed_forward.router.gate.weight"] = owned
        for module in hc_pre_modules:
            hc_mult = getattr(layer_config, module).hc_mult
            shardings[f"{prefix}.{module}.hc_fn"] = blocks(
                hc_mult, hc_mult, hc_mult * hc_mult
            )
        if attention.compressor is not None:
            shardings[f"{prefix}.attention.compressor.wkv.weight"] = blocks(
                nope_dim,
                attention.rope_head_dim,
                nope_dim,
                attention.rope_head_dim,
            )
            shardings[f"{prefix}.attention.compressor.wgate.weight"] = blocks(
                attention.head_dim
            )
        if attention.compressor_128 is not None:
            shardings[f"{prefix}.attention.compressor_128.wkv.weight"] = blocks(
                nope_dim, attention.rope_head_dim
            )
            shardings[f"{prefix}.attention.compressor_128.wgate.weight"] = owned
        if attention.indexer is not None:
            indexer = attention.indexer
            indexer_nope_dim = indexer.index_head_dim - indexer.rope_head_dim
            shardings[f"{prefix}.attention.indexer.wq_b.weight"] = blocks(
                indexer_nope_dim, indexer.rope_head_dim
            )
            shardings[f"{prefix}.attention.indexer.weights_proj.weight"] = owned
            shardings[f"{prefix}.attention.indexer.compressor.wkv.weight"] = blocks(
                indexer_nope_dim,
                indexer.rope_head_dim,
                indexer_nope_dim,
                indexer.rope_head_dim,
            )
            shardings[f"{prefix}.attention.indexer.compressor.wgate.weight"] = blocks(
                indexer.index_head_dim
            )
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

    # Keep layer 0 separate and pair subsequent layers, matching Kimi.
    bucket_layer_ids = [(0,)] + [
        tuple(range(first_layer_id, min(first_layer_id + 2, len(main_layer_shardings))))
        for first_layer_id in range(1, len(main_layer_shardings), 2)
    ]
    bucket_configs = []
    for prefix, shardings in (
        *(
            (
                "layers." + "-".join(map(str, layer_ids)),
                {
                    fqn: compute_sharding
                    for layer_id in layer_ids
                    for fqn, compute_sharding in main_layer_shardings[layer_id].items()
                },
            )
            for layer_ids in bucket_layer_ids
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

    # Additive gate-logit biases (ape) stay with AdamW.
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        r"attention\.compressor(?:_128)?\."
        rf"(?:{'|'.join(compressor_projections)})\.weight|"
        r"attention\.indexer\.(?:wq_b|weights_proj|compressor\.(?:wkv|wgate))\.weight|"
        rf"feed_forward\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"feed_forward\.routed_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        r"feed_forward\.router\.gate\.weight|"
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


@dataclass(kw_only=True, slots=True)
class _DeepSeekV4TrainerConfig(Trainer.Config):
    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        if (
            self.parallelism.tensor_parallel_degree > 1
            or self.parallelism.pipeline_parallel_degree > 1
        ) and any(
            isinstance(config, DistMuon.Config)
            for config in self.optim.optimizer.optimizers
        ):
            raise ValueError(
                "DeepSeek V4 DistMuon requires tensor_parallel_degree=1 and "
                "pipeline_parallel_degree=1: TP _StridedShard and PP stage-local "
                "parameter groups are not supported yet."
            )


def deepseek_v4_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig(
            expert_parallel_degree=1,
        )
    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=parallelism
    )
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
            # Fixed document capacity keeps the packed offsets sync-free.
            max_num_documents=64,
        ),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                muon_ns_steps=muon_ns_steps,
                muon_ns_coefficients=muon_ns_coefficients,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            # The stock all-to-all EP dispatcher syncs with the CPU, so it
            # cannot run under CUDA graphs.
            disable_accelerator_graphs=parallelism.expert_parallel_degree > 1,
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_remat=None,
        checkpointer=None,
    )


def deepseek_v4_debugmodel_fsdp4_ep4() -> Trainer.Config:
    config = deepseek_v4_debugmodel(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=4,
            expert_parallel_degree=4,
        )
    )
    config.training.steps = 1
    config.metrics.log_freq = 1
    config.training.disable_accelerator_graphs = True
    return config


def deepseek_v4_mtp_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    num_mtp_layers = 1
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    model_config = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        n_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
    )
    return _DeepSeekV4TrainerConfig(
        loss=MTPCrossEntropyLoss.Config(
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
            # Fixed document capacity keeps the packed offsets sync-free.
            max_num_documents=64,
            num_mtp_layers=num_mtp_layers,
        ),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                muon_ns_steps=muon_ns_steps,
                muon_ns_coefficients=muon_ns_coefficients,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_remat=None,
        checkpointer=None,
    )


def deepseek_v4_flash(
    seq_len: int | None = None,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    model_config = build_model_config(
        "deepseek_v4_flash", seq_len=seq_len, parallelism=parallelism
    )
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
            # Fixed document capacity keeps the packed offsets sync-free.
            max_num_documents=64,
        ),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                muon_ns_steps=muon_ns_steps,
                muon_ns_coefficients=muon_ns_coefficients,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_remat=None,
        checkpointer=None,
    )


def deepseek_v4_pro(
    seq_len: int | None = None,
    *,
    muon_ns_steps: int = 10,
    muon_ns_coefficients: _NSCoefficientSchedule = _DEFAULT_NS_COEFFICIENTS,
) -> Trainer.Config:
    parallelism = ParallelismConfig(expert_parallel_degree=1)
    model_config = build_model_config(
        "deepseek_v4_pro", seq_len=seq_len, parallelism=parallelism
    )
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
            # Fixed document capacity keeps the packed offsets sync-free.
            max_num_documents=64,
        ),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                muon_ns_steps=muon_ns_steps,
                muon_ns_coefficients=muon_ns_coefficients,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        activation_remat=None,
        checkpointer=None,
    )
