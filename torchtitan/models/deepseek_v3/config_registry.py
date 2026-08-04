# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor import Shard
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    assign_balanced_owners,
    BucketConfig,
)
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.distributed_optimizers.muon import Owned
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    BatchedMatrixComputeView,
    MuonComputeSharding,
)
from torchtitan.components.optimizer import (
    default_adamw,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry


def enable_fused_swiglu(config: Trainer.Config) -> None:
    # fused_swiglu.py registers two overrides (dense FeedForward + MoE grouped
    # experts); activate both by naming each factory.
    for override in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ):
        assert override not in config.override.imports
        config.override.imports.append(override)


def deepseek_v3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_debugmodel_mxfp8() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    # Quantize the MoE expert grouped GEMMs to MXFP8, plus the dense Linear
    # layers in attention, the shared experts, and the dense-layer feed-forward.
    # fqns is an include-list (substring match), so the MoE router gate
    # (moe.router.gate) and lm_head are left in bf16.
    # pad_multiple=128 is required by the CuTeDSL quantization kernel
    # on sm_100 (e.g. B200)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "shared_experts", "feed_forward"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_hybridep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_debugmodel_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_16b() -> Trainer.Config:
    model_spec = model_registry("16B", attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_16b_distributed_muon() -> Trainer.Config:
    """DSV3-16B with local-block and bucketed owner-compute Muon."""
    config = deepseek_v3_16b()
    owner_group_size = 8
    config.optimizer = _deepseek_v3_distributed_muon_optimizer(
        n_layers=27,
        num_matrices=16,
        wkv_a_matrix_shape=(576, 2048),
        owner_group_size=owner_group_size,
        lr=2.2e-4,
    )
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=owner_group_size,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=4,
        enable_sequence_parallel=False,
        spmd_backend="spmd_types",
    )
    return config


def _deepseek_v3_distributed_muon_optimizer(
    *,
    n_layers: int,
    num_matrices: int,
    wkv_a_matrix_shape: tuple[int, int],
    owner_group_size: int,
    lr: float,
) -> OptimizersContainer.Config:
    muon_kwargs = {
        "lr": lr,
        "weight_decay": 0.1,
        "fused": False,
        "foreach": False,
    }
    adamw_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
        "fused": False,
        "foreach": True,
    }
    param_groups = [
        ParamGroupConfig(
            pattern=r"attention\.wq\.weight$",
            optimizer_name="DistributedMuon",
            optimizer_kwargs={
                **muon_kwargs,
                "compute_sharding": MuonComputeSharding(
                    view_before_placement=BatchedMatrixComputeView(
                        num_matrices=num_matrices,
                        matrices_flattened_into_dim=0,
                    ),
                    placement=Shard(0),
                ),
            },
        ),
        ParamGroupConfig(
            pattern=r"attention\.wkv_a\.weight$",
            optimizer_name="DistributedMuon",
            optimizer_kwargs={
                **muon_kwargs,
                "compute_sharding": MuonComputeSharding(placement=Owned()),
            },
        ),
        ParamGroupConfig(
            pattern=r"attention\.wkv_b\.weight$",
            optimizer_name="DistributedMuon",
            optimizer_kwargs={
                **muon_kwargs,
                "compute_sharding": MuonComputeSharding(
                    view_before_placement=BatchedMatrixComputeView(
                        num_matrices=num_matrices,
                        matrices_flattened_into_dim=0,
                    ),
                    placement=Shard(0),
                ),
            },
        ),
    ]
    for projection in ("w1_EFD", "w2_EDF", "w3_EFD"):
        param_groups.append(
            ParamGroupConfig(
                pattern=rf"routed_experts\.inner_experts\.{projection}$",
                optimizer_name="DistributedMuon",
                optimizer_kwargs={
                    **muon_kwargs,
                    "compute_sharding": MuonComputeSharding(
                        placement=Shard(0)
                    ),
                },
            )
        )
    param_groups.append(
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name="AdamW",
            optimizer_kwargs=adamw_kwargs.copy(),
        )
    )

    def layer_fqns(layer_id: int) -> tuple[str, ...]:
        prefix = f"layers.{layer_id}"
        fqns = tuple(
            f"{prefix}.attention.{projection}.weight"
            for projection in ("wq", "wkv_a", "wkv_b")
        )
        if layer_id:
            fqns += tuple(
                f"{prefix}.moe.routed_experts.inner_experts.{projection}"
                for projection in ("w1_EFD", "w2_EDF", "w3_EFD")
            )
        return fqns

    layer_bucket_fqns = tuple(layer_fqns(layer_id) for layer_id in range(n_layers))
    owner_rank_by_bucket = assign_balanced_owners(
        layer_bucket_fqns,
        {
            f"layers.{layer_id}.attention.wkv_a.weight": (
                wkv_a_matrix_shape[0] * wkv_a_matrix_shape[1]
            )
            for layer_id in range(n_layers)
        },
        num_ranks=owner_group_size,
    )
    bucket_configs = tuple(
        BucketConfig(
            name=f"layers.{layer_id}",
            patterns=fqns,
            owner_rank_by_fqn=owners,
            mesh_axis="dp_shard",
        )
        for layer_id, (fqns, owners) in enumerate(
            zip(layer_bucket_fqns, owner_rank_by_bucket, strict=True)
        )
    )
    return OptimizersContainer.Config(
        implementation="foreach",
        param_groups=param_groups,
        optimizer_init_kwargs={
            "DistributedMuon": {
                "bucket_configs": bucket_configs,
            }
        },
    )


def deepseek_v3_16b_hybridep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_16b_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_671b() -> Trainer.Config:
    model_spec = model_registry(
        "671B",
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=2.2e-4),
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
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b_float8() -> Trainer.Config:
    config = deepseek_v3_671b()
    # Quantize the dense Linear layers and the MoE expert grouped GEMMs to
    # float8 (fp8). This requires torchao and is only supported on NVIDIA SM89+
    # or AMD MI300+; on other backends (e.g. Intel XPU) the converter raises at
    # build time, so use the plain deepseek_v3_671b config there.
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "671B",
        attn_backend="flex",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=model_compile_enabled,
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled
            ),
        ],
    )
    return config
