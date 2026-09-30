# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 model recipes."""

from dataclasses import replace
from typing import Literal

from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.dist_moe import DistMoeBackendConfig, DistMoeRoutedExperts
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    convert_config_type,
    DistMoeTransform,
    ModelConfigConverter,
    MXFP8LinearConverter,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import build_model_config, RoundRobinDeepSeekV3Router
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer


def deepseek_v3_mxfp8_linear_converter_config(
    *, model_compile_enabled: bool = False, include_lm_head: bool = False
) -> MXFP8LinearConverter.Config:
    """Build the dense MXFP8 policy shared by eager and GraphTrainer configs."""
    fqns = ["attention", "shared_experts", "feed_forward"]
    if include_lm_head:
        fqns.append("lm_head")
    return MXFP8LinearConverter.Config(
        model_compile_enabled=model_compile_enabled,
        fqns=fqns,
        linears_saving_inputs_for_backward_in_mxfp8=[
            "attention.wkv_b",
            "feed_forward.w2",
            "shared_experts.w2",
        ],
    )


def enable_deepseek_v3_dist_moe(
    config: Trainer.Config,
    *,
    flavor: str,
    seq_len: int | None,
    dtype: Literal["bf16", "mxfp8"],
    device_scratch_capacity_factor: float,
) -> Trainer.Config:
    """Replace routed experts while preserving the base training recipe."""
    block_scaled = None
    converters: list[ModelConfigConverter.Config] = []
    if dtype == "mxfp8":
        converters.append(
            deepseek_v3_mxfp8_linear_converter_config(
                model_compile_enabled=True,
                include_lm_head=True,
            )
        )
        block_scaled = DistMoeBlockScaledConfig(
            format=BlockScaledFormat.MXFP8_E4M3,
            fast_math=True,
            pipeline="staged",
        )
    elif dtype != "bf16":
        raise ValueError(f"Unsupported DistMoE dtype {dtype!r}")
    config.model = build_model_config(
        flavor,
        seq_len=seq_len,
        attn_backend="varlen",
        converters=converters,
    )
    config.dataloader.max_num_documents = 512
    config.training.disable_cuda_graphs = False
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "bfloat16"
    for optimizer in config.optim.optimizer.optimizers:
        assert isinstance(optimizer, AdamW.Config)
        optimizer.moment_dtype = "bfloat16"
    config.compile = LocalCompileConfig(regions=["loss"])
    return apply_transforms(
        config,
        [
            DistMoeTransform(
                backend=DistMoeBackendConfig(
                    device_scratch_capacity_factor=device_scratch_capacity_factor,
                    saved_activation_buffer_bytes="maximum_useful",
                    vmm_total_scratch_capacity_factor=None,
                    vmm_prefetch=False,
                    block_scaled=block_scaled,
                )
            )
        ],
    )


def configure_deepseek_v3_round_robin_routing(
    config: Trainer.Config,
) -> Trainer.Config:
    """Use exact round-robin routing for a controlled performance workload."""
    for _, moe, _, _ in config.model.traverse(MoE.Config):
        router = convert_config_type(moe.router, RoundRobinDeepSeekV3Router)
        assert isinstance(router, RoundRobinDeepSeekV3Router.Config)
        moe.router = router
    return config


def deepseek_v3_16b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config("16B", seq_len=seq_len, attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=1000,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
        compile=LocalCompileConfig(),
    )


def deepseek_v3_16b_dist_moe_bf16(
    seq_len: int | None = None,
    *,
    device_scratch_capacity_factor: float = 4.0,
) -> Trainer.Config:
    """Build the DSV3 16B recipe with BF16 DistMoE experts."""
    return enable_deepseek_v3_dist_moe(
        deepseek_v3_16b(seq_len=seq_len),
        flavor="16B",
        seq_len=seq_len,
        dtype="bf16",
        device_scratch_capacity_factor=device_scratch_capacity_factor,
    )


def deepseek_v3_16b_dist_moe_mxfp8(
    seq_len: int | None = None,
    *,
    device_scratch_capacity_factor: float = 4.0,
) -> Trainer.Config:
    """Build the DSV3 16B recipe with MXFP8 DistMoE experts and linears."""
    return enable_deepseek_v3_dist_moe(
        deepseek_v3_16b(seq_len=seq_len),
        flavor="16B",
        seq_len=seq_len,
        dtype="mxfp8",
        device_scratch_capacity_factor=device_scratch_capacity_factor,
    )


def deepseek_v3_671b(seq_len: int | None = None) -> Trainer.Config:
    model_config = build_model_config(
        "671B",
        seq_len=seq_len,
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4)]
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
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
        compile=LocalCompileConfig(),
    )


def deepseek_v3_671b_dist_moe_bf16(seq_len: int | None = None) -> Trainer.Config:
    """Build the DSV3 671B recipe with BF16 DistMoE experts."""
    return enable_deepseek_v3_dist_moe(
        deepseek_v3_671b(seq_len=seq_len),
        flavor="671B",
        seq_len=seq_len,
        dtype="bf16",
        device_scratch_capacity_factor=4.0,
    )


def deepseek_v3_671b_dist_moe_mxfp8(seq_len: int | None = None) -> Trainer.Config:
    """Build the DSV3 671B recipe with MXFP8 DistMoE experts and linears."""
    return enable_deepseek_v3_dist_moe(
        deepseek_v3_671b(seq_len=seq_len),
        flavor="671B",
        seq_len=seq_len,
        dtype="mxfp8",
        device_scratch_capacity_factor=4.0,
    )


def _deepseek_v3_671b_16_stage_modules() -> list[list[str]]:
    """Build the asymmetric 16-stage split used by Sanket's PP2 run."""
    num_layers = 61
    num_dense_layers = 3
    stages = [
        ["tok_embeddings"] + [f"layers.{layer}" for layer in range(num_dense_layers)]
    ]
    next_layer = num_dense_layers
    for _ in range(14):
        stages.append(
            [f"layers.{layer}" for layer in range(next_layer, next_layer + 4)]
        )
        next_layer += 4
    stages.append(
        [f"layers.{layer}" for layer in range(next_layer, num_layers)]
        + ["norm", "lm_head"]
    )
    assert len(stages) == 16
    return stages


def deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu() -> Trainer.Config:
    """Build the current-stack adaptation of Sanket's 256-GPU PP2 topology.

    The source run used one MTP block, but current public TorchTitan does not
    support MTP with pipeline parallelism. This recipe retains its PP2/VPP8,
    DP128, EP64, expert-FSDP2, batch, precision, and memory configuration with
    MTP disabled. The current eager runtime also rejects the source run's
    in-place WGrad mode because GraphRuntime owns those accumulators.
    """
    config = deepseek_v3_671b_dist_moe_mxfp8(seq_len=4096)
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the Sanket-topology recipe requires GrainDataLoader")
    if not isinstance(config.dataloader.dataset, ConcatThenSplitPackingConfig):
        raise TypeError("the Sanket-topology recipe requires packed text data")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=config.dataloader.dataset.dataset,
        mask_document_boundaries=False,
    )
    config.dataloader.max_num_documents = 1
    config.dataloader.num_prefetch_microbatches = 64
    for _, inner_attention, _, _ in config.model.traverse(VarlenInnerAttention.Config):
        inner_attention.fixed_length_rows = True
    for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
        experts.backend = replace(
            experts.backend,
            device_scratch_capacity_factor=1.0,
        )
    configure_deepseek_v3_round_robin_routing(config)

    if not isinstance(config.loss, ChunkedLossWrapper.Config):
        raise TypeError("the Sanket-topology recipe requires ChunkedLossWrapper")
    config.loss = config.loss.loss_fn
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.training.steps = 40
    config.training.disable_cuda_graphs = False
    config.optim.lr_scheduler.warmup_steps = 2
    config.optim.lr_scheduler.total_steps = 40
    config.activation_checkpoint = None
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 128
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = None
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        _deepseek_v3_671b_16_stage_modules()
    )
    config.parallelism.num_pp_microbatches = 32
    config.parallelism.pp_max_unsharded_active_stages = 8
    config.parallelism.pp_num_unshard_lookahead_factor = "auto"
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_symm_mem_scope = None
    config.metrics.log_freq = 1
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.compile = LocalCompileConfig(regions=[])
    for override in (
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu_feed_forward",
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
    ):
        if override not in config.override.imports:
            config.override.imports.append(override)
    return config


def deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile() -> (
    Trainer.Config
):
    """Profile step 41 of the current-stack Sanket-topology configuration."""
    config = deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu()
    config.training.steps = 41
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=41,
        profiler_warmup=0,
        profiler_active=1,
        profiler_repeat=1,
        enable_memory_snapshot=True,
        memory_snapshot_freq=41,
    )
    return config
