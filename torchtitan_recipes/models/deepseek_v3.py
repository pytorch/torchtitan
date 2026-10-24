# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 model recipes."""

import math
from dataclasses import fields, replace
from functools import partial
from importlib import import_module
from types import ModuleType
from typing import Any, cast, TYPE_CHECKING

import grain.python as grain
import torch
import torch.nn as nn
from torchtitan.components.checkpointer import CheckpointManager

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
    SourceConfig,
    TextCollator,
)
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.validator import Validator
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    MXFP8LinearConverter,
    NVFP4GroupedLinearConverter,
    NVFP4LinearConverter,
    TokenDispatcherTransform,
    transform_model_config_,
)
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common import Embedding, Linear, RMSNorm, Sigmoid
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import MoE, RoundRobinTokenChoiceTopKRouter
from torchtitan.models.common.moe.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.models.deepseek_v3 import build_model_config, DeepSeekV3Model
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.models.deepseek_v3.pipeline import MTPGradientClipper
from torchtitan.quantization.nvfp4 import nvfp4_bf16_tail_fqns, nvfp4_ffn_submodules
from torchtitan.trainer import Trainer

from torchtitan_recipes.overrides.dsv3_mlperf import UnbiasedTopKLoadBalanceLoss

if TYPE_CHECKING:
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime


def _mlperf_dsv3_model_policy(model, *, performance: bool) -> None:
    """Bind initialization, unbiased sequence balancing, and router precision."""
    output_suffixes = ("attention.wo", "feed_forward.w2", "shared_experts.w2")
    base_init = partial(nn.init.normal_, std=0.006)
    output_init = partial(nn.init.normal_, std=0.006 / math.sqrt(2 * len(model.layers)))
    for _, embedding, _, _ in model.traverse(Embedding.Config):
        embedding.param_init = {"weight": base_init}
    for config_type in (Linear.Config, GroupedLinear.Config):
        for fqn, linear, _, _ in model.traverse(config_type):
            linear.param_init = {
                "weight": output_init if fqn.endswith(output_suffixes) else base_init,
                "bias": nn.init.zeros_,
            }
    for _, norm, _, _ in model.traverse(RMSNorm.Config):
        norm.param_init = {"weight": nn.init.ones_}
    for _, moe, _, _ in model.traverse(MoE.Config):
        moe.center_expert_bias_update = False
        gate = moe.router.gate
        gate_kwargs: dict[str, Any] = {
            field.name: getattr(gate, field.name) for field in fields(Linear.Config)
        }
        gate_kwargs["inplace_wgrad_accum"] = False
        # MLPerf uses BF16 logits and backward GEMMs, not HiMidLo's FP32 path.
        moe.router.gate = Linear.Config(**gate_kwargs)
        moe.router.score_func = Sigmoid.Config(compute_dtype=torch.float32)
        moe.router.routing_weight_dtype = torch.bfloat16
        moe.router.aux_loss = UnbiasedTopKLoadBalanceLoss.Config(
            coeff=0.01, top_k=moe.router.top_k
        )
        if performance:
            moe.router = RoundRobinTokenChoiceTopKRouter.Config(
                **{
                    field.name: getattr(moe.router, field.name)
                    for field in fields(RoundRobinTokenChoiceTopKRouter.Config)
                }
            )


def _mlperf_dsv3_loader(
    source: SourceConfig, *, microbatches: int, readers: int
) -> GrainDataLoader.Config:
    """Consume frozen aligned rows directly, without tokenization or packing."""
    return GrainDataLoader.Config(
        dataset=SingleDatasetConfig(source=source),
        collator=TextCollator.Config(),
        max_num_documents=1,
        shuffle=False,
        map_dataset_dp_sharding="strided",
        read_options=grain.ReadOptions(
            num_threads=readers, prefetch_buffer_size=2 * readers
        ),
        num_prefetch_microbatches=microbatches,
        num_mtp_layers=1,
    )


def _deepseek_v3_671b_mtp1_mlperf(
    *,
    pp_degree: int,
    training_source: SourceConfig,
    validation_source: SourceConfig,
    hf_assets_path: str,
    performance: bool,
) -> Trainer.Config:
    """Compose the full MLPerf contract and its forced-performance variant."""
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    dist_moe = _require_dist_moe()
    dp_degree = 256 // pp_degree
    microbatches = 15360 // dp_degree
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=pp_degree,
        data_parallel_shard_degree=dp_degree,
        expert_parallel_degree=64,
        pipeline_parallel_schedule="Interleaved1F1B",
        num_pp_microbatches=microbatches,
        pipeline_parallel_max_outstanding_sends=16,
        pp_max_unsharded_active_stages=16 // pp_degree,
        fsdp_reshard_after_forward="never",
        fsdp_defer_gradient_reduction=True,
        pipeline_parallel_module_fqns_per_model_part=[
            ["tok_embeddings", *[f"layers.{i}" for i in range(4)]],
            *[
                [f"layers.{i}" for i in range(start, start + 4)]
                for start in range(4, 60, 4)
            ],
            ["layers.60", "norm", "lm_head", "mtp_layers.0", "tok_embeddings"],
        ],
    )
    model = build_model_config(
        "671B",
        seq_len=4096,
        parallelism=parallelism,
        num_mtp_layers=1,
        attn_backend="varlen",
    )
    _mlperf_dsv3_model_policy(model, performance=performance)
    model.coalesce_lm_head = True
    converter = deepseek_v3_mxfp8_linear_converter_config(include_lm_head=True)
    model = converter.build().convert(model)
    model = cast(
        DeepSeekV3Model.Config,
        transform_model_config_(
            model,
            [
                DistMoeTransform(
                    expert_precision="mxfp8",
                    block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
                )
            ],
        ),
    )
    runtime = _dist_moe_runtime_config(
        scratch_capacity_factor=1.0 if performance else 32.0
    )
    runtime.activation_slot_capacity_factor = 1.0 if performance else None
    common_optimizer = AdamW.Config(
        pattern=r".*",
        lr=2.3238e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
        moment_dtype="bfloat16",
        weight_decay=0.033,
    )
    config = Trainer.Config(
        model=model,
        hf_assets_path=hf_assets_path,
        loss=ChunkedLossWrapper.Config(
            num_chunks=1,
            validation_only=True,
            loss_fn=MTPLoss.Config(global_vocab_size=129280, mtp_scale=0.1),
        ),
        dataloader=_mlperf_dsv3_loader(
            training_source, microbatches=microbatches, readers=16
        ),
        validator=Validator.Config(
            freq=1,
            steps=1,
            global_batch_size=1024,
            dataloader=_mlperf_dsv3_loader(
                validation_source, microbatches=2, readers=4
            ),
        ),
        optim=Optim.Config(
            enable_cuda_graph=True,
            gradient_clipper=MTPGradientClipper.Config(),
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    replace(
                        common_optimizer,
                        pattern=r"(?:^|\.)(?:[^.]*norm\.weight|bias)$",
                        weight_decay=0.0,
                    ),
                    common_optimizer,
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=4,
                warmup_start_factor=0.0,
                total_steps=11996,
                decay_type="cosine",
                min_lr_factor=5e-6 / 2.3238e-5,
            ),
        ),
        parallelism=parallelism,
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4096,
            num_tokens_per_train_step=15360 * 4096,
            max_context_length=4096,
            steps=60,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
            disable_cuda_graphs=False,
            cuda_graph_warmup_steps=1,
        ),
        activation_remat=None,
        dist_moe_runtime=runtime,
        checkpointer=CheckpointManager.Config(
            initial_load_path=hf_assets_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            load_only=True,
        ),
    )
    config.debug.seed = 14536
    config.metrics.log_freq = 1
    config.override.imports = [
        "torchtitan_recipes.overrides.dense_causal_sdpa.dense_causal_fused_mla",
        "torchtitan_recipes.overrides.dense_causal_sdpa.unmasked_text_rows",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    return config


def deepseek_v3_671b_mtp1_pp2_mlperf(
    *,
    training_source: SourceConfig,
    validation_source: SourceConfig,
    hf_assets_path: str,
) -> Trainer.Config:
    """Build PP2/VPP8/DP128/EP64 training and fixed1024-row validation.

    Pass indexed source configs for the frozen already-aligned token rows and
    the warmed Hugging Face assets directory. Private storage adapters remain
    outside TorchTitan and implement its ordinary SourceConfig contract.
    """
    return _deepseek_v3_671b_mtp1_mlperf(
        pp_degree=2,
        training_source=training_source,
        validation_source=validation_source,
        hf_assets_path=hf_assets_path,
        performance=False,
    )


def deepseek_v3_671b_mtp1_pp4_mlperf(
    *,
    training_source: SourceConfig,
    validation_source: SourceConfig,
    hf_assets_path: str,
) -> Trainer.Config:
    """Build PP4/VPP4/DP64/EP64 with the same MLPerf objective/data contract."""
    return _deepseek_v3_671b_mtp1_mlperf(
        pp_degree=4,
        training_source=training_source,
        validation_source=validation_source,
        hf_assets_path=hf_assets_path,
        performance=False,
    )


def deepseek_v3_671b_mtp1_pp2_mlperf_perf(
    *,
    training_source: SourceConfig,
    validation_source: SourceConfig,
    hf_assets_path: str,
) -> Trainer.Config:
    """Build PP2's exact round-robin/CF1/max-useful performance variant."""
    return _deepseek_v3_671b_mtp1_mlperf(
        pp_degree=2,
        training_source=training_source,
        validation_source=validation_source,
        hf_assets_path=hf_assets_path,
        performance=True,
    )


def deepseek_v3_671b_mtp1_pp4_mlperf_perf(
    *,
    training_source: SourceConfig,
    validation_source: SourceConfig,
    hf_assets_path: str,
) -> Trainer.Config:
    """Build PP4's exact round-robin/CF1/max-useful performance variant."""
    return _deepseek_v3_671b_mtp1_mlperf(
        pp_degree=4,
        training_source=training_source,
        validation_source=validation_source,
        hf_assets_path=hf_assets_path,
        performance=True,
    )


def _require_dist_moe() -> ModuleType:
    """Import the optional Dist-MoE package for a selected Dist-MoE recipe."""
    try:
        return import_module("dist_moe")
    except ModuleNotFoundError as error:
        if error.name != "dist_moe":
            raise
        raise ModuleNotFoundError(
            "This recipe requires the optional dist_moe package. Install "
            "it from git+https://github.com/meta-pytorch/dist_moe.git@main "
            "before selecting it.",
            name="dist_moe",
        ) from error


def _dist_moe_runtime_config(
    *, scratch_capacity_factor: float
) -> "DistMoeRuntime.Config":
    """Build the rank-wide activation and scratch policy for Dist-MoE."""
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    return DistMoeRuntime.Config(
        activation_slot_capacity_factor=1.0,
        scratch_capacity_factor=scratch_capacity_factor,
    )


def deepseek_v3_mxfp8_linear_converter_config(
    *,
    include_lm_head: bool = False,
) -> MXFP8LinearConverter.Config:
    """Build the dense MXFP8 policy shared by eager and GraphTrainer recipes.

    KV up and FFN down projections have single-consumer inputs that no other
    operation retains for backward, so their columnwise MXFP8 representations
    replace duplicate BF16 storage. Shared-input and attention-output
    projections retain BF16 inputs. The language-model head is opt-in because
    some recipes intentionally keep logits on the high-precision path.
    """
    fqns = ["attention", "shared_experts", "feed_forward.w13", "feed_forward.w2"]
    if include_lm_head:
        fqns.append("lm_head")
    return MXFP8LinearConverter.Config(
        fqns=fqns,
        linears_saving_inputs_for_backward_in_mxfp8=[
            "attention.inner_attention.wkv_b",
            "feed_forward.w2",
            "shared_experts.w2",
        ],
    )


def deepseek_v3_671b(seq_len: int | None = None) -> Trainer.Config:
    parallelism = ParallelismConfig(
        pipeline_parallel_schedule="Interleaved1F1B",
        expert_parallel_degree=2,
    )
    model_config = build_model_config(
        "671B",
        seq_len=seq_len,
        parallelism=parallelism,
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
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def deepseek_v3_671b_dist_moe_bf16(seq_len: int = 4096) -> Trainer.Config:
    """Build the verified 671B BF16 Dist-MoE training recipe."""
    _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_671b(seq_len=seq_len)
    config.parallelism.data_parallel_shard_degree = 128
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    config.parallelism.num_pp_microbatches = 120
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.model = build_model_config(
        "671B",
        seq_len=seq_len,
        parallelism=config.parallelism,
    )
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model)
    )
    config.dataloader.max_num_documents = 32
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4, moment_dtype="bfloat16")]
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.model.max_context_length
    )
    config.training.num_tokens_per_train_step = 15360 * config.model.max_context_length
    config.training.disable_cuda_graphs = False
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "bfloat16"
    config.activation_remat = None
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    config.dist_moe_runtime = _dist_moe_runtime_config(scratch_capacity_factor=4.0)
    return apply_transforms(config, [DistMoeTransform()])


def deepseek_v3_671b_dist_moe_mxfp8(seq_len: int = 4096) -> Trainer.Config:
    """Build the verified 671B MXFP8 Dist-MoE training recipe."""
    dist_moe = _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_671b(seq_len=seq_len)
    config.parallelism.data_parallel_shard_degree = 64
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    config.parallelism.num_pp_microbatches = 240
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.model = build_model_config(
        "671B",
        seq_len=seq_len,
        parallelism=config.parallelism,
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(include_lm_head=True),
        ],
    )
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model)
    )
    config.dataloader.max_num_documents = 32
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4, moment_dtype="bfloat16")]
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.model.max_context_length
    )
    config.training.num_tokens_per_train_step = 15360 * config.model.max_context_length
    config.training.disable_cuda_graphs = False
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "bfloat16"
    config.training.mixed_precision_reduce = "bfloat16"
    config.activation_remat = None
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    config.dist_moe_runtime = _dist_moe_runtime_config(scratch_capacity_factor=4.0)
    return apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )


def deepseek_v3_671b_nvfp4_ffn_mxfp8_attn(
    bf16_tail_fraction: float = 0.0, *, seq_len: int | None = None
) -> Trainer.Config:
    config = deepseek_v3_671b(seq_len=seq_len)
    model_config = cast(DeepSeekV3Model.Config, config.model)
    layer_fqns = nvfp4_bf16_tail_fqns(len(model_config.layers), bf16_tail_fraction)
    config.model = build_model_config(
        "671B",
        seq_len=seq_len,
        parallelism=config.parallelism,
        converters=[
            NVFP4LinearConverter.Config(
                fqns=[
                    f"{layer}{submodule}"
                    for layer in layer_fqns
                    for submodule in nvfp4_ffn_submodules(model_config.layers)
                ],
            ),
            NVFP4GroupedLinearConverter.Config(
                fqns=layer_fqns,
            ),
            MXFP8LinearConverter.Config(
                fqns=["attention.wq", "attention.wo"],
            ),
        ],
    )
    config = apply_transforms(
        config,
        [
            TokenDispatcherTransform(
                dispatcher=HybridEPTokenDispatcher,
                # Each EP rank dispatches one CP x TP shard of a microbatch.
                num_max_tokens_per_rank=(
                    config.training.num_tokens_per_microbatch_per_dp_rank
                    // (
                        config.parallelism.context_parallel_degree
                        * config.parallelism.tensor_parallel_degree
                    )
                ),
                kwargs={"non_blocking_capacity_factor": 0.03125, "pad_multiple": 128},
            )
        ],
    )
    config.model.local_compile_regions = ["loss"]
    return config
