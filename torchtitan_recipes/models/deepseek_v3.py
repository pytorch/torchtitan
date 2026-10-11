# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 model recipes."""

from importlib import import_module
from types import ModuleType
from typing import cast

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import Configurable, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    MXFP8LinearConverter,
    NVFP4GroupedLinearConverter,
    NVFP4LinearConverter,
    TokenDispatcherTransform,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.models.deepseek_v3 import build_model_config, DeepSeekV3Model
from torchtitan.quantization.nvfp4 import nvfp4_bf16_tail_fqns, nvfp4_ffn_submodules
from torchtitan.trainer import Trainer


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


def _dist_moe_runtime_config(*, scratch_capacity_factor: float) -> Configurable.Config:
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
    fqns = ["attention", "shared_experts", "feed_forward"]
    if include_lm_head:
        fqns.append("lm_head")
    return MXFP8LinearConverter.Config(
        fqns=fqns,
        linears_saving_inputs_for_backward_in_mxfp8=[
            "attention.wkv_b",
            "feed_forward.w2",
            "shared_experts.w2",
        ],
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
    )


def deepseek_v3_671b_dist_moe_bf16(seq_len: int = 4096) -> Trainer.Config:
    """Build the verified 671B BF16 Dist-MoE training recipe."""
    _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_671b(seq_len=seq_len)
    config.model = build_model_config(
        "671B",
        seq_len=seq_len,
        attn_backend="varlen",
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
    config.parallelism.data_parallel_shard_degree = 128
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    config.parallelism.num_pp_microbatches = 120
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.activation_checkpoint = None
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
    config.model = build_model_config(
        "671B",
        seq_len=seq_len,
        attn_backend="varlen",
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
    config.parallelism.data_parallel_shard_degree = 64
    config.parallelism.pipeline_parallel_degree = 4
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = 4
    config.parallelism.num_pp_microbatches = 240
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.activation_checkpoint = None
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
        attn_backend="flex",
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
                kwargs={"non_blocking_capacity_factor": 0.03125, "pad_multiple": 128},
            )
        ],
    )
    config.model.local_compile_regions = ["loss"]
    return config
