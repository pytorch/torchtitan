# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 model configurations used by tests."""

from typing import Literal

from dist_moe import DistMoeBlockScaledConfig, DistMoeBlockScaledFormat

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader

from torchtitan.components.dist_moe import DistMoeBackendConfig
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
    DistMoeTransform,
    Float8GroupedLinearConverter,
    Float8LinearConverter,
    ModelConfigConverter,
    MXFP8GroupedLinearConverter,
    MXFP8LinearConverter,
    TokenDispatcherTransform,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher

from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def deepseek_v3_mxfp8_linear_converter_config(
    *, model_compile_enabled: bool = False, include_lm_head: bool = False
) -> MXFP8LinearConverter.Config:
    """Build the dense MXFP8 policy shared by eager and GraphTrainer configs.

    The KV up projection and FFN down projections have single-consumer inputs
    that are not saved elsewhere for backward, so their columnwise MXFP8
    representations replace BF16 storage. Shared-input and attention output
    projections use the conservative BF16 save format. This selection is based
    on activation ownership, not the activation-checkpointing policy.
    Checkpointing changes when the selected representation is recreated and how
    long it remains live.
    """
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


def _enable_dist_moe(
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
            format=DistMoeBlockScaledFormat.MXFP8_E4M3,
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


def deepseek_v3_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_config = build_model_config("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
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
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_debugmodel_mtp(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model = build_model_config("debugmodel", seq_len=seq_len, num_mtp_layers=1)
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(config.model),
        ),
    )
    return config


def deepseek_v3_debugmodel_mxfp8(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    # Quantize the MoE expert grouped GEMMs to MXFP8, plus the dense Linear
    # layers in attention, the shared experts, and the dense-layer feed-forward.
    # fqns is an include-list (substring match), so the MoE router gate
    # (moe.router.gate) and lm_head are left in bf16.
    # pad_multiple=128 is required by the CuTeDSL quantization kernel
    # on sm_100 (e.g. B200)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(),
            MXFP8GroupedLinearConverter.Config(
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_float8_grouped(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        converters=[Float8GroupedLinearConverter.Config()],
    )
    return config


def deepseek_v3_debugmodel_dist_moe_bf16(
    seq_len: int | None = None,
    *,
    device_scratch_capacity_factor: float = 1.0,
) -> Trainer.Config:
    """Build the debug DSV3 recipe with BF16 Dist-MoE experts."""
    return _enable_dist_moe(
        deepseek_v3_debugmodel(seq_len=seq_len),
        flavor="debugmodel",
        seq_len=seq_len,
        dtype="bf16",
        device_scratch_capacity_factor=device_scratch_capacity_factor,
    )


def deepseek_v3_debugmodel_dist_moe_mxfp8(
    seq_len: int | None = None,
    *,
    device_scratch_capacity_factor: float = 1.0,
) -> Trainer.Config:
    """Build the debug DSV3 recipe with MXFP8 Dist-MoE experts and linears."""
    return _enable_dist_moe(
        deepseek_v3_debugmodel(seq_len=seq_len),
        flavor="debugmodel",
        seq_len=seq_len,
        dtype="mxfp8",
        device_scratch_capacity_factor=device_scratch_capacity_factor,
    )


def deepseek_v3_debugmodel_hybridep(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
    )
    config.parallelism.expert_parallel_degree = 2
    return apply_transforms(
        config,
        [
            TokenDispatcherTransform(
                dispatcher=HybridEPTokenDispatcher,
                kwargs={"non_blocking_capacity_factor": 1.0},
            )
        ],
    )


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
    )


def deepseek_v3_16b_hybridep(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_16b(seq_len=seq_len)
    config.model = build_model_config(
        "16B",
        seq_len=seq_len,
        attn_backend="flex",
    )
    config.training.disable_cuda_graphs = False
    return apply_transforms(
        config,
        [
            TokenDispatcherTransform(
                dispatcher=HybridEPTokenDispatcher,
                kwargs={"non_blocking_capacity_factor": 1.0},
            )
        ],
    )


def deepseek_v3_16b_dist_moe_bf16(
    seq_len: int | None = None,
    *,
    device_scratch_capacity_factor: float = 4.0,
) -> Trainer.Config:
    """Build the DSV3 16B recipe with BF16 Dist-MoE experts."""
    return _enable_dist_moe(
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
    """Build the DSV3 16B recipe with MXFP8 Dist-MoE experts and linears."""
    return _enable_dist_moe(
        deepseek_v3_16b(seq_len=seq_len),
        flavor="16B",
        seq_len=seq_len,
        dtype="mxfp8",
        device_scratch_capacity_factor=device_scratch_capacity_factor,
    )


def configure_deepseek_v3_16b_dist_moe_local_4gpu(
    config: Trainer.Config,
) -> Trainer.Config:
    """Configure the shared two-row, four-GPU DistMoE performance workload."""
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the local DistMoE recipe requires GrainDataLoader")
    dataloader = config.dataloader
    num_fixed_rows = 2
    num_tokens_per_row = config.training.max_context_length
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        num_fixed_rows * num_tokens_per_row
    )
    config.training.steps = 10
    dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"],
        mask_document_boundaries=False,
    )
    dataloader.shuffle = False
    dataloader.repeat = True
    dataloader.max_num_documents = num_fixed_rows
    config.activation_checkpoint = None
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_symm_mem_scope = None
    config.debug.moe_force_load_balance = True

    for _, inner_attention, _, _ in config.model.traverse(VarlenInnerAttention.Config):
        inner_attention.fixed_length_rows = True

    config.training.num_tokens_per_train_step = (
        config.training.num_tokens_per_microbatch_per_dp_rank
        * config.parallelism.data_parallel_replicate_degree
        * config.parallelism.data_parallel_shard_degree
        * 16
    )
    return config


def deepseek_v3_16b_dist_moe_bf16_local_4gpu() -> Trainer.Config:
    """Build the matched eager four-GPU BF16 DistMoE configuration."""
    return configure_deepseek_v3_16b_dist_moe_local_4gpu(
        deepseek_v3_16b_dist_moe_bf16(
            seq_len=4096,
            device_scratch_capacity_factor=1.0,
        )
    )


def deepseek_v3_16b_dist_moe_mxfp8_local_4gpu() -> Trainer.Config:
    """Build the matched eager four-GPU MXFP8 DistMoE configuration."""
    return configure_deepseek_v3_16b_dist_moe_local_4gpu(
        deepseek_v3_16b_dist_moe_mxfp8(
            seq_len=4096,
            device_scratch_capacity_factor=1.0,
        )
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


def deepseek_v3_671b_float8(seq_len: int | None = None) -> Trainer.Config:
    config = deepseek_v3_671b(seq_len=seq_len)
    # Quantize the dense Linear layers and the MoE expert grouped GEMMs to
    # float8 (fp8). This requires torchao and is only supported on NVIDIA SM89+
    # or AMD MI300+; on other backends (e.g. Intel XPU) the converter raises at
    # build time, so use the plain deepseek_v3_671b config there.
    config.model = build_model_config(
        "671B",
        seq_len=seq_len,
        attn_backend="flex",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
            ),
            Float8GroupedLinearConverter.Config(),
        ],
    )
    return config


def deepseek_v3_671b_dist_moe_bf16(seq_len: int | None = None) -> Trainer.Config:
    """Build the DSV3 671B recipe with BF16 Dist-MoE experts."""
    return _enable_dist_moe(
        deepseek_v3_671b(seq_len=seq_len),
        flavor="671B",
        seq_len=seq_len,
        dtype="bf16",
        device_scratch_capacity_factor=4.0,
    )


def deepseek_v3_671b_dist_moe_mxfp8(seq_len: int | None = None) -> Trainer.Config:
    """Build the DSV3 671B recipe with MXFP8 Dist-MoE experts and linears."""
    return _enable_dist_moe(
        deepseek_v3_671b(seq_len=seq_len),
        flavor="671B",
        seq_len=seq_len,
        dtype="mxfp8",
        device_scratch_capacity_factor=4.0,
    )
