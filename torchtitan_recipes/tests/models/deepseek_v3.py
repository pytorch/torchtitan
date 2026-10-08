# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 model configurations used by tests."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
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
    MXFP8GroupedLinearConverter,
    MXFP8LinearConverter,
    NVFP4GroupedLinearConverter,
    NVFP4LinearConverter,
    TokenDispatcherTransform,
)
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher

from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.quantization.nvfp4 import nvfp4_bf16_tail_fqns, nvfp4_ffn_submodules
from torchtitan.trainer import Trainer

from torchtitan_recipes.models.deepseek_v3 import (
    _dist_moe_runtime_config,
    _require_dist_moe,
    deepseek_v3_mxfp8_linear_converter_config,
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
    num_mtp_layers = 1
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel", seq_len=seq_len, num_mtp_layers=num_mtp_layers
    )
    config.dataloader.num_mtp_layers = num_mtp_layers
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


def deepseek_v3_debugmodel_dist_moe_bf16(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    scratch_capacity_factor: float = 1.0,
) -> Trainer.Config:
    """Build the debug BF16 Dist-MoE test recipe."""
    _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend="varlen",
    )
    config.dataloader.max_num_documents = 512
    config.training.mixed_precision_reduce = "bfloat16"
    config.dist_moe = _dist_moe_runtime_config(
        scratch_capacity_factor=scratch_capacity_factor
    )
    return apply_transforms(config, [DistMoeTransform()])


def deepseek_v3_debugmodel_dist_moe_mxfp8(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    scratch_capacity_factor: float = 1.0,
) -> Trainer.Config:
    """Build the debug MXFP8 Dist-MoE test recipe."""
    dist_moe = _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_debugmodel(seq_len=seq_len)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend="varlen",
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(include_lm_head=True),
        ],
    )
    config.dataloader.max_num_documents = 512
    config.training.mixed_precision_reduce = "bfloat16"
    config.dist_moe = _dist_moe_runtime_config(
        scratch_capacity_factor=scratch_capacity_factor
    )
    return apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )


def deepseek_v3_16b_dist_moe_bf16(seq_len: int | None = None) -> Trainer.Config:
    """Build the 16B BF16 Dist-MoE test recipe."""
    _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_16b(seq_len=seq_len)
    config.model = build_model_config(
        "16B",
        seq_len=seq_len,
        attn_backend="varlen",
    )
    config.dataloader.max_num_documents = 512
    config.training.mixed_precision_reduce = "bfloat16"
    config.dist_moe = _dist_moe_runtime_config(scratch_capacity_factor=4.0)
    return apply_transforms(config, [DistMoeTransform()])


def deepseek_v3_16b_dist_moe_mxfp8(seq_len: int | None = None) -> Trainer.Config:
    """Build the 16B MXFP8 Dist-MoE test recipe."""
    dist_moe = _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_16b(seq_len=seq_len)
    config.model = build_model_config(
        "16B",
        seq_len=seq_len,
        attn_backend="varlen",
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(include_lm_head=True),
        ],
    )
    config.dataloader.max_num_documents = 512
    config.training.mixed_precision_reduce = "bfloat16"
    config.dist_moe = _dist_moe_runtime_config(scratch_capacity_factor=4.0)
    return apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
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


def deepseek_v3_debugmodel_nvfp4_ffn_mxfp8_attn(
    bf16_tail_fraction: float = 0.0,
    *,
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=seq_len)
    model_config = config.model
    layer_fqns = nvfp4_bf16_tail_fqns(len(model_config.layers), bf16_tail_fraction)
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
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
    config.model.local_compile_regions = ["loss"]
    return config


def deepseek_v3_16b_nvfp4_ffn_mxfp8_attn(
    bf16_tail_fraction: float = 0.0, *, seq_len: int | None = None
) -> Trainer.Config:
    config = deepseek_v3_16b(seq_len=seq_len)
    config.optim.lr_scheduler.warmup_steps = 200
    config.training.disable_cuda_graphs = False
    model_config = config.model
    layer_fqns = nvfp4_bf16_tail_fqns(len(model_config.layers), bf16_tail_fraction)
    config.model = build_model_config(
        "16B",
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
                kwargs={"non_blocking_capacity_factor": 0.1875, "pad_multiple": 128},
            )
        ],
    )
    config.model.local_compile_regions = ["loss"]
    return config
