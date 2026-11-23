# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 and GPT-OSS benchmark configurations used to capture the fusion problems."""

import dataclasses

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.optim.optimizer import MomentDType
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, MXFP8LinearConverter
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.gpt_oss.model import GraphTrainerGptOssModel
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.moe import (
    RoundRobinTokenChoiceTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.gpt_oss import build_model_config as build_gpt_oss_model_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity

# Every matched Megatron-Bridge recipe trains 4K-token sequences.
_SEQ_LEN = 4096


def _use_round_robin_routing(model_config: BaseModel.Config) -> None:
    """Route with exact balance, like Megatron's moe_router_force_load_balancing."""
    router_fields = dataclasses.fields(TokenChoiceTopKRouter.Config)
    for _fqn, router, parent, attr in list(
        model_config.traverse(TokenChoiceTopKRouter.Config)
    ):
        assert parent is not None and isinstance(attr, str)
        setattr(
            parent,
            attr,
            RoundRobinTokenChoiceTopKRouter.Config(
                **{f.name: getattr(router, f.name) for f in router_fields}
            ),
        )


def _benchmark_config(
    model_config: BaseModel.Config,
    *,
    parallelism: ParallelismConfig,
    micro_batch_size: int,
    global_batch_size: int,
    moment_dtype: MomentDType = "parameter",
) -> Trainer.Config:
    """Build the shared Megatron-Bridge benchmark policy around a model."""
    _use_round_robin_routing(model_config)
    num_tokens_per_microbatch = micro_batch_size * _SEQ_LEN
    return Trainer.Config(
        # Chunking the LM head and loss avoids materializing FP32 logits, as
        # Megatron's fused cross entropy does.
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(num_tokens_per_microbatch),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        # The learning rate does not affect throughput. Megatron-Bridge
        # benchmarks run 50 iterations with 10 warmup iterations.
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", moment_dtype=moment_dtype)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            num_tokens_per_train_step=global_batch_size * _SEQ_LEN,
            steps=50,
            # The stock all-to-all dispatcher synchronizes with the host.
            disable_accelerator_graphs=True,
        ),
        # Megatron-Bridge benchmarks keep FP32 main weights and reduce BF16
        # gradients (grad_reduce_in_fp32=False).
        precision=PrecisionConfig(
            dtype="float32", param_dtype="bfloat16", reduce_dtype="bfloat16"
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
    )


def _with_dist_moe_mxfp8(config: Trainer.Config) -> Trainer.Config:
    """Replace routed experts with MXFP8 Dist-MoE."""
    import dist_moe

    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    config.dist_moe_runtime = DistMoeRuntime.Config(
        # Retain every expert intermediate, as Megatron does without recompute.
        activation_slot_capacity_factor=1.0,
        # Matches Megatron's moe_expert_rank_capacity_factor.
        scratch_capacity_factor=1.5,
    )
    config = apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                # GraphTrainer traces WGRAD as a functional graph output.
                inplace_wgrad_accum=False,
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )
    # Dist-MoE dispatch never synchronizes with the host, so capture the step
    # like Megatron's full-iteration CUDA graphs.
    config.training.disable_accelerator_graphs = False
    return config


def _to_graph_trainer(
    config: Trainer.Config, model_config_cls: type[BaseModel.Config]
) -> GraphTrainer.Config:
    graph_config = to_graph_trainer_config(config, model_config_cls)
    # Megatron's matched recipes recompute nothing.
    graph_config.compile = GraphTrainerCompileConfig(memory_policy="none")
    return graph_config


def qwen3_30b_a3b_dist_moe_mxfp8_8gpu() -> Trainer.Config:
    """Match ``qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config`` with the eager ``Trainer``.

    Both run EP8 without PP on 8 GPUs with GBS 512 and MBS 8 (8 microbatches
    per step), a BF16 LM head, and CUDA graphs. EP spans every rank, so FSDP
    cannot shard expert optimizer state; BF16 Adam moments (Megatron keeps
    FP32) leave memory headroom without recompute.

    The model's local compile regions plus ``moe_aux_loss`` compile the glue
    that GraphTrainer captures with the whole step. Dist-MoE fuses SwiGLU into
    its expert GEMMs, so no activation override applies.
    """
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        expert_parallel_degree=8,
    )
    model_config = build_qwen3_model_config(
        "30B-A3B",
        seq_len=_SEQ_LEN,
        parallelism=parallelism,
        converters=[MXFP8LinearConverter.Config(fqns=["attention"])],
    )
    config = _benchmark_config(
        model_config,
        parallelism=parallelism,
        micro_batch_size=8,
        global_batch_size=512,
        moment_dtype="bfloat16",
    )
    config = _with_dist_moe_mxfp8(config)
    config.model.local_compile.regions.append("moe_aux_loss")
    return config


def graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu() -> GraphTrainer.Config:
    """``qwen3_30b_a3b_dist_moe_mxfp8_8gpu`` with GraphTrainer."""
    return _to_graph_trainer(
        qwen3_30b_a3b_dist_moe_mxfp8_8gpu(), GraphTrainerQwen3Model.Config
    )


def gpt_oss_120b_mxfp8_64gpu() -> Trainer.Config:
    """Match ``gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config`` where supported,
    with the eager ``Trainer``.

    Both run EP16 without PP on 64 GPUs with GBS 1280 and MBS 4 (5
    microbatches per step) and MXFP8 attention projections and LM head.
    Megatron also runs MXFP8 experts and CUDA graphs. Here experts stay BF16
    behind the all-to-all dispatcher, whose host-synchronized split sizes rule
    out CUDA graphs, so this recipe runs without them.

    The model's local compile regions (including ``fused_binary_activation``
    for its clamped SwiGLU) plus ``moe_aux_loss`` compile the glue that
    GraphTrainer captures with the whole step.
    """
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=64,
        expert_parallel_degree=16,
    )
    model_config = build_gpt_oss_model_config(
        "120b",
        seq_len=_SEQ_LEN,
        parallelism=parallelism,
        # TODO: Quantize experts once Dist-MoE supports expert biases or the
        # torchao MXFP8 grouped GEMM accepts GPT-OSS's dim=2880, which is not a
        # multiple of its 128-element quantization tile.
        converters=[MXFP8LinearConverter.Config(fqns=["attention", "lm_head"])],
    )
    config = _benchmark_config(
        model_config,
        parallelism=parallelism,
        micro_batch_size=4,
        global_batch_size=1280,
    )
    config.model.local_compile.regions.append("moe_aux_loss")
    return config


def graph_trainer_gpt_oss_120b_mxfp8_64gpu() -> GraphTrainer.Config:
    """``gpt_oss_120b_mxfp8_64gpu`` with GraphTrainer."""
    return _to_graph_trainer(gpt_oss_120b_mxfp8_64gpu(), GraphTrainerGptOssModel.Config)
