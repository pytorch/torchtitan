# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer throughput benchmarks matched to Megatron-Bridge GB300 recipes.

Each recipe mirrors one MXFP8 GB300 row of the Megatron-Bridge 26.08.01
performance summary
(https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)
and its ``megatron.bridge.perf_recipes.<family>.gb300`` recipe at
Megatron-Bridge v0.6.2, the release that produced that summary. Every recipe
keeps the GPU count, global and micro batch size, sequence length,
forced-balanced routing, and absence of activation recompute. They run
without pipeline parallelism, so Megatron's PP x VP ranks become FSDP ranks
and each step accumulates GBS / (DP * MBS) microbatches. Each recipe
docstring lists its remaining differences. Compare tokens/sec/GPU: torchtitan
and Megatron count attention FLOPs differently.

Megatron captures every matched recipe as full-iteration CUDA graphs. Here:

    Recipe            Routed experts                    CUDA graphs
    DeepSeek V3 671B  MXFP8 Dist-MoE                    yes
    Qwen3 235B-A22B   MXFP8 Dist-MoE                    yes
    Qwen3 30B-A3B     MXFP8 Dist-MoE                    yes
    GPT-OSS 120B      BF16 grouped GEMM, all-to-all     no

Benchmarks fix the model shape, so they read the repository's test tokenizer
and C4 sample: token content does not change the cost of a step. The same
recipe runs with real or fake process groups, for example:

    NGPU=256 COMM_BACKEND=fake \\
    MODULE=torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks \\
    CONFIG=graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_256gpu ./run_train.sh

Dist-MoE recipes require the optional ``dist_moe`` package; see
``torchtitan/models/common/dist_moe/README.md``. ``MEGATRON_BRIDGE_BENCHMARKS.md``
next to this file compares each recipe with its Megatron counterpart.
"""

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
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
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
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
    MTPCrossEntropyLoss,
)
from torchtitan.models.gpt_oss import build_model_config as build_gpt_oss_model_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.models.deepseek_v3 import (
    deepseek_v3_mxfp8_linear_converter_config,
)

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


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_256gpu() -> GraphTrainer.Config:
    """Match ``deepseek_v3_pretrain_256gpu_gb300_fp8mx_config``.

    Megatron runs TP1 PP2 VP8 EP32 (DP 128, 32 microbatches per step); this
    recipe runs FSDP 256 EP64 (16 microbatches per step). Both use GBS 4096,
    MBS 1, one MTP layer, an MXFP8 LM head, BF16 Adam moments, and CUDA
    graphs.

    Without PP, every rank keeps all 61 layers' unsharded parameters and
    gradients for the whole step. EP64, which Megatron uses for the same model
    on 256 GB200 GPUs, halves the local experts so the step fits without
    recompute.
    """
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=256,
        expert_parallel_degree=64,
    )
    num_mtp_layers = 1
    mxfp8_linears = deepseek_v3_mxfp8_linear_converter_config(include_lm_head=True)
    # Megatron also runs the MTP embedding/hidden projection under MXFP8.
    mxfp8_linears.fqns.append("eh_proj")
    model_config = build_deepseek_v3_model_config(
        "671B",
        seq_len=_SEQ_LEN,
        parallelism=parallelism,
        num_mtp_layers=num_mtp_layers,
        converters=[mxfp8_linears],
    )
    config = _benchmark_config(
        model_config,
        parallelism=parallelism,
        micro_batch_size=1,
        global_batch_size=4096,
        moment_dtype="bfloat16",
    )
    config.loss = ChunkedLossWrapper.Config(
        loss_fn=MTPCrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        ),
    )
    config.dataloader.num_mtp_layers = num_mtp_layers
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    return _to_graph_trainer(
        _with_dist_moe_mxfp8(config), GraphTrainerDeepSeekV3Model.Config
    )


def graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu() -> GraphTrainer.Config:
    """Match ``qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config``.

    Megatron runs TP1 PP4 VP12 EP32 (DP 64, 64 microbatches per step); this
    recipe runs FSDP 256 EP32 (16 microbatches per step). Both use GBS 8192,
    MBS 2, a BF16 LM head, and CUDA graphs.
    """
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=256,
        expert_parallel_degree=32,
    )
    model_config = build_qwen3_model_config(
        "235B-A22B",
        seq_len=_SEQ_LEN,
        parallelism=parallelism,
        converters=[MXFP8LinearConverter.Config(fqns=["attention"])],
    )
    config = _benchmark_config(
        model_config,
        parallelism=parallelism,
        micro_batch_size=2,
        global_batch_size=8192,
    )
    return _to_graph_trainer(
        _with_dist_moe_mxfp8(config), GraphTrainerQwen3Model.Config
    )


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
