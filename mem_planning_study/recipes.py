# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipes for the memory-planning study.

DeepSeek V3 671B, MXFP8, Dist-MoE, GraphTrainer, FSDP256 x EP64 (no PP),
one microbatch per rank (no gradient accumulation), run under the pure fake
process-group backend so one GPU represents logical rank 0.
"""

import dataclasses
import os

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.transform import apply_transforms
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.moe import (
    MoE,
    RoundRobinTokenChoiceTopKRouter,
    TokenChoiceTopKRouter,
)
from torchtitan.models.deepseek_v3 import build_model_config

from torchtitan_recipes.graph_trainer.deepseek_v3 import _dist_moe_graph_config
from torchtitan_recipes.models.deepseek_v3 import (
    _dist_moe_runtime_config,
    _require_dist_moe,
    deepseek_v3_671b,
    deepseek_v3_mxfp8_linear_converter_config,
)

# Registers memplan_dump_graph_info_pass.
from mem_planning_study import graph_info_pass  # noqa: F401

SEQ_LEN = 4096
DP_SHARD = 256
EP = 64


def _dsv3_671b_mxfp8_dist_moe(*, cuda_graph: bool, steps: int, pp: int = 1):
    dist_moe = _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform

    config = deepseek_v3_671b(seq_len=SEQ_LEN)
    p = config.parallelism
    dp_shard = (DP_SHARD * 1) // pp
    p.data_parallel_shard_degree = dp_shard
    p.pipeline_parallel_degree = pp
    if pp > 1:
        # 61 layers + embedding + output weights = 63 -> 8 virtual stages of 8
        # layers -> PP2 x VPP4 with Interleaved1F1B; 8 microbatches per step
        # cover rank 0's full interleaved warmup.
        p.pipeline_parallel_schedule = "Interleaved1F1B"
        p.pipeline_parallel_layers_per_stage = 8
        p.num_pp_microbatches = int(os.environ.get("MEMPLAN_PP_MICROBATCHES", "8"))
    else:
        p.pipeline_parallel_layers_per_stage = None
    p.expert_parallel_degree = EP
    # GraphTrainer rejects deferred gradient reduction.
    p.fsdp_defer_gradient_reduction = False
    config.model = build_model_config(
        "671B",
        seq_len=SEQ_LEN,
        parallelism=p,
        converters=[deepseek_v3_mxfp8_linear_converter_config(include_lm_head=True)],
    )
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model)
    )
    # Local test tokenizer and dataset: token values do not affect memory, and
    # Dist-MoE arenas are capacity-sized, so routing does not change sizes.
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.dataloader = GrainDataLoader.Config(
        max_num_documents=32,
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=2.2e-4, moment_dtype="bfloat16")]
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = SEQ_LEN
    config.training.num_tokens_per_train_step = (
        dp_shard * SEQ_LEN * (p.num_pp_microbatches if pp > 1 else 1)
    )
    config.training.steps = int(os.environ.get("MEMPLAN_STEPS", steps))
    config.training.disable_accelerator_graphs = not cuda_graph
    config.precision.dtype = "float32"
    config.precision.param_dtype = "bfloat16"
    config.precision.reduce_dtype = "bfloat16"
    config.activation_remat = None
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    config.dist_moe_runtime = _dist_moe_runtime_config(scratch_capacity_factor=4.0)
    config.metrics.loggers = []
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config = apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )
    config = _dist_moe_graph_config(config)
    # Fake PG all-gathers leave non-local weight shards unset, so the learned
    # router ties and sends every token to experts 0-7, overflowing Dist-MoE
    # scratch. Route round-robin with exact balance instead (same as the
    # GraphTrainer integration tests); Dist-MoE buffers are capacity-sized, so
    # allocation sizes match a balanced production run.
    router_fields = dataclasses.fields(TokenChoiceTopKRouter.Config)
    for layer in config.model.layers:
        if isinstance(layer.feed_forward, MoE.Config):
            router = layer.feed_forward.router
            layer.feed_forward.router = RoundRobinTokenChoiceTopKRouter.Config(
                **{f.name: getattr(router, f.name) for f in router_fields}
            )
    # "none" = no recompute, "default" = SAC (recipe default), "full" = full
    # recompute (saves only layer outputs).
    config.compile.memory_policy = os.environ.get("MEMPLAN_MEMORY_POLICY", "default")
    if os.environ.get("MEMPLAN_DUMP_GRAPH", "1") == "1":
        config.compile.extra_passes = ["memplan_dump_graph_info_pass"]
    return config


def qwen3_30b_a3b_fp8_distmoe_fsdp8_ep8():
    """Qwen3 30B-A3B, MXFP8 attention linears + MXFP8 Dist-MoE experts, FSDP8 x EP8,
    GBS 512 x MBS 8 x 4096 (8 microbatches per step per rank), no recompute.

    Port of ``graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu`` from the
    agentic_compiler branch (torchtitan_recipes/graph_trainer/
    megatron_bridge_benchmarks.py), which matches Megatron-Bridge's
    ``qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config``. Differences: CUDA graphs
    off (every allocation goes through the allocator under study), dump/metrics
    settings for this study.
    """
    import dist_moe

    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.components.optim import LRSchedulersContainer, Optim
    from torchtitan.config import PrecisionConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.config.transform import MXFP8LinearConverter
    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.experiments.graph_trainer.configs import (
        GraphTrainerCompileConfig,
        to_graph_trainer_config,
    )
    from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
    from torchtitan.models.qwen3 import build_model_config as build_qwen3_model_config
    from torchtitan.trainer import Trainer

    from torchtitan_recipes.data import c4_document_capacity

    parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        expert_parallel_degree=8,
    )
    model_config = build_qwen3_model_config(
        "30B-A3B",
        seq_len=SEQ_LEN,
        parallelism=parallelism,
        converters=[MXFP8LinearConverter.Config(fqns=["attention"])],
    )
    router_fields = dataclasses.fields(TokenChoiceTopKRouter.Config)
    for _fqn, router, parent, attr in list(
        model_config.traverse(TokenChoiceTopKRouter.Config)
    ):
        setattr(
            parent,
            attr,
            RoundRobinTokenChoiceTopKRouter.Config(
                **{f.name: getattr(router, f.name) for f in router_fields}
            ),
        )
    num_tokens_per_microbatch = 8 * SEQ_LEN
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config)
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(num_tokens_per_microbatch),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", moment_dtype="bfloat16")]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            num_tokens_per_train_step=512 * SEQ_LEN,
            steps=int(os.environ.get("MEMPLAN_STEPS", "5")),
            # The source recipe captures CUDA graphs; off by default so every
            # allocation goes through the allocator under study.
            disable_accelerator_graphs=os.environ.get("MEMPLAN_CUDA_GRAPH", "0") != "1",
        ),
        precision=PrecisionConfig(
            dtype="float32", param_dtype="bfloat16", reduce_dtype="bfloat16"
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
    )
    config.dist_moe_runtime = DistMoeRuntime.Config(
        activation_slot_capacity_factor=1.0,
        scratch_capacity_factor=1.5,
    )
    config = apply_transforms(
        config,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                inplace_wgrad_accum=False,
                block_scaled_config=dist_moe.BlockScaledConfig(fast_math=True),
            )
        ],
    )
    for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
        experts.inplace_wgrad_accum = False
    config = to_graph_trainer_config(config, GraphTrainerQwen3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        memory_policy=os.environ.get("MEMPLAN_MEMORY_POLICY", "none")
    )
    config.metrics.loggers = []
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    if os.environ.get("MEMPLAN_DUMP_GRAPH", "1") == "1":
        config.compile.extra_passes = ["memplan_dump_graph_info_pass"]
    return config


def dsv3_debug_fp8_distmoe_fsdp2_ep2_real():
    """Real 2-GPU numerics/throughput check: DSV3 debugmodel, MXFP8, Dist-MoE,
    FSDP2 x EP2 (the GraphTrainer Dist-MoE integration-test layout), learned
    router, deterministic, no CUDA graphs (same setting for both allocators)."""
    from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
        graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8,
    )
    from torchtitan_recipes.tests.models.deepseek_v3 import (
        configure_dist_moe_fsdp2_ep2,
        dist_moe_fsdp2_ep2_parallelism,
    )

    config = configure_dist_moe_fsdp2_ep2(
        graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8(
            seq_len=128, parallelism=dist_moe_fsdp2_ep2_parallelism()
        )
    )
    config.training.steps = int(os.environ.get("MEMPLAN_STEPS", "20"))
    config.training.disable_accelerator_graphs = True
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.loggers = []
    config.metrics.log_freq = 1
    config.comm.trace_buf_size = 0
    return config


def dsv3_671b_fp8_distmoe_eager_alloc():
    """No CUDA graphs: every allocation goes through the caching allocator."""
    return _dsv3_671b_mxfp8_dist_moe(cuda_graph=False, steps=4)


def dsv3_671b_fp8_distmoe_pp2_vpp4():
    """PP2 x VPP4 (Interleaved1F1B) x FSDP128 x EP64, no CUDA graphs."""
    return _dsv3_671b_mxfp8_dist_moe(cuda_graph=False, steps=4, pp=2)


def dsv3_671b_fp8_distmoe_cudagraph():
    """Production setting: the joint graph is captured into a CUDA graph."""
    return _dsv3_671b_mxfp8_dist_moe(cuda_graph=True, steps=5)
