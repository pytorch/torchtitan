# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAST-local adjustments for the two 256-GPU validation runs."""

import gc
from dataclasses import dataclass, replace

import torch

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.config.transform import convert_config_type
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    SPMDGradientAccumulationConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import MTPDecoder, MTPLoss
from torchtitan.models.deepseek_v3.flavors import _build_mtp_layers
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer
from torchtitan_recipes.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8 as _base_chien,
)
from torchtitan_recipes.models.deepseek_v3 import (
    deepseek_v3_671b_dist_moe_mxfp8 as _base_chien_eager,
)
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_16b_dist_moe_mxfp8 as _base_local_cat_verification,
)


class _RoundRobinDeepSeekV3Router(DeepSeekV3Router):
    """Route DeepSeek tokens round-robin while preserving its config contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Router.Config):
        pass

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        del expert_bias_E, router_kwargs
        num_tokens = scores_TE.shape[0]
        return (
            torch.arange(
                num_tokens * self.top_k,
                device=scores_TE.device,
                dtype=torch.int64,
            ).reshape(num_tokens, self.top_k)
            % self.num_experts
        )


def _prepare_outer_cuda_graph_capture() -> None:
    """Warm NCCL, then capture before Adam initialization.

    Capturing the first forward-backward directly fails because FSDP lazily
    creates split NCCL communicators, which CUDA forbids during capture. A
    normal training-step warmup initializes Adam and leaves too little
    allocator space for the graph pool. Instead, run one eager
    forward-backward inside the wrapper, clear its gradients, release unused
    blocks, and capture immediately before returning to the optimizer. Adam is
    then initialized only after the graph pool has established its layout.
    """
    from torchtitan import training_engine
    from torchtitan.components.optim import optim
    from torchtitan.distributed.cuda_graph import (
        CUDAGraphWrapper,
        run_eager_on_cuda_graph_stream,
    )
    from torchtitan.experiments.graph_trainer import trainer as graph_trainer

    # pyrefly: ignore [bad-assignment]
    training_engine.NUM_CUDA_GRAPH_WARMUP_STEPS = 0
    # GraphTrainer imports the warmup count into its own module namespace.
    # pyrefly: ignore [bad-assignment]
    graph_trainer.NUM_CUDA_GRAPH_WARMUP_STEPS = 0
    # pyrefly: ignore [bad-assignment]
    optim.NUM_CUDA_GRAPH_WARMUP_STEPS = 0

    if getattr(CUDAGraphWrapper, "_empties_cache_before_capture", False):
        return
    original_call = CUDAGraphWrapper.__call__

    def call_with_capture_headroom(self, *args):
        if self._graph is None and self._warmup_remaining == 0:
            if hasattr(self, "_gradient_state") and not getattr(
                self, "_prewarmed_before_adam", False
            ):
                self._prewarmed_before_adam = True
                warmup_output = run_eager_on_cuda_graph_stream(self._fn, *args)
                del warmup_output
                for parameter in self._gradient_state._parameters:
                    parameter.grad = None
                torch.cuda.synchronize()
                gc.collect()
            torch.cuda.empty_cache()
        return original_call(self, *args)

    CUDAGraphWrapper.__call__ = call_with_capture_headroom
    CUDAGraphWrapper._empties_cache_before_capture = True


def _historical_sanket_stage_modules() -> list[list[str]]:
    """Return Sanket's original 4/4/.../4/1 decoder-layer split."""
    stages = [
        (["tok_embeddings"] if stage_index == 0 else [])
        + [f"layers.{layer}" for layer in range(4 * stage_index, 4 * stage_index + 4)]
        for stage_index in range(15)
    ]
    stages.append(["layers.60", "norm", "lm_head"])
    return stages


def _historical_sanket_mtp1_stage_modules() -> list[list[str]]:
    """Place MTP1 and its embedding replica on Sanket's final stage."""
    stages = _historical_sanket_stage_modules()
    stages[-1].extend(["mtp_layers.0", "tok_embeddings"])
    return stages


def _local_sanket_stage_modules() -> list[list[str]]:
    """Split the 27-layer local model into 16 non-empty virtual stages."""
    num_layers_per_stage = [2] * 11 + [1] * 5
    stages: list[list[str]] = []
    first_layer = 0
    for num_layers in num_layers_per_stage:
        stages.append(
            [
                f"layers.{layer}"
                for layer in range(first_layer, first_layer + num_layers)
            ]
        )
        first_layer += num_layers
    stages[0].insert(0, "tok_embeddings")
    stages[-1].extend(["norm", "lm_head"])
    return stages


def _local_sanket_mtp1_stage_modules() -> list[list[str]]:
    """Add MTP1 and its embedding replica to the local final stage."""
    stages = _local_sanket_stage_modules()
    stages[-1].extend(["mtp_layers.0", "tok_embeddings"])
    return stages


def _configure_round_robin_routing(config: Trainer.Config) -> None:
    """Use deterministic balanced routing for performance comparisons."""
    for _, moe, _, _ in config.model.traverse(MoE.Config):
        router = convert_config_type(
            moe.router,
            _RoundRobinDeepSeekV3Router,
        )
        assert isinstance(router, _RoundRobinDeepSeekV3Router.Config)
        moe.router = router


def _configure_dist_moe_runtime(
    config: Trainer.Config,
    *,
    inplace_wgrad_accum: bool,
    activation_slot_capacity_factor: float = 1.0,
) -> None:
    """Match the validated activation, scratch, and WGrad policy."""
    runtime = config.dist_moe
    if not isinstance(runtime, DistMoeRuntime.Config):
        raise TypeError("the 256-GPU workloads require DistMoeRuntime")
    runtime.activation_slot_capacity_factor = activation_slot_capacity_factor
    runtime.scratch_capacity_factor = 1.0
    for _, experts, _, _ in config.model.traverse(DistMoeRoutedExperts.Config):
        experts.inplace_wgrad_accum = inplace_wgrad_accum


def _configure_mtp1(config: Trainer.Config) -> None:
    """Add the final-report MTP1 and sequence-auxiliary objectives.

    The report applies sequence-wise auxiliary loss with coefficient 0.01 to
    every routed depth: 58 main-model MoE layers plus the routed MTP layer.
    Build MTP from the last main layer first, then set the coefficient on all
    routed layers so the contract does not depend on inheritance details.
    """
    model = config.model
    if not isinstance(model, MTPDecoder.Config):
        raise TypeError("MTP1 requires an MTPDecoder model config")
    if model.mtp_layers:
        raise ValueError("MTP1 configuration expects a model without MTP layers")
    model.mtp_layers = _build_mtp_layers(
        model.layers[-1],
        dim=model.dim,
        num_mtp_layers=1,
    )
    config.dataloader.num_mtp_layers = 1
    for _, moe, _, _ in model.traverse(MoE.Config):
        if moe.router.aux_loss is None:
            raise ValueError(
                "the final-report MTP1 recipe requires sequence auxiliary loss"
            )
        moe.router.aux_loss.coeff = 0.01
    config.loss = MTPLoss.Config(
        global_vocab_size=decoder_vocab_size(model),
        mtp_scale=0.1,
    )


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_cat_verification_4gpu() -> (
    GraphTrainer.Config
):
    """Verify FSDP layout extraction with EP4/FSDP4 and outer CUDA graphs.

    Four accumulated microbatches exercise the repeated compute graph and the
    last-microbatch reduction graph separately. The profiler captures step 12,
    after whole-step CUDA graph capture and replay have stabilized.
    """
    base = _base_local_cat_verification(seq_len=512)
    _configure_dist_moe_runtime(base, inplace_wgrad_accum=False)
    _configure_round_robin_routing(base)
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
            fsdp_param_unshard_mode="first_microbatch",
            fsdp_grad_reduce_mode="last_microbatch",
            fuse_wgrad_accumulation="enabled",
        ),
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
    )
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.num_tokens_per_train_step = 512 * 4 * 4
    config.training.disable_cuda_graphs = False
    config.training.steps = 12
    config.activation_checkpoint = None
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"]
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True
    config.dataloader.max_num_documents = 1
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = False
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=12,
        profiler_warmup=0,
        profiler_active=1,
        profiler_repeat=1,
    )
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mtp1_pp1_verification_4gpu() -> (
    GraphTrainer.Config
):
    """Verify single-stage GraphTrainer MTP1 on four GB300s."""
    _prepare_outer_cuda_graph_capture()
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_cat_verification_4gpu()
    _configure_mtp1(config)
    _configure_dist_moe_runtime(
        config,
        inplace_wgrad_accum=False,
        activation_slot_capacity_factor=0.5,
    )
    config.training.steps = 10
    config.profiler = Profiler.Config(enable_profiling=False)
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_sanket_pp2_verification_4gpu() -> (
    GraphTrainer.Config
):
    """Verify the Cat fix through Sanket-shaped PP2/VPP8 GraphPP execution."""
    _prepare_outer_cuda_graph_capture()
    base = _base_local_cat_verification(seq_len=512)
    _configure_dist_moe_runtime(base, inplace_wgrad_accum=False)
    _configure_round_robin_routing(base)
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
            fuse_wgrad_accumulation="enabled",
        ),
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
    )
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = None
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        _local_sanket_stage_modules()
    )
    config.parallelism.num_pp_microbatches = 32
    config.parallelism.pp_max_unsharded_active_stages = 8
    config.parallelism.pp_num_unshard_lookahead_factor = "auto"
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = False
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.num_tokens_per_train_step = 512 * 32 * 2
    config.training.disable_cuda_graphs = False
    config.training.steps = 10
    config.activation_checkpoint = None
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the local Sanket control requires GrainDataLoader")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"]
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True
    config.dataloader.max_num_documents = 1
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = False
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=10,
        profiler_warmup=0,
        profiler_active=1,
        profiler_repeat=1,
    )
    return config


def deepseek_v3_16b_dist_moe_mxfp8_sanket_pp2_verification_4gpu() -> Trainer.Config:
    """Verify eager PP2/VPP8 in-place WGrad and outer CUDA graphs locally."""
    _prepare_outer_cuda_graph_capture()
    config = _base_local_cat_verification(seq_len=512)
    _configure_dist_moe_runtime(config, inplace_wgrad_accum=True)
    _configure_round_robin_routing(config)
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = None
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        _local_sanket_stage_modules()
    )
    config.parallelism.num_pp_microbatches = 32
    config.parallelism.pp_max_unsharded_active_stages = 8
    config.parallelism.pp_num_unshard_lookahead_factor = "auto"
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.num_tokens_per_train_step = 512 * 32 * 2
    config.training.disable_cuda_graphs = False
    config.training.steps = 10
    config.activation_checkpoint = None
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the local Sanket control requires GrainDataLoader")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"]
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True
    config.dataloader.max_num_documents = 1
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = False
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.profiler = Profiler.Config(enable_profiling=False)
    return config


def deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu() -> (
    Trainer.Config
):
    """Verify eager PP2/VPP8 MTP1 and shared embedding gradients locally."""
    config = deepseek_v3_16b_dist_moe_mxfp8_sanket_pp2_verification_4gpu()
    _configure_mtp1(config)
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        _local_sanket_mtp1_stage_modules()
    )
    return config


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu() -> (
    GraphTrainer.Config
):
    """Verify GraphPP MTP1 and shared embedding gradients on four GB300s."""
    base = deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu()
    _configure_dist_moe_runtime(base, inplace_wgrad_accum=False)
    base.parallelism.fsdp_defer_gradient_reduction = False
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
            fuse_wgrad_accumulation="enabled",
        ),
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
    )
    return config


def _configure_mtp1_pp2_numerics_16gpu(config: Trainer.Config) -> Trainer.Config:
    """Scale the local PP2 MTP1 gate to DP8/EP8 for MAST ablations."""
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.expert_parallel_degree = 8
    config.parallelism.fsdp_symm_mem_scope = None
    config.training.num_tokens_per_train_step = 512 * 32 * 8
    config.training.steps = 3
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    config.profiler = Profiler.Config(enable_profiling=False)
    return config


def deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_numerics_16gpu() -> (
    Trainer.Config
):
    """Run the deterministic eager PP2/VPP8/DP8/EP8 MTP1 control."""
    config = deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu()
    return _configure_mtp1_pp2_numerics_16gpu(config)


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_numerics_16gpu() -> (
    GraphTrainer.Config
):
    """Run the matched deterministic GraphTrainer PP2 MTP1 ablation."""
    config = (
        graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_sanket_mtp1_pp2_verification_4gpu()
    )
    return _configure_mtp1_pp2_numerics_16gpu(config)


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile() -> (
    GraphTrainer.Config
):
    """Profile Chien-Chin's topology with reference-equivalent metric cadence.

    Warm lazy kernels and communicators inside the forward-backward wrapper,
    then clear gradients and capture before the optimizer initializes Adam.
    Leave the recipe's original no-rematerialization policy and maximum-useful
    DistMoE activation arena intact. Log every ten steps, matching the original
    benchmark: scalar loss and grad-norm extraction synchronizes CUDA and costs
    almost one second when performed after every asynchronous graph replay.
    """
    _prepare_outer_cuda_graph_capture()
    config = _base_chien()
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the Chien-Chin workload requires GrainDataLoader")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"],
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True
    config.dataloader.max_num_documents = 1
    _configure_dist_moe_runtime(config, inplace_wgrad_accum=False)
    _configure_round_robin_routing(config)

    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.training.steps = 60
    config.training.disable_cuda_graphs = False
    config.activation_checkpoint = None
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = False
    config.parallelism.fsdp_symm_mem_scope = None
    config.metrics.log_freq = 10
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.compile = GraphTrainerCompileConfig(
        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
            fsdp_param_unshard_mode="first_microbatch",
            fsdp_grad_reduce_mode="last_microbatch",
            fuse_wgrad_accumulation="enabled",
        ),
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
        enable_fsdp_ag_rs_overlap=True,
        enable_fsdp_dense_region_overlap=False,
    )
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


def deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile() -> Trainer.Config:
    """Profile the Chien-Chin topology through eager full-step CUDA graphs.

    This control uses ordinary c10d FSDP communication to match GraphTrainer's
    SimpleFSDP backend, along with the same whole-step CUDA graph, model, batch,
    precision, routing, logging cadence, and profiler schedule. It otherwise
    corresponds to the historical R4 in-place WGrad configuration.
    """
    _prepare_outer_cuda_graph_capture()
    config = _base_chien_eager(seq_len=4096)
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the Chien-Chin control requires GrainDataLoader")
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"],
    )
    config.dataloader.shuffle = False
    config.dataloader.repeat = True
    config.dataloader.max_num_documents = 1
    _configure_dist_moe_runtime(config, inplace_wgrad_accum=True)
    _configure_round_robin_routing(config)

    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.training.steps = 60
    config.training.disable_cuda_graphs = False
    config.activation_checkpoint = None
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.parallelism.fsdp_symm_mem_scope = None
    config.metrics.log_freq = 10
    config.hf_assets_path = "./tests/assets/tokenizer"
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.model.local_compile_regions = []
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


def deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile() -> (
    Trainer.Config
):
    """Run the Chien-Chin PP1 control with MTP1 and matched memory capacity."""
    config = deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile()
    _configure_mtp1(config)
    _configure_dist_moe_runtime(
        config,
        inplace_wgrad_accum=True,
        activation_slot_capacity_factor=0.5,
    )
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile() -> (
    GraphTrainer.Config
):
    """Run Chien-Chin MTP1 through GraphTrainer with the same PP1 layout.

    Keep whole-step CUDA graphs enabled while reserving memory for MTP1 and
    Adam state. A half-sized DistMoE activation arena recomputes expert
    activations only after its soft capacity is exhausted; the model, batch,
    parallelism, and optimizer remain unchanged.
    """
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile()
    _configure_mtp1(config)
    _configure_dist_moe_runtime(
        config,
        inplace_wgrad_accum=False,
        activation_slot_capacity_factor=0.5,
    )
    return config


def deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance() -> (
    Trainer.Config
):
    """Run eager Chien-Chin MTP1 with profiling; exclude metrics interval 50."""
    return deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile()


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance() -> (
    GraphTrainer.Config
):
    """Run GraphTrainer Chien-Chin MTP1 with profiling; exclude interval 50."""
    return (
        graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile()
    )


def deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile() -> (
    Trainer.Config
):
    """Use Sanket's Manifold C4 mirror with the shipped topology recipe.

    The historical MAST run bound its C4 dataset to ``/mnt/mffuse/c4`` and
    disabled Hugging Face network access. This wrapper restores its explicit
    stage split and cluster-local dataset binding. It permits eight concurrent
    unsharded stages and retains the historical in-place WGrad accumulation
    and deferred FSDP reduction policies.
    """
    _prepare_outer_cuda_graph_capture()
    config = _base_chien_eager(seq_len=4096)
    _configure_dist_moe_runtime(config, inplace_wgrad_accum=True)
    _configure_round_robin_routing(config)
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = 4096 * 4096
    config.training.steps = 40
    config.training.disable_cuda_graphs = False
    config.activation_checkpoint = None
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 128
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_layers_per_stage = None
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        _historical_sanket_stage_modules()
    )
    config.parallelism.num_pp_microbatches = 32
    config.parallelism.pipeline_parallel_max_outstanding_sends = 16
    config.parallelism.pp_max_unsharded_active_stages = 8
    config.parallelism.pp_num_unshard_lookahead_factor = "auto"
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.parallelism.fsdp_symm_mem_scope = None
    config.model.local_compile_regions = []
    config.metrics.log_freq = 10
    config.comm.trace_buf_size = 0
    config.debug.seed = 42
    config.optim.lr_scheduler.warmup_steps = 2
    config.optim.lr_scheduler.total_steps = 40
    config.profiler = Profiler.Config(
        enable_profiling=True,
        profile_freq=43,
        profiler_warmup=3,
        profiler_active=2,
        profiler_repeat=1,
        enable_memory_snapshot=True,
        memory_snapshot_freq=40,
    )
    if not isinstance(config.dataloader, GrainDataLoader.Config):
        raise TypeError("the Sanket MAST config requires GrainDataLoader")
    config.dataloader.max_num_documents = 1
    config.dataloader.num_prefetch_microbatches = 64
    packing = config.dataloader.dataset
    if not isinstance(packing, ConcatThenSplitPackingConfig):
        raise TypeError("the Sanket MAST config requires concat-then-split packing")
    dataset = packing.dataset
    if not isinstance(dataset, SingleDatasetConfig):
        raise TypeError("the Sanket MAST config requires one C4 dataset")
    source = dataset.source
    if not isinstance(source, HuggingFaceStreamingSource.Config):
        raise TypeError("the Sanket MAST config requires a streaming source")
    config.dataloader.dataset = replace(
        packing,
        dataset=replace(
            dataset,
            source=replace(source, path="/mnt/mffuse/c4"),
        ),
    )
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile() -> (
    GraphTrainer.Config
):
    """Run Sanket's PP2/VPP8 topology through GraphTrainer GraphPP.

    This preserves the eager control's model, stage split, batch, parallelism,
    routing, data, and outer full-step CUDA graph. GraphPP owns FSDP reduction
    as explicit schedule actions, so its config must disable the eager-only
    ``fsdp_defer_gradient_reduction`` flag.
    """
    base = deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile()
    _configure_dist_moe_runtime(base, inplace_wgrad_accum=False)
    base.parallelism.fsdp_defer_gradient_reduction = False
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
            fuse_wgrad_accumulation="enabled",
        ),
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
    )
    return config


def deepseek_v3_671b_dist_moe_mxfp8_sanket_mtp1_256gpu_profile() -> Trainer.Config:
    """Run Sanket's PP2/VPP8 topology with one MTP depth and scale 0.1."""
    config = deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile()
    _configure_mtp1(config)
    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        _historical_sanket_mtp1_stage_modules()
    )
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_mtp1_256gpu_profile() -> (
    GraphTrainer.Config
):
    """Run Sanket MTP1 through GraphTrainer with the same PP2/VPP8 layout."""
    base = deepseek_v3_671b_dist_moe_mxfp8_sanket_mtp1_256gpu_profile()
    _configure_dist_moe_runtime(base, inplace_wgrad_accum=False)
    base.parallelism.fsdp_defer_gradient_reduction = False
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig(
        spmd_gradient_accumulation=SPMDGradientAccumulationConfig(
            fuse_wgrad_accumulation="enabled",
        ),
        memory_policy="none",
        inductor_compilation="regional",
        numerics_changing_optim=False,
    )
    return config


def _configure_sanket_final_report_measurement(config: Trainer.Config) -> None:
    """Match the final report's PP2-256 training-step geometry.

    The published PP2 point uses DP128, VPP8, 120 local microbatches, and
    global batch size 15,360 at sequence length 4,096. This benchmark keeps
    the report's training arithmetic and seed, but does not claim its
    convergence-only contract: the internal warmed checkpoint, AirStore
    validation stream, and validation pass are not available in this OSS
    reproduction.
    """
    config.training.num_tokens_per_train_step = (
        15_360 * config.training.max_context_length
    )
    config.training.steps = 60
    config.parallelism.num_pp_microbatches = 120
    config.metrics.log_freq = 10
    config.debug.seed = 14_536
    config.optim.lr_scheduler.total_steps = 60


def deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile() -> (
    Trainer.Config
):
    """Profile eager MTP1 with the final report's PP2-256 step geometry."""
    config = deepseek_v3_671b_dist_moe_mxfp8_sanket_mtp1_256gpu_profile()
    _configure_sanket_final_report_measurement(config)
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile() -> (
    GraphTrainer.Config
):
    """Profile GraphTrainer MTP1 with the final report's PP2-256 geometry."""
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_mtp1_256gpu_profile()
    _configure_sanket_final_report_measurement(config)
    return config


def deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance() -> (
    Trainer.Config
):
    """Run eager Sanket MTP1 with profiling; exclude metrics intervals 40 and 50."""
    return deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile()


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance() -> (
    GraphTrainer.Config
):
    """Run GraphTrainer Sanket MTP1 with profiling; exclude intervals 40 and 50."""
    return (
        graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile()
    )


def _configure_full_scale_numerics(config: Trainer.Config) -> None:
    """Record every step of a deterministic full-scale correctness run.

    These recipes preserve the production topology, batch, seed, objective,
    and outer full-step CUDA graph. They disable profiling so eager and
    GraphTrainer see the same execution contract, and enable TensorBoard at
    every step so loss and grad norm are retained at full precision.
    """
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.metrics.enable_tensorboard = True
    config.metrics.log_freq = 1
    config.profiler = Profiler.Config(enable_profiling=False)


def deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_numerics() -> (
    Trainer.Config
):
    """Run the eager PP1 MTP1 full-scale numerics control."""
    config = deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance()
    _configure_full_scale_numerics(config)
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_numerics() -> (
    GraphTrainer.Config
):
    """Run the matched GraphTrainer PP1 MTP1 full-scale numerics gate."""
    config = (
        graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance()
    )
    _configure_full_scale_numerics(config)
    return config


def deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_numerics() -> (
    Trainer.Config
):
    """Run the eager PP2/VPP8 MTP1 full-scale numerics control."""
    config = deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance()
    _configure_full_scale_numerics(config)
    return config


def graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_numerics() -> (
    GraphTrainer.Config
):
    """Run the matched GraphTrainer PP2/VPP8 MTP1 full-scale numerics gate."""
    config = (
        graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance()
    )
    _configure_full_scale_numerics(config)
    return config
