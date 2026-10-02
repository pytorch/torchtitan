# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import IntegrationTestDefinition
from tests.integration_tests.run_tests import run_tests
from torchtitan_recipes.graph_trainer.llama3 import graph_trainer_llama3_8b

from torchtitan_recipes.tests.graph_trainer import (
    b200 as b200_recipes,
    deepseek_v3 as deepseek_v3_recipes,
    llama3 as llama3_recipes,
    muse_glimmer as muse_glimmer_recipes,
    qwen3 as qwen3_recipes,
)

# TODO: Re-enable after regional_inductor can trace the CP load balancer's
# index-rearrange constants; it currently raises a FunctionalTensor error.
_FLEX_CP_INDUCTOR_DISABLED = True


def llama3_fsdp_tp_cp():
    config = llama3_recipes.graph_trainer_llama3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def llama3_fsdp_tp():
    config = llama3_recipes.graph_trainer_llama3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_spmd_gradient_accumulation():
    config = llama3_recipes.graph_trainer_llama3_debugmodel()
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 4096
    return config


def _llama3_fsdp_collectives(*, param_unshard_mode: str, grad_reduce_mode: str):
    config = llama3_recipes.graph_trainer_llama3_debugmodel()
    config.compile.spmd_gradient_accumulation.fsdp_param_unshard_mode = (
        param_unshard_mode
    )
    config.compile.spmd_gradient_accumulation.fsdp_grad_reduce_mode = grad_reduce_mode
    config.parallelism.data_parallel_shard_degree = 4
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 16384
    return config


def llama3_ga_per_microbatch_fsdp_collectives():
    return _llama3_fsdp_collectives(
        param_unshard_mode="every_microbatch",
        grad_reduce_mode="every_microbatch",
    )


def llama3_ga_first_unshard_last_reduce_fsdp_collectives():
    return _llama3_fsdp_collectives(
        param_unshard_mode="first_microbatch",
        grad_reduce_mode="last_microbatch",
    )


def llama3_fsdp_tp_sac_and_offload():
    config = llama3_fsdp_tp()
    config.compile.memory_policy = "sac_and_offload"
    return config


def llama3_fsdp_tp_regional_inductor():
    config = llama3_fsdp_tp()
    config.compile.inductor_compilation = "regional"
    return config


def deepseek_v3_fused_mla_swiglu_fsdp_tp_ep():
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel()
    config.training.disable_cuda_graphs = True
    config.compile.disable_passes = [
        "joint_transformer_block_bucketing_reordering_pass",
    ]
    config.override.imports = [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    return config


def deepseek_v3_fsdp_tp_cp_ep():
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    return config


def deepseek_v3_fsdp_tp_ep():
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    return config


def deepseek_v3_fsdp_tp_ep_regional_inductor():
    config = deepseek_v3_fsdp_tp_ep()
    config.compile.inductor_compilation = "regional"
    return config


def _deepseek_v3_ep_overlap(
    *, inductor_compilation: str, chunk_dim: str, module_fqn: str
):
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel()
    config.training.disable_cuda_graphs = True
    config.compile.inductor_compilation = inductor_compilation
    config.compile.ep_overlap.enabled = True
    config.compile.ep_overlap.chunk_dim = chunk_dim
    config.compile.ep_overlap.module_fqn = module_fqn
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 4
    return config


def deepseek_v3_regional_ep_overlap_transformer_batch():
    return _deepseek_v3_ep_overlap(
        inductor_compilation="regional", chunk_dim="batch", module_fqn="layers.*"
    )


def deepseek_v3_regional_ep_overlap_moe_batch():
    return _deepseek_v3_ep_overlap(
        inductor_compilation="regional",
        chunk_dim="batch",
        module_fqn="layers.*.moe",
    )


def deepseek_v3_regional_ep_overlap_moe_seq():
    return _deepseek_v3_ep_overlap(
        inductor_compilation="regional",
        chunk_dim="seq",
        module_fqn="layers.*.moe",
    )


def deepseek_v3_full_ep_overlap_transformer_batch():
    return _deepseek_v3_ep_overlap(
        inductor_compilation="full", chunk_dim="batch", module_fqn="layers.*"
    )


def deepseek_v3_full_ep_overlap_moe_batch():
    return _deepseek_v3_ep_overlap(
        inductor_compilation="full",
        chunk_dim="batch",
        module_fqn="layers.*.moe",
    )


def deepseek_v3_full_ep_overlap_moe_seq():
    return _deepseek_v3_ep_overlap(
        inductor_compilation="full", chunk_dim="seq", module_fqn="layers.*.moe"
    )


def deepseek_v3_graph_pp_interleaved_1f1b():
    return _deepseek_v3_graph_pp("Interleaved1F1B")


def deepseek_v3_graph_pp_zbv_zero_bubble():
    return _deepseek_v3_graph_pp("ZBVZeroBubble")


def deepseek_v3_graph_pp_dual_pipe_v():
    return _deepseek_v3_graph_pp("DualPipeV")


def _deepseek_v3_graph_pp(schedule: str):
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel()
    config.training.disable_cuda_graphs = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.compile.inductor_compilation = "full"
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = schedule
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    return config


def deepseek_v3_hybrid_ep():
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel_hybridep()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    return config


def qwen3_fsdp_tp_cp():
    config = qwen3_recipes.graph_trainer_qwen3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config


def qwen3_moe_fsdp_tp_ep():
    config = qwen3_recipes.graph_trainer_qwen3_debugmodel_moe()
    config.training.disable_cuda_graphs = True
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    return config


def muse_glimmer_fsdp():
    config = muse_glimmer_recipes.graph_trainer_muse_glimmer_debugmodel()
    config.parallelism.data_parallel_shard_degree = 8
    return config


def muse_glimmer_fsdp_tp():
    config = muse_glimmer_recipes.graph_trainer_muse_glimmer_debugmodel()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    return config


def llama3_fsdp_tp_async_tp():
    config = graph_trainer_llama3_8b(seq_len=512)
    config.compile.enable_async_tensor_parallel = True
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.hf_assets_path = "./tests/assets/tokenizer"
    return config


def llama3_autoparallel_fsdp_tp():
    config = llama3_recipes.graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss()
    config.compile.enable_autoparallel = True
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    return config


def deepseek_v3_autoparallel_edp_shard_ep():
    config = deepseek_v3_recipes.graph_trainer_deepseek_v3_debugmodel()
    config.compile.enable_autoparallel = True
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    return config


def _build_llama3_tests() -> list[IntegrationTestDefinition]:
    """Llama3-based integration tests (run on default A10 machines)."""
    return [
        IntegrationTestDefinition(
            configs=[llama3_recipes.graph_trainer_llama3_debugmodel_sdc_replay],
            test_descr="GraphTrainer SDC replay",
            test_name="sdc_replay",
            ngpu=1,
            skip_rocm_test=True,
        ),
        # === GraphRuntime tests ===
        # Note: aot_fx_trace applies CUDA graph by default, so skip_rocm_test=True.
        #
        # Disable cuda_graph: replaying coalesced FSDP collectives with CP fails
        # with "CUDA error: invalid argument".
        IntegrationTestDefinition(
            configs=[llama3_fsdp_tp_cp],
            test_descr="aot_fx_trace llama3 FSDP+TP+CP",
            test_name="aot_fx_trace_llama3_fsdp_tp_cp",
            ngpu=8,
            skip_rocm_test=True,
            disabled=_FLEX_CP_INDUCTOR_DISABLED,
        ),
        # async_tp test lives in graph_trainer_h100 suite (needs NVLink).
        IntegrationTestDefinition(
            configs=[llama3_fsdp_tp],
            test_descr="aot_fx_trace llama3 FSDP+TP",
            test_name="aot_fx_trace_llama3_fsdp_tp",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[llama3_spmd_gradient_accumulation],
            test_descr="aot_fx_trace llama3 SPMD gradient accumulation",
            test_name="aot_fx_trace_llama3_spmd_gradient_accumulation",
            ngpu=1,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_ga_per_microbatch_fsdp_collectives],
            test_descr="aot_fx_trace llama3 GA with per-microbatch FSDP collectives",
            test_name="aot_fx_trace_llama3_ga_per_microbatch_fsdp_collectives",
            ngpu=4,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_ga_first_unshard_last_reduce_fsdp_collectives],
            test_descr=(
                "aot_fx_trace llama3 GA with first-microbatch FSDP unshard and "
                "last-microbatch grad reduction"
            ),
            test_name="aot_fx_trace_llama3_ga_first_unshard_last_reduce_fsdp",
            ngpu=4,
            skip_rocm_test=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_fsdp_tp_sac_and_offload],
            test_descr="aot_fx_trace llama3 FSDP+TP+sac_and_offload",
            test_name="aot_fx_trace_llama3_fsdp_tp_sac_and_offload",
            ngpu=8,
            skip_rocm_test=True,
            # GraphRuntime must preserve offload/reload pairs when it
            # extracts scheduled graph callables.
            disabled=True,
        ),
        IntegrationTestDefinition(
            configs=[llama3_fsdp_tp_regional_inductor],
            test_descr="aot_fx_trace llama3 FSDP+TP+regional_inductor",
            test_name="aot_fx_trace_llama3_fsdp_tp_regional_inductor",
            ngpu=8,
        ),
    ]


def _build_deepseek_v3_tests() -> list[IntegrationTestDefinition]:
    """DeepSeek-v3-based integration tests (require H100 machines)."""
    ep_overlap_flex_tests = [
        # TODO(#4342): Remove transformer-level chunking. After the model batch
        # dimension was folded into the token dimension, splitting `layers.*`
        # in half cuts the packed token stream mid-document, so neither chunk
        # has full attention context. This variant aborts at step 1 with a
        # non-finite loss.
        (
            deepseek_v3_regional_ep_overlap_transformer_batch,
            "regional",
            "transformer_batch",
            True,
        ),
        (
            deepseek_v3_regional_ep_overlap_moe_batch,
            "regional",
            "moe_batch",
            True,
        ),
        (
            deepseek_v3_regional_ep_overlap_moe_seq,
            "regional",
            "moe_seq",
            True,
        ),
        # TODO(#4342): Remove transformer-level chunking, as above. Under full
        # Inductor the mid-document split surfaces earlier than the non-finite
        # loss: this variant aborts before step 1 on a Triton index-out-of-
        # bounds assertion.
        (
            deepseek_v3_full_ep_overlap_transformer_batch,
            "full",
            "transformer_batch",
            True,
        ),
        (
            deepseek_v3_full_ep_overlap_moe_batch,
            "full",
            "moe_batch",
            True,
        ),
        (
            deepseek_v3_full_ep_overlap_moe_seq,
            "full",
            "moe_seq",
            True,
        ),
    ]

    return [
        # === GraphRuntime tests ===
        # Note: standard DSv3 MoE load-balancing introduces CUDA-to-CPU
        # transfers incompatible with CUDA graph capture, so this fused test
        # explicitly disables CUDA graphs in both the trainer and graph passes.
        #
        # TODO: Re-enable FSDP bucketing when its stable topological sort
        # supports the fused MLA Q kernel's mutating custom-op boundary.
        IntegrationTestDefinition(
            configs=[deepseek_v3_fused_mla_swiglu_fsdp_tp_ep],
            test_descr="aot_fx_trace deepseek_v3 fused MLA+SwiGLU FSDP+TP+EP",
            test_name="aot_fx_trace_deepseek_v3_fused_mla_swiglu_fsdp_tp_ep",
            ngpu=4,
        ),
        # TODO: Re-enable after fixing the separate CP+EP mixed Tensor/DTensor
        # failure, in addition to the graph_trainer CP backend issue.
        IntegrationTestDefinition(
            configs=[deepseek_v3_fsdp_tp_cp_ep],
            test_descr="aot_fx_trace deepseek_v3 FSDP+TP+CP+EP",
            test_name="aot_fx_trace_deepseek_v3_fsdp_tp_cp_ep",
            ngpu=8,
            disabled=True,
        ),
        # TODO: Disabled — flaky/hanging EP all-to-all. The mesh_ep
        # ALLTOALL_BASE collective times out (NCCL watchdog, 100s) and the job
        # hangs to the workflow timeout; this also caused H100 job timeouts on
        # main. Likely an upstream MoE-EP all-to-all / collective-ordering
        # instability (intermittent; also seen as a "Split sizes" crash, and the
        # full_inductor EP variant below has passed in the same run). Re-enable
        # once the EP all-to-all instability is resolved upstream.
        IntegrationTestDefinition(
            configs=[deepseek_v3_fsdp_tp_ep],
            test_descr="aot_fx_trace deepseek_v3 FSDP+TP+EP",
            test_name="aot_fx_trace_deepseek_v3_fsdp_tp_ep",
            ngpu=8,
            disabled=True,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_fsdp_tp_ep_regional_inductor],
            test_descr="aot_fx_trace deepseek_v3 FSDP+TP+EP+regional_inductor",
            test_name="aot_fx_trace_deepseek_v3_fsdp_tp_ep_regional_inductor",
            ngpu=8,
            # TODO(#4047): Re-enable once FSDP bucketing no longer creates a
            # cyclic region for this DeepSeekV3 FSDP+TP+EP configuration.
            disabled=True,
        ),
        *[
            IntegrationTestDefinition(
                configs=[config_fn],
                test_descr=(
                    "aot_fx_trace deepseek_v3 FlexAttn "
                    f"{inductor_compilation}_inductor ep_overlap {variant}"
                ),
                test_name=(
                    "aot_fx_trace_deepseek_v3_flexattn_"
                    f"{inductor_compilation}_inductor_ep_overlap_{variant}"
                ),
                ngpu=8,
                # TODO(#4052): These were disabled under graph chunking with
                # dense-region FSDP overlap; re-validate the MoE variants with
                # eager chunking before re-enabling.
                disabled=disabled,
            )
            for (
                config_fn,
                inductor_compilation,
                variant,
                disabled,
            ) in ep_overlap_flex_tests
        ],
        IntegrationTestDefinition(
            configs=[deepseek_v3_graph_pp_interleaved_1f1b],
            test_descr=(
                "aot_fx_trace deepseek_v3 GraphPP Interleaved1F1B full_inductor"
            ),
            test_name=(
                "aot_fx_trace_deepseek_v3_graph_pp_interleaved_1f1b_full_inductor"
            ),
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_graph_pp_zbv_zero_bubble],
            test_descr="aot_fx_trace deepseek_v3 GraphPP ZBVZeroBubble full_inductor",
            test_name="aot_fx_trace_deepseek_v3_graph_pp_zbv_zero_bubble_full_inductor",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_graph_pp_dual_pipe_v],
            test_descr="aot_fx_trace deepseek_v3 GraphPP DualPipeV full_inductor",
            test_name="aot_fx_trace_deepseek_v3_graph_pp_dual_pipe_v_full_inductor",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[deepseek_v3_hybrid_ep],
            test_descr="aot_fx_trace deepseek_v3 FSDP+TP+HybridEP",
            test_name="aot_fx_trace_deepseek_v3_hybridep",
            ngpu=4,
            disabled=True,
        ),
    ]


def _build_qwen3_tests() -> list[IntegrationTestDefinition]:
    """Qwen3-based integration tests (dense + MoE)."""
    return [
        # Disable cuda_graph: replaying coalesced FSDP collectives with CP fails
        # with "CUDA error: invalid argument".
        IntegrationTestDefinition(
            configs=[qwen3_fsdp_tp_cp],
            test_descr="aot_fx_trace qwen3 FSDP+TP+CP",
            test_name="aot_fx_trace_qwen3_fsdp_tp_cp",
            ngpu=8,
            disabled=_FLEX_CP_INDUCTOR_DISABLED,
        ),
        IntegrationTestDefinition(
            configs=[qwen3_moe_fsdp_tp_ep],
            test_descr="aot_fx_trace qwen3 MoE FSDP+TP+EP",
            test_name="aot_fx_trace_qwen3_moe_fsdp_tp_ep",
            ngpu=8,
        ),
    ]


def _build_muse_glimmer_tests() -> list[IntegrationTestDefinition]:
    """MuseGlimmer integration tests."""
    return [
        IntegrationTestDefinition(
            configs=[muse_glimmer_fsdp],
            test_descr="aot_fx_trace muse_glimmer FSDP",
            test_name="aot_fx_trace_muse_glimmer_fsdp",
            ngpu=8,
        ),
        IntegrationTestDefinition(
            configs=[muse_glimmer_fsdp_tp],
            test_descr="aot_fx_trace muse_glimmer FSDP+TP",
            test_name="aot_fx_trace_muse_glimmer_fsdp_tp",
            ngpu=8,
        ),
    ]


def build_graph_trainer_test_list() -> list[IntegrationTestDefinition]:
    """All graph_trainer integration tests."""
    return (
        _build_llama3_tests()
        + _build_deepseek_v3_tests()
        + _build_qwen3_tests()
        + _build_muse_glimmer_tests()
    )


def build_graph_trainer_default_test_list() -> list[IntegrationTestDefinition]:
    """Dense-model tests for default A10 machines."""
    return _build_llama3_tests() + _build_muse_glimmer_tests()


def _build_async_tp_tests() -> list[IntegrationTestDefinition]:
    """Async TP tests (require NVLink for symmetric memory)."""
    return [
        IntegrationTestDefinition(
            configs=[llama3_fsdp_tp_async_tp],
            # async_tp (micro_pipeline_tp) requires shard_dim >= 1024.
            # 8B (dim=4096) with TP=2 gives shard=2048, above threshold.
            test_descr="aot_fx_trace llama3 FSDP+TP+async_tp",
            test_name="aot_fx_trace_llama3_fsdp_tp_asynctp",
            ngpu=8,
            skip_rocm_test=True,
            # TODO: Disabled — async_tp (micro_pipeline_tp) fails with an
            # inductor stride mismatch: assert_size_stride on the fused
            # collective-matmul input expects a different stride than produced
            # ("expected size 2==2, stride 2048==1048576 at dim=0"), i.e. a bad
            # meta/fake kernel for the async-TP fused op. Likely an upstream
            # inductor / async-TP regression. Re-enable once fixed upstream.
            disabled=True,
        ),
    ]


def _build_autoparallel_tests() -> list[IntegrationTestDefinition]:
    """AutoParallel integration tests for default runners."""
    return [
        # Uses the SDPA backend: AutoParallel's dynamo export
        # (_dynamo_graph_capture_for_export) pytree-flattens the default
        # FlexInnerAttention BlockMask to plain (Fake)Tensors, so flex_attention then
        # fails with "'FakeTensor' object has no attribute 'BLOCK_SIZE'". SDPA is
        # maskless (is_causal) and carries no BlockMask, and its input_fn
        # (tokens, positions) binds correctly now that Decoder.forward lists
        # positions before attention_metadata.
        # TODO: re-test on FlexInnerAttention once BlockMask survives AutoParallel
        # graph capture.
        # TODO: Disabled due to upstream AutoParallel/PyTorch API skew. PyTorch
        # #186754 (2026-06-24) removed propagate_single_input_strategy in favor
        # of propagate_single_input_single_dim_strategy, but AutoParallel's
        # convert_element_type_rule still imports the old name, so the sharding
        # optimizer fails with ImportError. Re-enable once AutoParallel migrates.
        # https://github.com/pytorch/torchtitan/issues/3699
        IntegrationTestDefinition(
            configs=[llama3_autoparallel_fsdp_tp],
            test_descr="autoparallel llama3 FSDP+TP",
            test_name="autoparallel_llama3_fsdp_tp",
            ngpu=4,
            disabled=True,
        ),
    ]


def _build_autoparallel_h100_tests() -> list[IntegrationTestDefinition]:
    """AutoParallel integration tests that require H100 runners."""
    return [
        # TODO: Disabled due to upstream AutoParallel regression in PyTorch
        # nightly dev20260508. AutoParallel rejects FakeTensor device
        # mismatch (traced on meta vs actual cuda). Re-enable once fixed.
        IntegrationTestDefinition(
            configs=[deepseek_v3_autoparallel_edp_shard_ep],
            test_descr="autoparallel deepseek_v3 edp_shard+ep",
            test_name="autoparallel_deepseek_v3_edp_shard_ep",
            ngpu=4,
            disabled=True,
        ),
    ]


def build_graph_trainer_h100_test_list() -> list[IntegrationTestDefinition]:
    """DeepSeek-v3 + Qwen3 + async_tp tests (for H100 machines)."""
    return _build_deepseek_v3_tests() + _build_qwen3_tests() + _build_async_tp_tests()


def build_graph_trainer_b200_test_list() -> list[IntegrationTestDefinition]:
    """Dist-MoE tests that require B200-class hardware."""
    return [
        IntegrationTestDefinition(
            configs=[
                b200_recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16_fsdp2_ep2
            ],
            test_descr="GraphTrainer BF16 Dist-MoE with FSDP and EP",
            test_name="graph_trainer_dist_moe_fsdp_ep",
            ngpu=2,
            use_real_pg=True,
        ),
        IntegrationTestDefinition(
            configs=[
                b200_recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8_fsdp2_ep2_pp2
            ],
            test_descr="GraphPP MXFP8 Dist-MoE activation-slot reuse",
            test_name="graph_trainer_dist_moe_mxfp8_fsdp_ep_pp",
            ngpu=4,
            use_real_pg=True,
        ),
    ]


def build_graph_trainer_autoparallel_test_list() -> list[IntegrationTestDefinition]:
    """AutoParallel tests for default runners."""
    return _build_autoparallel_tests()


def build_graph_trainer_autoparallel_h100_test_list() -> list[
    IntegrationTestDefinition
]:
    """AutoParallel tests that require H100 runners."""
    return _build_autoparallel_h100_tests()


_TEST_SUITES_FUNCTION = {
    "graph_trainer": build_graph_trainer_test_list,
    "graph_trainer_default": build_graph_trainer_default_test_list,
    "graph_trainer_h100": build_graph_trainer_h100_test_list,
    "graph_trainer_b200": build_graph_trainer_b200_test_list,
    "graph_trainer_autoparallel": build_graph_trainer_autoparallel_test_list,
    "graph_trainer_autoparallel_h100": build_graph_trainer_autoparallel_h100_test_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--gpu_arch_type",
        default="cuda",
        choices=["cuda", "rocm"],
        help="GPU architecture type. Must be specified as either 'cuda' or 'rocm'.",
    )
    parser.add_argument(
        "--test_suite",
        default="graph_trainer",
        choices=list(_TEST_SUITES_FUNCTION.keys()),
        help="Which test suite to run (default: graph_trainer, which runs all tests)",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
