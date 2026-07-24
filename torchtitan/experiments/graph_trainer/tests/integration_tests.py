# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests

# TODO: JIT mode tests are disabled due to an upstream PyTorch
# partitioner regression ("Node tangents_2 was invalid, but is output")
# triggered by the full DTensor change (#2149). Re-enable once the
# partitioner issue is resolved.
_JIT_DISABLED = True

def _graph_capture_args(gpu_arch_type: str) -> list[str]:
    """Return the explicit graph-capture option for the accelerator."""
    if gpu_arch_type == "cuda":
        return ["--compile.enable_cudagraph"]
    if gpu_arch_type == "xpu":
        return ["--compile.enable_xpugraph"]
    return []

def _build_llama3_tests(gpu_arch_type: str = "cuda",) -> list[OverrideDefinitions]:
    """Llama3-based integration tests (run on default A10 machines)."""
    return [
        # === JIT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=auto_bucketing",
                ],
            ],
            "JIT 1D+auto_bucketing",
            "jit_1d_auto_bucketing",
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=transformer_block_bucketing",
                ],
            ],
            "JIT 1D+transformer_block_bucketing",
            "jit_1d_transformer_block_bucketing",
            disabled=_JIT_DISABLED,
        ),
        # TODO: re-enable this test once the async TP issue is fixed
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.enable_async_tensor_parallel",
                ],
            ],
            "JIT 2D async TP",
            "jit_2d_asynctp",
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--training.steps 20",
                    "--checkpoint.enable",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "JIT PP+DP+TP 3D test with save/load resume ckpt",
            "jit_pp_dp_tp",
            ngpu=8,
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ]
            ],
            "JIT HSDP+TP",
            "jit_hsdp+tp",
            ngpu=8,
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ]
            ],
            "JIT HSDP+CP (with dp_shard)",
            "jit_hsdp+cp_with_dp_shard",
            ngpu=8,
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ]
            ],
            "JIT FSDP+TP+CP",
            "jit_fsdp+tp+cp",
            ngpu=8,
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--training.steps 10",
                ],
                # Save at [dp:4] and load at [dp:2, tp:2]. Note that the dataloader should be
                # excluded during loading to avoid errors caused by mismatched dp_degree.
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 2",
                    "--training.steps 20",
                ],
                # load at [tp:4].
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode jit",
                    "--checkpoint.enable",
                    "--checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer",
                    "--parallelism.tensor_parallel_degree 4",
                    "--training.steps 30",
                ],
            ],
            "JIT Optional checkpoint",
            "jit_optional_checkpoint",
            ngpu=4,
            disabled=_JIT_DISABLED,
        ),
        # === aot_fx_trace mode tests ===
        # Note: aot_fx_trace applies cudagraph by default, so skip_rocm_test=True.
        #
        # Uses the SDPA backend: the default FlexAttention + CP +
        # regional_inductor combination is not yet supported — the CP load
        # balancer injects an index-rearrange constant (torch
        # _context_parallel/_attention.py qkv_rearrange_indices) that
        # regional_inductor's make_fx re-trace cannot lift ("Attempting to use
        # FunctionalTensor on its own"). SDPA has native CP support and no such
        # constant, so it exercises the CP graph path. cudagraph is disabled
        # here: CUDA-graph replay of the coalesced FSDP collectives fails under
        # CP with "CUDA error: invalid argument".
        # TODO: re-test on FlexAttention once flex + CP + regional_inductor is
        # supported upstream.
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel_sdpa",
                    "--compile.mode aot_fx_trace",
                    "--compile.disable_passes cudagraph_pass",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "aot_fx_trace llama3 FSDP+TP+CP+graph_capture",
            "aot_fx_trace_llama3_fsdp_tp_cp",
            ngpu=8,
            skip_rocm_test=True,
        ),
        # async_tp test lives in graph_trainer_h100 suite (needs NVLink).
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    *_graph_capture_args(gpu_arch_type),
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "aot_fx_trace llama3 FSDP+TP",
            "aot_fx_trace_llama3_fsdp_tp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.memory_policy sac_and_offload",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "aot_fx_trace llama3 FSDP+TP+sac_and_offload",
            "aot_fx_trace_llama3_fsdp_tp_sac_and_offload",
            ngpu=8,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.inductor_compilation regional",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "aot_fx_trace llama3 FSDP+TP+regional_inductor",
            "aot_fx_trace_llama3_fsdp_tp_regional_inductor",
            ngpu=8,
        ),
    ]


def _build_deepseek_v3_tests() -> list[OverrideDefinitions]:
    """DeepSeek-v3-based integration tests (require H100 machines)."""
    ep_overlap_flex_tests = [
        (
            "regional",
            "batch",
            "layers.*",
            "transformer_batch",
        ),
        (
            "regional",
            "batch",
            "layers.*.moe",
            "moe_batch",
        ),
        (
            "regional",
            "seq",
            "layers.*.moe",
            "moe_seq",
        ),
        (
            "full",
            "batch",
            "layers.*",
            "transformer_batch",
        ),
        (
            "full",
            "batch",
            "layers.*.moe",
            "moe_batch",
        ),
        (
            "full",
            "seq",
            "layers.*.moe",
            "moe_seq",
        ),
    ]

    def ep_overlap_parallelism() -> list[str]:
        return [
            "--parallelism.data_parallel_shard_degree 8",
            "--parallelism.tensor_parallel_degree 1",
            "--parallelism.expert_parallel_degree 4",
        ]

    return [
        # === JIT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 8",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "JIT FSDP+EP",
            "jit_fsdp+ep",
            ngpu=8,
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "JIT FSDP+CP",
            "jit_fsdp+cp",
            ngpu=8,
            disabled=_JIT_DISABLED,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend inductor",
                    "--parallelism.tensor_parallel_degree 1",
                    "--parallelism.expert_parallel_degree 8",
                    "--compile.passes=auto_bucketing",
                ]
            ],
            "jit_deepseekv3_auto_bucketing",
            ngpu=8,
            # JIT mode is deprecated; gate with the other JIT flavors. It also
            # currently hits an upstream inductor bug (control_deps lowering
            # calls realize() on an ir.Subgraph: "realize NYI on Subgraph").
            disabled=_JIT_DISABLED,
        ),
        # === aot_fx_trace mode tests ===
        # Note: cudagraph is auto-skipped for DSv3 because MoE load-balancing
        # introduces CUDA→CPU transfers incompatible with CUDA graph capture.
        #
        # TODO: FSDP+TP+CP+EP is disabled: tracing fails with "aten.add.Tensor
        # got mixed torch.Tensor and DTensor" — a separate CP+EP issue,
        # unrelated to the empty_strided shadow-node fix. Re-enable once fixed.
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "aot_fx_trace deepseek_v3 FSDP+TP+CP+EP",
            "aot_fx_trace_deepseek_v3_fsdp_tp_cp_ep",
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
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "aot_fx_trace deepseek_v3 FSDP+TP+EP",
            "aot_fx_trace_deepseek_v3_fsdp_tp_ep",
            ngpu=8,
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.inductor_compilation regional",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "aot_fx_trace deepseek_v3 FSDP+TP+EP+regional_inductor",
            "aot_fx_trace_deepseek_v3_fsdp_tp_ep_regional_inductor",
            ngpu=8,
        ),
        *[
            OverrideDefinitions(
                [
                    [
                        "--module graph_trainer.deepseek_v3",
                        "--config graph_trainer_deepseek_v3_debugmodel",
                        "--compile.mode aot_fx_trace",
                        f"--compile.inductor_compilation {inductor_compilation}",
                        "--compile.ep_overlap.enabled",
                        "--compile.ep_overlap.strategy graph",
                        f"--compile.ep_overlap.chunk_dim {mode}",
                        f"--compile.ep_overlap.module_fqn {modules}",
                        *(
                            ["--compile.enable_fsdp_dense_region_overlap"]
                            if modules == "layers.*.moe"
                            else []
                        ),
                        *ep_overlap_parallelism(),
                    ],
                ],
                f"aot_fx_trace deepseek_v3 FlexAttn {inductor_compilation}_inductor ep_overlap {variant}",
                f"aot_fx_trace_deepseek_v3_flexattn_{inductor_compilation}_inductor_ep_overlap_{variant}",
                ngpu=8,
            )
            for inductor_compilation, mode, modules, variant in ep_overlap_flex_tests
        ],
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.inductor_compilation full",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "aot_fx_trace deepseek_v3 GraphPP Interleaved1F1B full_inductor",
            "aot_fx_trace_deepseek_v3_graph_pp_interleaved_1f1b_full_inductor",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.inductor_compilation full",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule ZBVZeroBubble",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "aot_fx_trace deepseek_v3 GraphPP ZBVZeroBubble full_inductor",
            "aot_fx_trace_deepseek_v3_graph_pp_zbv_zero_bubble_full_inductor",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.inductor_compilation full",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule DualPipeV",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "aot_fx_trace deepseek_v3 GraphPP DualPipeV full_inductor",
            "aot_fx_trace_deepseek_v3_graph_pp_dual_pipe_v_full_inductor",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel_hybridep",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "aot_fx_trace deepseek_v3 FSDP+TP+HybridEP",
            "aot_fx_trace_deepseek_v3_hybridep",
            ngpu=4,
            disabled=True,
        ),
        # MinimalAsyncEP avoids the standard all-to-all load-balancing path and
        # is expected to remain CUDA-graphable under its constrained topology.
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel_minimal_async_ep",
                    "--compile.mode aot_fx_trace",
                    "--compile.memory_policy full",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "aot_fx_trace deepseek_v3 MinimalAsyncEP",
            "aot_fx_trace_deepseek_v3_minimal_async_ep",
            ngpu=4,
        ),
    ]


def _build_qwen3_tests() -> list[OverrideDefinitions]:
    """Qwen3-based integration tests (dense + MoE)."""
    return [
        # TODO: Disabled — this uses the default FlexAttention backend, and
        # FlexAttention + CP + regional_inductor is unsupported: the CP load
        # balancer injects an index-rearrange constant (torch
        # _context_parallel/_attention.py qkv_idx_restore) that
        # regional_inductor's make_fx re-trace cannot lift ("Attempting to use
        # FunctionalTensor on its own"). This is the same upstream issue noted
        # for the llama3 CP test above, which works around it with an SDPA
        # config. To re-enable, add a qwen3 SDPA debug config and switch to it
        # (mirroring aot_fx_trace_llama3_fsdp_tp_cp), or wait for flex + CP +
        # regional_inductor support upstream.
        #
        # cudagraph is also disabled here (kept for when this is re-enabled):
        # CUDA-graph replay of the coalesced FSDP collectives fails under
        # context parallelism with "CUDA error: invalid argument".
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.qwen3",
                    "--config graph_trainer_qwen3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.disable_passes cudagraph_pass",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "aot_fx_trace qwen3 FSDP+TP+CP",
            "aot_fx_trace_qwen3_fsdp_tp_cp",
            ngpu=8,
            disabled=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.qwen3",
                    "--config graph_trainer_qwen3_debugmodel_moe",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "aot_fx_trace qwen3 MoE FSDP+TP+EP",
            "aot_fx_trace_qwen3_moe_fsdp_tp_ep",
            ngpu=8,
        ),
    ]


def build_graph_trainer_test_list() -> list[OverrideDefinitions]:
    """All graph_trainer integration tests (Llama3 + DeepSeek-v3 + Qwen3)."""
    return _build_llama3_tests() + _build_deepseek_v3_tests() + _build_qwen3_tests()


def build_graph_trainer_default_test_list() -> list[OverrideDefinitions]:
    """Llama3 tests only (for default A10 machines)."""
    return _build_llama3_tests()


def _build_async_tp_tests() -> list[OverrideDefinitions]:
    """Async TP tests (require NVLink for symmetric memory)."""
    return [
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_8b",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.enable_async_tensor_parallel",
                    "--training.local_batch_size 2",
                    "--training.seq_len 512",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--hf_assets_path ./tests/assets/tokenizer",
                ],
            ],
            # async_tp (micro_pipeline_tp) requires shard_dim >= 1024.
            # 8B (dim=4096) with TP=2 gives shard=2048, above threshold.
            "aot_fx_trace llama3 FSDP+TP+async_tp",
            "aot_fx_trace_llama3_fsdp_tp_asynctp",
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


def _build_autoparallel_tests() -> list[OverrideDefinitions]:
    """AutoParallel integration tests for default runners."""
    return [
        # Uses the SDPA backend: AutoParallel's dynamo export
        # (_dynamo_graph_capture_for_export) pytree-flattens the default
        # FlexAttention BlockMask to plain (Fake)Tensors, so flex_attention then
        # fails with "'FakeTensor' object has no attribute 'BLOCK_SIZE'". SDPA is
        # maskless (is_causal) and carries no BlockMask, and its input_fn
        # (tokens, positions) binds correctly now that Decoder.forward lists
        # positions before attention_masks.
        # TODO: re-test on FlexAttention once BlockMask survives AutoParallel
        # graph capture.
        # TODO: Disabled due to upstream AutoParallel/PyTorch API skew. PyTorch
        # #186754 (2026-06-24) removed propagate_single_input_strategy in favor
        # of propagate_single_input_single_dim_strategy, but AutoParallel's
        # convert_element_type_rule still imports the old name, so the sharding
        # optimizer fails with ImportError. Re-enable once AutoParallel migrates.
        # https://github.com/pytorch/torchtitan/issues/3699
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.llama3",
                    "--config graph_trainer_llama3_debugmodel_sdpa_cross_entropy_loss",
                    "--compile.mode aot_fx_trace",
                    "--compile.enable_autoparallel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                ],
            ],
            "autoparallel llama3 FSDP+TP",
            "autoparallel_llama3_fsdp_tp",
            ngpu=4,
            disabled=True,
        ),
    ]


def _build_autoparallel_h100_tests() -> list[OverrideDefinitions]:
    """AutoParallel integration tests that require H100 runners."""
    return [
        # TODO: Disabled due to upstream AutoParallel regression in PyTorch
        # nightly dev20260508. AutoParallel rejects FakeTensor device
        # mismatch (traced on meta vs actual cuda). Re-enable once fixed.
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.deepseek_v3",
                    "--config graph_trainer_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--compile.enable_autoparallel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "autoparallel deepseek_v3 EFSDP+EP",
            "autoparallel_deepseek_v3_efsdp_ep",
            ngpu=4,
            disabled=True,
        ),
    ]


def build_graph_trainer_h100_test_list() -> list[OverrideDefinitions]:
    """DeepSeek-v3 + Qwen3 + async_tp tests (for H100 machines)."""
    return _build_deepseek_v3_tests() + _build_qwen3_tests() + _build_async_tp_tests()


def build_graph_trainer_autoparallel_test_list() -> list[OverrideDefinitions]:
    """AutoParallel tests for default runners."""
    return _build_autoparallel_tests()


def build_graph_trainer_autoparallel_h100_test_list() -> list[OverrideDefinitions]:
    """AutoParallel tests that require H100 runners."""
    return _build_autoparallel_h100_tests()


_TEST_SUITES_FUNCTION = {
    "graph_trainer": build_graph_trainer_test_list,
    "graph_trainer_default": build_graph_trainer_default_test_list,
    "graph_trainer_h100": build_graph_trainer_h100_test_list,
    "graph_trainer_autoparallel": build_graph_trainer_autoparallel_test_list,
    "graph_trainer_autoparallel_h100": build_graph_trainer_autoparallel_h100_test_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--gpu_arch_type",
        default="cuda",
        choices=["cuda","rocm","xpu"],
        help="GPU architecture type. Must be specified as either 'cuda','rocm' or 'xpu'.",
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
