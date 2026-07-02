# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses

from tests.integration_tests import OverrideDefinitions


def _enable_spmd_backend(t: OverrideDefinitions, backend: str) -> OverrideDefinitions:
    """Inject ``--parallelism.spmd_backend`` into every model test variant."""
    test_name = f"{t.test_name}_{backend}"
    new_args = []
    for variant in t.override_args:
        variant = tuple(
            arg.replace(f"{t.test_name}/", f"{test_name}/") for arg in variant
        )
        prefix = [f"--parallelism.spmd_backend {backend}"]
        suffix = []
        # Compile, PP, and explicit AC modes are not compatible with SPMD
        # typechecking yet; keep those as backend-only coverage.
        if backend == "spmd_types" and not any(
            token in arg
            for arg in variant
            for token in (
                "compile.enable",
                "pipeline_parallel_degree",
                "activation-checkpoint:",
            )
        ):
            prefix.append("--debug.spmd_typechecking")
            suffix.append("activation-checkpoint:none")
        new_args.append(tuple(prefix) + tuple(variant) + tuple(suffix))
    return dataclasses.replace(
        t,
        override_args=tuple(new_args),
        test_name=test_name,
    )


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build a dictionary of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will broadly cover
    all the supported model parallelism patterns on all the supported models.

    Returns:
        A dictionary where each key is a model name and value is a list of OverrideDefinitions
    """
    model_tests = [
        # Integration Test Cases for DeepSeek V3
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "DeepSeek V3 FSDP+EP+compile",
            "deepseek_v3_fsdp+ep+compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "DeepSeek V3 PP+FSDP+TP+EP",
            "deepseek_v3_pp+fsdp+tp+ep",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module deepseek_v3 --config deepseek_v3_debugmodel",
                    "--parallelism.data_parallel_replicate_degree 2",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                ],
            ],
            "DeepSeek V3 HSDP+EP",
            "deepseek_v3_hsdp+ep",
            ngpu=4,
        ),
        # Integration Test Cases for Qwen3 dense and MoE model
        OverrideDefinitions(
            [
                [
                    "--module qwen3 --config qwen3_debugmodel_moe_param_groups",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "Qwen3 MoE FSDP+TP+EP (param groups)",
            "qwen3_moe_fsdp+tp+ep_param_groups",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module qwen3 --config qwen3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.no-enable-sequence-parallel",
                    "--parallelism.context_parallel_degree 2",
                ],
                [
                    "--module qwen3 --config qwen3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "Qwen3 FSDP+TP+CP (SP disabled)",
            "qwen3_fsdp+tp+cp_no_sp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module qwen3 --config qwen3_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--compile.enable",
                    "--override.imports torchtitan.overrides.helion_rope",
                ],
            ],
            "Qwen3 fused QKV FSDP+TP+CP + compile + Helion RoPE override",
            "qwen3_fused_qkv_fsdp+tp+cp_compile_helion_rope",
            ngpu=8,
            # The Helion fused cos/sin RoPE kernel is CUDA-only and its autotuned
            # configs are tuned for NVIDIA H100; skip on ROCm where it is
            # unvalidated (see torchtitan/overrides/helion_rope.py).
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module qwen3 --config qwen3_debugmodel_non_fused_qkv",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            # Reverse test: fused QKV is the debugmodel default, so exercise the
            # separate wq/wk/wv projection path under FSDP+TP+CP.
            "Qwen3 non-fused QKV FSDP+TP+CP",
            "qwen3_non_fused_qkv_fsdp+tp+cp",
            ngpu=8,
        ),
        # Integration Test Cases for Qwen3.5
        OverrideDefinitions(
            [
                [
                    "--module qwen3_5 --config qwen35_debugmodel_moe",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                ],
            ],
            "Qwen3.5 MoE FSDP+TP+EP+PP",
            "qwen3_5_moe_fsdp+tp+ep+pp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module qwen3_5 --config qwen35_debugmodel_varlen_attn",
                    "--parallelism.data_parallel_shard_degree=4",
                    "--training.steps 10",
                    "--comm.train_timeout_seconds 600",
                    "activation-checkpoint:selective",
                ]
            ],
            "Qwen3.5 FSDP+VARLEN_ATTN + per op SAC",
            "qwen3_5_fsdp+varlen_attn+per_op_sac",
            ngpu=4,
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            [
                [
                    "--module qwen3_5 --config qwen35_debugmodel_varlen_attn",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.data_parallel_shard_degree 1",
                    "--training.steps 10",
                    "--comm.train_timeout_seconds 600",
                    "activation-checkpoint:selective",
                ]
            ],
            "Qwen3.5 TP+VARLEN_ATTN + per op SAC",
            "qwen3_5_tp+varlen_attn+per_op_sac",
            ngpu=2,
            skip_rocm_test=True,
        ),
        # Integration Test Cases for gpt-oss
        OverrideDefinitions(
            [
                [
                    "--module gpt_oss --config gpt_oss_debugmodel",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 4",
                    "--compile.enable",
                ],
            ],
            "Gpt-oss FSDP+TP+EP+compile",
            "gpt_oss_fsdp+tp+ep+compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module gpt_oss --config gpt_oss_debugmodel_flex",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule Interleaved1F1B",
                    "--parallelism.expert_parallel_degree 4",
                    "activation-checkpoint:selective",
                ],
            ],
            "Gpt-oss PP+FSDP+EP+SACOP",
            "gpt_oss_pp+fsdp+ep+sacop",
            ngpu=8,
        ),
    ]

    return [
        *model_tests,
        *[_enable_spmd_backend(t, "full_dtensor") for t in model_tests],
        *[_enable_spmd_backend(t, "spmd_types") for t in model_tests],
    ]
