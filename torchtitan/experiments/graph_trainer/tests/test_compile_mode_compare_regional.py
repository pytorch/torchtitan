# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from torchtitan.experiments.graph_trainer.tests.test_compile_mode_compare import (
    CompareCase,
    run_compare_case,
)

LLAMA3_REGIONAL_AOT_VS_AOT_FX_TRACE = CompareCase(
    test_name="llama3_regional_aot_vs_aot_fx_trace",
    module="graph_trainer.llama3",
    config="graph_trainer_llama3_debugmodel_flex_attn",
    baseline_name="aot_regional_inductor",
    baseline_args=(
        "--compile.mode",
        "aot",
        "--compile.passes",
        "regional_inductor",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
    ),
    candidate_name="aot_fx_trace_regional_inductor",
    candidate_args=(
        "--compile.mode",
        "aot_fx_trace",
        "--compile.passes",
        "regional_inductor",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
    ),
    requires_h100=True,
)

DEEPSEEK_V3_REGIONAL_AOT_VS_AOT_FX_TRACE = CompareCase(
    test_name="deepseek_v3_regional_aot_vs_aot_fx_trace",
    module="graph_trainer.deepseek_v3",
    config="graph_trainer_deepseek_v3_debugmodel_flex_attn",
    baseline_name="aot_regional_inductor",
    baseline_args=(
        "--compile.mode",
        "aot",
        "--compile.passes",
        "regional_inductor",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.expert_parallel_degree",
        "4",
        "--parallelism.expert_tensor_parallel_degree",
        "1",
    ),
    candidate_name="aot_fx_trace_regional_inductor",
    candidate_args=(
        "--compile.mode",
        "aot_fx_trace",
        "--compile.passes",
        "regional_inductor",
        "--parallelism.data_parallel_shard_degree",
        "4",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.expert_parallel_degree",
        "4",
        "--parallelism.expert_tensor_parallel_degree",
        "1",
    ),
    requires_h100=True,
)


def test_llama3_regional_aot_vs_aot_fx_trace(tmp_path: Path) -> None:
    run_compare_case(case=LLAMA3_REGIONAL_AOT_VS_AOT_FX_TRACE, tmp_path=tmp_path)


def test_deepseek_v3_regional_aot_vs_aot_fx_trace(tmp_path: Path) -> None:
    run_compare_case(case=DEEPSEEK_V3_REGIONAL_AOT_VS_AOT_FX_TRACE, tmp_path=tmp_path)
