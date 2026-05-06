# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def _build_flex_shard_tests() -> list[OverrideDefinitions]:
    """FlexShard integration tests (1D mesh + multi-mesh)."""
    return [
        # === FSDP + TP multi-mesh tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.data_parallel_shard_degree=2",
                ],
            ],
            "FlexShard JIT 2D FSDP+TP",
            "flex_shard_jit_2d_fsdp_tp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode aot",
                    "--parallelism.tensor_parallel_degree=2",
                    "--parallelism.data_parallel_shard_degree=2",
                ],
            ],
            "FlexShard AOT 2D FSDP+TP",
            "flex_shard_aot_2d_fsdp_tp",
            ngpu=4,
        ),
        # === JIT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=auto_bucketing",
                ],
            ],
            "FlexShard JIT 1D+auto_bucketing",
            "flex_shard_jit_1d_auto_bucketing",
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.passes=transformer_block_bucketing",
                ],
            ],
            "FlexShard JIT 1D+transformer_block_bucketing",
            "flex_shard_jit_1d_transformer_block_bucketing",
        ),
        # === AOT mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode aot",
                ],
            ],
            "FlexShard AOT 1D",
            "flex_shard_aot_1d",
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode aot",
                    "--compile.passes auto_bucketing",
                ],
            ],
            "FlexShard AOT 1D+auto_bucketing",
            "flex_shard_aot_1d_auto_bucketing",
        ),
        # === aot_fx_trace mode tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode aot_fx_trace",
                ],
            ],
            "FlexShard aot_fx_trace 1D",
            "flex_shard_aot_fx_trace_1d",
        ),
        # === CPU offload tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--training.enable_cpu_offload",
                ],
            ],
            "FlexShard JIT 1D+cpu_offload",
            "flex_shard_jit_1d_cpu_offload",
        ),
        # === Mixed precision tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--training.mixed_precision_param bfloat16",
                    "--training.mixed_precision_reduce float32",
                ],
            ],
            "FlexShard JIT 1D+mixed_precision",
            "flex_shard_jit_1d_mixed_precision",
        ),
        # === Phase 5: Owned (param_boundary) placement tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.shard_placement param_boundary",
                ],
            ],
            "FlexShard JIT 1D Owned (param_boundary)",
            "flex_shard_jit_1d_owned",
        ),
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode aot",
                    "--compile.shard_placement param_boundary",
                ],
            ],
            "FlexShard AOT 1D Owned (param_boundary)",
            "flex_shard_aot_1d_owned",
        ),
        # === Phase 5: Uneven Shard tests (3 GPUs forces uneven splits) ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                ],
            ],
            "FlexShard JIT 1D uneven (3 GPUs)",
            "flex_shard_jit_1d_uneven",
            ngpu=3,
        ),
        # === Phase 5: RaggedShard placement tests ===
        OverrideDefinitions(
            [
                [
                    "--module graph_trainer.flex_shard_llama3",
                    "--config graph_trainer_flex_shard_llama3_debugmodel",
                    "--compile.mode jit",
                    "--compile.backend aot_eager",
                    "--compile.shard_placement ragged",
                ],
            ],
            "FlexShard JIT 1D RaggedShard",
            "flex_shard_jit_1d_ragged",
        ),
    ]


_TEST_SUITES_FUNCTION = {
    "flex_shard": _build_flex_shard_tests,
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
        default="flex_shard",
        choices=list(_TEST_SUITES_FUNCTION.keys()),
        help="Which test suite to run (default: flex_shard)",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="test to run, acceptable values: `test_name` in test list (default: all)",
    )
    parser.add_argument("--ngpu", default=4, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION[args.test_suite]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
