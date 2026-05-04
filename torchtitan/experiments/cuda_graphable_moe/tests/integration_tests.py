# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def _build_paged_stash_tests() -> list[OverrideDefinitions]:
    """Paged stash integration tests (require NVLink GPUs + DeepEP)."""
    return [
        OverrideDefinitions(
            [
                [
                    "--module cuda_graphable_moe.deepseek_v3",
                    "--config paged_stash_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.memory_policy paged_stash_save_only",
                ],
            ],
            "aot_fx_trace deepseek_v3 paged_stash_save_only (baseline)",
            "paged_stash_save_only",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module cuda_graphable_moe.deepseek_v3",
                    "--config paged_stash_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.memory_policy paged_stash",
                ],
            ],
            "aot_fx_trace deepseek_v3 paged_stash",
            "paged_stash",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module cuda_graphable_moe.deepseek_v3",
                    "--config paged_stash_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.memory_policy paged_stash_spillover",
                ],
            ],
            "aot_fx_trace deepseek_v3 paged_stash_spillover",
            "paged_stash_spillover",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module cuda_graphable_moe.deepseek_v3",
                    "--config paged_stash_deepseek_v3_debugmodel",
                    "--compile.mode aot_fx_trace",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--compile.memory_policy paged_stash_overflow_test",
                ],
            ],
            "aot_fx_trace deepseek_v3 paged_stash_overflow_test",
            "paged_stash_overflow_test",
            ngpu=4,
        ),
    ]


def build_cuda_graphable_moe_test_list() -> list[OverrideDefinitions]:
    """All cuda_graphable_moe integration tests."""
    return _build_paged_stash_tests()


_TEST_SUITES_FUNCTION = {
    "cuda_graphable_moe": build_cuda_graphable_moe_test_list,
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
        default="cuda_graphable_moe",
        choices=list(_TEST_SUITES_FUNCTION.keys()),
        help="Which test suite to run (default: cuda_graphable_moe)",
    )
    parser.add_argument(
        "--test_name",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
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
