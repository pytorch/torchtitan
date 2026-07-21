# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_transformers_modeling_backend_test_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.

    Each entry exercises several parallelism axes at once to keep the matrix
    small. Coverage: T1 covers FSDP/TP/EP/CP for MoE + flex attention; T2 and
    T3 cover the dense path and PP (PP is not wired for MoE); T4 covers SFT.
    """
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [
                    "--module transformers_modeling_backend",
                    "--config transformers_modeling_backend_debugmodel_moe",
                    "--parallelism.data_parallel_shard_degree -1",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.expert_parallel_degree 2",
                    "--parallelism.context_parallel_degree 2",
                    "--parallelism.context_parallel_load_balancer ptrr",
                    "--training.steps 2",
                ],
            ],
            "Transformers Backend MoE FSDP+TP+EP+CP",
            "transformers_modeling_backend_moe_fsdp+tp+ep+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module transformers_modeling_backend",
                    "--config transformers_modeling_backend_debugmodel",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--training.steps 2",
                ],
            ],
            "Transformers Backend Dense FSDP+TP+PP",
            "transformers_modeling_backend_dense_fsdp+tp+pp",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--module transformers_modeling_backend",
                    "--config transformers_modeling_backend_debugmodel",
                    "--parallelism.data_parallel_shard_degree 1",
                    "--parallelism.context_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--parallelism.pipeline_parallel_schedule 1F1B",
                    "--parallelism.context_parallel_load_balancer ptrr",
                    "--training.steps 2",
                ],
            ],
            "Transformers Backend Dense CP+PP",
            "transformers_modeling_backend_dense_cp+pp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module transformers_modeling_backend",
                    "--config transformers_modeling_backend_sft_debugmodel",
                    "--training.steps 2",
                ],
            ],
            "Transformers Backend SFT ChatDataset",
            "transformers_modeling_backend_sft",
            ngpu=2,
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "transformers_modeling_backend": build_transformers_modeling_backend_test_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
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

    test_list = _TEST_SUITES_FUNCTION["transformers_modeling_backend"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
