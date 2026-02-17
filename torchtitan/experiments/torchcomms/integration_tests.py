# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_torchcomms_test_list() -> list[OverrideDefinitions]:
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [],
            ],
            "1D FSDP",
            "1d",
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.tensor_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "FSDP+TP+PP+compile",
            "3d_dp+tp+pp+compile",
            ngpu=8,
        ),
        OverrideDefinitions(
            [
                [
                    "--parallelism.context_parallel_degree 2",
                    "--parallelism.pipeline_parallel_degree 2",
                    "--compile.enable",
                ],
            ],
            "DP+CP+PP+compile",
            "3d_dp+cp+pp+compile",
            ngpu=8,
        ),
        # TODO: Enable async TP test once fixes are available for running on CI nodes.
        # OverrideDefinitions(
        #    [
        #        [
        #            "--compile.enable",
        #            "--parallelism.context_parallel_degree 2",
        #            "--parallelism.tensor_parallel_degree 2",
        #            "--parallelism.enable_async_tensor_parallel",
        #        ],
        #    ],
        #    "3D CP+async TP compile",
        #    "3d_cp+asynctp_compile",
        #    ngpu=8,
        # ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "torchcomms": build_torchcomms_test_list,
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

    test_list = _TEST_SUITES_FUNCTION["torchcomms"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
