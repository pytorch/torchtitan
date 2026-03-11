# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_sft_test_list() -> list[OverrideDefinitions]:
    integration_tests_flavors = [
        OverrideDefinitions(
            [
                [
                    "--module sft",
                    "--config sft_debugmodel",
                    "--training.steps 10",
                ],
            ],
            "SFT training with debugmodel",
            "sft_debugmodel",
            ngpu=2,
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "sft": build_sft_test_list,
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

    test_list = _TEST_SUITES_FUNCTION["sft"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
