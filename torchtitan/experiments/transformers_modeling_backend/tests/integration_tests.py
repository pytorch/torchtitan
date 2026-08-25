# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torchtitan_recipes.tests.transformers_modeling_backend as recipes

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def build_transformers_modeling_backend_test_list() -> list[OverrideDefinitions]:
    """
    Build the Transformers modeling backend integration tests.

    Each entry exercises several parallelism axes at once to keep the matrix
    small. Coverage: T1 covers FSDP/TP/EP/CP for MoE + flex attention; T2 and
    T3 cover the dense path and PP (PP is not wired for MoE); T4 covers SFT.
    """
    return [
        OverrideDefinitions(
            configs=[recipes.transformers_backend_moe_fsdp_tp_ep_cp],
            test_descr="Transformers Backend MoE FSDP+TP+EP+CP",
            test_name="transformers_modeling_backend_moe_fsdp+tp+ep+cp",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.transformers_backend_dense_fsdp_tp_pp],
            test_descr="Transformers Backend Dense FSDP+TP+PP",
            test_name="transformers_modeling_backend_dense_fsdp+tp+pp",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.transformers_backend_dense_cp_pp],
            test_descr="Transformers Backend Dense CP+PP",
            test_name="transformers_modeling_backend_dense_cp+pp",
            ngpu=4,
        ),
        OverrideDefinitions(
            configs=[recipes.transformers_backend_sft],
            test_descr="Transformers Backend SFT ChatDataset",
            test_name="transformers_modeling_backend_sft",
            ngpu=2,
        ),
    ]


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
