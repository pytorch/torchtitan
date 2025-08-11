# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import subprocess

from tests.integration_tests.features import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests

from torchtitan.tools.logging import logger


def build_flux_test_list():
    """
    key is the config file name and value is a list of OverrideDefinitions
    that is used to generate variations of integration tests based on the
    same root config file.
    """
    integration_tests_flavors = [
        # basic tests
        OverrideDefinitions(
            [
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--profiling.enable_profiling",
                    "--metrics.enable_tensorboard",
                ],
            ],
            "default",
            "default",
        ),
        # Checkpointing tests.
        OverrideDefinitions(
            [
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--checkpoint.enable_checkpoint",
                ],
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--checkpoint.enable_checkpoint",
                    "--training.steps 20",
                ],
            ],
            "Checkpoint Integration Test - Save Load Full Checkpoint",
            "full_checkpoint",
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--checkpoint.enable_checkpoint",
                    "--checkpoint.last_save_model_only",
                ],
            ],
            "Checkpoint Integration Test - Save Model Only fp32",
            "last_save_model_only_fp32",
        ),
        # Parallelism tests.
        OverrideDefinitions(
            [
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--parallelism.data_parallel_shard_degree 4",
                    "--parallelism.data_parallel_replicate_degree 1",
                ]
            ],
            "FSDP",
            "fsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--parallelism.data_parallel_shard_degree 2",
                    "--parallelism.data_parallel_replicate_degree 2",
                ]
            ],
            "HSDP",
            "hsdp",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--model.name flux",
                    "--training.test_mode",
                    "--encoder.clip_encoder torchtitan/experiments/flux/tests/assets/clip-vit-large-patch14/",
                    "--encoder.t5_encoder torchtitan/experiments/flux/tests/assets/t5-v1_1-xxl/",
                    "--model.tokenizer_path tests/assets/tokenizer",
                    "--validation.enabled",
                ]
            ],
            "Flux Validation Test",
            "validation",
        ),
    ]
    return integration_tests_flavors


_TEST_SUITES_FUNCTION = {
    "flux": build_flux_test_list,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--config_path",
        default="./torchtitan/experiments/flux/train_configs/debug_model.toml",
        help="Base config path for integration tests. This is the config that will be used as a base for all tests.",
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

    test_list = _TEST_SUITES_FUNCTION["flux"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
