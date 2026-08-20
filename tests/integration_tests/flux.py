# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torchtitan_recipes.tests.flux as recipes

from torchtitan.tools.logging import logger

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import _run_cmd


def build_flux_test_list() -> list[OverrideDefinitions]:
    """
    Build the list of Flux integration tests.

    Each entry names one configuration per run; see ``torchtitan_recipes.tests.flux``.
    """
    return [
        OverrideDefinitions(
            configs=[
                recipes.flux_debugmodel_hsdp2x2_cp2_validation,
                recipes.flux_debugmodel_test,
            ],
            test_descr="HSDP+CP+Validation+Inference",
            test_name="hsdp+cp+validation+inference",
            ngpu=8,
        ),
        OverrideDefinitions(
            configs=[recipes.flux_debugmodel_compile],
            test_descr="Flux FSDP+compile",
            test_name="flux_fsdp+compile",
        ),
    ]


_TEST_SUITES_FUNCTION = {
    "flux": build_flux_test_list,
}


def run_single_test(test_flavor: OverrideDefinitions, output_dir: str):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--dump_folder {output_dir}/{test_name}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))

    for idx, override_arg in enumerate(test_flavor.override_args):
        config_fn = test_flavor.configs[idx]
        env = (
            f"MODULE={config_fn.__module__} CONFIG={config_fn.__name__} "
            f"NGPU={test_flavor.ngpu} LOG_RANK={all_ranks}"
        )
        cmd = f"{env} ./run_train.sh"
        # dump compile trace for debugging purpose
        cmd = f'TORCH_TRACE="{output_dir}/{test_name}/compile_trace" ' + cmd

        # save checkpoint (idx == 0) and load it for generation (idx == 1)
        if test_name == "hsdp+cp+validation+inference" and idx == 1:
            # For flux generation, test using inference script
            cmd = f"{env} torchtitan/models/flux/run_infer.sh"

        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + " ".join(override_arg)

        logger.info(
            f"=====Flux Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )
        result = _run_cmd(cmd)
        logger.info(result.stdout)
        if result.returncode != 0:
            raise Exception(
                f"Flux Integration test failed, flavor : {test_flavor.test_descr}, command : {cmd}"
            )


def run_tests(args, test_list: list[OverrideDefinitions]):
    """Run all integration tests to test the core features of TorchTitan
    Override the run_tests function in run_tests.py because FLUX model
    uses different train.py in command to run the model"""

    for test_flavor in test_list:
        # Filter by test_name if specified
        if args.test_name != "all" and test_flavor.test_name != args.test_name:
            continue

        # Check if we have enough GPUs
        if args.ngpu < test_flavor.ngpu:
            logger.info(
                f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
                f" because --ngpu arg is {args.ngpu}"
            )
        else:
            run_single_test(test_flavor, args.output_dir)


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

    test_list = _TEST_SUITES_FUNCTION["flux"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
