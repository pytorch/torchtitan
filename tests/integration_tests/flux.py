# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shlex

import torchtitan_recipes.tests.flux as recipes

from torchtitan.tools.logging import logger

from tests.integration_tests import OverrideDefinitions, validate_fake_pg_compatibility
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
            use_real_pg=True,
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


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    *,
    use_fake_pg: bool,
):
    # run_test supports sequence of tests.
    test_name = test_flavor.test_name
    dump_folder_arg = f"--dump_folder {output_dir}/{test_name}"

    all_ranks = ",".join(map(str, range(test_flavor.ngpu)))
    base_env = os.environ.copy()
    base_env["NGPU"] = str(test_flavor.ngpu)
    base_env["LOG_RANK"] = all_ranks
    base_env.pop("COMM_MODE", None)
    if use_fake_pg:
        base_env["COMM_MODE"] = "fake_backend"

    for idx, override_arg in enumerate(test_flavor.override_args):
        config_fn = test_flavor.configs[idx]
        if use_fake_pg:
            validate_fake_pg_compatibility(test_flavor, config_fn())
        env = base_env.copy()
        env["MODULE"] = config_fn.__module__
        env["CONFIG"] = config_fn.__name__
        cmd = "./run_train.sh"
        # dump compile trace for debugging purpose
        env["TORCH_TRACE"] = f"{output_dir}/{test_name}/compile_trace"

        # save checkpoint (idx == 0) and load it for generation (idx == 1)
        if test_name == "hsdp+cp+validation+inference" and idx == 1:
            # For flux generation, test using inference script
            cmd = "torchtitan/models/flux/run_infer.sh"

        cmd += " " + dump_folder_arg
        if override_arg:
            cmd += " " + shlex.join(override_arg)

        logger.info(
            f"=====Flux Integration test, flavor : {test_flavor.test_descr}, command : {cmd}====="
        )
        result = _run_cmd(cmd, env=env)
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

        use_fake_pg = args.execution_mode == "fake_pg"
        if use_fake_pg and test_flavor.use_real_pg:
            continue
        if args.test_scope == "real_pg_required" and not test_flavor.use_real_pg:
            continue
        if not use_fake_pg and args.ngpu < test_flavor.ngpu:
            logger.info(
                f"Skipping test {test_flavor.test_name} that requires {test_flavor.ngpu} gpus,"
                f" because --ngpu arg is {args.ngpu}"
            )
        else:
            run_single_test(
                test_flavor,
                args.output_dir,
                use_fake_pg=use_fake_pg,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument(
        "--test_name",
        default="all",
        help="test to run, acceptable values: `test_name` in `build_test_list` (default: all)",
    )
    parser.add_argument("--ngpu", default=8, type=int)
    parser.add_argument(
        "--execution_mode",
        choices=("fake_pg", "real_pg"),
        default="real_pg",
        help="Communication mode used to execute the Flux tests.",
    )
    parser.add_argument(
        "--test_scope",
        choices=("all", "real_pg_required"),
        default="all",
        help="Run every selected test or only tests marked use_real_pg=True.",
    )
    args = parser.parse_args()

    if args.execution_mode == "fake_pg" and args.test_scope == "real_pg_required":
        parser.error("real_pg_required test scope requires --execution_mode real_pg")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        raise RuntimeError("Please provide an empty output directory.")

    test_list = _TEST_SUITES_FUNCTION["flux"]()
    run_tests(args, test_list)


if __name__ == "__main__":
    main()
