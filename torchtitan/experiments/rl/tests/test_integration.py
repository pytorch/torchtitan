# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end RL GRPO integration tests.

Each ``OverrideDefinitions`` entry below is a separate parametrized
pytest test that spawns ``python grpo.py`` with the given config
overrides. The marker (``gpu_test`` or ``h100_test``) decides which
CI lane picks the case up. Adding a new config = adding to
``build_rl_test_list`` / ``build_rl_h100_test_list``; no YAML edit
needed.
"""

import os
import shlex
import subprocess
import sys
import time

import pytest

from tests.integration_tests import OverrideDefinitions

from torchtitan.experiments.rl.tests.utils import gpu_test, h100_test
from torchtitan.tools.logging import logger


def build_rl_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            [
                [
                    "--module rl",
                    "--config rl_grpo_qwen3_0_6b",
                    "--trainer.parallelism.tensor_parallel_degree 2",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--generator.sampling.n 2",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                    "--compile.no-enable",
                    "--generator.cudagraph.no-enable",
                ],
            ],
            "RL GRPO TP=2 no compile",
            "rl_grpo_tp2_no_compile",
            ngpu=4,
        ),
        OverrideDefinitions(
            [
                [
                    "--module rl",
                    "--config rl_grpo_qwen3_0_6b",
                    "--trainer.parallelism.tensor_parallel_degree 2",
                    "--generator.parallelism.tensor_parallel_degree 2",
                    "--generator.sampling.n 2",
                    "--trainer.debug.no_batch_invariant",
                    "--generator.debug.no_batch_invariant",
                ],
            ],
            "RL GRPO TP=2 compile",
            "rl_grpo_tp2_compile",
            ngpu=4,
        ),
    ]


def build_rl_h100_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            [
                [
                    "--module rl",
                    "--config rl_grpo_qwen3_0_6b_batch_invariant",
                ],
            ],
            "RL GRPO TP=2 batch-invariant + deterministic",
            "rl_grpo_tp2_batch_invariant",
            ngpu=4,
        ),
    ]


def run_single_test(
    test_flavor: OverrideDefinitions,
    output_dir: str,
    hf_assets_path: str = "",
) -> None:
    """Spawn ``python grpo.py`` with the given config overrides.

    Unlike the standard ``./run_train.sh`` (torchrun) pattern, this
    runs ``python grpo.py`` directly: the RL script manages its own
    distributed setup via Monarch.
    """
    test_name = test_flavor.test_name
    dump_folder = os.path.join(output_dir, test_name)

    for override_arg in test_flavor.override_args:
        cmd = [
            sys.executable,
            "torchtitan/experiments/rl/grpo.py",
            "--dump_folder",
            dump_folder,
        ]
        if hf_assets_path:
            cmd.extend(["--hf_assets_path", hf_assets_path])
        for arg in override_arg:
            cmd.extend(shlex.split(arg))

        logger.info(
            f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"RL integration test: {test_flavor.test_descr}, "
            f"command: {shlex.join(cmd)} ====="
        )

        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise AssertionError(
                f"RL integration test failed: {test_flavor.test_descr}, "
                f"command: {shlex.join(cmd)}"
            )


def _output_dir(tmp_path) -> str:
    return os.environ.get("RL_TEST_OUTPUT_DIR", str(tmp_path))


class TestRLGRPOIntegration:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(case, marks=gpu_test(num_gpus=case.ngpu), id=case.test_name)
            for case in build_rl_test_list()
        ],
    )
    def test_grpo(self, case, tmp_path):
        run_single_test(
            case, _output_dir(tmp_path), os.environ.get("HF_ASSETS_PATH", "")
        )

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(case, marks=h100_test(num_gpus=case.ngpu), id=case.test_name)
            for case in build_rl_h100_test_list()
        ],
    )
    def test_grpo_h100(self, case, tmp_path):
        run_single_test(
            case, _output_dir(tmp_path), os.environ.get("HF_ASSETS_PATH", "")
        )
