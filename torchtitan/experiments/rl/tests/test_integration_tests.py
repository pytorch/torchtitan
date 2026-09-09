# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.rl.tests.integration_tests import build_rl_test_list


def test_rl_ci_keeps_zero_std_reward_groups() -> None:
    expected_override = (
        "--async-loop.training-sample-builder.no-drop-zero-std-reward-groups"
    )

    for test in build_rl_test_list():
        for override_args in test.override_args:
            assert expected_override in override_args, test.test_name
