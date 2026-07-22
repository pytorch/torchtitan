# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from argparse import Namespace
from unittest.mock import patch

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import _filter_tests


class TestIntegrationTestFiltering(unittest.TestCase):
    def test_filters_missing_optional_dependencies(self):
        args = Namespace(test_name="all", exclude=None, gpu_arch_type="cuda", ngpu=4)
        test_list = [
            OverrideDefinitions(test_name="core"),
            OverrideDefinitions(test_name="fa3", requires_fa3=True),
            OverrideDefinitions(test_name="torchcomms", requires_torchcomms=True),
            OverrideDefinitions(test_name="deep_ep", requires_deep_ep=True),
            OverrideDefinitions(test_name="torchao", requires_torchao=True),
        ]

        with (
            patch(
                "tests.integration_tests.run_tests.num_available_gpus",
                return_value=4,
            ),
            patch(
                "tests.integration_tests.run_tests.has_cuda_capability",
                return_value=True,
            ),
            patch("tests.integration_tests.run_tests.has_fa3", return_value=False),
            patch(
                "tests.integration_tests.run_tests.has_torchcomms",
                return_value=False,
            ),
            patch(
                "tests.integration_tests.run_tests.has_deep_ep",
                return_value=False,
            ),
            patch("tests.integration_tests.run_tests.has_torchao", return_value=False),
        ):
            runnable, skipped_ngpu = _filter_tests(args, test_list)

        self.assertEqual([test.test_name for test in runnable], ["core"])
        self.assertEqual(skipped_ngpu, [])

    def test_caps_requested_gpus_by_visible_devices(self):
        args = Namespace(test_name="all", exclude=None, gpu_arch_type="cuda", ngpu=8)
        test_list = [
            OverrideDefinitions(test_name="four_gpu", ngpu=4),
            OverrideDefinitions(test_name="eight_gpu", ngpu=8),
        ]

        with patch(
            "tests.integration_tests.run_tests.num_available_gpus", return_value=4
        ):
            runnable, skipped_ngpu = _filter_tests(args, test_list)

        self.assertEqual([test.test_name for test in runnable], ["four_gpu"])
        self.assertEqual(
            [test.test_name for test in skipped_ngpu],
            ["eight_gpu"],
        )

    def test_fa3_is_only_required_on_sm90_or_newer(self):
        args = Namespace(test_name="all", exclude=None, gpu_arch_type="cuda", ngpu=4)
        test_list = [OverrideDefinitions(test_name="fa3", requires_fa3=True)]

        with (
            patch(
                "tests.integration_tests.run_tests.num_available_gpus",
                return_value=4,
            ),
            patch(
                "tests.integration_tests.run_tests.has_cuda_capability",
                return_value=False,
            ),
            patch("tests.integration_tests.run_tests.has_fa3", return_value=False),
        ):
            runnable, skipped_ngpu = _filter_tests(args, test_list)

        self.assertEqual([test.test_name for test in runnable], ["fa3"])
        self.assertEqual(skipped_ngpu, [])


if __name__ == "__main__":
    unittest.main()
