# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.experiments.graph_trainer.passes import resolve_registered_graph_passes
from torchtitan.experiments.graph_trainer.registry import (
    GRAPH_PASS_REGISTRY,
    register_graph_pass,
)


def _first_pass(gm, example_inputs=None):
    return gm


def _second_pass(gm, example_inputs=None):
    return gm


class TestRegisteredGraphPasses(unittest.TestCase):
    def setUp(self):
        register_graph_pass(_first_pass)
        register_graph_pass(_second_pass)

    def tearDown(self):
        GRAPH_PASS_REGISTRY.pop(_first_pass.__name__)
        GRAPH_PASS_REGISTRY.pop(_second_pass.__name__)

    def test_resolves_in_config_order(self):
        names = [_second_pass.__name__, _first_pass.__name__]
        self.assertEqual(
            resolve_registered_graph_passes(names), [_second_pass, _first_pass]
        )

    def test_rejects_invalid_selections(self):
        with self.assertRaisesRegex(ValueError, "not a registered graph pass"):
            resolve_registered_graph_passes(["missing_pass"])
        with self.assertRaisesRegex(ValueError, "more than once"):
            resolve_registered_graph_passes(
                [_first_pass.__name__, _first_pass.__name__]
            )

    def test_rejects_conflicting_registration(self):
        def _first_pass(gm, example_inputs=None):
            return gm

        with self.assertRaisesRegex(ValueError, "conflicts with"):
            register_graph_pass(_first_pass)


if __name__ == "__main__":
    unittest.main()
