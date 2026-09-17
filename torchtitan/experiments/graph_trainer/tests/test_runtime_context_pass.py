# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest

import torch

from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    GraphPassRuntimeContext,
    RuntimeContextGraphPass,
)


class TestRuntimeContextGraphPass(unittest.TestCase):
    def setUp(self) -> None:
        self.gm = torch.fx.symbolic_trace(lambda x: x + 1)
        self.example_inputs = (torch.ones(2),)
        self.runtime_context = GraphPassRuntimeContext(
            traced_result=object(),
            module=torch.nn.Linear(2, 2),
            args=(torch.ones(2),),
            train_context=contextlib.nullcontext,
        )

    def test_context_is_only_passed_to_opted_in_pass(self) -> None:
        calls = []

        def ordinary_pass(gm, example_inputs):
            calls.append(("ordinary", example_inputs))
            return gm

        def runtime_pass(gm, example_inputs, *, runtime_context):
            calls.append(("runtime", runtime_context))
            return gm

        result = apply_graph_passes(
            self.gm,
            self.example_inputs,
            [ordinary_pass, RuntimeContextGraphPass(runtime_pass)],
            runtime_context=self.runtime_context,
        )

        self.assertIs(result, self.gm)
        self.assertEqual(calls[0][0], "ordinary")
        self.assertIs(calls[1][1], self.runtime_context)

    def test_context_aware_pass_requires_context(self) -> None:
        def runtime_pass(gm, example_inputs, *, runtime_context):
            return gm

        with self.assertRaisesRegex(
            RuntimeError, "requires a graph pass runtime context"
        ):
            apply_graph_passes(
                self.gm,
                self.example_inputs,
                [RuntimeContextGraphPass(runtime_pass)],
            )


if __name__ == "__main__":
    unittest.main()
