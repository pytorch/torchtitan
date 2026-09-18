# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import unittest

import torch

from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    GraphPassRuntimeContext,
)


class TestGraphPassRuntimeContext(unittest.TestCase):
    def test_pipeline_can_bind_runtime_context_with_partial(self) -> None:
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        example_inputs = (torch.ones(2),)
        traced_result = object()
        runtime_context = GraphPassRuntimeContext(
            module=torch.nn.Linear(2, 2),
            args=(torch.ones(2),),
            train_context=contextlib.nullcontext,
        )
        received = []

        def runtime_pass(
            gm,
            example_inputs,
            *,
            traced_result,
            runtime_context,
        ):
            received.append((traced_result, runtime_context))
            return gm

        def pipeline_fn(
            traced_result,
            config,
            *,
            parallel_dims=None,
            runtime_context=None,
        ):
            return [
                functools.partial(
                    runtime_pass,
                    traced_result=traced_result,
                    runtime_context=runtime_context,
                )
            ]

        passes = pipeline_fn(
            traced_result,
            None,
            runtime_context=runtime_context,
        )

        result = apply_graph_passes(gm, example_inputs, passes)

        self.assertIs(result, gm)
        self.assertEqual(received, [(traced_result, runtime_context)])


if __name__ == "__main__":
    unittest.main()
