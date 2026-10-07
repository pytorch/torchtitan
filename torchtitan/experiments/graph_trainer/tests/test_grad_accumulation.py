# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.grad_accumulation import (
    insert_graph_gradient_accumulation_before_reduction,
    insert_graph_gradient_accumulation_from_outputs,
)


class TestGraphGradientAccumulation(unittest.TestCase):
    @staticmethod
    def _grad_graph(*, reduce: bool = False) -> tuple[fx.GraphModule, str]:
        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(2, dtype=torch.float64)
        grad = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, 2))
        grad.meta["val"] = torch.empty(2, dtype=torch.float64)
        result = grad
        if reduce:
            result = graph.call_function(torch.ops.aten.mul.Tensor, args=(grad, 3))
            result.meta["val"] = torch.empty(2, dtype=torch.float64)
        graph.output((result,))
        return fx.GraphModule(torch.nn.Module(), graph), grad.name

    def test_last_graph_accumulates_before_reduction_without_wgrad_fusion(self) -> None:
        first, _ = self._grad_graph()
        middle, _ = self._grad_graph()
        last, raw_grad_name = self._grad_graph(reduce=True)
        self.assertEqual(
            insert_graph_gradient_accumulation_from_outputs(
                middle, num_param_grads=1, device=torch.device("cpu")
            ),
            (0,),
        )
        self.assertEqual(
            insert_graph_gradient_accumulation_before_reduction(
                last,
                param_grad_output_names=(raw_grad_name,),
                reduce_grad_input_names=(raw_grad_name,),
                accumulators=(0,),
                device=torch.device("cpu"),
            ),
            (0,),
        )

        x0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x1 = torch.tensor([3.0, 4.0], dtype=torch.float64)
        x2 = torch.tensor([5.0, 6.0], dtype=torch.float64)
        (accumulator,) = first(x0)
        (updated,) = middle(x1, accumulator)
        self.assertIs(updated, accumulator)
        (reduced,) = last(x2, accumulator)
        torch.testing.assert_close(accumulator, 2 * (x0 + x1 + x2))
        torch.testing.assert_close(reduced, 3 * accumulator)


if __name__ == "__main__":
    unittest.main()
