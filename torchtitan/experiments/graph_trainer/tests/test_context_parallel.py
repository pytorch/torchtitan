# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.nn as nn

from torchtitan.experiments.graph_trainer.common_utils import apply_context_parallel


class _InnerAttention(nn.Module):
    pass


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = SimpleNamespace(inner_attention=_InnerAttention())


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Block(), "1": _Block()})


class TestGraphTrainerContextParallel(unittest.TestCase):
    def test_apply_context_parallel_wraps_inner_attention_modules(self):
        model = _Model()
        mesh = object()
        parallel_dims = SimpleNamespace(
            cp_enabled=True,
            get_mesh=MagicMock(return_value=mesh),
        )

        with patch(
            "torchtitan.distributed.context_parallel.apply_cp_to_forward"
        ) as mock_apply:
            apply_context_parallel(model, parallel_dims)

        expected_modules = [
            block.attention.inner_attention for block in model.layers.values()
        ]
        mock_apply.assert_called_once_with(expected_modules, mesh)
        parallel_dims.get_mesh.assert_called_once_with("cp")

    def test_apply_context_parallel_skips_when_cp_disabled(self):
        model = _Model()
        parallel_dims = SimpleNamespace(
            cp_enabled=False,
            get_mesh=MagicMock(),
        )

        with patch(
            "torchtitan.distributed.context_parallel.apply_cp_to_forward"
        ) as mock_apply:
            apply_context_parallel(model, parallel_dims)

        mock_apply.assert_not_called()
        parallel_dims.get_mesh.assert_not_called()

    def test_graph_trainer_parallelizers_apply_cp_before_tp_parallelize(self):
        from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
            parallelize_deepseekv3,
        )
        from torchtitan.experiments.graph_trainer.llama3.parallelize import (
            parallelize_llama,
        )
        from torchtitan.experiments.graph_trainer.qwen3.parallelize import (
            parallelize_qwen3,
        )

        for parallelize_fn in (
            parallelize_llama,
            parallelize_qwen3,
            parallelize_deepseekv3,
        ):
            with self.subTest(parallelize_fn=parallelize_fn.__name__):
                source = inspect.getsource(parallelize_fn)
                self.assertIn("apply_context_parallel", source)
                self.assertIn("model.parallelize", source)
                self.assertLess(
                    source.index("apply_context_parallel"),
                    source.index("model.parallelize"),
                )


if __name__ == "__main__":
    unittest.main()
