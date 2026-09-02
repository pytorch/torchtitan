# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.components.optimizer.optimizer import OptimizersContainer
from torchtitan.experiments.torchft import optimizer as opt_mod
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer


def _make_container(use_ft_optimizer: bool = False):
    # torchft is an optional dependency; stub it so this test runs on CPU
    # without the package installed.
    stub = types.ModuleType("torchft")
    stub.Optimizer = mock.MagicMock()
    opt_mod.torchft = stub

    model = nn.Linear(4, 4)
    config = TorchFTOptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="Adam",
                optimizer_kwargs={"lr": 1e-3},
            )
        ],
        implementation="for-loop",
    )
    ft_manager = SimpleNamespace(
        manager=mock.MagicMock(), use_async_quorum=use_ft_optimizer
    )
    container = TorchFTOptimizersContainer(
        config, model_parts=[model], ft_manager=ft_manager
    )
    return model, container


class TestTorchFTOptimizerState(unittest.TestCase):
    def test_state_dict_usable_before_init_cache(self):
        # state_dict() must lazily build instead of returning an empty cache.
        _, container = _make_container()
        sd = container.state_dict()
        self.assertTrue(len(sd) > 0)

    def test_state_dict_reflects_steps(self):
        model, container = _make_container()
        container.init_cache_state_dict()
        before = copy.deepcopy(container.state_dict())

        out = model(torch.randn(2, 4))
        out.sum().backward()
        container.step()

        after = container.state_dict()
        live = OptimizersContainer.state_dict(container)
        self.assertEqual(set(after.keys()), set(live.keys()))

        def _same(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.equal(a, b)
            return a == b

        for k in live:
            self.assertTrue(
                _same(after[k], live[k]),
                f"cached state for {k} does not match live optimizer state",
            )
        self.assertTrue(
            any(not _same(after[k], before[k]) for k in live),
            "optimizer state did not change across step()",
        )

    def test_load_state_dict_refreshes_cache(self):
        _, container = _make_container()
        container.init_cache_state_dict()
        sd = copy.deepcopy(container.state_dict())
        container.load_state_dict(sd)
        self.assertTrue(container._cache_valid)
        self.assertTrue(len(container.state_dict()) > 0)


if __name__ == "__main__":
    unittest.main()
