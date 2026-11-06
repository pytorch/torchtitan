# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer dispatch configuration without a native DeepEP installation."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, PropertyMock

import pytest
import torch

from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.common.moe.token_dispatcher import DeepEPTokenDispatcher


@pytest.mark.parametrize(
    "overlap,disable_passes",
    [
        (False, []),
        (True, []),
        (True, ["deepep_dispatch_overlap_pass"]),
    ],
)
def test_compile_config_controls_dispatch_split(monkeypatch, overlap, disable_passes):
    backend_name = "torchtitan.distributed.deepep.deepep"
    backend = ModuleType(backend_name)
    x_TD = torch.zeros(4, 8)
    counts_E = torch.zeros(4, dtype=torch.int32)
    dispatch_tokens = Mock(return_value=(x_TD, counts_E, object()))
    backend.dispatch_tokens = dispatch_tokens
    monkeypatch.setitem(sys.modules, backend_name, backend)

    dispatcher = DeepEPTokenDispatcher(
        DeepEPTokenDispatcher.Config(
            num_experts=4, top_k=2, num_max_tokens_per_rank=4, hidden_dim=8
        )
    )
    model = torch.nn.Sequential(torch.nn.Sequential(dispatcher))
    mesh = Mock()
    mesh.get_group.return_value.size.return_value = 2
    inputs = (
        x_TD,
        torch.ones(4, 2),
        torch.zeros(4, 2, dtype=torch.int64),
        counts_E,
    )
    with patch.object(
        DeepEPTokenDispatcher, "ep_mesh", new_callable=PropertyMock
    ) as ep_mesh:
        ep_mesh.return_value = mesh
        dispatcher.dispatch(*inputs)
        assert dispatch_tokens.call_args.kwargs["split_launch_wait"] is False

        config = GraphTrainerCompileConfig(
            deepep_dispatch_overlap=overlap, disable_passes=disable_passes
        )
        assert (
            apply_compile(
                model,
                compile_config=config,
                parallelism_context=SimpleNamespace(tp_enabled=False),
            )
            is model
        )
        dispatcher.dispatch(*inputs)
        assert dispatch_tokens.call_args.kwargs["split_launch_wait"] is overlap
