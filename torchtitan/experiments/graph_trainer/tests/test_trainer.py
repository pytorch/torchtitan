# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from unittest.mock import MagicMock

from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphPipelineRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


def test_graph_pp_runtime_does_not_scope_python_loss_calls() -> None:
    trainer = GraphTrainer.__new__(GraphTrainer)
    trainer.pp_schedule = GraphPipelineRuntime.__new__(GraphPipelineRuntime)
    trainer.loss_fn = MagicMock()

    with trainer._pp_loss_step_context(num_loss_calls=2):
        pass

    trainer.loss_fn.step_context.assert_not_called()


def test_graph_trainer_eager_pp_scopes_python_loss_calls() -> None:
    context_active = False

    @contextmanager
    def step_context(*, num_loss_calls: int):
        nonlocal context_active
        assert num_loss_calls == 2
        context_active = True
        try:
            yield
        finally:
            context_active = False

    trainer = GraphTrainer.__new__(GraphTrainer)
    trainer.pp_schedule = MagicMock()
    trainer.loss_fn = MagicMock()
    trainer.loss_fn.step_context.side_effect = step_context

    with trainer._pp_loss_step_context(num_loss_calls=2):
        assert context_active

    assert not context_active
    trainer.loss_fn.step_context.assert_called_once_with(num_loss_calls=2)
