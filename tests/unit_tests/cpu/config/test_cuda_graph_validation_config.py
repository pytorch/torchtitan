# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from torchtitan.components.validator import Validator
from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_manager,
    accelerator_graph_teardown,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.tests.graph_trainer.muse_glimmer import (
    graph_trainer_muse_glimmer_debugmodel,
)

from tests.integration_tests.features import (
    muse_glimmer_debugmodel_shared_eval_cuda_graph,
)


def test_shared_eval_cuda_graph_recipe_enables_capture() -> None:
    config = muse_glimmer_debugmodel_shared_eval_cuda_graph()
    assert not config.training.disable_cuda_graphs
    assert config.optim.enable_cuda_graph
    assert not hasattr(config.validator, "enable_cuda_graphs")


@pytest.fixture(autouse=True)
def reset_cuda_graph_manager():
    accelerator_graph_teardown()
    yield
    accelerator_graph_teardown()


@pytest.mark.parametrize("engine_cls", [TrainingEngine, GraphTrainingEngine])
def test_engine_closes_graphs_after_pending_work(engine_cls):
    events = []
    engine = object.__new__(engine_cls)
    if engine_cls is GraphTrainingEngine:
        engine._pinned_pool_ctx = SimpleNamespace(
            __exit__=lambda *args: events.append("offload")
        )
    engine.close_profiler = lambda: events.append("profiler")
    engine._dist_moe_runtime = SimpleNamespace(close=lambda: events.append("moe"))
    engine.checkpointer = SimpleNamespace(close=lambda: events.append("checkpoint"))
    with patch(
        "torchtitan.training_engine.accelerator_graph_teardown",
        side_effect=lambda: events.append("graphs"),
    ):
        engine.close()
    expected = ["profiler", "moe", "checkpoint", "graphs"]
    if engine_cls is GraphTrainingEngine:
        expected.insert(0, "offload")
        assert engine._pinned_pool_ctx is None
    assert events == expected
    assert engine._dist_moe_runtime is None


def test_graph_trainer_accepts_eager_validation_with_training_graphs():
    config = graph_trainer_muse_glimmer_debugmodel()
    config.validator = Validator.Config(steps=1)
    config.__post_init__()


def test_graph_trainer_preserves_eager_pp_validation(monkeypatch):
    stage = SimpleNamespace(state=MagicMock(), clear_runtime_states=MagicMock())
    schedule = SimpleNamespace(_stages=[stage])

    def evaluate(**kwargs):
        kwargs["losses"].append(torch.tensor(2.0))

    schedule.eval = MagicMock(side_effect=evaluate)
    runtime = object.__new__(GraphRuntime)
    runtime.schedule = schedule
    runtime.is_spmd = False
    validator = object.__new__(Validator)
    validator.parallelism_context = SimpleNamespace(
        pp_enabled=True, activate_spmd=lambda: nullcontext()
    )
    validator.pp_schedule = runtime
    validator.pp_has_first_stage = True
    validator.pp_has_last_stage = True
    model = torch.nn.Identity()
    inputs, labels = torch.ones(1, 4), torch.zeros(1, 4)
    monkeypatch.setattr("torchtitan.components.validator.utils.device_type", "cpu")

    with torch.no_grad():
        loss = validator._validate_body([model], [(inputs, labels, {})])

    torch.testing.assert_close(loss, torch.tensor(2.0), rtol=0, atol=0)
    schedule.eval.assert_called_once()
    assert schedule.eval.call_args.kwargs["return_outputs"] is False
    stage.state.clear.assert_called_once()
    stage.clear_runtime_states.assert_called_once()


def test_graph_engine_pp_does_not_initialize_eager_schedule():
    config = graph_trainer_muse_glimmer_debugmodel()
    config.parallelism.pipeline_parallel_degree = 2
    assert not config.training.disable_cuda_graphs
    with patch.object(TrainingEngine, "_initialize_distributed_runtime"):
        engine = GraphTrainingEngine(
            config,
            model_config=config.model,
            max_num_documents=None,
            output_dir="",
        )
    assert not engine._forward_backward_cuda_graph_enabled


@pytest.mark.parametrize(
    "loaded_step,freq,training_steps,capture_mode,expected",
    [
        pytest.param(
            0,
            2,
            4,
            "enabled",
            [
                "train checkpoint eval:blocked",
                "train checkpoint eval:allowed",
                "train checkpoint",
                "train checkpoint eval:allowed",
            ],
            id="fresh",
        ),
        pytest.param(
            3,
            2,
            3,
            "enabled",
            [
                "train checkpoint eval:blocked",
                "train checkpoint eval:allowed",
                "train checkpoint eval:allowed",
            ],
            id="resume",
        ),
        pytest.param(
            0,
            5,
            3,
            "enabled",
            [
                "train checkpoint eval:blocked",
                "train checkpoint eval:allowed",
                "train checkpoint",
            ],
            id="eval-not-due",
        ),
        pytest.param(
            0,
            2,
            1,
            "enabled",
            ["train checkpoint eval:blocked"],
            id="short-run",
        ),
        pytest.param(
            0,
            2,
            3,
            "disabled",
            [
                "train checkpoint eval:allowed",
                "train checkpoint eval:allowed",
                "train checkpoint",
            ],
            id="capture-disabled",
        ),
        pytest.param(
            0,
            2,
            3,
            "unsupported",
            [
                "train checkpoint eval:allowed",
                "train checkpoint eval:allowed",
                "train checkpoint",
            ],
            id="unsupported-platform",
        ),
    ],
)
def test_trainer_keeps_scheduled_eval_during_warmup(
    loaded_step, freq, training_steps, capture_mode, expected
):
    accelerator_graph_manager.capture_allowed = capture_mode != "enabled"
    events = []
    step_events = []
    model = torch.nn.Identity()
    validator = object.__new__(Validator)
    validator.config = Validator.Config(steps=1, freq=freq)
    validator._cuda_graph_enabled = capture_mode == "enabled"
    validator._run_eval = None
    validator._cuda_graph_annotation_config = None
    validator.parallelism_context = SimpleNamespace(pp_enabled=False)
    validator._validate_body = MagicMock()

    def validate(*args):
        evaluate = validator._init_validate_fn([model])
        step_events.append(
            "eval:allowed"
            if accelerator_graph_manager.capture_allowed
            else "eval:blocked"
        )
        evaluate([])

    def save_checkpoint(**kwargs):
        step_events.append("checkpoint")

    def finish_step():
        events.append(" ".join(step_events))
        step_events.clear()

    validator.validate = validate
    engine = SimpleNamespace(
        num_completed_steps=loaded_step,
        model_parts=[model],
        parallelism_context=MagicMock(),
        load_checkpoint=lambda: None,
        start_profiler=lambda: None,
        close_profiler=lambda: None,
        save_checkpoint=save_checkpoint,
        checkpointer=SimpleNamespace(maybe_wait_for_staging=MagicMock()),
        step_profiler=finish_step,
    )
    trainer = object.__new__(Trainer)
    trainer.engine = engine
    trainer.validator = validator
    trainer.dataloader = []
    trainer.config = SimpleNamespace(
        validator=validator.config,
        checkpointer=object(),
        training=SimpleNamespace(steps=loaded_step + training_steps),
        comm=SimpleNamespace(train_timeout_seconds=30),
    )

    def train_step(*args):
        step_events.append("train")
        engine.num_completed_steps += 1

    trainer.train_step = train_step
    with (
        patch(
            "torchtitan.components.validator.wrap_with_accelerator_graph",
            side_effect=lambda fn, **kwargs: fn,
        ),
        patch.object(
            type(accelerator_graph_manager),
            "all_wrappers_warmed_up",
            new_callable=PropertyMock,
            side_effect=lambda: engine.num_completed_steps - loaded_step >= 2,
        ),
        patch("torchtitan.trainer.dist_utils.set_pg_timeouts"),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        trainer.train()
    assert events == expected
    engine.checkpointer.maybe_wait_for_staging.assert_not_called()
    assert engine.num_completed_steps == loaded_step + training_steps
