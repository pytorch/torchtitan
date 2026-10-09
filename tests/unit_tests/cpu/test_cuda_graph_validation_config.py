# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.components.validate import Validator
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    is_cuda_graph_capture_enabled,
    set_cuda_graph_capture_enabled,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.tests.graph_trainer.muse_glimmer import (
    graph_trainer_muse_glimmer_debugmodel,
)
from torchtitan_recipes.tests.suites.features import (
    muse_glimmer_debugmodel_shared_eval_cuda_graph,
    muse_glimmer_debugmodel_shared_eval_cuda_graph_pp2,
)

from tests.integration_tests import validate_fake_pg_compatibility
from tests.integration_tests.features import build_features_test_list


def test_shared_eval_cuda_graph_integration_supports_fake_pg() -> None:
    tests_by_name = {test.test_name: test for test in build_features_test_list()}
    test = tests_by_name["shared_eval_cuda_graph"]
    config = test.configs[0]()

    assert not test.use_real_pg
    assert test.ngpu == 2
    assert config.parallelism.data_parallel_shard_degree == 2
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.training.steps == 10
    assert config.debug.seed == 42
    assert config.debug.deterministic
    assert not config.training.disable_cuda_graphs
    assert config.optim.enable_cuda_graph
    assert config.validator.enable_cuda_graphs
    assert "shared_eval_cuda_graph_pp" not in tests_by_name
    validate_fake_pg_compatibility(test, config)


@pytest.mark.parametrize(
    "schedule",
    [
        "GPipe",
        "1F1B",
        "Interleaved1F1B",
        "LoopedBFS",
        "InterleavedZeroBubble",
        "gpipe",
        "interleavedzerobubble",
    ],
)
def test_verified_pp_validation_graph_schedules(schedule):
    config = muse_glimmer_debugmodel_shared_eval_cuda_graph_pp2()
    config.parallelism.pipeline_parallel_schedule = schedule
    config.__post_init__()
    assert config.training.steps == 10
    assert config.debug.seed == 42
    assert config.debug.deterministic
    assert config.validator.enable_cuda_graphs
    assert config.optim.enable_cuda_graph


@pytest.fixture(autouse=True)
def reset_cuda_graph_manager():
    cuda_graph_teardown()
    yield
    cuda_graph_teardown()


def test_validation_graph_accepts_csv_schedule():
    config = muse_glimmer_debugmodel_shared_eval_cuda_graph_pp2()
    config.parallelism.pipeline_parallel_schedule = "PipelineScheduleMulti"
    config.parallelism.pipeline_parallel_schedule_csv = "schedule.csv"
    config.__post_init__()


def test_eval_graph_does_not_require_training_graph():
    config = muse_glimmer_debugmodel_shared_eval_cuda_graph()
    config.training.disable_cuda_graphs = True
    config.optim.enable_cuda_graph = False
    config.__post_init__()


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
        "torchtitan.training_engine.cuda_graph_teardown",
        side_effect=lambda: events.append("graphs"),
    ):
        engine.close()
    expected = ["profiler", "moe", "checkpoint", "graphs"]
    if engine_cls is GraphTrainingEngine:
        expected.insert(0, "offload")
        assert engine._pinned_pool_ctx is None
    assert events == expected
    assert engine._dist_moe_runtime is None


def test_graph_trainer_rejects_eval_graphs():
    config = graph_trainer_muse_glimmer_debugmodel()
    config.validator = Validator.Config(enable_cuda_graphs=True, steps=1)
    with pytest.raises(ValueError, match="GraphTrainer does not support"):
        config.__post_init__()


def test_graph_trainer_preserves_eager_pp_validation(monkeypatch):
    config = graph_trainer_muse_glimmer_debugmodel()
    config.training.disable_cuda_graphs = True
    config.parallelism.pipeline_parallel_degree = 2
    config.validator = Validator.Config(steps=1)
    config.__post_init__()

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
    monkeypatch.setattr("torchtitan.components.validate.utils.device_type", "cpu")

    with torch.no_grad():
        loss = validator._evaluate_body([model], [(inputs, labels, {})])

    torch.testing.assert_close(loss, torch.tensor(2.0), rtol=0, atol=0)
    schedule.eval.assert_called_once_with(
        arg_mbs=[(inputs,)],
        kwarg_mbs=[{}],
        target_mbs=[labels],
        losses=[torch.tensor(2.0)],
        return_outputs=False,
    )
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
    engine.parallelism_context = SimpleNamespace(
        pp_enabled=True, activate_spmd=lambda **kwargs: nullcontext()
    )
    engine.device = torch.device("cpu")
    engine.garbage_collector = SimpleNamespace(run=MagicMock())
    engine.optim = SimpleNamespace(zero_grad=MagicMock())
    engine.pp_schedule = object.__new__(GraphRuntime)
    inputs, labels = torch.ones(1, 4), torch.zeros(1, 4)
    prepared = [([(inputs,)], [{}], [labels])]
    engine._preprocess_microbatch_groups = MagicMock(return_value=prepared)
    engine._run_forward_backward = MagicMock(
        return_value=SimpleNamespace(loss=torch.tensor(1.0))
    )
    token_counts = torch.tensor(4)
    with patch("torchtitan.training_engine.cuda_graphs_supported", return_value=True):
        engine.forward_backward(
            microbatch_groups=[[MagicMock()]],
            global_loss_token_counts=token_counts,
            global_routing_token_counts=torch.tensor([4]),
        )
    engine._run_forward_backward.assert_called_once_with(prepared, token_counts)


@pytest.mark.parametrize("loaded_step", [0, 3, 10])
@pytest.mark.parametrize("freq", [1, 2, 5])
@pytest.mark.parametrize("capture_mode", ["enabled", "disabled", "unsupported"])
@pytest.mark.parametrize("training_steps", [1, 5])
def test_trainer_keeps_scheduled_eval_during_warmup(
    loaded_step, freq, capture_mode, training_steps
):
    capture_enabled = capture_mode == "enabled"
    set_cuda_graph_capture_enabled(not capture_enabled)
    events = []
    validator = object.__new__(Validator)
    validator.config = Validator.Config(
        enable_cuda_graphs=capture_mode != "disabled", steps=1, freq=freq
    )

    def validate(*args):
        events.append(
            ("validate", engine.num_completed_steps, is_cuda_graph_capture_enabled())
        )

    validator.validate = validate
    engine = SimpleNamespace(
        num_completed_steps=loaded_step,
        model_parts=[],
        parallelism_context=MagicMock(),
        load_checkpoint=lambda: None,
        start_profiler=lambda: None,
        close_profiler=lambda: None,
        save_checkpoint=lambda **kwargs: events.append("checkpoint"),
        checkpointer=SimpleNamespace(
            maybe_wait_for_staging=lambda: events.append("wait")
        ),
        step_profiler=lambda: None,
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
        events.append("train")
        engine.num_completed_steps += 1

    trainer.train_step = train_step
    with (
        patch(
            "torchtitan.trainer.cuda_graphs_supported",
            return_value=capture_mode != "unsupported",
        ),
        patch(
            "torchtitan.trainer.is_cuda_graph_warmup_complete",
            side_effect=lambda: engine.num_completed_steps - loaded_step >= 2,
        ) as warmup_complete,
        patch("torchtitan.trainer.dist_utils.set_pg_timeouts"),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        trainer.train()
    assert events[:2] == ["train", "checkpoint"]
    validation_events = [event for event in events if isinstance(event, tuple)]
    if capture_enabled:
        expected = []
        if validator.should_validate(loaded_step + 1):
            expected.append(("validate", loaded_step + 1, False))
        if training_steps >= 2:
            expected.extend(
                [
                    ("validate", loaded_step + 2, False),
                    ("validate", loaded_step + 2, True),
                ]
            )
        first_scheduled_step = loaded_step + 3
        assert warmup_complete.call_count == min(training_steps, 2)
    else:
        expected = []
        first_scheduled_step = loaded_step + 1
        warmup_complete.assert_not_called()
    expected.extend(
        ("validate", step, True)
        for step in range(first_scheduled_step, loaded_step + training_steps + 1)
        if validator.should_validate(step)
    )
    assert validation_events == expected
    assert events.count("train") == training_steps


def test_eval_replay_waits_for_checkpointed_buffers():
    set_cuda_graph_capture_enabled(True)
    model = torch.nn.Module()
    model.register_buffer("counter", torch.tensor(0))
    pending_states = []
    saved_counters = []

    def save_checkpoint(**kwargs):
        pending_states.append(model.state_dict())

    def wait_for_staging():
        if pending_states:
            saved_counters.append(pending_states.pop()["counter"].clone())

    validator = object.__new__(Validator)
    validator.config = Validator.Config(enable_cuda_graphs=True, steps=1, freq=1)
    validator.validate = lambda parts, step: parts[0].counter.add_(1)
    engine = SimpleNamespace(
        num_completed_steps=3,
        model_parts=[model],
        parallelism_context=MagicMock(),
        load_checkpoint=lambda: None,
        start_profiler=lambda: None,
        close_profiler=lambda: None,
        save_checkpoint=save_checkpoint,
        checkpointer=SimpleNamespace(maybe_wait_for_staging=wait_for_staging),
        step_profiler=lambda: None,
        close=wait_for_staging,
    )
    trainer = object.__new__(Trainer)
    trainer.engine = engine
    trainer.validator = validator
    trainer.dataloader = []
    trainer.config = SimpleNamespace(
        validator=validator.config,
        checkpointer=object(),
        training=SimpleNamespace(steps=4),
        comm=SimpleNamespace(train_timeout_seconds=30),
    )

    def train_step(*args):
        engine.num_completed_steps += 1

    trainer.train_step = train_step
    with (
        patch("torchtitan.trainer.cuda_graphs_supported", return_value=True),
        patch("torchtitan.trainer.dist_utils.set_pg_timeouts"),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        trainer.train()
        trainer.close()

    assert model.counter.item() == 1
    assert len(saved_counters) == 1
    assert saved_counters[0].item() == 0
