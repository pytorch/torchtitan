# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpoint import (
    AsyncMode,
    CheckpointManager,
    MODEL,
    ModelWrapper,
)
from torchtitan.components.external_eval import ExternalEval


def test_external_eval_rolling_merge_schedule():
    external_eval = ExternalEval(
        ExternalEval.Config(
            enable=True,
            freq=10,
            path="eval.py",
            tasks="task",
            eval_raw=False,
            merge_checkpoint_count=4,
            merge_checkpoint_interval=10,
        )
    )

    assert external_eval.merge_checkpoint_steps(10) == []
    assert external_eval.merge_checkpoint_steps(20) == []
    assert external_eval.merge_checkpoint_steps(30) == []
    assert external_eval.merge_checkpoint_steps(40) == [10, 20, 30, 40]
    assert external_eval.merge_checkpoint_steps(50) == [20, 30, 40, 50]
    assert external_eval.merge_checkpoint_steps(55) == []


def test_external_eval_close_waits_for_results():
    external_eval = ExternalEval(ExternalEval.Config())
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    process.pid = 123
    external_eval.processes = [process]

    external_eval.close()

    process.wait.assert_called_once_with()
    assert external_eval.processes == []


def _build_checkpoint_manager(
    folder: Path, model: torch.nn.Module
) -> CheckpointManager:
    manager = object.__new__(CheckpointManager)
    manager.enable = True
    manager.load_only = False
    manager.folder = str(folder)
    manager.states = {MODEL: ModelWrapper(model)}
    manager.async_mode = AsyncMode.DISABLED
    manager.save_future = None
    manager.staging_future = None
    manager.stager = None
    manager.sd_adapter = None
    manager.purge_thread = None
    return manager


def test_save_averaged_model_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    model_wrapper = manager.states[MODEL]

    for step in range(10, 50, 10):
        with torch.no_grad():
            model.weight.fill_(step / 10)
            model.bias.fill_(step / 10)
        dcp.save(
            model_wrapper.state_dict(),
            checkpoint_id=manager.get_checkpoint_id(step),
        )

    output_path = manager.save_averaged_model_checkpoint(
        source_steps=[10, 20, 30, 40],
        curr_step=40,
    )

    averaged_state = {
        "weight": torch.empty_like(model.weight),
        "bias": torch.empty_like(model.bias),
    }
    dcp.load(averaged_state, checkpoint_id=output_path)

    torch.testing.assert_close(
        averaged_state["weight"], torch.full_like(model.weight, 2.5)
    )
    torch.testing.assert_close(averaged_state["bias"], torch.full_like(model.bias, 2.5))
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4.0))
    torch.testing.assert_close(model.bias, torch.full_like(model.bias, 4.0))


def test_save_averaged_model_checkpoint_decay_weights_recent_more(tmp_path):
    """decay < 1 must weight newer checkpoints more, and stay normalized."""
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    model_wrapper = manager.states[MODEL]

    for step in range(10, 50, 10):
        with torch.no_grad():
            model.weight.fill_(step / 10)
            model.bias.fill_(step / 10)
        dcp.save(
            model_wrapper.state_dict(),
            checkpoint_id=manager.get_checkpoint_id(step),
        )

    decay = 0.5
    output_path = manager.save_averaged_model_checkpoint(
        source_steps=[10, 20, 30, 40],
        curr_step=40,
        decay=decay,
    )

    # Ages are 3, 2, 1, 0 for values 1, 2, 3, 4.
    weights = [decay**3, decay**2, decay, 1.0]
    expected = sum(w * v for w, v in zip(weights, [1.0, 2.0, 3.0, 4.0])) / sum(weights)

    averaged_state = {
        "weight": torch.empty_like(model.weight),
        "bias": torch.empty_like(model.bias),
    }
    dcp.load(averaged_state, checkpoint_id=output_path)

    torch.testing.assert_close(
        averaged_state["weight"], torch.full_like(model.weight, expected)
    )
    # Strictly between the uniform mean and the newest checkpoint.
    assert 2.5 < expected < 4.0
    # Decayed merges live in their own tree so a decay sweep cannot collide.
    assert Path(output_path).parent.name == "ema-4-0.5"
    # Live model is still the newest checkpoint, not the average.
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4.0))


def test_save_averaged_model_checkpoint_rejects_invalid_decay(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)

    for decay in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="decay must be in"):
            manager.save_averaged_model_checkpoint(
                source_steps=[10, 20], curr_step=20, decay=decay
            )


def test_merge_decay_config_validation():
    def build(decay):
        return ExternalEval.Config(
            enable=True, path="/p.py", tasks="t", merge_decay=decay
        )

    for decay in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="merge_decay must be in"):
            build(decay)

    assert build(1.0).merge_decay == 1.0
    assert build(0.9).merge_decay == 0.9


def test_merge_window_spanning_warmup_warns_once(caplog):
    external_eval = ExternalEval(
        ExternalEval.Config(
            enable=True,
            path="/p.py",
            tasks="t",
            merge_checkpoint_count=4,
            merge_checkpoint_interval=1000,
        )
    )
    trainer_config = SimpleNamespace(lr_scheduler=SimpleNamespace(warmup_steps=2000))

    with caplog.at_level(logging.WARNING):
        external_eval._warn_if_merge_window_spans_warmup(
            [1000, 2000, 3000, 4000], trainer_config
        )
    assert "reaches back before LR warmup" in caplog.text
    # Suggests a start step that puts the whole window past warmup.
    assert "merge_start_step >= 5000" in caplog.text

    # Warn-once: a second violating window stays quiet.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        external_eval._warn_if_merge_window_spans_warmup([1000, 2000], trainer_config)
    assert caplog.text == ""


def test_merge_window_after_warmup_does_not_warn(caplog):
    external_eval = ExternalEval(
        ExternalEval.Config(enable=True, path="/p.py", tasks="t")
    )
    trainer_config = SimpleNamespace(lr_scheduler=SimpleNamespace(warmup_steps=2000))

    with caplog.at_level(logging.WARNING):
        external_eval._warn_if_merge_window_spans_warmup(
            [2000, 3000, 4000, 5000], trainer_config
        )
    assert caplog.text == ""


def test_merge_window_warning_tolerates_missing_scheduler(caplog):
    """trainer_config is typed Any; a config without lr_scheduler must not crash."""
    external_eval = ExternalEval(
        ExternalEval.Config(enable=True, path="/p.py", tasks="t")
    )
    with caplog.at_level(logging.WARNING):
        external_eval._warn_if_merge_window_spans_warmup([10, 20], SimpleNamespace())
    assert caplog.text == ""


def test_ema_source_steps_are_non_overlapping():
    """Each checkpoint must enter the running EMA exactly once."""
    external_eval = ExternalEval(
        ExternalEval.Config(
            enable=True,
            path="/p.py",
            tasks="t",
            freq=1000,
            merge_checkpoint_interval=1000,
            merge_checkpoint_count=4,
            merge_start_step=5000,
            ema_decay=0.9,
        )
    )
    # freq == interval -> exactly one new checkpoint per merge, and consecutive
    # merges must not share any source.
    assert external_eval.ema_source_steps(5000) == [5000]
    assert external_eval.ema_source_steps(6000) == [6000]
    # Gated before the first merge step, and on non-eval steps.
    assert external_eval.ema_source_steps(4000) == []
    assert external_eval.ema_source_steps(5500) == []

    # freq spanning several checkpoint intervals folds each one exactly once.
    dense = ExternalEval(
        ExternalEval.Config(
            enable=True,
            path="/p.py",
            tasks="t",
            freq=1000,
            merge_checkpoint_interval=250,
            merge_start_step=1000,
            ema_decay=0.9,
        )
    )
    assert dense.ema_source_steps(2000) == [1250, 1500, 1750, 2000]
    assert dense.ema_source_steps(3000) == [2250, 2500, 2750, 3000]


def test_ema_disabled_by_default_keeps_soup_path():
    external_eval = ExternalEval(
        ExternalEval.Config(enable=True, path="/p.py", tasks="t")
    )
    assert external_eval.config.ema_decay == 0.0
    assert external_eval.ema_source_steps(4000) == []


def test_previous_ema_step_gated_by_first_merge():
    external_eval = ExternalEval(
        ExternalEval.Config(
            enable=True, path="/p.py", tasks="t", freq=1000, merge_start_step=5000
        )
    )
    assert external_eval.previous_ema_step(5000) is None  # first merge bootstraps
    assert external_eval.previous_ema_step(6000) == 5000


def test_ema_decay_config_validation():
    def build(decay):
        return ExternalEval.Config(enable=True, path="/p.py", tasks="t", ema_decay=decay)

    for decay in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="ema_decay must be in"):
            build(decay)
    assert build(0.0).ema_decay == 0.0
    assert build(0.9).ema_decay == 0.9


def _seed_regular_checkpoints(manager, model, steps):
    model_wrapper = manager.states[MODEL]
    for step in steps:
        with torch.no_grad():
            model.weight.fill_(step / 1000)
            model.bias.fill_(step / 1000)
        dcp.save(
            model_wrapper.state_dict(), checkpoint_id=manager.get_checkpoint_id(step)
        )


def _read_merged_value(path, model):
    state = {"weight": torch.empty_like(model.weight)}
    dcp.load(state, checkpoint_id=path)
    return state["weight"].flatten()[0].item()


def test_save_ema_model_checkpoint_follows_recurrence(tmp_path):
    """The on-disk chain must reproduce W <- d*W + (1-d)*w exactly."""
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    steps = list(range(1000, 5000, 1000))
    _seed_regular_checkpoints(manager, model, steps)

    decay = 0.9
    expected = None
    for step in steps:
        output_path = manager.save_ema_model_checkpoint(
            new_steps=[step],
            curr_step=step,
            decay=decay,
            prev_ema_step=step - 1000 if step > 1000 else None,
        )
        value = step / 1000
        # First merge bootstraps from the observation, so there is no init bias.
        expected = value if expected is None else decay * expected + (1 - decay) * value
        assert abs(_read_merged_value(output_path, model) - expected) < 1e-5

    assert Path(output_path).parent.name == "ema-d0.9"
    # Live model is left at the newest regular checkpoint, not the EMA.
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4.0))


def test_save_ema_model_checkpoint_is_idempotent(tmp_path):
    """Re-running a completed merge must reuse, never fold the same step twice.

    This is the MAST retry path: an attempt is abandoned and the job replays a
    step that was already merged.
    """
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    _seed_regular_checkpoints(manager, model, [1000, 2000])

    manager.save_ema_model_checkpoint(new_steps=[1000], curr_step=1000, decay=0.9)
    first = manager.save_ema_model_checkpoint(
        new_steps=[2000], curr_step=2000, decay=0.9, prev_ema_step=1000
    )
    first_value = _read_merged_value(first, model)

    second = manager.save_ema_model_checkpoint(
        new_steps=[2000], curr_step=2000, decay=0.9, prev_ema_step=1000
    )
    assert second == first
    assert _read_merged_value(second, model) == first_value


def test_save_ema_model_checkpoint_bootstraps_on_chain_break(tmp_path, caplog):
    """A missing predecessor must restart the chain, not crash or read garbage."""
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    _seed_regular_checkpoints(manager, model, [1000, 2000])

    with caplog.at_level(logging.WARNING):
        output_path = manager.save_ema_model_checkpoint(
            new_steps=[2000],
            curr_step=2000,
            decay=0.9,
            prev_ema_step=1000,  # never written
        )
    assert "Restarting the EMA chain" in caplog.text
    # Bootstrapped: plain value of the new checkpoint, no history mixed in.
    assert abs(_read_merged_value(output_path, model) - 2.0) < 1e-5


def test_save_ema_model_checkpoint_rejects_invalid_input(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)

    for decay in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="EMA decay must be in"):
            manager.save_ema_model_checkpoint(
                new_steps=[1000], curr_step=1000, decay=decay
            )
    with pytest.raises(ValueError, match="must equal curr_step"):
        manager.save_ema_model_checkpoint(
            new_steps=[1000], curr_step=2000, decay=0.9
        )
    with pytest.raises(ValueError, match="at least one new source step"):
        manager.save_ema_model_checkpoint(new_steps=[], curr_step=1000, decay=0.9)


def test_external_eval_request_only_queues_command(tmp_path):
    external_eval = ExternalEval(
        ExternalEval.Config(
            enable=True,
            path="/package/run_external_eval.py",
            tasks="task",
            request_only=True,
            eval_cuda_visible_devices="0,1",
        )
    )
    trainer_config = SimpleNamespace(
        dump_folder=str(tmp_path),
        hf_assets_path="/package/assets",
        model_spec=SimpleNamespace(name="olmo3", flavor="7B"),
    )

    external_eval._launch_checkpoint(
        step=4000,
        checkpoint_dir="/checkpoints/step-4000",
        output_name="step-4000-merged-4",
        source_steps=[1000, 2000, 3000, 4000],
        trainer_config=trainer_config,
    )

    output_dir = tmp_path / "eval" / "step-4000-merged-4"
    with (output_dir / "eval_request.json").open() as f:
        request = json.load(f)

    assert request["command"][0] == sys.executable
    assert request["command"][1] == "/package/run_external_eval.py"
    assert request["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert not (output_dir / "launch.log").exists()

    external_eval.close()
    assert (tmp_path / "eval" / "_TRAINING_COMPLETE").exists()
