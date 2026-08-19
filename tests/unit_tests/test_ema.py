# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpoint import (
    AsyncMode,
    CheckpointManager,
    MODEL,
    ModelWrapper,
)
from torchtitan.components.ema import EMA


def _build_checkpoint_manager(
    folder: Path,
    model: torch.nn.Module,
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


def _build_ema(
    manager: CheckpointManager,
    **config_kwargs,
) -> EMA:
    return EMA(EMA.Config(**config_kwargs), manager)


def _seed_regular_checkpoints(manager, model, steps, divisor=1000):
    model_wrapper = manager.states[MODEL]
    for step in steps:
        with torch.no_grad():
            model.weight.fill_(step / divisor)
            model.bias.fill_(step / divisor)
        dcp.save(
            model_wrapper.state_dict(),
            checkpoint_id=manager.get_checkpoint_id(step),
        )


def _read_checkpoint_value(path, model):
    state = {"weight": torch.empty_like(model.weight)}
    dcp.load(state, checkpoint_id=path)
    return state["weight"].flatten()[0].item()


def test_ema_rolling_window_schedule():
    ema = EMA(
        EMA.Config(
            enable=True,
            freq=10,
            checkpoint_count=4,
            checkpoint_interval=10,
        ),
        CheckpointManager.__new__(CheckpointManager),
    )

    assert ema.window_source_steps(10) == []
    assert ema.window_source_steps(20) == []
    assert ema.window_source_steps(30) == []
    assert ema.window_source_steps(40) == [10, 20, 30, 40]
    assert ema.window_source_steps(50) == [20, 30, 40, 50]
    assert ema.window_source_steps(55) == []


def test_ema_config_validation():
    for decay in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="EMA decay must be in"):
            EMA.Config(decay=decay)

    for decay in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="EMA stateful_decay must be in"):
            EMA.Config(stateful_decay=decay)

    assert EMA.Config(decay=0.9).decay == 0.9
    assert EMA.Config(stateful_decay=0.9).stateful_decay == 0.9


def test_ema_window_spanning_warmup_warns_once(caplog):
    ema = EMA(
        EMA.Config(
            enable=True,
            checkpoint_count=4,
            checkpoint_interval=1000,
        ),
        CheckpointManager.__new__(CheckpointManager),
    )

    with caplog.at_level(logging.WARNING):
        ema._warn_if_window_spans_warmup(
            [1000, 2000, 3000, 4000],
            warmup_steps=2000,
        )
    assert "reaches back before LR warmup" in caplog.text
    assert "ema.start_step >= 5000" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ema._warn_if_window_spans_warmup([1000, 2000], warmup_steps=2000)
    assert caplog.text == ""


def test_save_window_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)
    _seed_regular_checkpoints(manager, model, range(10, 50, 10), divisor=10)

    output_path = ema.save_window_checkpoint(
        source_steps=[10, 20, 30, 40],
        curr_step=40,
    )

    assert abs(_read_checkpoint_value(output_path, model) - 2.5) < 1e-5
    assert Path(output_path).parent.name == "avg-4"
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4.0))


def test_save_window_checkpoint_decay_weights_recent_more(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)
    _seed_regular_checkpoints(manager, model, range(10, 50, 10), divisor=10)

    decay = 0.5
    output_path = ema.save_window_checkpoint(
        source_steps=[10, 20, 30, 40],
        curr_step=40,
        decay=decay,
    )
    weights = [decay**3, decay**2, decay, 1.0]
    expected = sum(
        weight * value
        for weight, value in zip(weights, [1.0, 2.0, 3.0, 4.0], strict=True)
    ) / sum(weights)

    assert abs(_read_checkpoint_value(output_path, model) - expected) < 1e-5
    assert 2.5 < expected < 4.0
    assert Path(output_path).parent.name == "window-4-d0.5"
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4.0))


def test_save_window_checkpoint_rejects_invalid_decay(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)

    for decay in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="decay must be in"):
            ema.save_window_checkpoint(
                source_steps=[10, 20],
                curr_step=20,
                decay=decay,
            )


def test_stateful_source_steps_are_non_overlapping():
    ema = EMA(
        EMA.Config(
            enable=True,
            freq=1000,
            checkpoint_interval=1000,
            checkpoint_count=4,
            start_step=5000,
            stateful_decay=0.9,
        ),
        CheckpointManager.__new__(CheckpointManager),
    )
    assert ema.stateful_source_steps(5000) == [5000]
    assert ema.stateful_source_steps(6000) == [6000]
    assert ema.stateful_source_steps(4000) == []
    assert ema.stateful_source_steps(5500) == []
    assert ema.previous_stateful_step(5000) is None
    assert ema.previous_stateful_step(6000) == 5000

    dense = EMA(
        EMA.Config(
            enable=True,
            freq=1000,
            checkpoint_interval=250,
            start_step=1000,
            stateful_decay=0.9,
        ),
        CheckpointManager.__new__(CheckpointManager),
    )
    assert dense.stateful_source_steps(2000) == [1250, 1500, 1750, 2000]
    assert dense.stateful_source_steps(3000) == [2250, 2500, 2750, 3000]


def test_save_stateful_checkpoint_follows_recurrence(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)
    steps = list(range(1000, 5000, 1000))
    _seed_regular_checkpoints(manager, model, steps)

    decay = 0.9
    expected = None
    for step in steps:
        output_path = ema.save_stateful_checkpoint(
            new_steps=[step],
            curr_step=step,
            decay=decay,
            prev_ema_step=step - 1000 if step > 1000 else None,
        )
        value = step / 1000
        expected = value if expected is None else decay * expected + (1 - decay) * value
        assert abs(_read_checkpoint_value(output_path, model) - expected) < 1e-5

    assert Path(output_path).parent.name == "stateful-d0.9"
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 4.0))


def test_save_stateful_checkpoint_is_idempotent(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)
    _seed_regular_checkpoints(manager, model, [1000, 2000])

    ema.save_stateful_checkpoint(new_steps=[1000], curr_step=1000, decay=0.9)
    first = ema.save_stateful_checkpoint(
        new_steps=[2000],
        curr_step=2000,
        decay=0.9,
        prev_ema_step=1000,
    )
    first_value = _read_checkpoint_value(first, model)
    second = ema.save_stateful_checkpoint(
        new_steps=[2000],
        curr_step=2000,
        decay=0.9,
        prev_ema_step=1000,
    )

    assert second == first
    assert _read_checkpoint_value(second, model) == first_value


def test_save_stateful_checkpoint_bootstraps_on_chain_break(tmp_path, caplog):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)
    _seed_regular_checkpoints(manager, model, [1000, 2000])

    with caplog.at_level(logging.WARNING):
        output_path = ema.save_stateful_checkpoint(
            new_steps=[2000],
            curr_step=2000,
            decay=0.9,
            prev_ema_step=1000,
        )
    assert "Restarting the EMA chain" in caplog.text
    assert abs(_read_checkpoint_value(output_path, model) - 2.0) < 1e-5


def test_save_stateful_checkpoint_rejects_invalid_input(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    ema = _build_ema(manager)

    for decay in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="EMA decay must be in"):
            ema.save_stateful_checkpoint(
                new_steps=[1000],
                curr_step=1000,
                decay=decay,
            )
    with pytest.raises(ValueError, match="must equal curr_step"):
        ema.save_stateful_checkpoint(
            new_steps=[1000],
            curr_step=2000,
            decay=0.9,
        )
    with pytest.raises(ValueError, match="at least one new source step"):
        ema.save_stateful_checkpoint(new_steps=[], curr_step=1000, decay=0.9)


def test_maybe_save_returns_checkpoint_for_external_consumers(tmp_path):
    model = torch.nn.Linear(2, 2)
    manager = _build_checkpoint_manager(tmp_path, model)
    _seed_regular_checkpoints(manager, model, [10, 20], divisor=10)
    ema = _build_ema(
        manager,
        enable=True,
        freq=10,
        checkpoint_count=2,
        checkpoint_interval=10,
    )

    checkpoint = ema.maybe_save(20)

    assert checkpoint is not None
    assert checkpoint.output_name == "step-20-averaged-2"
    assert checkpoint.source_steps == (10, 20)
