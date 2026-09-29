# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The controller sizes the trainer's LR schedule to the RL run."""

import asyncio
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch


def _controller_module():
    pytest.importorskip("vllm")
    pytest.importorskip("renderers")
    from torchtitan.rl import controller

    return controller


def _alphabet_sort_config():
    _controller_module()
    from torchtitan.rl.examples.alphabet_sort.config_registry import (
        rl_grpo_gpt_oss_20b_varlen,
    )

    # Linear warmup + decay to min_lr_factor with no explicit total_steps.
    return rl_grpo_gpt_oss_20b_varlen()


def _trainer_config_for_run(config):
    """The trainer config the controller spawns the trainer with."""
    return _controller_module()._with_rl_lr_horizon(
        config.trainer, config.async_loop.num_training_steps
    )


def _lr_factors(trainer_config, num_steps: int) -> list[float]:
    """LR multiplier before each of ``num_steps + 1`` scheduler steps.

    Builds the schedulers the way TrainingEngine does, from
    ``lr_scheduler`` and ``training.steps``.
    """
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    lr_schedulers = trainer_config.lr_scheduler.build(
        optimizers=[optimizer], training_steps=trainer_config.training.steps
    )
    factors = []
    for _ in range(num_steps + 1):
        factors.append(lr_schedulers.schedulers[0].get_last_lr()[0])
        optimizer.step()
        lr_schedulers.step()
    return factors


def _assert_schedule_spans(trainer_config, num_steps: int) -> None:
    lr_config = trainer_config.lr_scheduler
    factors = _lr_factors(trainer_config, num_steps)
    # Warmup completes inside the run ...
    assert max(factors[:num_steps]) == pytest.approx(1.0)
    assert factors[num_steps - 1] < 1.0
    # ... and decay ends exactly where the run ends: the scheduler reaches its
    # floor after the last optimizer step and never goes below it.
    assert factors[num_steps] == pytest.approx(lr_config.min_lr_factor)
    assert min(factors) == pytest.approx(lr_config.min_lr_factor)


def test_lr_schedule_spans_the_rl_run() -> None:
    config = _alphabet_sort_config()
    num_steps = config.async_loop.num_training_steps
    assert config.trainer.lr_scheduler.total_steps is None
    configured_steps = config.trainer.training.steps
    assert configured_steps != num_steps
    trainer_config = _trainer_config_for_run(config)
    assert trainer_config.training.steps == num_steps
    _assert_schedule_spans(trainer_config, num_steps)
    # The controller's own config is not modified.
    assert config.trainer.training.steps == configured_steps


def test_lr_schedule_follows_a_step_count_set_after_construction() -> None:
    # The horizon is taken from the config the controller runs, so a step count
    # edited after the config was built is used, shorter or longer.
    for num_steps in (7, 30):
        config = _alphabet_sort_config()
        config.async_loop.num_training_steps = num_steps
        trainer_config = _trainer_config_for_run(config)
        assert trainer_config.training.steps == num_steps
        _assert_schedule_spans(trainer_config, num_steps)


def test_explicit_lr_total_steps_opts_out() -> None:
    config = _alphabet_sort_config()
    config.trainer.lr_scheduler.total_steps = 4
    config.trainer.training.steps = 123
    assert config.async_loop.num_training_steps != 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer_config = _trainer_config_for_run(config)
    assert trainer_config is config.trainer
    # The schedule follows the pinned horizon, not the RL step count.
    _assert_schedule_spans(trainer_config, 4)


def test_explicit_training_steps_is_replaced_with_a_warning() -> None:
    config = _alphabet_sort_config()
    config.trainer.training.steps = 123
    with pytest.warns(UserWarning, match="lr_scheduler.total_steps"):
        trainer_config = _trainer_config_for_run(config)
    assert trainer_config.training.steps == config.async_loop.num_training_steps


def test_default_training_steps_is_replaced_without_a_warning() -> None:
    config = _alphabet_sort_config()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer_config = _trainer_config_for_run(config)
    assert trainer_config.training.steps == config.async_loop.num_training_steps


def test_setup_async_spawns_the_trainer_with_the_rl_horizon(monkeypatch) -> None:
    controller_module = _controller_module()
    config = _alphabet_sort_config()
    config.async_loop.num_training_steps = 7
    controller = object.__new__(controller_module.Controller)
    controller.config = config

    class _TrainerSpawned(Exception):
        pass

    spawned = {}

    def spawn(name, actor_cls, actor_config, **kwargs):
        spawned[name] = actor_config
        raise _TrainerSpawned  # Stop setup_async at the first actor spawn.

    host = SimpleNamespace(spawn_procs=lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(controller_module, "this_host", lambda: host)
    monkeypatch.setattr(controller_module, "setup_torch_elastic_env_async", AsyncMock())
    with pytest.raises(_TrainerSpawned):
        asyncio.run(
            controller.setup_async(
                trainer_mesh=SimpleNamespace(spawn=spawn),
                generator_meshes=[SimpleNamespace(spawn=spawn)],
            )
        )
    assert set(spawned) == {"trainer"}
    assert spawned["trainer"].training.steps == 7
