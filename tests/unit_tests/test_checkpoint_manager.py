# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.checkpoint import CheckpointManager
from torchtitan.config_manager import JobConfig


class ModelDummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.layer.weight = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):
        return self.layer(x)


class StatefulDummy(Stateful):
    def __init__(self, state: Dict[str, Any]):
        self.state = state

    def state_dict(self):
        return self.state

    # simulate SchedulersContainer
    def get_lr_scheduler_state(self):
        return {"lr_scheduler": self}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        assert set(state_dict.keys()) == set(self.state.keys())
        for k, v in state_dict.items():
            self.state[k] = v

    def __repr__(self) -> str:
        return f"StatefulDummy({self.state})"


class TestCheckpointManager:
    @pytest.fixture
    def job_config(self, tmp_path: Path) -> JobConfig:
        config = JobConfig()
        config.parse_args(["--job.config_file", "./train_configs/debug_model.toml"])

        config.job.dump_folder = os.path.join(str(tmp_path), "output")
        config.checkpoint.enable_checkpoint = True
        config.checkpoint.async_mode = "disabled"
        config.checkpoint.folder = "checkpoint"
        config.checkpoint.interval_type = "steps"

        return config

    @pytest.fixture
    def initial_state(self) -> Dict[str, Any]:
        return dict(
            dataloader=StatefulDummy({"dataloader": 1}),
            # need cuda here because async checkpointing works incorrectly on CPU
            # https://github.com/pytorch/pytorch/issues/144657
            model_parts=ModelDummy().cuda(),
            optimizers=StatefulDummy({"optimizer": 1}),
            lr_schedulers=StatefulDummy({"lr_schedulers": 1}),
        )

    @pytest.fixture
    def process_group(self, tmp_path) -> Generator[None, None, None]:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group()
        yield None
        dist.destroy_process_group(None)

    def _set_state_values(self, state, value):
        for part in state.values():
            if isinstance(part, StatefulDummy):
                for key in part.state.keys():
                    part.state[key] = value
            elif isinstance(part, ModelDummy):
                with torch.no_grad():
                    state["model_parts"].layer.weight.fill_(value)
            else:
                raise ValueError(
                    f"expected StatefulDummy or ModelDummy, got {type(part)}"
                )

    def _check_state_values(self, state, value):
        for part in state.values():
            if isinstance(part, StatefulDummy):
                for key in part.state.keys():
                    assert part.state[key] == value, f"expected 1 for {key} state"
            elif isinstance(part, ModelDummy):
                assert state["model_parts"].layer.weight.data[0, 0] == value
            else:
                raise ValueError(
                    f"expected StatefulDummy or ModelDummy, got {type(part)}"
                )

    def test_saveload(
        self,
        initial_state: Dict[str, Any],
        job_config: JobConfig,
        process_group: None,
    ):
        state = initial_state

        manager = CheckpointManager(
            **state,
            states={},
            job_config=job_config,
        )

        # nothing saved yet, nothing to load
        assert not manager.load()

        self._set_state_values(state, 10)
        manager.save(10)
        self._set_state_values(state, 11)

        # should load step 10
        assert manager.load()
        self._check_state_values(state, 10)

    def test_loading_last_checkpoint(
        self,
        initial_state: Dict[str, Any],
        job_config: JobConfig,
        process_group: None,
    ):
        state = initial_state

        manager = CheckpointManager(
            **state,
            states={},
            job_config=job_config,
        )

        # nothing saved yet, nothing to load
        assert not manager.load()

        for step in [20, 30, 200, 400]:
            self._set_state_values(state, step)
            manager.save(step)

        # corrupt checkpoint 400
        os.remove(os.path.join(manager._create_checkpoint_id(400), ".metadata"))

        # no step passed means last successful checkpoint
        assert manager.load()
        self._check_state_values(state, 200)

        # if an existing step passed, load it
        assert manager.load(30)
        self._check_state_values(state, 30)

        # if a corrupted step passed, don't load
        assert not manager.load(400)
        self._check_state_values(state, 30)

        # if a non-existant step passed, don't load
        assert not manager.load(300)
        self._check_state_values(state, 30)

    def test_keep_latest_k(
        self,
        initial_state: Dict[str, Any],
        job_config: JobConfig,
        process_group: None,
    ):
        state = initial_state

        job_config.checkpoint.keep_latest_k = 3
        manager = CheckpointManager(
            **state,
            states={},
            job_config=job_config,
        )

        steps = [0, 10, 20, 30, 40, 50]

        for i, step in enumerate(steps):
            self._set_state_values(state, step)
            manager.save(step)

            found = sorted(os.listdir(manager.folder))

            expected_steps_left = steps[: i + 1][-job_config.checkpoint.keep_latest_k :]
            assert found == [f"step-{step}" for step in expected_steps_left]

    @pytest.mark.parametrize("async_mode", ["async", "async_with_pinned_mem"])
    def test_async_modes(
        self,
        initial_state: Dict[str, Any],
        job_config: JobConfig,
        process_group: None,
        async_mode: str,
    ):
        state = initial_state

        job_config.checkpoint.async_mode = async_mode
        manager = CheckpointManager(
            **state,
            states={},
            job_config=job_config,
        )

        steps = [10, 20, 30, 40, 50]
        for step in steps:
            manager.maybe_wait_for_staging()
            self._set_state_values(state, step)
            # force on the last step to ensure the last checkpoint is written
            manager.save(step, force=step == 50)

        for step in steps:
            assert manager.load(step)
            self._check_state_values(state, step)
