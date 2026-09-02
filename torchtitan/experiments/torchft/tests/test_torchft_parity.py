# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import time
import unittest
from contextlib import nullcontext
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtitan.components.checkpointer import DATALOADER
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import (
    maybe_semi_sync_training,
    TorchFTManager,
)


class FakeOptimizersContainer:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass

    def init_cache_state_dict(self):
        pass


class FakeLRSchedulersContainer:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class FakeDataLoader(DataLoader):
    def __init__(self):
        super().__init__(dataset=[], batch_size=1)

    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class DummyFTManager:
    """Mimics TorchFTManager for testing without requiring torchft."""

    def __init__(self, enabled=True, replica_id=0, participating_rank=0):
        self._enabled = enabled
        self.replica_id = replica_id
        if enabled:
            self.manager = mock.MagicMock()
            self.manager.participating_rank.return_value = participating_rank
        else:
            self.manager = None

    @property
    def enabled(self):
        return self._enabled


def _manager_config(test_folder, **overrides):
    kwargs = dict(
        enable=True,
        async_mode="disabled",
        folder=test_folder,
        interval=1,
        keep_latest_k=0,
        last_save_model_only=False,
        export_dtype="float32",
        exclude_from_loading=[],
        initial_load_path=None,
        initial_load_model_only=False,
        enable_ft_dataloader_checkpoints=True,
    )
    kwargs.update(overrides)
    return TorchFTCheckpointManager.Config(**kwargs)


class TestFTCheckpointParity(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.base_temp_dir, self._testMethodName)
        os.makedirs(self.test_folder, exist_ok=True)
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()
        self.patcher_destroy = mock.patch("torch.distributed.destroy_process_group")
        self.patcher_destroy.start()

    def tearDown(self):
        self.patcher_group.stop()
        self.patcher_destroy.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    def _make_manager(self, ft_manager, config=None, states=None):
        return TorchFTCheckpointManager(
            config or _manager_config(self.test_folder),
            dataloader=FakeDataLoader(),
            model_parts=[nn.Linear(2, 2)],
            optimizers=FakeOptimizersContainer(),
            lr_schedulers=FakeLRSchedulersContainer(),
            states=states or {"trainer": torch.tensor([1.0])},
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=ft_manager,
        )

    def test_ft_folder_naming(self):
        manager = self._make_manager(DummyFTManager(replica_id=3))
        self.assertTrue(manager._ft_folder().endswith("ft-replica-3"))
        self.assertNotIn("replicat-", manager._ft_folder())
        manager.close()

    def test_states_to_load_excludes_dataloader_when_ft_enabled(self):
        states = {"trainer": torch.tensor([1.0]), DATALOADER: FakeDataLoader()}
        manager = self._make_manager(DummyFTManager(), states=states)
        self.assertNotIn(DATALOADER, manager._states_to_load(model_only=False))
        manager.close()

    def test_states_to_load_keeps_dataloader_when_ft_disabled(self):
        states = {"trainer": torch.tensor([1.0]), DATALOADER: FakeDataLoader()}
        manager = self._make_manager(DummyFTManager(enabled=False), states=states)
        self.assertIn(DATALOADER, manager._states_to_load(model_only=False))
        manager.close()

    def test_dataloader_checkpoints_require_dataloader(self):
        manager = TorchFTCheckpointManager(
            _manager_config(self.test_folder),
            dataloader=None,
            model_parts=[nn.Linear(2, 2)],
            optimizers=FakeOptimizersContainer(),
            lr_schedulers=FakeLRSchedulersContainer(),
            states={"trainer": torch.tensor([1.0])},
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=DummyFTManager(),
        )
        self.assertFalse(manager.enable_ft_dataloader_checkpoints)
        manager.close()

    def test_should_purge_only_on_participating_rank_zero(self):
        config = _manager_config(self.test_folder, keep_latest_k=2)
        with mock.patch("torch.distributed.get_rank", return_value=0):
            rank0 = self._make_manager(
                DummyFTManager(participating_rank=0), config=config
            )
            self.assertTrue(rank0._should_purge())
            rank0.close()
            bystander = self._make_manager(
                DummyFTManager(participating_rank=1), config=config
            )
            self.assertFalse(bystander._should_purge())
            bystander.close()


class TestFTManagerParity(unittest.TestCase):
    def test_get_dp_info_scales_with_group_size(self):
        manager = TorchFTManager.__new__(TorchFTManager)
        manager._manager = mock.MagicMock()
        manager.group_size = 4
        manager.replica_id = 1
        # dp_degree ranks per replica group; replica_id offsets into global DP.
        self.assertEqual(manager.get_dp_info(8, 3), (32, 11))

    def test_get_dp_info_disabled_is_identity(self):
        manager = TorchFTManager.__new__(TorchFTManager)
        manager._manager = None
        self.assertEqual(manager.get_dp_info(8, 3), (8, 3))

    def test_maybe_semi_sync_training_disabled_returns_nullcontext(self):
        manager = TorchFTManager.__new__(TorchFTManager)
        manager._manager = None
        ctx = maybe_semi_sync_training(
            FaultTolerance(),
            ft_manager=manager,
            model=mock.MagicMock(),
            n_layers=0,
            optimizer=mock.MagicMock(),
        )
        self.assertIsInstance(ctx, nullcontext)
        with ctx as value:
            self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()
