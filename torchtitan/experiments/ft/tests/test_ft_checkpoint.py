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
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtitan.experiments.ft.checkpoint import FTCheckpointManager


class FakeOptimizersContainer:
    def __init__(self):
        self._fake_param = torch.tensor([1.0], dtype=torch.float32)

    def state_dict(self):
        return {"fake_param": self._fake_param}

    def load_state_dict(self, sd: dict):
        if "fake_param" in sd:
            self._fake_param = sd["fake_param"]

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


class DummyFuture:
    def __init__(self):
        self.result = mock.Mock()


def fake_async_save(*args, **kwargs):
    return DummyFuture()


class DummyFTManager:
    """Mimics FTManager for testing without requiring torchft."""

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


class TestFTCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.base_temp_dir, self._testMethodName)
        os.makedirs(self.test_folder, exist_ok=True)
        self.model_parts = [nn.Linear(2, 2)]
        self.states = {"trainer": torch.tensor([1.2347])}
        self.optimizers = FakeOptimizersContainer()
        self.lr_schedulers = FakeLRSchedulersContainer()
        self.data_loader = FakeDataLoader()
        self.ft_manager = DummyFTManager(enabled=True, participating_rank=0)
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()

    def tearDown(self):
        self.patcher_group.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    @mock.patch("torch.cuda.Stream")
    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_ft_async_save_calls_async_wait(
        self,
        mock_async_save,
        mock_new_group,
        mock_cuda_stream,
    ):
        """
        Test that with FT enabled, AsyncMode.ASYNC via FT triggers correct waits.
        """
        config = FTCheckpointManager.Config(
            enable=True,
            async_mode="async",
            folder="",
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=True,
        )
        manager = FTCheckpointManager(
            config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=self.ft_manager,
        )

        # Initially no future
        self.assertIsNone(manager.save_future)
        manager.save(curr_step=5, last_step=False)
        self.assertIsNotNone(manager.save_future)

        manager.save_future.result.assert_not_called()
        prev_future = manager.save_future
        manager.save(curr_step=6, last_step=False)
        prev_future.result.assert_called_once()
        self.assertIsNotNone(manager.save_future)
        manager.save_future.result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
