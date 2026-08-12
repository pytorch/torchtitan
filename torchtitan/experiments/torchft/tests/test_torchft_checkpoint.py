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
from concurrent.futures import Future
from unittest import mock

import torch
import torch.distributed.checkpoint as dist_checkpoint
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtitan.components.checkpoint import DATALOADER, CheckpointManager
from torchtitan.experiments.torchft.checkpoint import (
    TORCHFT_MANAGER,
    TorchFTCheckpointManager,
)


class FakeOptimizersContainer:
    def __init__(self):
        self._fake_param = torch.tensor([1.0], dtype=torch.float32)
        self.cache_initialized = False

    def state_dict(self):
        return {"fake_param": self._fake_param}

    def load_state_dict(self, sd: dict):
        if "fake_param" in sd:
            self._fake_param = sd["fake_param"]

    def init_cache_state_dict(self):
        self.cache_initialized = True


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
    def __new__(cls):
        # Return a Mock that mimics Future instead of an instance of this class
        # That allows isinstance(DummyFuture, Future) to pass
        instance = mock.Mock(spec=Future)
        instance.result = mock.Mock()

        return instance


def fake_async_save(*args, **kwargs):
    return DummyFuture()


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
        # Patch process group destruction
        self.patcher_destroy = mock.patch("torch.distributed.destroy_process_group")
        self.patcher_destroy.start()

    def tearDown(self):
        self.patcher_group.stop()
        self.patcher_destroy.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    def _config(
        self,
        *,
        enable: bool = True,
        enable_ft_dataloader_checkpoints: bool = False,
    ) -> TorchFTCheckpointManager.Config:
        return TorchFTCheckpointManager.Config(
            enable=enable,
            async_mode="disabled",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=enable_ft_dataloader_checkpoints,
        )

    def _manager(
        self,
        config: TorchFTCheckpointManager.Config,
        *,
        participating_rank: int | None = 0,
        replica_id: int = 0,
    ) -> TorchFTCheckpointManager:
        ft_manager = DummyFTManager(
            enabled=True,
            replica_id=replica_id,
            participating_rank=participating_rank,
        )
        return TorchFTCheckpointManager(
            config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=ft_manager,
        )

    def test_registers_healing_state_when_persistent_checkpoint_is_disabled(self):
        manager = self._manager(self._config(enable=False))

        self.assertTrue(self.optimizers.cache_initialized)
        manager.ft_manager.register_state_dict_fn.assert_called_once()
        _, _, save_state = manager.ft_manager.register_state_dict_fn.call_args.args
        self.assertTrue(
            manager.ft_manager.register_state_dict_fn.call_args.kwargs[
                "state_dict_on_training_thread"
            ]
        )
        self.assertIn("model", save_state())

        manager.close()

    @mock.patch.object(CheckpointManager, "_save")
    def test_disabled_checkpoint_save_is_a_noop(self, mock_save):
        manager = self._manager(self._config(enable=False))

        self.assertFalse(manager.save(curr_step=1, last_step=False))

        mock_save.assert_not_called()
        manager.ft_manager.participating_rank.assert_not_called()
        manager.close()

    def test_full_checkpoint_owns_manager_state_but_not_dataloader(self):
        manager = self._manager(self._config())

        self.assertIs(manager.states[TORCHFT_MANAGER], manager.ft_manager)
        self.assertNotIn(DATALOADER, manager._flattened_model_states_sd())
        self.assertNotIn(DATALOADER, manager._states_to_load(model_only=False))

        manager.close()

    @mock.patch.object(CheckpointManager, "_save")
    def test_nonparticipating_replica_skips_full_checkpoint(self, mock_save):
        manager = self._manager(self._config(), participating_rank=1)

        self.assertFalse(manager.save(curr_step=2, last_step=True))

        mock_save.assert_not_called()
        manager.close()

    @mock.patch.object(CheckpointManager, "_save", return_value=True)
    def test_static_replica_zero_owns_checkpoint_before_first_quorum(self, mock_save):
        manager = self._manager(
            self._config(),
            participating_rank=None,
            replica_id=0,
        )

        self.assertTrue(manager.save(curr_step=1, last_step=False))

        mock_save.assert_called_once_with(1, False)
        manager.close()

    @mock.patch.object(CheckpointManager, "_save")
    def test_static_nonzero_replica_skips_checkpoint_before_first_quorum(
        self, mock_save
    ):
        manager = self._manager(
            self._config(),
            participating_rank=None,
            replica_id=1,
        )

        self.assertFalse(manager.save(curr_step=1, last_step=False))

        mock_save.assert_not_called()
        manager.close()

    @mock.patch("torch.cuda.Stream")
    @mock.patch.object(
        dist_checkpoint,
        "async_save",
        side_effect=fake_async_save,
    )
    def test_torchft_async_save_calls_maybe_wait_for_saving(
        self,
        mock_async_save,
        mock_cuda_stream,
    ):
        """
        Test that with FT enabled, AsyncMode.ASYNC via FT triggers correct waits.
        """
        config = TorchFTCheckpointManager.Config(
            enable=True,
            async_mode="async",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=True,
        )
        manager = TorchFTCheckpointManager(
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

        manager.close()


if __name__ == "__main__":
    unittest.main()
