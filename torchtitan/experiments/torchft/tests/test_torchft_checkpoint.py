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

from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager


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

    def _manager(
        self, participating_rank: int, keep_latest_k: int = 0
    ) -> TorchFTCheckpointManager:
        config = TorchFTCheckpointManager.Config(
            enable=True,
            async_mode="disabled",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=keep_latest_k,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=True,
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
            ft_manager=DummyFTManager(
                enabled=True, participating_rank=participating_rank
            ),
        )

    @mock.patch("torch.cuda.Stream")
    @mock.patch.object(dist_checkpoint, "async_save", side_effect=fake_async_save)
    def test_save_returns_whether_the_full_checkpoint_was_written(
        self,
        mock_async_save,
        mock_cuda_stream,
    ):
        # BaseCheckpointManager.save returns _save's result, so this override has
        # to report a bool. The per-replica dataloader checkpoint is a side
        # channel and does not count as writing the checkpoint.
        with mock.patch.object(dist_checkpoint, "save"):
            participating = self._manager(participating_rank=0)
            self.assertIs(True, participating.save(curr_step=5))
            participating.close()

            bystander = self._manager(participating_rank=1)
            self.assertIs(False, bystander.save(curr_step=5))
            bystander.close()

    def _make_ft_steps(self, manager, folder, steps):
        for step in steps:
            path = os.path.join(folder, f"step-{step}")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, ".metadata"), "w") as f:
                f.write("{}")

    def _drain_queue(self, manager):
        paths = []
        while not manager.purge_queue.empty():
            paths.append(manager.purge_queue.get_nowait())
        return paths

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_ft_folder_purge_keeps_latest_k(self, _mock_rank):
        manager = self._manager(participating_rank=0, keep_latest_k=2)
        try:
            self._make_ft_steps(manager, manager._ft_folder(), [1, 2, 3, 4])
            manager._purge_stale_checkpoints()
            purged = self._drain_queue(manager)
            self.assertEqual(
                sorted(purged),
                sorted(os.path.join(manager._ft_folder(), f"step-{s}") for s in (1, 2)),
            )
        finally:
            manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_ft_folder_purge_skipped_for_nonzero_participating_rank(self, _mock_rank):
        manager = self._manager(participating_rank=1, keep_latest_k=2)
        try:
            self._make_ft_steps(manager, manager._ft_folder(), [1, 2, 3, 4])
            manager._purge_stale_checkpoints()
            self.assertTrue(manager.purge_queue.empty())
        finally:
            manager.close()

    def test_ft_load_falls_back_to_legacy_folder(self):
        manager = self._manager(participating_rank=0)
        try:
            legacy_folder = manager._legacy_ft_folder()
            self._make_ft_steps(manager, legacy_folder, [3])
            with mock.patch.object(TorchFTCheckpointManager, "dcp_load") as mock_load:
                manager._ft_load()
            (call,) = mock_load.call_args_list
            self.assertIn("ft-replicat-0", call.kwargs["checkpoint_id"])
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
