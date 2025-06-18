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
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtitan.components.checkpoint import CheckpointManager, MODEL
from torchtitan.config_manager import Checkpoint as CheckpointConfig


class FakeOptimizersContainer:
    """A fake OptimizersContainer that returns fake state dicts."""

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
    """A fake LRSchedulersContainer that does nothing."""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class FakeDataLoader(DataLoader):
    """A fake DataLoader that returns a fake batch."""

    def __init__(self):
        super().__init__(dataset=[], batch_size=1)

    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class DummyFTManager:
    """A fake FTManager-like object with enabled=False."""

    def __init__(self):
        self.enabled = False
        self.manager = None


class DummyFuture:
    def __init__(self):
        self.result = mock.Mock()


def fake_async_save(*args, **kwargs):
    return DummyFuture()


class DummyJobConfig:
    def __init__(self, job):
        self.job = job
        self.checkpoint = CheckpointConfig(
            enable_checkpoint=True,
            async_mode="disabled",
            folder="",
            interval=1,
            keep_latest_k=0,
            last_save_model_weights_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_weights_only=False,
        )
        self.fault_tolerance = SimpleNamespace(replica_id=0)


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.base_temp_dir, self._testMethodName)
        os.makedirs(self.test_folder, exist_ok=True)

        self.model_part = nn.Linear(2, 2)
        self.model_parts = [self.model_part]
        # TODO: Use a real OptimizerContainer here so that we can actually verify
        # some optimizer.state_dict() behavior (e.g., the key being the parameter name.)
        self.optimizers = FakeOptimizersContainer()
        self.lr_schedulers = FakeLRSchedulersContainer()
        self.states = {}
        self.data_loader = FakeDataLoader()
        self.ft_manager = DummyFTManager()

        ckpt_cfg = CheckpointConfig(
            enable_checkpoint=True,
            async_mode="DISABLED",
            folder="",
            interval=1,
            keep_latest_k=2,
            last_save_model_weights_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_weights_only=False,
        )
        ft_ns = SimpleNamespace(replica_id=0)
        job_ns = SimpleNamespace(dump_folder=self.test_folder)
        self.job_config = SimpleNamespace(
            checkpoint=ckpt_cfg,
            fault_tolerance=ft_ns,
            job=job_ns,
        )

        # Patch process group creation
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()

    def tearDown(self):
        self.patcher_group.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    def fake_save(self, state_dict: dict, checkpoint_id: str):
        os.makedirs(checkpoint_id, exist_ok=True)
        sd_to_save = {}
        for key, val in state_dict.items():
            if hasattr(val, "state_dict"):
                sd_to_save[key] = val.state_dict()
            elif isinstance(val, torch.Tensor):
                sd_to_save[key] = val
        torch.save(sd_to_save, os.path.join(checkpoint_id, "state_dict.pt"))

    def fake_load(self, states: dict, checkpoint_id=None):
        path = os.path.join(checkpoint_id, "state_dict.pt")
        loaded = torch.load(path, weights_only="False")
        for key, val in loaded.items():
            if key in states and hasattr(states[key], "load_state_dict"):
                states[key].load_state_dict(val)
            elif key in states and isinstance(states[key], torch.Tensor):
                states[key] = val

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_save_load_restores_state(self, mock_load, mock_save, mock_rank):
        mock_save.side_effect = self.fake_save
        mock_load.side_effect = self.fake_load
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )

        w0 = self.model_part.weight.clone()
        b0 = self.model_part.bias.clone()
        p0 = self.optimizers._fake_param.clone()
        manager.save(curr_step=1)
        with torch.no_grad():
            self.model_part.weight.zero_()
            self.model_part.bias.zero_()
        self.optimizers._fake_param = torch.tensor([42.0], dtype=torch.float32)
        manager.load(step=1)

        self.assertTrue(torch.equal(self.model_part.weight, w0))
        self.assertTrue(torch.equal(self.model_part.bias, b0))
        self.assertTrue(torch.equal(self.optimizers._fake_param, p0))
        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_save_and_purge_keeps_last_k_checkpoints(
        self, mock_load, mock_save, mock_rank
    ):
        mock_save.side_effect = self.fake_save
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )

        manager.save(curr_step=1)
        manager.save(curr_step=2)
        manager.save(curr_step=3)
        deadline = time.time() + 5.0

        while True:
            exist = sorted(os.listdir(self.test_folder))
            if exist == ["step-2", "step-3"]:
                break
            if time.time() > deadline:
                self.fail(f"Purge timed out; found {exist}")
            time.sleep(0.05)

        self.assertListEqual(sorted(os.listdir(self.test_folder)), ["step-2", "step-3"])
        calls = [c.kwargs.get("checkpoint_id") for c in mock_save.call_args_list]
        expected = [os.path.join(self.test_folder, f"step-{i}") for i in (1, 2, 3)]
        self.assertListEqual(calls, expected)
        sd = torch.load(
            os.path.join(self.test_folder, "step-3", "state_dict.pt"),
            weights_only=False,
        )
        self.assertIn("optimizer", sd)
        torch.testing.assert_close(sd["optimizer"]["fake_param"], torch.tensor([1.0]))
        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=1)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_nonzero_rank_does_not_purge_or_save(self, mock_load, mock_save, mock_rank):
        mock_save.side_effect = self.fake_save
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )
        manager.save(curr_step=1)
        manager.save(curr_step=2)
        manager.save(curr_step=3)
        time.sleep(1)
        self.assertListEqual(
            sorted(os.listdir(self.test_folder)), ["step-1", "step-2", "step-3"]
        )
        self.assertEqual(len(mock_save.call_args_list), 3)
        manager.close()

    def test_load_returns_false_when_no_checkpoint_folder(self):
        cfg = self.job_config.checkpoint
        cfg.folder = "nonexistent"
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )
        self.assertFalse(manager.load(step=-1))
        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_load_finds_latest_and_calls_dcp_load(self, mock_load, mock_rank):
        ckpt_folder = os.path.join(self.test_folder, "checkpoints")
        os.makedirs(ckpt_folder, exist_ok=True)
        for s in (2, 5):
            d = os.path.join(ckpt_folder, f"step-{s}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, ".metadata"), "w").close()
        cfg = self.job_config.checkpoint
        cfg.folder = "checkpoints"
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )
        res = manager.load(step=-1)
        expected = os.path.join(ckpt_folder, "step-5")
        mock_load.assert_called_once()
        args, kwargs = mock_load.call_args
        self.assertEqual(args[0], manager._states_to_load(model_only=False))
        self.assertEqual(kwargs.get("checkpoint_id"), expected)
        self.assertTrue(res)
        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_interval_respects_interval(self, mock_load, mock_save, mock_rank):
        """
        Test that save() only triggers on step 1 and multiples of interval, skipping others,
        but respects force flag to override interval.
        """
        cfg = self.job_config.checkpoint
        cfg.interval = 3
        cfg.keep_latest_k = 0
        mock_save.side_effect = self.fake_save
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )
        manager.save(curr_step=1)
        self.assertEqual(mock_save.call_count, 0)
        manager.save(curr_step=2)
        self.assertEqual(mock_save.call_count, 0)
        manager.save(curr_step=2, force=True)
        self.assertEqual(mock_save.call_count, 1)
        manager.save(curr_step=3)
        self.assertEqual(mock_save.call_count, 2)
        manager.save(curr_step=4)
        self.assertEqual(mock_save.call_count, 2)
        manager.save(curr_step=4, force=True)
        self.assertEqual(mock_save.call_count, 3)
        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_last_save_model_weights_only_and_initial_load_model_weights_only(
        self, mock_load, mock_save, mock_rank
    ):
        mock_save.side_effect = self.fake_save
        mock_load.side_effect = self.fake_load
        # Phase 1: save model weights only
        self.job_config.checkpoint.last_save_model_weights_only = True
        manager1 = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={MODEL: self.model_part},
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )
        manager1.save(curr_step=1, force=True)
        path1 = os.path.join(self.test_folder, "step-1")
        self.assertTrue(os.path.isdir(path1))
        # Phase 2: initial load from step-1
        cfg = self.job_config.checkpoint
        cfg.last_save_model_weights_only = False
        cfg.initial_load_model_weights_only = True
        cfg.initial_load_path = path1
        cfg.folder = ""
        self.job_config.job.dump_folder = self.test_folder
        manager2 = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={MODEL: self.model_part},
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )
        r1 = manager2.load(step=1)
        self.assertTrue(r1)
        mock_load.assert_called_once()
        args1, kwargs1 = mock_load.call_args
        self.assertEqual(kwargs1.get("checkpoint_id"), path1)
        # Phase 3: save new step under default folder, then load that
        manager2.save(curr_step=2, force=True)
        # Default folder is test_folder, so step-2 under that
        step2_dir = os.path.join(self.test_folder, "step-2")
        self.assertTrue(os.path.isdir(step2_dir))
        r2 = manager2.load(step=2)
        self.assertTrue(r2)
        self.assertEqual(mock_load.call_count, 2)
        args2, kwargs2 = mock_load.call_args_list[1]
        self.assertEqual(kwargs2.get("checkpoint_id"), step2_dir)
        manager1.close()
        manager2.close()

    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_async_save_calls_async_wait(self, mock_async_save, mock_new_group):
        """
        Test that in AsyncMode.ASYNC, save() waits on previous async future.
        """
        # Configure async mode
        job_config = DummyJobConfig(job=self.job_config.job)
        job_config.checkpoint.async_mode = "async"
        ft_manager = DummyFTManager()
        states = {"trainer": torch.tensor([0])}
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=states,
            job_config=job_config,
            ft_manager=ft_manager,
        )

        # First save schedules async
        manager.save(curr_step=10, force=False)
        future = manager.async_future
        future.result.assert_not_called()

        # Second save should wait
        manager.save(curr_step=20, force=False)
        future.result.assert_called_once()

        # New future created
        new_future = manager.async_future
        new_future.result.assert_not_called()

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
        job_config = DummyJobConfig(job=self.job_config.job)
        job_config.checkpoint.async_mode = "disabled"
        ft_manager = mock.Mock()
        ft_manager.enabled = True
        states = {"trainer": torch.tensor([0])}
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=states,
            job_config=job_config,
            ft_manager=ft_manager,
        )

        # Initially no future
        self.assertIsNone(manager.async_future)
        manager.save(curr_step=5, force=False)
        self.assertIsNotNone(manager.async_future)

        manager.async_future.result.assert_not_called()
        prev_future = manager.async_future
        manager.save(curr_step=6, force=False)
        prev_future.result.assert_called_once()
        self.assertIsNotNone(manager.async_future)
        manager.async_future.result.assert_not_called()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    def test_enable_first_step_checkpoint(self, mock_save, mock_rank):
        """
        Test that enable_first_step_checkpoint triggers checkpoint save at step 1.
        """
        mock_save.side_effect = self.fake_save

        # Test with enable_first_step_checkpoint=False (default case)
        cfg = self.job_config.checkpoint
        cfg.interval = 10  # Set interval to 10 so step 1 wouldn't normally trigger save
        cfg.keep_latest_k = 0  # Disable purging to avoid confusion

        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )

        # Step 1 should not trigger save when enable_first_step_checkpoint=False
        # and not at interval
        manager.save(curr_step=1)
        self.assertEqual(mock_save.call_count, 0)

        # Step 10 should trigger save due to interval
        manager.save(curr_step=10)
        self.assertEqual(mock_save.call_count, 1)

        manager.close()

        # Test with enable_first_step_checkpoint=True
        mock_save.reset_mock()
        cfg.enable_first_step_checkpoint = True

        manager2 = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            job_config=self.job_config,
            ft_manager=self.ft_manager,
        )

        # Step 1 should trigger save due to enable_first_step_checkpoint=True
        manager2.save(curr_step=1)
        self.assertEqual(mock_save.call_count, 1)

        # Step 2 should not trigger save (not at interval and not forced)
        manager2.save(curr_step=2)
        self.assertEqual(mock_save.call_count, 1)

        # Step 10 should trigger save due to interval
        manager2.save(curr_step=10)
        self.assertEqual(mock_save.call_count, 2)

        manager2.close()


if __name__ == "__main__":
    unittest.main()
