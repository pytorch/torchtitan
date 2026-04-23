# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
from torch.utils.data import DataLoader
from torchtitan.components.checkpoint import (
    AsyncCheckpointerType,
    AsyncMode,
    CheckpointManager,
    MODEL,
    OPTIMIZER,
    Terminate,
)


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


class DummyFuture:
    def __init__(self):
        self.result = mock.Mock()


class DummyAsyncResult(AsyncSaveResponse):
    """Mock object that mimics the return value of dcp.async_save with pinned memory"""

    def __init__(self):
        self.upload_completion = DummyFuture()
        self.staging_completion = DummyFuture()


def fake_async_save(*args, **kwargs):
    # Check if this is async_with_pinned_mem mode by looking for async_stager parameter
    if "async_stager" in kwargs:
        return DummyAsyncResult()
    else:
        return DummyFuture()


class DummyTrainerConfig:
    def __init__(self, dump_folder):
        self.dump_folder = dump_folder
        self.checkpoint = CheckpointManager.Config(
            enable=True,
            async_mode="disabled",
            folder="",
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
        )


class CheckpointTestBase:
    """
    Base class providing common setup and teardown logic for checkpointing tests.

    Note: This class does not inherit from `unittest.TestCase` to prevent the
    unittest runner from attempting to execute it as a standalone test suite.
    Actual test classes should use multiple inheritance:
    `class TestSpecificLogic(CheckpointTestBase, unittest.TestCase)`.
    """

    def setUp(self):
        self.base_temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.base_temp_dir, self._testMethodName)
        os.makedirs(self.test_folder, exist_ok=True)

        self.model_part = nn.Linear(2, 2)
        self.model_parts = [self.model_part]
        self.states = {"trainer": torch.tensor([1.2347])}
        # TODO: Use a real OptimizerContainer here so that we can actually verify
        # some optimizer.state_dict() behavior (e.g., the key being the parameter name.)
        self.optimizers = FakeOptimizersContainer()
        self.lr_schedulers = FakeLRSchedulersContainer()
        self.data_loader = FakeDataLoader()

        ckpt_cfg = CheckpointManager.Config(
            enable=True,
            async_mode="DISABLED",
            folder="",
            interval=1,
            keep_latest_k=2,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
        )
        self.trainer_config = SimpleNamespace(
            checkpoint=ckpt_cfg,
            dump_folder=self.test_folder,
        )

        self.manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        # Patch process group creation
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()

    def tearDown(self):
        """
        Ensures that background threads and processes are cleaned up
        even if a test assertion fails.
        """
        try:
            if hasattr(self, "manager") and self.manager is not None:
                self.manager.close()
        except Exception as e:
            print(f"Error closing manager during tearDown: {e}")
        finally:
            if hasattr(self, "patcher_group"):
                self.patcher_group.stop()

            if hasattr(self, "base_temp_dir") and os.path.exists(self.base_temp_dir):
                shutil.rmtree(self.base_temp_dir)

            if hasattr(super(), "tearDown"):
                super().tearDown()

    def fake_save(self, state_dict: dict, checkpoint_id: str, storage_writer=None):
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
                states[key].copy_(val)


class TestCheckpointManager(CheckpointTestBase, unittest.TestCase):
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
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
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
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
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
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
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
        cfg = self.trainer_config.checkpoint
        cfg.folder = "nonexistent"
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
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
        cfg = self.trainer_config.checkpoint
        cfg.folder = "checkpoints"
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )
        res = manager.load(step=-1)
        expected = os.path.join(ckpt_folder, "step-5")
        mock_load.assert_called_once()
        args, kwargs = mock_load.call_args
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
        cfg = self.trainer_config.checkpoint
        cfg.interval = 3
        cfg.keep_latest_k = 0
        mock_save.side_effect = self.fake_save
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )
        manager.save(curr_step=1)
        self.assertEqual(mock_save.call_count, 0)
        manager.save(curr_step=2)
        self.assertEqual(mock_save.call_count, 0)
        manager.save(curr_step=2, last_step=True)
        self.assertEqual(mock_save.call_count, 1)
        manager.save(curr_step=3)
        self.assertEqual(mock_save.call_count, 2)
        manager.save(curr_step=4)
        self.assertEqual(mock_save.call_count, 2)
        manager.save(curr_step=4, last_step=True)
        self.assertEqual(mock_save.call_count, 3)
        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_last_save_model_only_and_initial_load_model_only(
        self, mock_load, mock_save, mock_rank
    ):
        mock_save.side_effect = self.fake_save
        mock_load.side_effect = self.fake_load
        # Phase 1: save model weights only
        self.trainer_config.checkpoint.last_save_model_only = True
        manager1 = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )
        manager1.save(curr_step=1, last_step=True)
        path1 = os.path.join(self.test_folder, "step-1")
        self.assertTrue(os.path.isdir(path1))
        # Phase 2: initial load from step-1
        cfg = self.trainer_config.checkpoint
        cfg.last_save_model_only = False
        cfg.initial_load_model_only = True
        cfg.initial_load_path = path1
        cfg.folder = ""
        self.trainer_config.dump_folder = self.test_folder
        manager2 = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )
        r1 = manager2.load(step=1)
        self.assertTrue(r1)
        mock_load.assert_called_once()
        args1, kwargs1 = mock_load.call_args
        self.assertEqual(kwargs1.get("checkpoint_id"), path1)
        # Phase 3: save new step under default folder, then load that
        manager2.save(curr_step=2, last_step=True)
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

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.cuda.Stream")
    @mock.patch("torchtitan.components.checkpoint.DefaultStager")
    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_async_save_with_pinned_mem_sets_staging_flag(
        self,
        mock_async_save,
        mock_new_group,
        mock_default_stager,
        mock_cuda_stream,
        mock_rank,
    ):
        """
        Test that AsyncMode.ASYNC_WITH_PINNED_MEM correctly sets staging flag.

        This test verifies the bug fix where self.staging was not being set to True
        when using ASYNC_WITH_PINNED_MEM mode, which caused maybe_wait_for_staging()
        to not wait properly for staging completion.
        """
        # Configure async mode with pinned memory
        trainer_config = DummyTrainerConfig(dump_folder=self.trainer_config.dump_folder)
        checkpoint_config = trainer_config.checkpoint
        checkpoint_config.async_mode = "async_with_pinned_mem"

        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=checkpoint_config,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        # Initially staging should be False
        self.assertFalse(manager.staging)

        # After save, staging should be set to True
        manager.save(curr_step=1, last_step=False)
        self.assertTrue(manager.staging)

        # Verify that staging_future exists
        self.assertIsNotNone(manager.staging_future)

        # Verify that maybe_wait_for_staging actually waits when staging is True
        manager.maybe_wait_for_staging()
        # After waiting, staging should be set back to False
        self.assertFalse(manager.staging)

        manager.close()

    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_async_save_calls_async_wait(self, mock_async_save, mock_new_group):
        """
        Test that in AsyncMode.ASYNC, save() waits on previous async future.
        """
        # Configure async mode
        trainer_config = DummyTrainerConfig(dump_folder=self.trainer_config.dump_folder)
        checkpoint_config = trainer_config.checkpoint
        checkpoint_config.async_mode = "async"
        states = {"trainer": torch.tensor([0])}
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=states,
            config=checkpoint_config,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        # First save schedules async
        manager.save(curr_step=10, last_step=False)
        future = manager.save_future
        future.result.assert_not_called()

        # Second save should wait
        manager.save(curr_step=20, last_step=False)
        future.result.assert_called_once()

        # New future created
        new_future = manager.save_future
        new_future.result.assert_not_called()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    def test_enable_first_step_checkpoint(self, mock_save, mock_rank):
        """
        Test that enable_first_step_checkpoint triggers checkpoint save at step 1.
        """
        mock_save.side_effect = self.fake_save

        # Test with enable_first_step_checkpoint=False (default case)
        cfg = self.trainer_config.checkpoint
        cfg.interval = 10  # Set interval to 10 so step 1 wouldn't normally trigger save
        cfg.keep_latest_k = 0  # Disable purging to avoid confusion

        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
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
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
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

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    def test_non_persist_buffer_not_saved(self, mock_save, mock_rank):
        """Test that freqs_cis is not saved"""

        # Create a fake model with freqs_cis and other parameters
        class FakeModelWithFreqsCis(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))
                self.bias = nn.Parameter(torch.randn(2))
                # Register freqs_cis as a buffer (common pattern in transformer models)
                self.register_buffer("freqs_cis", torch.randn(10, 5), persistent=False)
                self.other_param = nn.Parameter(torch.randn(3, 3))

        fake_model = FakeModelWithFreqsCis()
        mock_save.side_effect = self.fake_save

        cfg = self.trainer_config.checkpoint
        cfg.keep_latest_k = 0  # Disable purging

        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=[fake_model],
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        manager.save(curr_step=1)
        self.assertEqual(mock_save.call_count, 1)
        checkpoint_path = os.path.join(self.test_folder, "step-1", "state_dict.pt")
        saved_data = torch.load(checkpoint_path, weights_only=False)

        # Verify that freqs_cis is NOT in the saved state dict
        self.assertNotIn("freqs_cis", saved_data)
        # Verify that other parameters ARE in the saved state dict
        self.assertIn("weight", saved_data)
        self.assertIn("bias", saved_data)
        self.assertIn("other_param", saved_data)

        manager.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    def test_load_only_prevents_saving(self, mock_save, mock_rank):
        """
        Test that load_only=True prevents checkpoint saving.
        """
        mock_save.side_effect = self.fake_save

        # Configure load_only=True
        cfg = self.trainer_config.checkpoint
        cfg.load_only = True
        cfg.interval = 1  # Set low interval to ensure saves would normally trigger

        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        # Test various save conditions that would normally trigger saves
        manager.save(curr_step=1)  # Regular step save
        self.assertEqual(mock_save.call_count, 0)

        manager.save(curr_step=5)  # Interval-based save
        self.assertEqual(mock_save.call_count, 0)

        manager.save(curr_step=10, last_step=True)  # Last step save
        self.assertEqual(mock_save.call_count, 0)

        manager.close()

        # Verify that saves work normally when load_only=False
        mock_save.reset_mock()
        cfg.load_only = False

        manager2 = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        manager2.save(curr_step=1)  # Should trigger save now
        self.assertEqual(mock_save.call_count, 1)

        manager2.close()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    def test_verify_prefix(self, mock_save, mock_load, mock_rank):
        def fake_save(state_dict: dict, checkpoint_id: str, storage_writer=None):
            self.assertIn("bias", state_dict)
            self.assertIn("weight", state_dict)
            # No model prefix
            self.assertNotIn("model", state_dict)
            if "step-1" in checkpoint_id:
                self.assertIn("optimizer", state_dict)
                self.fake_save(state_dict, checkpoint_id)
            else:
                self.assertNotIn("optimizer", state_dict)
            return

        def fake_load(state_dict: dict, checkpoint_id=None):
            self.assertIn("bias", state_dict)
            self.assertIn("weight", state_dict)
            # No model prefix
            self.assertNotIn("model", state_dict)
            self.assertIn("optimizer", state_dict)

        self.trainer_config.checkpoint.last_save_model_only = True
        self.trainer_config.checkpoint.initial_load_model_only = False
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=self.trainer_config.checkpoint,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        mock_save.side_effect = fake_save
        mock_load.side_effect = fake_load
        manager.save(curr_step=1)
        manager.save(curr_step=2, last_step=True)
        manager.load(step=1)


class TestClose(CheckpointTestBase, unittest.TestCase):
    """
    Validates the close() method and its ability to clean up background
    processes, threads, and stagers, including its invocation via __del__.
    """

    def setUp(self):
        super().setUp()

        # Injected mocks for background components
        self.manager.mp = mock.MagicMock()
        self.manager.mp.is_alive.return_value = True
        self.manager.mp_queue_send = mock.MagicMock()

        self.manager.purge_thread = mock.MagicMock()
        self.manager.purge_thread.is_alive.return_value = True
        self.manager.purge_queue = mock.MagicMock()

        self.manager.stager = mock.MagicMock()

    def test_del_calls_close(self):
        """Verify that the destructor triggers the close method."""
        with mock.patch.object(CheckpointManager, "close") as mock_close:
            self.manager.__del__()
            mock_close.assert_called_once()

    def test_close_terminates_all_components(self):
        # Act
        self.manager.close()

        # Assert - Multiprocessing cleanup
        put_args = self.manager.mp_queue_send.put.call_args[0][0]
        self.assertIsInstance(put_args, Terminate)
        self.manager.mp.join.assert_called_once()

        # Assert - Purge thread cleanup
        put_purge_args = self.manager.purge_queue.put.call_args[0][0]
        self.assertIsInstance(put_purge_args, Terminate)
        self.manager.purge_thread.join.assert_called_once()

        # Assert - Stager cleanup
        self.manager.stager.close.assert_called_once()

    def test_close_skips_inactive_components(self):
        """Ensure close() doesn't crash if components are already dead or None."""
        self.manager.mp.is_alive.return_value = False
        self.manager.purge_thread.is_alive.return_value = False
        self.manager.stager = None

        # Should not raise any errors
        self.manager.close()

        self.manager.mp_queue_send.put.assert_not_called()
        self.manager.mp.join.assert_not_called()


@mock.patch("torchtitan.components.checkpoint.dcp")
@mock.patch("torchtitan.components.checkpoint.HuggingFaceStorageWriter")
@mock.patch(
    "torchtitan.components.checkpoint.consolidate_safetensors_files_on_every_rank"
)
@mock.patch("torchtitan.components.checkpoint.GarbageCollection")
class TestDcpSave(CheckpointTestBase, unittest.TestCase):
    """
    Validates the dcp_save method branching logic for sync/async modes
    and Hugging Face export integration.
    """

    def setUp(self):
        super().setUp()
        self.manager.sd_adapter = mock.MagicMock()

    def test_sync_save_standard(self, mock_gc, mock_consolidate, mock_writer, mock_dcp):
        """Verify standard synchronous dcp.save call."""
        state_dict = self.manager.states
        checkpoint_id = "test_step_10"

        self.manager.dcp_save(state_dict, checkpoint_id, AsyncMode.DISABLED)

        mock_dcp.save.assert_called_once_with(
            state_dict, storage_writer=None, checkpoint_id=checkpoint_id
        )

    def test_async_mode_standard(
        self, mock_gc, mock_consolidate, mock_writer, mock_dcp
    ):
        """Verify that AsyncMode.ASYNC calls dcp.async_save with basic async params."""
        state_dict = self.manager.states
        checkpoint_id = "async_standard"

        self.manager.dcp_save(state_dict, checkpoint_id, AsyncMode.ASYNC)

        mock_dcp.async_save.assert_called_once()
        _, kwargs = mock_dcp.async_save.call_args

        self.assertEqual(kwargs["checkpoint_id"], checkpoint_id)
        self.assertEqual(kwargs["process_group"], self.manager.pg)
        # Ensure the pinned-mem specific args are NOT passed here
        self.assertNotIn("async_checkpointer_type", kwargs)
        self.assertNotIn("async_stager", kwargs)

    def test_async_mode_with_pinned_mem(
        self, mock_gc, mock_consolidate, mock_writer, mock_dcp
    ):
        """Verify that AsyncMode.ASYNC_WITH_PINNED_MEM passes the stager and process type."""

        state_dict = self.manager.states
        checkpoint_id = "async_pinned"

        # Ensure the manager has a stager mock (set in CheckpointTestBase or setUp)
        self.manager.stager = mock.MagicMock()

        self.manager.dcp_save(
            state_dict, checkpoint_id, AsyncMode.ASYNC_WITH_PINNED_MEM
        )

        mock_dcp.async_save.assert_called_once()
        _, kwargs = mock_dcp.async_save.call_args

        self.assertEqual(kwargs["process_group"], self.manager.pg)
        self.assertEqual(
            kwargs["async_checkpointer_type"], AsyncCheckpointerType.PROCESS
        )
        self.assertEqual(kwargs["async_stager"], self.manager.stager)

    def test_hf_save_consolidated_path(
        self, mock_gc, mock_consolidate, mock_writer, mock_dcp
    ):
        """Verify HF path with no mapping triggers consolidation on one rank."""
        state_dict = self.manager.states
        checkpoint_id = "hf_consolidated"

        # 1. Setup: No mapping -> consolidated path
        self.manager.sd_adapter.to_hf.return_value = {"hf_model": torch.tensor([42.69])}
        self.manager.sd_adapter.fqn_to_index_mapping = None

        self.manager.dcp_save(state_dict, checkpoint_id, AsyncMode.DISABLED, to_hf=True)

        # 2. Assert: check writer config
        # Expectation: enable_consolidation=True, path is root checkpoint_id
        _, kwargs = mock_writer.call_args
        self.assertEqual(kwargs["path"], checkpoint_id)
        self.assertTrue(kwargs["enable_consolidation"])

        # Verify consolidate function was not called manually (handled by writer)
        mock_consolidate.assert_not_called()

    def test_hf_save_sharded_path(
        self, mock_gc, mock_consolidate, mock_writer, mock_dcp
    ):
        """Verify HF path WITH mapping uses sharded path and manual consolidation."""
        state_dict = self.manager.states
        checkpoint_id = "hf_sharded"
        dummy_mapping = {"layer1.weight": torch.tensor([42.69])}

        # 1. Setup: Mapping exists -> sharded path + manual consolidation
        self.manager.sd_adapter.to_hf.return_value = {"hf_model": torch.tensor([42])}
        self.manager.sd_adapter.fqn_to_index_mapping = dummy_mapping

        self.manager.dcp_save(state_dict, checkpoint_id, AsyncMode.DISABLED, to_hf=True)

        # 2. Assert: check writer config
        # Expectation: enable_consolidation=False, path includes "sharded"
        _, kwargs = mock_writer.call_args
        self.assertEqual(kwargs["path"], os.path.join(checkpoint_id, "sharded"))
        self.assertFalse(kwargs["enable_consolidation"])
        self.assertEqual(kwargs["fqn_to_index_mapping"], dummy_mapping)

        # 3. Assert: Verify manual consolidation trigger
        mock_consolidate.assert_called_once_with(
            input_dir=os.path.join(checkpoint_id, "sharded"),
            output_dir=checkpoint_id,
            fqn_to_index_mapping=dummy_mapping,
            num_threads=5,
        )

    def test_hf_without_adapter_raises_assertion(
        self, mock_gc, mock_consolidate, mock_writer, mock_dcp
    ):
        """Verify assertion error if to_hf is True but sd_adapter is missing."""
        self.manager.sd_adapter = None
        with self.assertRaisesRegex(AssertionError, "sd_adapter is not provided"):
            self.manager.dcp_save({}, "test", AsyncMode.DISABLED, to_hf=True)

    def test_garbage_collection_trigger(
        self, mock_gc, mock_consolidate, mock_writer, mock_dcp
    ):
        """Verify GC is called when enable_garbage_collection is True."""
        self.manager.dcp_save(
            {}, "gc_test", AsyncMode.DISABLED, enable_garbage_collection=True
        )
        mock_gc.collect.assert_called_once()


@mock.patch("torchtitan.components.checkpoint.dcp")
class TestDcpLoad(CheckpointTestBase, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.manager.sd_adapter = mock.MagicMock()
        # Manually swap the model state with a mock after initialization
        # This ensures the manager's internal dictionary points to our mock.
        self.mock_model_obj = mock.MagicMock()
        self.manager.states[MODEL] = self.mock_model_obj

    def test_standard_dcp_load(self, mock_dcp):
        """Verify standard DCP load and subsequent model state_dict application."""
        state_dict = {"layer1.weight": torch.randn(2, 2)}
        checkpoint_id = "standard_step_10"

        self.manager.dcp_load(
            state_dict, checkpoint_id, from_hf=False, from_quantized=False
        )

        # Verify dcp.load call
        mock_dcp.load.assert_called_once_with(state_dict, checkpoint_id=checkpoint_id)

        # Verify the manager called load_state_dict on its internal model state
        self.mock_model_obj.load_state_dict.assert_called_once_with(state_dict)

    def test_hf_load_roundtrip(self, mock_dcp):
        """
        Verify the full HF orchestration:
        to_hf -> get_reader -> dcp.load -> from_hf -> load_state_dict
        """
        # 1. Setup specific test data
        initial_state = {"input": "raw"}
        mock_hf_state = {"hf": "transformed"}
        mock_restored_state = {"output": "final"}
        checkpoint_id = "hf_checkpoint_dir"
        mock_reader = mock.MagicMock()

        # Configure the adapter mocks
        self.manager.sd_adapter.to_hf.return_value = mock_hf_state
        self.manager.sd_adapter.get_hf_storage_reader.return_value = mock_reader
        self.manager.sd_adapter.from_hf.return_value = mock_restored_state

        # 2. Act
        self.manager.dcp_load(
            initial_state,
            checkpoint_id,
            from_hf=True,
            from_quantized=True,  # Testing the flag propagation
        )

        # 3. Assert - The Transformation Pipeline
        # Verify initial conversion
        self.manager.sd_adapter.to_hf.assert_called_once_with(initial_state)

        # Verify storage reader creation with the correct flags
        self.manager.sd_adapter.get_hf_storage_reader.assert_called_once_with(
            checkpoint_id, True
        )

        # 4. Assert - The DCP Load Call
        # Critical: Verify dcp.load was called with the transformed dict and the HF reader
        mock_dcp.load.assert_called_once_with(mock_hf_state, storage_reader=mock_reader)

        # 5. Assert - The Restoration Pipeline
        # Verify conversion back from HF format using the dict filled by dcp.load
        self.manager.sd_adapter.from_hf.assert_called_once_with(mock_hf_state)

        # Verify the final state dict was applied to the model
        self.mock_model_obj.load_state_dict.assert_called_once_with(mock_restored_state)

    def test_hf_load_without_adapter_raises(self, mock_dcp):
        """Verify assertion error if from_hf is True but no adapter is present."""
        self.manager.sd_adapter = None
        with self.assertRaises(AssertionError):
            self.manager.dcp_load({}, "id", from_hf=True, from_quantized=False)


class TestSave(CheckpointTestBase, unittest.TestCase):
    """
    Validates the high-level save() orchestration, including async future management
    and the last_step override.
    """

    def setUp(self):
        super().setUp()
        # Mock internal helpers to isolate the save() logic
        self.manager._should_save = mock.MagicMock()
        self.manager._create_checkpoint_id = mock.MagicMock(return_value="step_10")
        self.manager._async_wait = mock.MagicMock()
        self.manager._save_last_step = mock.MagicMock()
        self.manager._purge_stale_checkpoints = mock.MagicMock()
        self.manager._flattened_model_states_sd = mock.MagicMock(
            return_value={"state": 1}
        )
        self.manager.dcp_save = mock.MagicMock()

    def test_save_early_exit(self):
        """If _should_save is False, save() should do nothing."""
        self.manager._should_save.return_value = False
        self.manager.save(curr_step=10)

        self.manager.dcp_save.assert_not_called()
        self.manager._async_wait.assert_not_called()

    def test_save_last_step_branch(self):
        """When last_step is True, verify _save_last_step is called and normal save is skipped."""
        self.manager._should_save.return_value = True

        self.manager.save(curr_step=100, last_step=True)

        self.manager._async_wait.assert_called_once()
        self.manager._save_last_step.assert_called_once_with(100)
        # Normal save path should be bypassed
        self.manager.dcp_save.assert_not_called()

    def test_save_sync_mode(self):
        """Verify standard synchronous save behavior."""
        self.manager._should_save.return_value = True
        self.manager.async_mode = AsyncMode.DISABLED

        self.manager.save(curr_step=10)

        self.manager.dcp_save.assert_called_once_with(
            {"state": 1},
            checkpoint_id="step_10",
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
        )
        self.manager._purge_stale_checkpoints.assert_called_once()

    @mock.patch("torchtitan.components.checkpoint.GarbageCollection")
    def test_save_async_standard(self, mock_gc):
        """Verify GC is called before and after dcp_save in standard ASYNC mode."""
        self.manager._should_save.return_value = True
        self.manager.async_mode = AsyncMode.ASYNC
        self.manager.dcp_save.return_value = "future_obj"

        self.manager.save(curr_step=10)

        # GC should be called twice in this branch
        self.assertEqual(mock_gc.collect.call_count, 2)
        self.assertEqual(self.manager.save_future, "future_obj")

    @mock.patch("torchtitan.components.checkpoint.DefaultStager")
    def test_save_async_pinned_memory(self, mock_stager_class):
        """Verify stager setup and future assignment for PINNED_MEM mode."""
        self.manager._should_save.return_value = True
        self.manager.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        self.manager.stager = None

        # Mock the response from dcp_save
        mock_res = mock.MagicMock(spec=AsyncSaveResponse)
        mock_res.upload_completion = "future_save"
        mock_res.staging_completion = "future_staging"
        self.manager.dcp_save.return_value = mock_res

        self.manager.save(curr_step=10)

        # Verify the stager class was instantiated
        mock_stager_class.assert_called_once()
        self.assertEqual(self.manager.save_future, "future_save")
        self.assertTrue(self.manager.staging)


@mock.patch("torchtitan.components.checkpoint.os.path.exists")
@mock.patch("torchtitan.components.checkpoint.os.path.isdir")
class TestLoad(CheckpointTestBase, unittest.TestCase):
    """
    Validates the high-level load() orchestration, including initial loads,
    HF transitions, and standard resume logic.
    """

    def setUp(self):
        super().setUp()
        # Mock internal helpers
        self.manager.sd_adapter = mock.MagicMock()
        self.manager.dcp_load = mock.MagicMock()
        self.manager._states_to_load = mock.MagicMock(return_value={"state": 1})
        self.manager._find_load_step = mock.MagicMock(return_value=10)
        self.manager._create_checkpoint_id = mock.MagicMock(
            side_effect=lambda s: f"step_{s}"
        )

    def test_load_disabled(self, mock_isdir, mock_exists):
        """Should return False immediately if checkpointing is disabled."""
        self.manager.enable = False
        self.assertFalse(self.manager.load())
        self.manager.dcp_load.assert_not_called()

    def test_load_standard_resume(self, mock_isdir, mock_exists):
        """Test resuming from an existing checkpoint folder."""
        # Folder exists, checkpoint dir exists
        mock_exists.return_value = True
        mock_isdir.return_value = True

        success = self.manager.load(step=-1)  # Load latest

        self.assertTrue(success)
        self.manager._find_load_step.assert_called_once()
        self.manager.dcp_load.assert_called_once_with(
            {"state": 1}, checkpoint_id="step_10", from_hf=False, from_quantized=False
        )

    def test_load_initial_path_valid(self, mock_isdir, mock_exists):
        """Test loading from a specific initial_load_path when main folder is missing."""
        mock_exists.return_value = False  # Main folder doesn't exist
        mock_isdir.return_value = True  # But the initial path does

        self.manager.initial_load_path = "/custom/path"

        self.assertTrue(self.manager.load())
        self.manager.dcp_load.assert_called_once_with(
            {"state": 1},
            checkpoint_id="/custom/path",
            from_hf=False,
            from_quantized=False,
        )

    def test_load_from_hf_assets(self, mock_isdir, mock_exists):
        """Test the logic for loading weights from HF assets via sd_adapter."""
        mock_exists.return_value = False
        mock_isdir.return_value = True

        # Configure for HF
        self.manager.initial_load_in_hf = True
        self.manager.initial_load_model_only = True
        self.manager.initial_load_path = None
        self.manager.sd_adapter.hf_assets_path = "/hf/weights"

        self.assertTrue(self.manager.load())
        self.manager._states_to_load.assert_called_once_with(True)  # model_only=True
        self.manager.dcp_load.assert_called_once_with(
            {"state": 1},
            checkpoint_id="/hf/weights",
            from_hf=True,
            from_quantized=False,
        )

    def test_load_hf_validation_errors(self, mock_isdir, mock_exists):
        """Verify assertions for incorrect HF configurations."""
        mock_exists.return_value = False

        # 1. HF without model_only=True
        self.manager.initial_load_in_hf = True
        self.manager.initial_load_model_only = False
        with self.assertRaisesRegex(AssertionError, "Only model can be loaded"):
            self.manager.load()

        # 2. Quantized without HF
        self.manager.initial_load_model_only = True
        self.manager.initial_load_in_hf = False
        self.manager.initial_load_in_hf_quantized = True
        with self.assertRaisesRegex(AssertionError, "only be loaded from HuggingFace"):
            self.manager.load()


@mock.patch("torch.distributed.get_rank", return_value=0)
class TestMaybeWaitForStaging(CheckpointTestBase, unittest.TestCase):
    """
    Validates that the checkpointer correctly waits for GPU-to-CPU staging
    to complete before proceeding.
    """

    def setUp(self):
        super().setUp()
        self.manager.staging_future = mock.MagicMock()

    def test_staging_wait_success(self, mock_rank):
        """Verify that we wait for the future and reset the staging flag."""
        self.manager.enable_staging = True
        self.manager.staging = True

        self.manager.maybe_wait_for_staging()

        # Verify result() was called to block until finished
        self.manager.staging_future.result.assert_called_once()
        # Verify the flag is cleared
        self.assertFalse(self.manager.staging)

    def test_no_wait_if_staging_disabled(self, mock_rank):
        """Verify we don't wait if enable_staging is False."""
        self.manager.enable_staging = False
        self.manager.staging = True

        self.manager.maybe_wait_for_staging()

        self.manager.staging_future.result.assert_not_called()
        self.assertTrue(self.manager.staging)

    def test_no_wait_if_not_currently_staging(self, mock_rank):
        """Verify we don't wait if the staging flag is already False."""
        self.manager.enable_staging = True
        self.manager.staging = False

        self.manager.maybe_wait_for_staging()

        self.manager.staging_future.result.assert_not_called()

    def test_assertion_if_future_is_missing(self, mock_rank):
        """Verify that an error is raised if staging is True but future is None."""
        self.manager.enable_staging = True
        self.manager.staging = True
        self.manager.staging_future = None

        with self.assertRaises(AssertionError):
            self.manager.maybe_wait_for_staging()


@mock.patch("torchtitan.components.checkpoint.os.path.isdir")
@mock.patch("torchtitan.components.checkpoint.os.listdir")
@mock.patch("torchtitan.components.checkpoint.os.path.isfile")
class TestFindLoadStep(CheckpointTestBase, unittest.TestCase):
    """
    Validates the logic for scanning the checkpoint folder and identifying
    the latest valid training step.
    """

    def test_find_step_empty_or_missing_folder(
        self, mock_isfile, mock_listdir, mock_isdir
    ):
        """Should return -1 if the folder does not exist or is empty."""
        mock_isdir.return_value = False
        self.assertEqual(self.manager._find_load_step(), -1)

        mock_isdir.return_value = True
        mock_listdir.return_value = []
        self.assertEqual(self.manager._find_load_step(), -1)

    def test_find_step_valid_dcp(self, mock_isfile, mock_listdir, mock_isdir):
        """Should find the maximum step when valid DCP metadata is present."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["step-10", "step-20", "not-a-step", "step-5"]

        # Simulate .metadata exists for all step folders
        mock_isfile.side_effect = lambda path: ".metadata" in path

        self.assertEqual(self.manager._find_load_step(), 20)

    def test_find_step_valid_safetensors(self, mock_isfile, mock_listdir, mock_isdir):
        """Should find the step when safetensors index is present instead of DCP metadata."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["step-10", "step-20", "not-a-step", "step-5"]

        # Simulate only safetensors index exists
        mock_isfile.side_effect = lambda path: "model.safetensors.index.json" in path

        self.assertEqual(self.manager._find_load_step(), 20)

    def test_ignore_invalid_metadata(self, mock_isfile, mock_listdir, mock_isdir):
        """Should ignore folders that match the pattern but lack valid metadata probes."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["step-100", "step-200"]

        # Simulate only step-100 is "valid" (has .metadata)
        # step-200 is "corrupt" or "incomplete" (no metadata file)
        def isfile_side_effect(path):
            return "step-100" in path and ".metadata" in path

        mock_isfile.side_effect = isfile_side_effect

        self.assertEqual(self.manager._find_load_step(), 100)

    def test_custom_folder_argument(self, mock_isfile, mock_listdir, mock_isdir):
        """Verify the method respects the 'folder' argument if provided."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["step-99"]
        mock_isfile.return_value = True

        custom_folder = "/tmp/other_checkpoints"
        step = self.manager._find_load_step(folder=custom_folder)

        # Verify listdir was called with the custom folder, not self.folder
        mock_listdir.assert_called_once_with(custom_folder)
        self.assertEqual(step, 99)


@mock.patch("torch.distributed.get_rank", return_value=0)
class TestCreateCheckpointId(CheckpointTestBase, unittest.TestCase):
    def test_create_checkpoint_id(self, _):
        """Verify path generation for default and custom folders."""
        # 1. Test default folder
        path = self.manager._create_checkpoint_id(10)
        self.assertEqual(path, os.path.join(self.manager.folder, "step-10"))

        # 2. Test custom folder override
        path = self.manager._create_checkpoint_id(20, folder="/tmp/custom")
        self.assertEqual(path, "/tmp/custom/step-20")

        # 3. Test step 0
        self.assertTrue(self.manager._create_checkpoint_id(0).endswith("step-0"))


class TestFlattenedModelStates(CheckpointTestBase, unittest.TestCase):
    """
    Validates that the model wrapper is correctly unpacked into the top-level
    state dictionary, respecting that Checkpointer initializes ModelWrapper internally.
    """

    def setUp(self):
        super().setUp()

        self.MODEL_KEY = MODEL
        self.OPTIMIZER_KEY = OPTIMIZER

        # Manually swap the real ModelWrapper with a Mock.
        self.mock_model_wrapper = mock.MagicMock()
        self.mock_weights = {"layer.weight": torch.ones(1)}
        self.mock_model_wrapper.state_dict.return_value = self.mock_weights

        self.manager.states[self.MODEL_KEY] = self.mock_model_wrapper

    def test_flatten_with_internal_states(self):
        """Verify flattening using the manager's live states dictionary."""
        # Execute
        flattened = self.manager._flattened_model_states_sd()

        # 1. Ensure 'model' key is removed and replaced by its contents
        self.assertNotIn(self.MODEL_KEY, flattened)
        self.assertEqual(flattened["layer.weight"], self.mock_weights["layer.weight"])

        # 2. Ensure other keys injected during __init__ are preserved
        self.assertIn(self.OPTIMIZER_KEY, flattened)
        self.assertEqual(flattened["trainer"], torch.tensor([1.2347]))

        # 3. Verify the mock wrapper was actually used
        self.mock_model_wrapper.state_dict.assert_called_once()

    def test_flatten_with_external_override(self):
        """Verify flattening when an external dictionary is passed to the method."""
        alt_mock = mock.MagicMock()
        alt_weights = {"alt_param": torch.tensor([10.0])}
        alt_mock.state_dict.return_value = alt_weights

        external_sd = {self.MODEL_KEY: alt_mock, "external_only": True}

        # Execute
        flattened = self.manager._flattened_model_states_sd(state_dict=external_sd)

        # 1. Check weight presence
        self.assertEqual(flattened["alt_param"], 10.0)
        self.assertTrue(flattened["external_only"])

        # 2. Ensure it did not use the manager's internal states
        self.assertNotIn("trainer", flattened)
        self.assertNotIn(self.OPTIMIZER_KEY, flattened)

        # 3. Verify correct mock was called
        alt_mock.state_dict.assert_called_once()
        self.mock_model_wrapper.state_dict.assert_not_called()


class TestStatesToLoad(CheckpointTestBase, unittest.TestCase):
    """
    Validates the state selection logic, ensuring correct behavior for
    model-only loading and excluded components.
    """

    def setUp(self):
        super().setUp()

        self.MODEL_KEY = MODEL
        self.OPTIMIZER_KEY = OPTIMIZER

        # Setup a Mock ModelWrapper
        self.mock_model_weights = {"layer.weight": torch.ones(1)}
        self.mock_model_wrapper = mock.MagicMock()
        self.mock_model_wrapper.state_dict.return_value = self.mock_model_weights

        # Inject our mock into the manager's live state
        self.manager.states[self.MODEL_KEY] = self.mock_model_wrapper

    def test_states_to_load_model_only(self):
        """Verify that model_only=True returns only the model state_dict."""
        # This branch bypasses exclusion logic and flattening calls
        states = self.manager._states_to_load(model_only=True)

        self.assertEqual(states, self.mock_model_weights)
        self.assertNotIn("trainer", states)
        self.assertNotIn(self.OPTIMIZER_KEY, states)
        self.mock_model_wrapper.state_dict.assert_called_once()

    def test_states_to_load_with_exclusions(self):
        """Verify that excluded keys are stripped before flattening."""
        # Configure manager to exclude the optimizer
        self.manager.exclude_from_loading = [self.OPTIMIZER_KEY]

        self.manager._flattened_model_states_sd = mock.MagicMock()
        self.manager._flattened_model_states_sd.return_value = {"flattened": True}

        result = self.manager._states_to_load(model_only=False)

        # Verify the dict passed to flattening did not contain the optimizer
        passed_dict = self.manager._flattened_model_states_sd.call_args[0][0]
        self.assertNotIn(self.OPTIMIZER_KEY, passed_dict)
        self.assertIn("trainer", passed_dict)
        self.assertIn(self.MODEL_KEY, passed_dict)
        self.assertEqual(result, {"flattened": True})

    def test_states_to_load_invalid_exclude_key(self):
        """Verify that a ValueError is raised if an exclusion key is not in states."""
        self.manager.exclude_from_loading = ["non_existent_key"]

        with self.assertRaisesRegex(ValueError, "not found in state_dict"):
            self.manager._states_to_load(model_only=False)

    def test_states_to_load_full_integration(self):
        """Verify the full flow: exclude -> flatten -> return."""
        # No exclusions
        self.manager.exclude_from_loading = []

        states = self.manager._states_to_load(model_only=False)

        # Check that we have weights (from flattening) and metadata
        self.assertIn("layer.weight", states)
        self.assertIn("trainer", states)
        self.assertIn(self.OPTIMIZER_KEY, states)
        # Ensure the wrapper itself is gone (flattened)
        self.assertNotIn(self.MODEL_KEY, states)


class TestSaveLastStep(CheckpointTestBase, unittest.TestCase):
    """
    Validates final step checkpointing, specifically handling dtype conversion,
    model-only filtering, and HF format assertions.
    """

    def setUp(self):
        super().setUp()

        self.MODEL_KEY = MODEL

        # 1. Setup Mock ModelWrapper with float32 weights
        self.mock_weights = {"weight1": torch.tensor([42, 69], dtype=torch.float32)}
        self.mock_model_wrapper = mock.MagicMock()
        self.mock_model_wrapper.state_dict.return_value = self.mock_weights
        self.manager.states[self.MODEL_KEY] = self.mock_model_wrapper

        # 2. Mock internal dependencies
        self.manager.dcp_save = mock.MagicMock()
        self.manager._create_checkpoint_id = mock.MagicMock(return_value="final_step")
        self.manager._flattened_model_states_sd = mock.MagicMock(
            return_value={"full": "state"}
        )

    def test_save_last_step_full_checkpoint(self):
        """Verify full checkpoint save (no model-only filter, no dtype conversion)."""
        self.manager.last_save_model_only = False

        self.manager._save_last_step(curr_step=100)

        # Should call flattened_model_states_sd and not directly access model state_dict
        self.manager._flattened_model_states_sd.assert_called_once()
        self.manager.dcp_save.assert_called_once_with(
            {"full": "state"},
            checkpoint_id="final_step",
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=False,
        )

    def test_save_last_step_model_only_with_dtype_cast(self):
        """Verify model-only save with bfloat16 export conversion."""
        self.manager.last_save_model_only = True
        self.manager.export_dtype = torch.bfloat16

        self.manager._save_last_step(curr_step=100)

        # Capture the state dict passed to dcp_save
        args, kwargs = self.manager.dcp_save.call_args
        saved_states = args[0]

        # Verify weight precision was converted
        self.assertEqual(saved_states["weight1"].dtype, torch.bfloat16)
        self.assertEqual(saved_states["weight1"][0], 42)

        # Verify HF flag is passed correctly (default is False)
        self.assertFalse(kwargs["to_hf"])

    def test_save_last_step_hf_assertion(self):
        """Verify that saving in HF format requires last_save_model_only to be True."""
        self.manager.last_save_in_hf = True
        self.manager.last_save_model_only = False

        with self.assertRaisesRegex(AssertionError, "Only model can be saved"):
            self.manager._save_last_step(curr_step=100)

    def test_save_last_step_hf_success(self):
        """Verify successful HF save path."""
        self.manager.last_save_in_hf = True
        self.manager.last_save_model_only = True
        self.manager.export_dtype = torch.float32  # No conversion

        self.manager._save_last_step(curr_step=100)

        self.manager.dcp_save.assert_called_once_with(
            self.mock_weights,
            checkpoint_id="final_step",
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=True,
        )


class TestShouldSave(CheckpointTestBase, unittest.TestCase):
    """
    Validates the decision logic for triggering a checkpoint save.
    """

    def setUp(self):
        super().setUp()
        # Set a predictable interval for testing
        self.manager.interval = 100

    def test_should_save_disabled_or_load_only(self):
        """Verify that disabling or setting load_only overrides everything else."""
        self.manager.enable = False
        self.assertFalse(self.manager._should_save(curr_step=100))

        self.manager.enable = True
        self.manager.load_only = True
        self.assertFalse(self.manager._should_save(curr_step=100))

    def test_should_save_first_step(self):
        """Verify the logic for the initial step (step 1)."""
        self.manager.enable_first_step_checkpoint = True
        self.assertTrue(self.manager._should_save(curr_step=1))

        self.manager.enable_first_step_checkpoint = False
        self.assertFalse(self.manager._should_save(curr_step=1))

    def test_should_save_last_step(self):
        """Verify that the last_step flag always triggers a save if enabled."""
        self.manager.enable = True
        self.manager.load_only = False
        # Even if not at a multiple of the interval
        self.assertTrue(self.manager._should_save(curr_step=42, last_step=True))

    def test_should_save_interval(self):
        """Verify that saving triggers correctly at the specified step intervals."""
        # On interval
        self.assertTrue(self.manager._should_save(curr_step=100))
        self.assertTrue(self.manager._should_save(curr_step=200))

        # Off interval
        self.assertFalse(self.manager._should_save(curr_step=50))
        self.assertFalse(self.manager._should_save(curr_step=101))

    def test_should_save_default_false(self):
        """Verify the final fallback is False."""
        self.assertFalse(self.manager._should_save(curr_step=7))


class TestAsyncWait(CheckpointTestBase, unittest.TestCase):
    """
    Validates that the checkpointer correctly blocks until asynchronous
    storage operations are complete.
    """

    def setUp(self):
        super().setUp()
        self.mock_save_future = mock.MagicMock()

    def test_async_wait_standard(self):
        """Verify blocking and future clearing in standard ASYNC mode."""
        self.manager.async_mode = AsyncMode.ASYNC
        self.manager.save_future = self.mock_save_future

        self.manager._async_wait()

        self.mock_save_future.result.assert_called_once()
        self.assertIsNone(self.manager.save_future)

    def test_async_wait_pinned_mem(self):
        """Verify blocking in PINNED_MEM mode."""
        self.manager.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        self.manager.save_future = self.mock_save_future

        self.manager._async_wait()

        self.mock_save_future.result.assert_called_once()
        self.assertIsNotNone(self.manager.save_future)

    def test_async_wait_error_state(self):
        """Verify RuntimeError if a future exists while async is disabled."""
        self.manager.async_mode = AsyncMode.DISABLED
        self.manager.save_future = self.mock_save_future

        with self.assertRaisesRegex(RuntimeError, "self.save_future is not None"):
            self.manager._async_wait()

    def test_async_wait_no_op(self):
        """Verify no action is taken if there is no future."""
        self.manager.async_mode = AsyncMode.ASYNC
        self.manager.save_future = None

        # This should not raise any errors or call result()
        self.manager._async_wait()


@mock.patch("torchtitan.components.checkpoint.dist.get_rank")
@mock.patch("torchtitan.components.checkpoint.os.path.isdir")
class TestShouldPurge(CheckpointTestBase, unittest.TestCase):
    def test_should_purge_logic(self, mock_isdir, mock_rank):
        """Verify the three-way gate: keep_latest_k, rank 0, and directory existence."""
        # Case 1: All conditions met
        self.manager.keep_latest_k = 5
        mock_rank.return_value = 0
        mock_isdir.return_value = True
        self.assertTrue(self.manager._should_purge())

        # Case 2: keep_latest_k is 0 (disabled)
        self.manager.keep_latest_k = 0
        self.assertFalse(self.manager._should_purge())

        # Case 3: Not rank 0
        self.manager.keep_latest_k = 5
        mock_rank.return_value = 1
        self.assertFalse(self.manager._should_purge())

        # Case 4: Directory does not exist
        mock_rank.return_value = 0
        mock_isdir.return_value = False
        self.assertFalse(self.manager._should_purge())


@mock.patch("torchtitan.components.checkpoint.os.listdir")
@mock.patch("torchtitan.components.checkpoint.re.search")
class TestPurgeStaleCheckpoints(CheckpointTestBase, unittest.TestCase):
    """
    Validates that the purge logic correctly identifies the oldest checkpoints
    and enqueues them for deletion.
    """

    def setUp(self):
        super().setUp()

        # Mock the purge infrastructure
        self.manager.purge_queue = queue.Queue()
        self.manager.purge_thread = mock.MagicMock()

        # Mock _should_purge to always pass for these tests
        self.manager._should_purge = mock.MagicMock(return_value=True)

    def test_purge_logic_correct_subset(self, mock_re_search, mock_listdir):
        """Verify that only the oldest checkpoints are enqueued when keep_latest_k=2."""
        self.manager.keep_latest_k = 2

        # Simulate a folder with 4 checkpoints (unsorted in the listdir)
        mock_listdir.return_value = ["step-10", "step-30", "step-20", "step-40"]

        # Configure regex mock to simulate step extraction
        def re_side_effect(pattern, string):
            match = mock.MagicMock()
            match.group.return_value = string.split("-")[1]
            return match

        mock_re_search.side_effect = re_side_effect

        self.manager._purge_stale_checkpoints()

        # With 4 total and keep_latest_k=2, we expect 2 to be deleted (the oldest ones)
        self.assertEqual(self.manager.purge_queue.qsize(), 2)

        # Checkpoints are sorted as (10, 20, 30, 40).
        # [: -2] gives [10, 20].
        enqueued_paths = []
        while not self.manager.purge_queue.empty():
            enqueued_paths.append(self.manager.purge_queue.get())

        self.assertTrue(any("step-10" in p for p in enqueued_paths))
        self.assertTrue(any("step-20" in p for p in enqueued_paths))
        self.assertFalse(any("step-30" in p for p in enqueued_paths))
        self.assertFalse(any("step-40" in p for p in enqueued_paths))

    def test_purge_skips_when_below_threshold(self, mock_re_search, mock_listdir):
        """If we have fewer checkpoints than keep_latest_k, nothing should be enqueued."""
        self.manager.keep_latest_k = 5
        mock_listdir.return_value = ["step-10", "step-20"]

        self.manager._purge_stale_checkpoints()

        self.assertEqual(self.manager.purge_queue.qsize(), 0)

    def test_purge_assertion_error_no_thread(self, mock_re_search, mock_listdir):
        """Verify that an assertion error is raised if the background thread is missing."""
        self.manager.keep_latest_k = 1
        mock_listdir.return_value = ["step-1", "step-2"]
        self.manager.purge_thread = None  # Missing thread

        with self.assertRaises(AssertionError):
            self.manager._purge_stale_checkpoints()


if __name__ == "__main__":
    unittest.main()
