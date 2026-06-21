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
import uuid
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
from torch.utils.data import DataLoader
from torchtitan.components import fs
from torchtitan.components.checkpoint import CheckpointManager, MODEL, ModelWrapper


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
    def __new__(cls):
        # Return a Mock that mimics Future instead of an instance of this class
        # That allows isinstance(DummyFuture, Future) to pass
        instance = mock.Mock(spec=Future)

        # Add a custom attribute to the mock the done state
        instance.finished = False

        # When result() is called, it flips the finished flag
        def side_effect_result(*args, **kwargs):
            instance.finished = True
            return None

        instance.done.side_effect = lambda: instance.finished
        instance.result.side_effect = side_effect_result
        instance.result.return_value = None

        return instance


class DummyAsyncResult(AsyncSaveResponse):
    """Mock object that mimics the return value of dcp.async_save with pinned memory"""

    def __init__(self):
        self.upload_completion = DummyFuture()
        self.staging_completion = DummyFuture()


def fake_async_save(*args, **kwargs):
    # Check if this is async_with_pinned_mem mode by looking for async_stager parameter
    if kwargs.get("async_stager"):
        return DummyAsyncResult()
    else:
        return DummyFuture()


class DummyTrainerConfig:
    def __init__(self, dump_folder):
        self.dump_folder = dump_folder
        self.checkpoint = CheckpointManager.Config(
            enable=True,
            async_mode="disabled",
            folder="test_folder",
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
        )


class TestCheckpointManager(unittest.TestCase):
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
            folder=self.test_folder,
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

        # Patch process group creation
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()

    def tearDown(self):
        self.patcher_group.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    def fake_save(self, state_dict: dict, checkpoint_id: str, storage_writer=None):
        os.makedirs(checkpoint_id, exist_ok=True)
        sd_to_save = {}
        for key, val in state_dict.items():
            if hasattr(val, "state_dict"):
                sd_to_save[key] = val.state_dict()
            elif isinstance(val, torch.Tensor):
                sd_to_save[key] = val
        torch.save(sd_to_save, os.path.join(checkpoint_id, "state_dict.pt"))

    def fake_load(self, states: dict, checkpoint_id=None, storage_reader=None):
        path = os.path.join(checkpoint_id, "state_dict.pt")
        loaded = torch.load(path, weights_only="False")
        for key, val in loaded.items():
            if key in states and hasattr(states[key], "load_state_dict"):
                states[key].load_state_dict(val)
            elif key in states and isinstance(states[key], torch.Tensor):
                states[key].copy_(val)

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

    def test_dcp_save_load_memory_fs_checkpoint(self):
        root = f"memory://torchtitan-checkpoint-test/{uuid.uuid4()}"
        cfg = CheckpointManager.Config(
            enable=True,
            async_mode="disabled",
            folder=root,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
        )
        manager = CheckpointManager(
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            config=cfg,
            sd_adapter=None,
            base_folder="",
        )

        try:
            checkpoint_id = manager._create_checkpoint_id(1)
            expected = torch.tensor([3.0])
            manager.dcp_save(
                {"trainer": expected},
                checkpoint_id=checkpoint_id,
                async_mode=manager.async_mode,
            )

            actual = {"trainer": torch.zeros_like(expected)}
            manager.dcp_load(
                actual,
                checkpoint_id=checkpoint_id,
                from_hf=False,
                from_quantized=False,
            )

            torch.testing.assert_close(actual["trainer"], expected)
            self.assertTrue(manager._checkpoint_exists(checkpoint_id, from_hf=False))
            self.assertEqual(manager._find_load_step(), 1)
        finally:
            manager.close()
            fs.rm(root, recursive=True)

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

    @mock.patch("torchtitan.components.checkpoint.logger")
    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torch.cuda.Stream")
    @mock.patch("torchtitan.components.checkpoint.DefaultStager")
    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_async_save_with_pinned_mem_assigns_staging_future(
        self,
        mock_async_save,
        mock_new_group,
        mock_default_stager,
        mock_cuda_stream,
        mock_rank,
        mock_logger,
    ):
        """
        Test that AsyncMode.ASYNC_WITH_PINNED_MEM correctly assigns the staging future.

        This test verifies that when using ASYNC_WITH_PINNED_MEM mode, the
        staging_future is properly captured from the save response. This handle
        is critical for ensuring that subsequent operations wait for the
        GPU-to-CPU transfer to complete before proceeding.
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

        # Initially staging_future should be None
        self.assertIsNone(manager.staging_future)

        manager.save(curr_step=1, last_step=False)
        # After save, staging_future shouldn't be None ...
        self.assertIsNotNone(manager.staging_future)
        # ... and staging should be running
        self.assertFalse(manager.staging_future.done())

        # Verify that `maybe_wait_for_staging` actually waits for staging future to complete
        staging_future = manager.staging_future
        manager.maybe_wait_for_staging()
        staging_future.result.assert_called_once()

        # After waiting, the staging future should be None
        self.assertIsNone(manager.staging_future)

        manager.close()

    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_async_save_calls_maybe_wait_for_saving(
        self, mock_async_save, mock_new_group
    ):
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

        def fake_load(state_dict: dict, checkpoint_id=None, storage_reader=None):
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

    def test_maybe_wait_for_staging_when_checkpoint_disabled(self):
        """Verify that calling maybe_wait_for_staging succeeds without errors when the manager is disabled."""

        config = CheckpointManager.Config(enable=False)
        manager = CheckpointManager(
            config=config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        manager.maybe_wait_for_staging()

    def test_maybe_wait_for_saving_when_checkpoint_disabled(self):
        """Verify that calling maybe_wait_for_saving succeeds without errors when the manager is disabled."""

        config = CheckpointManager.Config(enable=False)
        manager = CheckpointManager(
            config=config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.trainer_config.dump_folder,
        )

        manager.maybe_wait_for_saving()


class TestConfigPostInit(unittest.TestCase):
    def test_valid_default_config(self):
        """Verify that default values pass initialization."""
        try:
            CheckpointManager.Config()
        except Exception as e:
            self.fail(f"Default Config raised {type(e).__name__} unexpectedly!")

    def test_sanity_and_range_checks(self):
        """Test basic field validation like empty strings and negative numbers."""
        # Folder cannot be empty
        with self.assertRaisesRegex(ValueError, "folder.*cannot be empty"):
            CheckpointManager.Config(folder="   ")

        # Interval must be >= 1
        with self.assertRaisesRegex(ValueError, "interval.*at least 1"):
            CheckpointManager.Config(interval=0)

        # keep_latest_k range checks
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            CheckpointManager.Config(keep_latest_k=-1)
        with self.assertRaisesRegex(ValueError, "at least 2 checkpoint replicas"):
            CheckpointManager.Config(keep_latest_k=1)

        with self.assertRaisesRegex(
            ValueError, f"{MODEL} key shouldn't be in exclude_from_loading."
        ):
            CheckpointManager.Config(exclude_from_loading=[MODEL])

    def test_path_normalization(self):
        """Test that paths are stripped and must be absolute."""
        # Test leading/trailing whitespace stripping
        cfg = CheckpointManager.Config(initial_load_path="  /absolute/path/step-100  ")
        self.assertEqual(cfg.initial_load_path, "/absolute/path/step-100")

        # Test relative path rejection
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            CheckpointManager.Config(initial_load_path="relative/path/step-100")

    def test_dependency_assertions(self):
        """Test logic where one field requires another to be set."""
        # HF load needs model_only=True; initial_load_path stays optional.
        with self.assertRaisesRegex(ValueError, "requires initial_load_model_only"):
            CheckpointManager.Config(
                initial_load_in_hf=True, initial_load_model_only=False
            )
        CheckpointManager.Config(initial_load_in_hf=True, initial_load_path=None)

        # HF quantized requires HF enabled
        with self.assertRaisesRegex(ValueError, "requires initial_load_in_hf"):
            CheckpointManager.Config(
                initial_load_in_hf_quantized=True,
                initial_load_in_hf=False,
                initial_load_path="/path/step-1",
            )

        # HF last save requires model_only
        with self.assertRaisesRegex(ValueError, "requires last_save_model_only=True"):
            CheckpointManager.Config(last_save_in_hf=True, last_save_model_only=False)

    def test_mode_normalization(self):
        """Test that async_mode is case-normalized."""
        cfg = CheckpointManager.Config(async_mode="ASYNC")
        self.assertEqual(cfg.async_mode, "async")

        with self.assertRaisesRegex(ValueError, "Invalid async_mode"):
            CheckpointManager.Config(async_mode="invalid_mode")

    @mock.patch("torchtitan.components.checkpoint.logger")
    def test_warnings(self, mock_logger):
        """Test that logical redundancies trigger warnings but don't crash."""

        # Redundant load_only vs first_step
        CheckpointManager.Config(load_only=True, enable_first_step_checkpoint=True)
        mock_logger.warning.assert_any_call(
            "checkpoint.load_only is True; enable_first_step_checkpoint will be ignored."
        )

        # model_only=True without a path
        CheckpointManager.Config(initial_load_model_only=True, initial_load_path=None)
        mock_logger.warning.assert_any_call(
            "initial_load_model_only=True has no effect without an initial_load_path."
        )


class TestModelWrapper(unittest.TestCase):
    """ModelWrapper.state_dict() must keep stable tensor storage across calls so
    the async pinned-memory stager (keyed by source storage) reuses its host
    buffers, while still reflecting current parameter values -- including for
    modules whose state_dict hooks emit freshly allocated tensors. CPU-only:
    checks storage identity and values, no actual staging.
    """

    def test_plain_param_cached_and_reflects_updates(self):
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4))

        model = Plain()
        wrapper = ModelWrapper(model)

        sd1 = wrapper.state_dict()
        ptr = sd1["w"].untyped_storage().data_ptr()

        with torch.no_grad():
            model.w.fill_(2.0)

        sd2 = wrapper.state_dict()
        # Same dict and same storage; the cached view shares the parameter's
        # storage, so the update shows through with no copy.
        self.assertIs(sd2, sd1)
        self.assertEqual(sd2["w"].untyped_storage().data_ptr(), ptr)
        self.assertTrue(torch.all(sd2["w"] == 2.0))

    def test_hook_tensor_storage_stable_and_refreshed(self):
        class HookedModule(nn.Module):
            # Slice a non-leading dim so .contiguous() allocates new storage
            # disconnected from the parameter, mirroring FusedSwiGLU's split.
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(4, 2, 3))
                self.register_state_dict_post_hook(self._split)

            @staticmethod
            def _split(module, state_dict, prefix, local_metadata):
                w = state_dict.pop(f"{prefix}w")
                state_dict[f"{prefix}a"] = w[:, 0].contiguous()
                state_dict[f"{prefix}b"] = w[:, 1].contiguous()

        model = HookedModule()
        wrapper = ModelWrapper(model)

        sd1 = wrapper.state_dict()
        self.assertIn("a", sd1)
        self.assertNotIn("w", sd1)
        ptr_a = sd1["a"].untyped_storage().data_ptr()
        self.assertTrue(torch.all(sd1["a"] == 0.0))

        with torch.no_grad():
            model.w.fill_(1.0)

        sd2 = wrapper.state_dict()
        # The storage object is reused (pinned staging buffers stay valid) ...
        self.assertEqual(sd2["a"].untyped_storage().data_ptr(), ptr_a)
        # ... and the in-place refresh picked up the updated parameter.
        self.assertTrue(torch.all(sd2["a"] == 1.0))


if __name__ == "__main__":
    unittest.main()
