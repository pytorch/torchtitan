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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from unittest import mock

import torch

from torchtitan.components.checkpoint import CheckpointManager


def fake_dcp_save(state, checkpoint_id):
    state = {k: v.state_dict() for k, v in state.items()}
    os.makedirs(checkpoint_id, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_id, "state.pt"))


def fake_dcp_load(state, checkpoint_id):
    state["trainer"].dcp_load_is_called = 7312


def fake_async_save(state, checkpoint_id, process_group):
    def run_save():
        fake_dcp_save(state, checkpoint_id)

    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(run_save)

    mock_future = mock.Mock()
    mock_future.result = mock.Mock(side_effect=f.result)
    return mock_future


def fake_get_model_state_dict(model, *args, **kwargs):
    return model.state_dict()


@dataclass
class DummyCheckpointConfig:
    enable_checkpoint: bool = True
    folder: str = "dummy_folder"
    interval: int = 10
    async_mode: str = "disabled"
    keep_latest_k: int = 0
    model_weights_only: bool = False
    export_dtype: str = "float32"
    exclude_from_loading = []


@dataclass
class DummyJob:
    dump_folder: str = "dummy_folder"


@dataclass
class DummyFaultTolerance:
    replica_id = 0
    group_size = 1


@dataclass
class DummyJobConfig:
    checkpoint: DummyCheckpointConfig = field(default_factory=DummyCheckpointConfig)
    job: DummyJob = field(default_factory=DummyJob)
    fault_tolerance: DummyFaultTolerance = field(default_factory=DummyFaultTolerance)
    ft_manager = None


# Dummy instances to supply as constructor arguments.
dummy_dataloader = mock.Mock()
dummy_dataloader.state_dict = mock.Mock(side_effect=lambda: {"dataloader": 1})
dummy_model_parts = [mock.Mock()]
dummy_model_parts[0].state_dict = mock.Mock(side_effect=lambda: {"model": 2})
dummy_optimizers = mock.Mock()
dummy_optimizers.state_dict = mock.Mock(side_effect=lambda: {"optimizer": 3})
dummy_lr_schedulers = mock.Mock()
dummy_lr_schedulers.state_dict = mock.Mock(side_effect=lambda: {"lr_scheduler": 4})


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.dummy_job = DummyJob(dump_folder=self.temp_dir)
        self.job_config = DummyJobConfig(job=self.dummy_job)
        self.checkpoint_folder = os.path.join(
            self.dummy_job.dump_folder, self.job_config.checkpoint.folder
        )
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        self.trainer_state = mock.Mock()
        self.trainer_state.state_dict = mock.Mock(side_effect=lambda: {"my_state": 765})

    def tearDown(self):
        # Remove the temporary directory after each test.
        shutil.rmtree(self.temp_dir)

    @mock.patch(
        "torchtitan.components.checkpoint.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    )
    @mock.patch("torchtitan.components.checkpoint.dcp.save", side_effect=fake_dcp_save)
    def test_save(self, *_):
        """Test that calling save() writes a checkpoint file to disk."""
        job_config = DummyJobConfig(job=self.dummy_job)
        ft_manager = mock.Mock()
        ft_manager.enabled = False
        manager = CheckpointManager(
            dummy_dataloader,
            dummy_model_parts,
            dummy_optimizers,
            dummy_lr_schedulers,
            {"trainer": self.trainer_state},
            job_config,
            ft_manager,
        )
        step = 20
        manager.save(curr_step=step, force=True)
        state_file = self._checkpoint_id(step)
        self.assertTrue(
            os.path.exists(state_file), "The checkpoint file was not created on disk."
        )
        loaded_state = torch.load(state_file, weights_only=False)
        self.assertEqual(
            loaded_state["trainer"]["my_state"],
            765,
            "Saved state does not match expected value.",
        )

    @mock.patch(
        "torchtitan.components.checkpoint.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    )
    @mock.patch("torchtitan.components.checkpoint.dcp.load", side_effect=fake_dcp_load)
    @mock.patch("torchtitan.components.checkpoint.dcp.save", side_effect=fake_dcp_save)
    def test_load(self, *_):
        """Test that load() properly reads the checkpoint file from disk and restores state."""
        job_config = DummyJobConfig(job=self.dummy_job)
        ft_manager = mock.Mock()
        ft_manager.enabled = False
        manager = CheckpointManager(
            dummy_dataloader,
            dummy_model_parts,
            dummy_optimizers,
            dummy_lr_schedulers,
            {"trainer": self.trainer_state},
            job_config,
            ft_manager,
        )
        step = 30
        manager.save(curr_step=step, force=True)
        # Simulate a state change.
        manager.states["test"] = 999
        success = manager.load(step=step)
        self.assertTrue(
            success,
            "The load() method should have returned True for an existing checkpoint.",
        )
        self.assertTrue(hasattr(manager.states["trainer"], "dcp_load_is_called"))

        self.assertEqual(
            manager.states["trainer"].dcp_load_is_called,
            7312,
            "The state was not correctly restored after loading.",
        )

    @mock.patch("torchtitan.components.checkpoint.dist.get_rank", return_value=0)
    @mock.patch(
        "torchtitan.components.checkpoint.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    )
    @mock.patch("torchtitan.components.checkpoint.dcp.save", side_effect=fake_dcp_save)
    def test_purge_stale_checkpoints_rank_zero(self, *_):
        """
        Test that when keep_latest_k is 3 and dist.get_rank() returns 0, stale checkpoints
        are purged by placing the correct paths into the purge queue.
        """
        job_config = DummyJobConfig(job=self.dummy_job)
        job_config.checkpoint.keep_latest_k = 3
        ft_manager = mock.Mock()
        ft_manager.enabled = False
        manager = CheckpointManager(
            dummy_dataloader,
            dummy_model_parts,
            dummy_optimizers,
            dummy_lr_schedulers,
            {"trainer": self.trainer_state},
            job_config,
            ft_manager,
        )
        steps = [10, 20, 30, 40, 50]
        for s in steps:
            manager.save(curr_step=s, force=False)
        while not manager.purge_queue.empty():
            time.sleep(1)
        time.sleep(1)
        os.sync()
        expected_paths = [
            os.path.join(self.checkpoint_folder, "step-30"),
            os.path.join(self.checkpoint_folder, "step-40"),
            os.path.join(self.checkpoint_folder, "step-50"),
        ]
        for step in [10, 20]:
            self.assertFalse(
                os.path.exists(self._checkpoint_id(step)),
                "The checkpoint is not purged.",
            )

        for step in [30, 40, 50]:
            self.assertTrue(
                os.path.exists(self._checkpoint_id(step)), "The checkpointis purged."
            )

    @mock.patch("torchtitan.components.checkpoint.dist.get_rank", return_value=1)
    @mock.patch(
        "torchtitan.components.checkpoint.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    )
    @mock.patch("torchtitan.components.checkpoint.dcp.save", side_effect=fake_dcp_save)
    def test_purge_stale_checkpoints_rank_nonzero(self, *_):
        """
        Test that when dist.get_rank() returns a non-zero value, the purge logic does not
        place any paths in the purge queue.
        """
        job_config = DummyJobConfig(job=self.dummy_job)
        job_config.checkpoint.keep_latest_k = 3
        ft_manager = mock.Mock()
        ft_manager.enabled = False
        manager = CheckpointManager(
            dummy_dataloader,
            dummy_model_parts,
            dummy_optimizers,
            dummy_lr_schedulers,
            {"trainer": self.trainer_state},
            job_config,
            ft_manager,
        )
        steps = [10, 20, 30, 40, 50]
        for s in steps:
            manager.save(curr_step=s, force=False)
        while not manager.purge_queue.empty():
            time.sleep(1)
        time.sleep(1)
        os.sync()

        for step in [10, 20, 30, 40, 50]:
            self.assertTrue(
                os.path.exists(self._checkpoint_id(step)), "The checkpointis purged."
            )

    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    )
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_async_save_calls_async_wait(self, *_):
        """
        Test that in async mode (AsyncMode.ASYNC), calling save() twice correctly waits
        on the previous async future via _async_wait().
        """
        # Set async_mode to "async" in the job configuration.
        job_config = DummyJobConfig(job=self.dummy_job)
        job_config.checkpoint.async_mode = "async"
        ft_manager = mock.Mock()
        ft_manager.enabled = False
        manager = CheckpointManager(
            dummy_dataloader,
            dummy_model_parts,
            dummy_optimizers,
            dummy_lr_schedulers,
            {"trainer": self.trainer_state},
            job_config,
            ft_manager,
        )
        # First save: should schedule an async save.
        manager.save(curr_step=10, force=False)
        f = manager.async_future
        f.result.assert_not_called()
        manager.save(curr_step=20, force=False)
        f.result.assert_called_once()
        f = manager.async_future
        f.result.assert_not_called()

    @mock.patch("torch.cuda.Stream")
    @mock.patch("torchtitan.components.checkpoint.dist.new_group")
    @mock.patch(
        "torchtitan.components.checkpoint.get_model_state_dict",
        side_effect=fake_get_model_state_dict,
    )
    @mock.patch(
        "torchtitan.components.checkpoint.dcp.async_save", side_effect=fake_async_save
    )
    def test_ft_async_save_calls_async_wait(self, *_):
        """
        Test that in async mode (AsyncMode.ASYNC), calling save() twice correctly waits
        on the previous async future via _async_wait().
        """
        # Set async_mode to "async" in the job configuration.
        job_config = DummyJobConfig(job=self.dummy_job)
        job_config.checkpoint.async_mode = "disabled"
        ft_manager = mock.Mock()
        ft_manager.enabled = True
        manager = CheckpointManager(
            dummy_dataloader,
            dummy_model_parts,
            dummy_optimizers,
            dummy_lr_schedulers,
            {"trainer": self.trainer_state},
            job_config,
            ft_manager,
        )
        # First save: should schedule an async save.
        self.assertTrue(manager.async_future is None)
        manager.save(curr_step=5, force=False)
        self.assertTrue(manager.async_future is not None)
        manager.async_future.result.assert_not_called()

        # Keep the previous future as it will be waited and replaced.
        async_future = manager.async_future
        manager.save(curr_step=6, force=False)
        async_future.result.assert_called_once()
        self.assertTrue(manager.async_future is not None)
        manager.async_future.result.assert_not_called()

    def _checkpoint_id(self, step):
        checkpoint_id = os.path.join(self.checkpoint_folder, f"step-{step}")
        state_file = os.path.join(checkpoint_id, "state.pt")
        return state_file


if __name__ == "__main__":
    unittest.main()
