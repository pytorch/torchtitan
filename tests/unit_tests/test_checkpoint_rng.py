# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import torch
import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager, TRAIN_STATE


def fake_dcp_save(state, checkpoint_id):
    """Mock implementation of dcp.save that saves the state to a file."""
    state_dict = {}
    for k, v in state.items():
        if hasattr(v, "state_dict"):
            state_dict[k] = v.state_dict()
        else:
            state_dict[k] = v
    os.makedirs(checkpoint_id, exist_ok=True)
    torch.save(state_dict, os.path.join(checkpoint_id, "state.pt"))


def fake_dcp_load(state, checkpoint_id):
    """Mock implementation of dcp.load that loads the state from a file."""
    state_file = os.path.join(checkpoint_id, "state.pt")
    if not os.path.exists(state_file):
        return False

    loaded_state = torch.load(state_file)
    for k, v in loaded_state.items():
        if k in state:
            if hasattr(state[k], "load_state_dict"):
                state[k].load_state_dict(v)
            else:
                state[k] = v
    return True


class MockTrainState:
    """Mock implementation of the train state with RNG state handling."""

    def __init__(self):
        # Initialize with random RNG states
        random_bytes_cpu = np.frombuffer(np.random.bytes(100), dtype=np.uint8)
        random_bytes_device = np.frombuffer(np.random.bytes(100), dtype=np.uint8)
        self.cpu_rng_state = torch.ByteTensor(random_bytes_cpu)
        self.device_rng_state = torch.ByteTensor(random_bytes_device)

    def state_dict(self):
        return {
            "cpu_rng_states": self.cpu_rng_state,
            "device_rng_states": self.device_rng_state,
            "other_state": torch.tensor([1, 2, 3]),
        }

    def load_state_dict(self, state_dict):
        self.cpu_rng_state = state_dict["cpu_rng_states"]
        self.device_rng_state = state_dict["device_rng_states"]


class TestCheckpointRNGState(unittest.TestCase):
    """Test cases for saving and loading RNG state in CheckpointManager."""

    def setUp(self):
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        # Create mock objects needed for CheckpointManager
        self.dataloader = mock.Mock()
        self.dataloader.state_dict = mock.Mock(return_value={"dataloader": 1})

        self.model_parts = [SimpleModel()]

        self.optimizers = mock.Mock()
        self.optimizers.state_dict = mock.Mock(return_value={"optimizer": 3})

        self.lr_schedulers = mock.Mock()
        self.lr_schedulers.state_dict = mock.Mock(return_value={"lr_scheduler": 4})

        # Create a mock train state with RNG state
        self.train_state = MockTrainState()

        # Create a job config
        self.job_config = mock.Mock()
        self.job_config.checkpoint.enable_checkpoint = True
        self.job_config.checkpoint.folder = "checkpoints"
        self.job_config.checkpoint.interval = 10
        self.job_config.checkpoint.async_mode = "disabled"
        self.job_config.checkpoint.keep_latest_k = 0
        self.job_config.checkpoint.model_weights_only = False
        self.job_config.checkpoint.export_dtype = "float32"
        self.job_config.checkpoint.exclude_from_loading = []

        self.job_config.job.dump_folder = self.temp_dir

        self.job_config.fault_tolerance.replica_id = 0

        # Create a mock FT manager
        self.ft_manager = mock.Mock()
        self.ft_manager.enabled = False

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    @mock.patch("torchtitan.components.checkpoint.dcp.save", side_effect=fake_dcp_save)
    @mock.patch("torchtitan.components.checkpoint.dcp.load", side_effect=fake_dcp_load)
    def test_save_load_rng_state(self, mock_load, mock_save):
        """Test that RNG state is correctly saved and loaded."""
        # Create a CheckpointManager with our mock objects
        manager = CheckpointManager(
            self.dataloader,
            self.model_parts,
            self.optimizers,
            self.lr_schedulers,
            {TRAIN_STATE: self.train_state},
            self.job_config,
            self.ft_manager,
        )

        # Save the initial RNG state
        step = 10
        initial_cpu_rng_state = self.train_state.cpu_rng_state.clone()
        initial_device_rng_state = self.train_state.device_rng_state.clone()

        # Save a checkpoint
        manager.save(curr_step=step, force=True)

        # Verify that save was called
        mock_save.assert_called_once()

        # Change the RNG state
        random_bytes_cpu = np.frombuffer(np.random.bytes(100), dtype=np.uint8)
        random_bytes_device = np.frombuffer(np.random.bytes(100), dtype=np.uint8)
        self.train_state.cpu_rng_state = torch.ByteTensor(random_bytes_cpu)
        self.train_state.device_rng_state = torch.ByteTensor(random_bytes_device)

        # Verify that the RNG state has changed
        self.assertFalse(
            torch.all(self.train_state.cpu_rng_state.eq(initial_cpu_rng_state))
        )
        self.assertFalse(
            torch.all(self.train_state.device_rng_state.eq(initial_device_rng_state))
        )

        # Load the checkpoint
        success = manager.load(step=step)

        # Verify that load was called and succeeded
        mock_load.assert_called_once()
        self.assertTrue(success)

        # Verify that the RNG state has been restored
        self.assertTrue(
            torch.all(self.train_state.cpu_rng_state.eq(initial_cpu_rng_state))
        )
        self.assertTrue(
            torch.all(self.train_state.device_rng_state.eq(initial_device_rng_state))
        )

    @mock.patch("torchtitan.components.checkpoint.dcp.save", side_effect=fake_dcp_save)
    def test_save_rng_state_multiple_steps(self, mock_save):
        """Test that RNG state is correctly saved at different steps."""
        # Create a CheckpointManager with our mock objects
        manager = CheckpointManager(
            self.dataloader,
            self.model_parts,
            self.optimizers,
            self.lr_schedulers,
            {TRAIN_STATE: self.train_state},
            self.job_config,
            self.ft_manager,
        )

        # Save RNG state at step 10
        step1 = 10
        rng_state1_cpu = self.train_state.cpu_rng_state.clone()
        rng_state1_device = self.train_state.device_rng_state.clone()
        manager.save(curr_step=step1, force=True)

        # Change RNG state and save at step 20
        random_bytes_cpu = np.frombuffer(np.random.bytes(100), dtype=np.uint8)
        random_bytes_device = np.frombuffer(np.random.bytes(100), dtype=np.uint8)
        self.train_state.cpu_rng_state = torch.ByteTensor(random_bytes_cpu)
        self.train_state.device_rng_state = torch.ByteTensor(random_bytes_device)
        step2 = 20
        rng_state2_cpu = self.train_state.cpu_rng_state.clone()
        rng_state2_device = self.train_state.device_rng_state.clone()
        manager.save(curr_step=step2, force=True)

        # Verify that save was called twice
        self.assertEqual(mock_save.call_count, 2)

        # Load checkpoint from step 10
        with mock.patch(
            "torchtitan.components.checkpoint.dcp.load", side_effect=fake_dcp_load
        ):
            manager.load(step=step1)

            # Verify that the RNG state from step 10 is restored
            self.assertTrue(
                torch.all(self.train_state.cpu_rng_state.eq(rng_state1_cpu))
            )
            self.assertTrue(
                torch.all(self.train_state.device_rng_state.eq(rng_state1_device))
            )

            # Load checkpoint from step 20
            manager.load(step=step2)

            # Verify that the RNG state from step 20 is restored
            self.assertTrue(
                torch.all(self.train_state.cpu_rng_state.eq(rng_state2_cpu))
            )
            self.assertTrue(
                torch.all(self.train_state.device_rng_state.eq(rng_state2_device))
            )


if __name__ == "__main__":
    unittest.main()
