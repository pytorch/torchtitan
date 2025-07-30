# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
from torch.optim import Adam

from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ConfigManager


class TestLRScheduler(unittest.TestCase):
    def setUp(self):
        # Create a simple model with parameters
        self.model = torch.nn.Linear(10, 10)
        # Create an optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.1)

        # We don't actually call `optimizer.step()` which will cause a warning
        # from PyTorch. Avoid the warnings that may confuse people.
        self.optimizer._opt_called = True

        # Create an optimizer container
        self.optimizer_container = MagicMock(spec=OptimizersContainer)
        self.optimizer_container.__iter__.return_value = iter([self.optimizer])
        self.optimizer_container.__len__.return_value = 1

    def create_job_config(
        self,
        training_steps=10,
        warmup_steps=None,
        decay_ratio=None,
        decay_type=None,
        min_lr_factor=None,
    ):
        # Create a job config with the specified parameters
        args = [
            "--training.steps",
            str(training_steps),
        ]

        args += (
            ["--lr_scheduler.warmup_steps", str(warmup_steps)]
            if warmup_steps is not None
            else []
        )
        args += (
            ["--lr_scheduler.decay_ratio", str(decay_ratio)]
            if decay_ratio is not None
            else []
        )
        args += (
            ["--lr_scheduler.decay_type", decay_type] if decay_type is not None else []
        )
        args += (
            ["--lr_scheduler.min_lr_factor", str(min_lr_factor)]
            if min_lr_factor is not None
            else []
        )

        config_manager = ConfigManager()
        # Create base config with parameters passed directly
        config = config_manager.parse_args(args)

        return config

    def test_linear_warmup_decay(self):
        """Test the linear warmup followed by linear decay schedule."""
        # Create a job config with 10 steps, 2 warmup steps, and linear decay
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,  # Use default decay: start decay immediately
            decay_type=None,
            min_lr_factor=None,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(
            self.optimizer_container, config.lr_scheduler, config.training.steps
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.5,  # Step 0: 50% of max LR (warmup)
            1.0,  # Step 1: 100% of max LR (warmup complete)
            1.0,  # Step 2: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            7.0 / 8.0,  # Step 3: 7/8 of max LR
            6.0 / 8.0,  # Step 4: 3/4 of max LR
            5.0 / 8.0,  # Step 5: 5/8 of max LR
            4.0 / 8.0,  # Step 6: 1/2 of max LR
            3.0 / 8.0,  # Step 7: 3/8 of max LR
            2.0 / 8.0,  # Step 8: 1/4 of max LR
            1.0 / 8.0,  # Step 9: 1/8 of max LR
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            # The LambdaLR multiplies the base lr by the factor
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_warmup_stable_decay(self):
        """Test warmup followed by stable phase and then decay."""
        # Create a job config with 10 steps, 2 warmup steps, 3 stable steps, and 5 decay steps
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=0.5,  # 50% of steps for decay
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(
            self.optimizer_container, config.lr_scheduler, config.training.steps
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.5,  # Step 0: 50% of max LR (warmup)
            1.0,  # Step 1: 100% of max LR (warmup complete)
            1.0,  # Step 2: Stable phase
            1.0,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
            1.0,  # Step 5: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            0.8,  # Step 6: Linear decay starts (80% of max LR)
            0.6,  # Step 7: 60% of max LR
            0.4,  # Step 8: 40% of max LR
            0.2,  # Step 9: 20% of max LR
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_min_lr(self):
        """Test that the learning rate doesn't go below the minimum."""
        # Create a job config with a minimum learning rate
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,
            decay_type="linear",
            min_lr_factor=0.2,  # 20% of base LR as minimum
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(
            self.optimizer_container, config.lr_scheduler, config.training.steps
        )

        # Step through all steps
        for _ in range(10):
            lr_scheduler.step()

        # After all steps, LR should be at minimum (0.1 * 0.2 = 0.02)
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.02, places=6)

    def test_warmup_exceeds_training(self):
        """Test when warmup steps exceed training steps."""
        # Create a job config where warmup steps > training steps
        config = self.create_job_config(
            training_steps=5,
            warmup_steps=10,  # More than training steps
            decay_ratio=None,
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler - should adjust warmup steps
        lr_scheduler = build_lr_schedulers(
            self.optimizer_container, config.lr_scheduler, config.training.steps
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.2,  # Step 0: 50% of max LR (warmup)
            0.4,  # Step 1: 100% of max LR (warmup complete)
            0.6,  # Step 2: Stable phase
            0.8,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_warmup_stable_only(self):
        """Test warmup followed by stable phase only, with no decay phase."""
        # Create a job config with 10 steps, 2 warmup steps, and no decay phase
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=0.0,  # 0% of steps for decay (no decay)
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(
            self.optimizer_container, config.lr_scheduler, config.training.steps
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.5,  # Step 0: 50% of max LR (warmup)
            1.0,  # Step 1: 100% of max LR (warmup complete)
            1.0,  # Step 2: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            1.0,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
            1.0,  # Step 5: Stable phase
            1.0,  # Step 6: Stable phase
            1.0,  # Step 7: Stable phase
            1.0,  # Step 8: Stable phase
            1.0,  # Step 9: Stable phase
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_warmup_plus_decay_exceeds_training(self):
        """Test when warmup + decay steps exceed training steps."""
        # Create a job config where warmup + decay steps > training steps
        # Expected behaviro: warmup steps = 5, decay steps = 5
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=5,
            decay_ratio=0.8,  # 80% of steps for decay (8 steps)
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler - should adjust warmup steps
        lr_scheduler = build_lr_schedulers(
            self.optimizer_container, config.lr_scheduler, config.training.steps
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.2,  # Step 0: 50% of max LR (warmup)
            0.4,  # Step 1: 100% of max LR (warmup complete)
            0.6,  # Step 2: Stable phase
            0.8,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
            1.0,  # Step 5: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            0.8,  # Step 6: Linear decay starts (80% of max LR)
            0.6,  # Step 7: 60% of max LR
            0.4,  # Step 8: 40% of max LR
            0.2,  # Step 9: 20% of max LR
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()


if __name__ == "__main__":
    unittest.main()
