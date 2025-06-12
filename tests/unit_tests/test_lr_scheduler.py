# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.optim import Adam

from torchtitan.components.lr_scheduler import (
    build_lr_schedulers,
    LRSchedulersContainer,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config_manager import JobConfig, LRScheduler, Training


class TestLRScheduler(unittest.TestCase):
    def setUp(self):
        # Create a simple model with parameters
        self.model = torch.nn.Linear(10, 10)
        # Create an optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.1)
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
        lr_min=None,
    ):
        # Create a job config with the specified parameters
        from torchtitan.config_manager import ConfigManager

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
        args += ["--lr_scheduler.lr_min", str(lr_min)] if lr_min is not None else []

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
            lr_min=None,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

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
            lr_min=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

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

    def test_cosine_decay(self):
        """Test the cosine decay schedule."""
        # Create a job config with cosine decay. warmup_steps=2, decay_steps=8.
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,
            decay_type="cosine",
            lr_min=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

        # Skip warmup steps
        lr_scheduler.step()
        # After warmup (step 2), LR should be at max
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.1, places=6)
        lr_scheduler.step()

        # We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.1, places=6)

        # Step through and check cosine decay values
        lr_scheduler.step()  # Step 2
        # Expected: initial_lr * (0.5 * (1 + cos(π * 1/8)))
        expected_lr = 0.1 * 0.5 * (1 + math.cos(math.pi * 1 / 8))
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"], expected_lr, places=6
        )

        # Middle of decay (step 6)
        lr_scheduler.step()  # Step 3
        lr_scheduler.step()  # Step 4
        lr_scheduler.step()  # Step 5
        lr_scheduler.step()  # Step 6
        # Expected: 0.1 * (0.5 * (1 + cos(π * 5/8)))
        expected_lr = 0.1 * 0.5 * (1 + math.cos(math.pi * 5 / 8))
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"], expected_lr, places=6
        )

    def test_sqrt_decay(self):
        """Test the sqrt decay schedule."""
        # Create a job config with sqrt decay. warmup_steps=2, decay_steps=8.
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,
            decay_type="sqrt",
            lr_min=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

        # Skip warmup steps
        lr_scheduler.step()  # lr: 0.5 -> 1.0
        # After warmup (step 2), LR should be at max
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.1, places=6)
        lr_scheduler.step()

        # We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.1, places=6)

        # Step through and check sqrt decay values
        lr_scheduler.step()
        # Expected: initial_lr * (1 - sqrt(1/8))
        expected_lr = 0.1 * (1 - math.sqrt(1 / 8))
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"], expected_lr, places=6
        )

    def test_min_lr(self):
        """Test that the learning rate doesn't go below the minimum."""
        # Create a job config with a minimum learning rate
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,
            decay_type="linear",
            lr_min=0.2,  # 20% of base LR as minimum
        )

        # Build the lr scheduler
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

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
            lr_min=0.0,
        )

        # Build the lr scheduler - should adjust warmup steps
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

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

    def test_warmup_plus_decay_exceeds_training(self):
        """Test when warmup + decay steps exceed training steps."""
        # Create a job config where warmup + decay steps > training steps
        # Expected behaviro: warmup steps = 5, decay steps = 5
        config = self.create_job_config(
            training_steps=10,
            warmup_steps=5,
            decay_ratio=0.8,  # 80% of steps for decay (8 steps)
            decay_type="linear",
            lr_min=0.0,
        )

        # Build the lr scheduler - should adjust warmup steps
        lr_scheduler = build_lr_schedulers(self.optimizer_container, config)

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
