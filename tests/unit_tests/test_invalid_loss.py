# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.trainer import Trainer


class TestInvalidLoss(unittest.TestCase):
    """Trainer.train_step crashes on a non-finite loss, aligned with logging.

    The finiteness check reuses the device-to-host copy that logging already
    performs, so it only runs on steps where metrics are logged.
    """

    def _make_trainer(self, loss_value: float, should_log: bool) -> Trainer:
        # Build a bare Trainer and inject only the collaborators train_step
        # touches on the non-distributed (single-rank) path.
        trainer = object.__new__(Trainer)

        trainer.optimizers = MagicMock()
        trainer.lr_schedulers = MagicMock()
        trainer.lr_schedulers.get_metrics.return_value = {}
        trainer.checkpointer = MagicMock()
        trainer.model_parts = []
        trainer.config = MagicMock()
        trainer.config.training.max_norm = 1.0
        trainer.device = torch.device("cpu")
        trainer.gradient_accumulation_steps = 1
        trainer.step = 1
        trainer.ntokens_seen = 0

        parallel_dims = MagicMock()
        parallel_dims.dp_enabled = False
        parallel_dims.dp_cp_enabled = False
        parallel_dims.ep_enabled = False
        parallel_dims.get_optional_mesh.return_value = None
        trainer.parallel_dims = parallel_dims

        trainer.metrics_processor = MagicMock()
        trainer.metrics_processor.should_log.return_value = should_log

        # Shadow the bound method so forward/backward returns a canned loss.
        trainer.forward_backward_step = MagicMock(return_value=torch.tensor(loss_value))
        return trainer

    def _data_iterator(self):
        labels = torch.tensor([1, 2, IGNORE_INDEX])
        input_dict = {"input": torch.tensor([1, 2, 3])}
        while True:
            yield input_dict, labels

    def _run_step(self, loss_value: float, should_log: bool) -> None:
        trainer = self._make_trainer(loss_value, should_log)
        # sl.* are logging side effects; clip_grad_norm_ needs real params.
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(1.0),
        ):
            trainer.train_step(self._data_iterator())

    def test_nan_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("nan"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))

    def test_inf_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("inf"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))

    def test_finite_loss_does_not_raise(self):
        self._run_step(1.5, should_log=True)

    def test_nan_loss_ignored_when_not_logging(self):
        # Detection is gated on logging, so a non-log step must not raise even
        # when the loss is NaN.
        self._run_step(float("nan"), should_log=False)


if __name__ == "__main__":
    unittest.main()
