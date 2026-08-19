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
    """Trainer.train_step stops before updates when a step is non-finite."""

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
        trainer.num_pipeline_parallel_microbatches = 1
        trainer.step = 1
        trainer.ntokens_seen = 0

        parallel_dims = MagicMock()
        parallel_dims.dp_enabled = False
        parallel_dims.pp_enabled = False
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

    def _run_step(
        self,
        loss_value: float,
        should_log: bool,
        *,
        grad_norm: float = 1.0,
    ) -> Trainer:
        trainer = self._make_trainer(loss_value, should_log)
        # sl.* are logging side effects; clip_grad_norm_ needs real params.
        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(grad_norm),
        ):
            trainer.train_step(self._data_iterator())
        return trainer

    def _assert_updates_not_started(self, trainer: Trainer) -> None:
        trainer.checkpointer.maybe_wait_for_staging.assert_not_called()
        trainer.optimizers.step.assert_not_called()
        trainer.lr_schedulers.step.assert_not_called()

    def test_nan_loss_raises_on_log_step(self):
        trainer = self._make_trainer(float("nan"), should_log=True)
        with self.assertRaises(RuntimeError) as ctx:
            with patch("torchtitan.trainer.sl", MagicMock()), patch(
                "torchtitan.trainer.dist_utils.clip_grad_norm_",
                return_value=torch.tensor(1.0),
            ):
                trainer.train_step(self._data_iterator())
        self.assertIn("not finite", str(ctx.exception))
        self._assert_updates_not_started(trainer)

    def test_inf_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("inf"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))

    def test_finite_loss_does_not_raise(self):
        trainer = self._run_step(1.5, should_log=True)
        trainer.checkpointer.maybe_wait_for_staging.assert_called_once_with()
        trainer.optimizers.step.assert_called_once_with()
        trainer.lr_schedulers.step.assert_called_once_with()

    def test_nan_loss_raises_when_not_logging(self):
        trainer = self._make_trainer(float("nan"), should_log=False)
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            with patch("torchtitan.trainer.sl", MagicMock()), patch(
                "torchtitan.trainer.dist_utils.clip_grad_norm_",
                return_value=torch.tensor(1.0),
            ):
                trainer.train_step(self._data_iterator())
        self._assert_updates_not_started(trainer)

    def test_nonfinite_grad_norm_stops_before_updates(self):
        trainer = self._make_trainer(1.0, should_log=False)
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            with patch("torchtitan.trainer.sl", MagicMock()), patch(
                "torchtitan.trainer.dist_utils.clip_grad_norm_",
                return_value=torch.tensor(float("nan")),
            ):
                trainer.train_step(self._data_iterator())
        self._assert_updates_not_started(trainer)

    def test_nonfinite_dtensor_local_loss_stops_before_updates(self):
        class FakeDTensor:
            def detach(self):
                return self

            def to_local(self):
                return torch.tensor(float("nan"))

        trainer = self._make_trainer(1.0, should_log=False)
        trainer.forward_backward_step.return_value = FakeDTensor()
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            with patch("torchtitan.trainer.DTensor", FakeDTensor), patch(
                "torchtitan.trainer.sl", MagicMock()
            ), patch(
                "torchtitan.trainer.dist_utils.clip_grad_norm_",
                return_value=torch.tensor(1.0),
            ):
                trainer.train_step(self._data_iterator())
        self._assert_updates_not_started(trainer)

    def test_loss_reductions_use_loss_and_pp_meshes(self):
        trainer = self._make_trainer(1.0, should_log=False)
        trainer.parallel_dims.pp_enabled = True
        trainer.pp_has_last_stage = True
        loss_mesh = MagicMock()
        pp_mesh = MagicMock()
        loss_group = loss_mesh.get_group.return_value
        pp_group = pp_mesh.get_group.return_value
        trainer.parallel_dims.get_optional_mesh.side_effect = {
            "loss": loss_mesh,
            "pp": pp_mesh,
        }.get

        with patch("torchtitan.trainer.sl", MagicMock()), patch(
            "torchtitan.trainer.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(1.0),
        ), patch("torchtitan.trainer.torch.distributed.all_reduce") as all_reduce:
            trainer.train_step(self._data_iterator())

        self.assertEqual(all_reduce.call_count, 2)
        self.assertIs(all_reduce.call_args_list[0].kwargs["group"], loss_group)
        self.assertIs(all_reduce.call_args_list[1].kwargs["group"], pp_group)

    def test_remote_pp_loss_failure_stops_non_last_stage(self):
        trainer = self._make_trainer(-1.0, should_log=False)
        trainer.parallel_dims.pp_enabled = True
        trainer.pp_has_last_stage = False
        loss_mesh = MagicMock()
        pp_mesh = MagicMock()
        pp_group = pp_mesh.get_group.return_value
        trainer.parallel_dims.get_optional_mesh.side_effect = {
            "loss": loss_mesh,
            "pp": pp_mesh,
        }.get

        def propagate_nonfinite_loss(flag, **kwargs):
            flag.zero_()

        with self.assertRaisesRegex(RuntimeError, "not finite"):
            with patch("torchtitan.trainer.sl", MagicMock()), patch(
                "torchtitan.trainer.dist_utils.clip_grad_norm_",
                return_value=torch.tensor(1.0),
            ), patch(
                "torchtitan.trainer.torch.distributed.all_reduce",
                side_effect=propagate_nonfinite_loss,
            ) as all_reduce:
                trainer.train_step(self._data_iterator())

        all_reduce.assert_called_once()
        self.assertIs(all_reduce.call_args.kwargs["group"], pp_group)
        loss_mesh.get_group.assert_not_called()
        self._assert_updates_not_started(trainer)


if __name__ == "__main__":
    unittest.main()
