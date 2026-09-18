# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine


class TestInvalidLoss(unittest.TestCase):
    """Trainer.train_step crashes on a non-finite step before updates."""

    def _make_trainer(self, loss_value: float, should_log: bool) -> Trainer:
        # Build a bare loop and engine, then inject only what train_step
        # touches on the non-distributed (single-rank) path.
        loop = object.__new__(Trainer)
        trainer = object.__new__(TrainingEngine)

        trainer.optimizers = MagicMock()
        trainer.lr_schedulers = MagicMock()
        trainer.lr_schedulers.get_metrics.return_value = {}
        trainer.checkpointer = MagicMock()
        trainer.model_parts = [
            SimpleNamespace(
                preprocess_inputs=lambda input_dict, **kwargs: (
                    input_dict["input"],
                    input_dict["labels"],
                    {},
                ),
                parameters=lambda: iter(()),
            )
        ]
        trainer.max_num_documents = None
        trainer.preprocess_inputs_kwargs = {}
        trainer.config = MagicMock()
        trainer.config.training.max_norm = 1.0
        trainer.config.training.disable_cuda_graphs = True
        trainer.sdc_replayer = None
        trainer.device = torch.device("cpu")
        trainer.num_completed_steps = 1
        trainer.ntokens_seen = 0
        trainer._num_optimizer_steps_since_cuda_graph_init = 0
        trainer.gc_handler = MagicMock()
        trainer._deferred_cuda_graph_options = None

        parallel_dims = MagicMock()
        parallel_dims.dp_enabled = False
        parallel_dims.pp_enabled = False
        parallel_dims.dp_cp_enabled = False
        parallel_dims.ep_enabled = False
        parallel_dims.dp_replicate_enabled = False
        parallel_dims.get_optional_mesh.return_value = None
        trainer.parallel_dims = parallel_dims

        loop.engine = trainer
        loop.config = trainer.config
        loop.gradient_accumulation_steps = 1
        loop.num_pp_microbatches = 1
        loop.metrics_processor = MagicMock()
        loop.metrics_processor.should_log.return_value = should_log

        trainer.forward_backward_body_fn = MagicMock(
            return_value=torch.tensor(loss_value)
        )

        return loop

    def _data_iterator(self):
        labels = torch.tensor([1, 2, IGNORE_INDEX])
        while True:
            yield TokenizedTrainingMicrobatch(
                input=torch.tensor([1, 2, 3]),
                labels=labels,
                positions=torch.arange(3),
                padding_mask=torch.zeros(3, dtype=torch.bool),
                num_valid_tokens=2,
            )

    def _run_step(self, loss_value: float, should_log: bool) -> Trainer:
        trainer = self._make_trainer(loss_value, should_log)
        # sl.* are logging side effects; clip_grad_norm_ needs real params.
        with patch("torchtitan.training_engine.sl", MagicMock()), patch(
            "torchtitan.training_engine.dist_utils.clip_grad_norm_",
            return_value=torch.tensor(1.0),
        ):
            trainer.train_step(self._data_iterator())
        return trainer

    def test_nan_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("nan"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))
        self.assertIn("step 2", str(ctx.exception))

    def test_inf_loss_raises_on_log_step(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("inf"), should_log=True)
        self.assertIn("not finite", str(ctx.exception))
        self.assertIn("step 2", str(ctx.exception))

    def test_finite_loss_does_not_raise(self):
        trainer = self._run_step(1.5, should_log=True)
        self.assertEqual(trainer.engine.num_completed_steps, 2)

    def test_nan_loss_raises_when_not_logging(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run_step(float("nan"), should_log=False)
        self.assertIn("not finite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
