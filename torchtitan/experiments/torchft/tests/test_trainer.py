# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer


def test_train_step_replay_checks_whole_accumulation():
    run_gradient_accumulation = MagicMock(return_value=torch.tensor(1.0))
    replayer = SimpleNamespace(
        run_fwd_bwd=MagicMock(side_effect=lambda execute, **kwargs: execute()),
    )
    trainer = FaultTolerantTrainer.__new__(FaultTolerantTrainer)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            disable_cuda_graphs=True,
            max_norm=1.0,
            max_context_length=2048,
        ),
        parallelism="PARA",
    )
    trainer.dataloader = SimpleNamespace(max_num_documents=None)
    trainer.optimizers = MagicMock()
    trainer.lr_schedulers = SimpleNamespace(
        schedulers=[SimpleNamespace(get_last_lr=lambda: [0.1])],
        step=MagicMock(),
    )
    trainer.parallel_dims = SimpleNamespace(
        dp_enabled=False,
        pp_enabled=False,
        dp_cp_enabled=False,
        ep_enabled=False,
        get_optional_mesh=lambda name: None,
    )
    trainer.gradient_accumulation_steps = 2
    trainer.num_pp_microbatches = 1
    trainer.device = torch.device("cpu")
    trainer._run_gradient_accumulation = run_gradient_accumulation
    trainer.sdc_replayer = replayer
    trainer.model_parts = [
        SimpleNamespace(
            preprocess_inputs=lambda input_dict, **kwargs: (
                input_dict["input"],
                input_dict["labels"],
                {},
            ),
            parameters=lambda: [],
        )
    ]
    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=MagicMock())
    trainer.metrics_processor = SimpleNamespace(
        should_log=MagicMock(return_value=False)
    )
    trainer.step = 1
    trainer.ntokens_seen = 0
    batches = [
        (
            {"input": torch.ones(1), "num_valid_tokens": torch.tensor(1)},
            torch.ones(1, dtype=torch.long),
        )
        for _ in range(2)
    ]

    with patch(
        "torchtitan.experiments.torchft.trainer.dist_utils.clip_grad_norm_",
        return_value=torch.tensor(1.0),
    ):
        trainer.train_step(iter(batches))

    replayer.run_fwd_bwd.assert_called_once()
    assert replayer.run_fwd_bwd.call_args.kwargs == {"step": 1}
    run_gradient_accumulation.assert_called_once()
    trainer.optimizers.step.assert_called_once_with()
