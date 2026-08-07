# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import torch

from torchtitan.trainer import Trainer


def test_pp_forward_backward_step_returns_sentinel_without_last_stage():
    trainer = cast(
        Trainer,
        SimpleNamespace(
            pp_has_first_stage=False,
            pp_has_last_stage=False,
            pp_schedule=SimpleNamespace(step=lambda **kwargs: None),
            train_context=nullcontext,
            post_dataloading_process=lambda input_dict, labels: (
                input_dict["input"],
                labels,
                {},
            ),
            device=torch.device("cpu"),
        ),
    )

    loss = Trainer.pp_forward_backward_step(
        trainer,
        input_dict_mbs=[{"input": torch.ones(1)}],
        label_mbs=[torch.ones(1)],
        global_valid_tokens=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor([-1.0]))


def test_cuda_graph_fwd_bwd_clones_reused_cuda_graph_output():
    graph_loss = torch.tensor(0.0)
    trainer = cast(
        Trainer,
        SimpleNamespace(
            _cg_wrapper=lambda *args: graph_loss,
            _cg_extra_input_spec=SimpleNamespace(flatten=lambda kwargs: []),
            device=torch.device("cpu"),
        ),
    )

    accumulated_losses = []
    for value in (1.0, 2.0, 3.0):
        graph_loss.fill_(value)
        loss = Trainer._cuda_graph_fwd_bwd(
            trainer,
            inputs=torch.ones(1),
            labels=torch.ones(1),
            global_valid_tokens=1.0,
            extra_kwargs={},
        )
        accumulated_losses.append(loss.detach())

    torch.testing.assert_close(
        torch.sum(torch.stack(accumulated_losses)), torch.tensor(6.0)
    )
