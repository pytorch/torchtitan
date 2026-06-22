# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn


def run_looped_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    extra_kwargs: dict[str, Any],
    *,
    steps: int,
    use_exit_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a trainer-controlled recurrent decoder forward.

    The model is treated as a black-box compatible decoder: it only needs the
    four loop hook methods installed on ``Decoder``. This keeps recurrence
    independent of dense, MoE, or linear-attention layer internals.
    """
    for method_name in (
        "prepare_loop_inputs",
        "loop_step",
        "project_logits",
        "exit_gate",
    ):
        if not callable(getattr(model, method_name, None)):
            raise TypeError(
                f"{type(model).__name__} does not support looped training; "
                f"missing method {method_name!r}."
            )

    hidden = model.prepare_loop_inputs(inputs, **extra_kwargs)
    logits_by_step = []
    gate_logits_by_step = []

    for _ in range(steps):
        hidden = model.loop_step(hidden, **extra_kwargs)
        logits_by_step.append(model.project_logits(hidden))
        if use_exit_gate:
            gate_logits_by_step.append(model.exit_gate(hidden))

    logits = torch.stack(logits_by_step, dim=0)
    gates = torch.stack(gate_logits_by_step, dim=0) if use_exit_gate else None
    return logits, gates
