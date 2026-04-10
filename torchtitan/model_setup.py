# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.protocols import BaseModel


def materialize_model(
    model: BaseModel,
    *,
    init_device: str | torch.device,
    buffer_device: torch.device | None,
) -> BaseModel:
    """Materialize a meta-initialized model and run init_weights().

    Moves the model from the meta device to ``init_device``, initializes
    weights (under ``torch.no_grad()``), and sets the model to training
    mode.
    """
    model.to_empty(device=init_device)
    with torch.no_grad():
        # TODO: Change this back to init_weights once
        # autoparallel contains the wrap_init_states
        model.init_weights(buffer_device=buffer_device)
    model.train()
    return model
