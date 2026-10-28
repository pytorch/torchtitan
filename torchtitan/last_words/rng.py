# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Rank-local random streams committed with each logical replica's data cursor."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

RNG_STATE = "last_words_rng"
RNG_SCHEMA = 1


class RNGCheckpointState:
    """Export a frozen training-thread snapshot, never a background RNG read.

    DCP keys are disjoint across replica-local ranks. Saving one replicated key
    for different rank-local generators would silently keep only one stream.
    Only the current device is read; no other device context is initialized.
    """

    def __init__(self, device: torch.device, rank: int):
        self.device = torch.device(device)
        self.key = f"rank_{rank}"
        self._snapshot = None

    def capture(self):
        algorithm, keys, position, has_gaussian, gaussian = np.random.get_state()
        state = dict(
            schema=RNG_SCHEMA,
            device_type=self.device.type,
            python=random.getstate(),
            cpu=torch.get_rng_state().clone(),
            numpy=dict(
                algorithm=algorithm,
                keys=torch.from_numpy(keys.astype(np.int64)),
                position=position,
                has_gaussian=has_gaussian,
                gaussian=gaussian,
            ),
        )
        if self.device.type != "cpu":
            device_module = getattr(torch, self.device.type)
            state["accelerator"] = (
                device_module.get_rng_state(self.device).cpu().clone()
            )
        self._snapshot = state

    def state_dict(self):
        if self._snapshot is None:
            raise RuntimeError(
                "RNG state must be captured at a complete training boundary"
            )
        return {self.key: copy.deepcopy(self._snapshot)}

    def load_state_dict(self, state):
        value = state[self.key]
        if value["schema"] != RNG_SCHEMA or value["device_type"] != self.device.type:
            raise ValueError("Checkpoint RNG schema or device type differs")
        if self.device.type != "cpu":
            stream = value.get("accelerator")
            if (
                not isinstance(stream, torch.Tensor)
                or stream.dtype != torch.uint8
                or stream.ndim != 1
                or stream.numel() == 0
            ):
                raise ValueError(
                    "Checkpoint accelerator RNG stream is missing or invalid"
                )
        self._snapshot = copy.deepcopy(value)

    def restore(self):
        state = self.state_dict()[self.key]
        random.setstate(state["python"])
        torch.set_rng_state(state["cpu"])
        numpy = state["numpy"]
        np.random.set_state(
            (
                numpy["algorithm"],
                numpy["keys"].numpy().astype(np.uint32),
                numpy["position"],
                numpy["has_gaussian"],
                numpy["gaussian"],
            )
        )
        if self.device.type != "cpu":
            getattr(torch, self.device.type).set_rng_state(
                state["accelerator"], self.device
            )

    def fork(self):
        other = RNGCheckpointState(self.device, int(self.key.removeprefix("rank_")))
        other.load_state_dict(self.state_dict())
        return other
