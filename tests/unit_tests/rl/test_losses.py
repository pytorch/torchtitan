# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.losses.dapo import _normalize


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_valid_tokens = torch.tensor(7, dtype=torch.int64)

    normalized = _normalize(value, global_valid_tokens)

    assert torch.equal(
        normalized,
        value * global_valid_tokens.clamp_min(1).reciprocal(),
    )
