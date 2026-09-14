# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.losses.dapo import _normalize


def test_eager_loss_normalization_preserves_scalar_division() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)

    normalized = _normalize(value, 7)

    assert torch.equal(normalized, value / 7)
    assert not torch.equal(normalized, value * (1 / 7))
