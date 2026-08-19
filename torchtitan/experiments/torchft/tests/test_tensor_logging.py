# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer


def test_fault_tolerant_trainer_rejects_tensor_logging() -> None:
    config = SimpleNamespace(
        metrics=SimpleNamespace(tensor_logging=SimpleNamespace(enabled=True))
    )

    with pytest.raises(NotImplementedError, match="FaultTolerantTrainer"):
        FaultTolerantTrainer(config)
