# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest


@pytest.fixture(autouse=True)
def _isolate_aux_loss_registry():
    """Undo the process-global registration that building an MoE model does.

    ``AuxLoss.__init__`` increments the class-level ``AuxLoss._group_counts``,
    and ``collect_aux_loss_metrics`` later allocates on the accelerator for
    every key in it. Without this, a test that builds an MoE model with an
    auxiliary loss makes unrelated tests in the same process try to allocate
    CUDA memory, which fails on a CPU-only build.
    """
    from torchtitan.models.common.aux_loss import AuxLoss

    group_counts = dict(AuxLoss._group_counts)
    group_acc = dict(AuxLoss.group_acc)
    yield
    AuxLoss._group_counts.clear()
    AuxLoss._group_counts.update(group_counts)
    AuxLoss.group_acc.clear()
    AuxLoss.group_acc.update(group_acc)
