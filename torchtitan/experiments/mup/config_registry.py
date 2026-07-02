# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import getpass
from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager

from ..path.config_registry import _vit
from ..path.trainer import PathTrainer

BIG_FLAVOR = "w2048"
BIG_STEPS = 15360
BIG_LR = 1e-2


def vit_mup_w2048_ckpt() -> PathTrainer.Config:
    cfg = _vit(BIG_FLAVOR, mup=True, lr=BIG_LR)
    return replace(
        cfg,
        training=replace(cfg.training, steps=BIG_STEPS),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder=f"/raid.unprotected/reports/{getpass.getuser()}_reports"
            f"/prune_10m/vit/checkpoints/{BIG_FLAVOR}",
            interval=BIG_STEPS,
            keep_latest_k=0,
        ),
    )
