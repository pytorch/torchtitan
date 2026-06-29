# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-model muP sweep specs. The config-name and training-id schemes here MUST match the model's
config_registry and whatever launcher submits the runs."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass

REPORTERV2_API_URL = "https://reporterv2.comma.life"
TRAIN_LOSS_KEY = (
    "loss_metrics/global_avg_loss"  # the torchtitan trainer's train-loss metric key
)


@dataclass(frozen=True)
class MuPSweepSpec:
    name: str  # also the training-id and report-subdir stem
    module: str  # torchtitan --module / MODULE env (e.g. "path")
    config_prefix: str  # config = {prefix}_{mode}_w{width}
    base_width: int  # muP base width; the lr transfers for widths >= this
    widths: tuple  # ascending ints
    lrs: tuple  # base-lr tokens as STRINGS, so the launcher and routine share one spelling
    loss_key: str = TRAIN_LOSS_KEY

    def config_name(self, mode: str, width: int) -> str:
        return f"{self.config_prefix}_{mode}_w{width}"

    def training_id(self, mode: str, width: int, lr: str) -> str:
        return f"{self.name}_sweep_{mode}_w{width}_lr{lr}"

    @property
    def report_dir(self) -> str:
        # per-user by default so this is not bound to any one person's report mount; override with MUP_REPORT_DIR.
        return os.getenv("MUP_REPORT_DIR") or (
            f"/raid.unprotected/reports/{getpass.getuser()}_reports/mup/{self.name}"
        )

    @property
    def report_url(self) -> str:
        return f"https://research-reports.comma.life/{getpass.getuser()}_reports/mup/{self.name}/mutransfer.html"


GRID = dict(
    widths=(256, 512, 1024, 2048),
    lrs=("1e-6", "1e-5", "1e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"),
    base_width=256,
)

SPECS = {
    "plan_vit": MuPSweepSpec(
        name="plan_vit",
        module="path",
        config_prefix="vit",
        **GRID,
    ),
}
