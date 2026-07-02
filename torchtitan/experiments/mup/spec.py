# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass

REPORTERV2_API_URL = "https://reporterv2.comma.life"
TRAIN_LOSS_KEY = "loss_metrics/global_avg_loss"
MODES = ("standard", "mup")


def _report_user() -> str:
    return os.getenv("REPORT_USER") or getpass.getuser()


@dataclass(frozen=True)
class MuPSweepSpec:
    name: str
    base_width: int
    module: str = "path"
    slug: str = ""
    widths: tuple = (256, 512, 1024, 2048)
    loss_key: str = TRAIN_LOSS_KEY
    ready: bool = True

    def config_name(self, mode: str, width: int) -> str:
        return f"{self.slug or self.name}_{mode}_w{width}"

    @property
    def manifest_path(self) -> str:
        slug = self.slug or self.name
        return os.getenv("MUP_MANIFEST") or (
            f"/raid.unprotected/reports/{getpass.getuser()}_reports/prune_10m/{slug}/sweep_manifest.txt"
        )

    @property
    def report_dir(self) -> str:
        return os.getenv("MUP_REPORT_DIR") or (
            f"/raid.unprotected/reports/{_report_user()}_reports/mup/{self.name}"
        )

    def report_url(self, page: str = "mutransfer") -> str:
        return f"https://research-reports.comma.life/{_report_user()}_reports/mup/{self.name}/{page}.html"


SPECS = {
    "plan_vit": MuPSweepSpec(name="plan_vit", base_width=256, slug="vit"),
    "convnext": MuPSweepSpec(name="convnext", base_width=256),
    "fastvit": MuPSweepSpec(name="fastvit", base_width=256, ready=False),
}
