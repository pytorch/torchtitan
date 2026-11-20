# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .types import Max, Metric


def ram_metrics(role: str) -> list[Metric]:
    """Return the RAM this process uses now (VmRSS) and the most it has used (VmHWM), in GiB.
    Shared libraries count in full.

    Example, in a rollout worker:

        ram_metrics("rollout_worker")
        # -> [Metric("rollout_worker/ram_gib", Max(1.7)),
        #     Metric("rollout_worker/peak_ram_gib", Max(2.1))]
    """
    # Each line reads e.g. "VmRSS:  1745628 kB".
    with open("/proc/self/status") as status:
        kib = {
            line.split(":")[0]: int(line.split()[1])
            for line in status
            if line.startswith(("VmRSS:", "VmHWM:"))
        }
    return [
        Metric(f"{role}/ram_gib", Max(kib["VmRSS"] / 2**20)),
        Metric(f"{role}/peak_ram_gib", Max(kib["VmHWM"] / 2**20)),
    ]
