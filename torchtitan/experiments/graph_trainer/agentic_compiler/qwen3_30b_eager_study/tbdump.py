# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print TensorBoard scalars (full precision) of run dirs: tbdump.py <run dir>..."""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [
    "loss_metrics/global_avg_loss",
    "grad_norm",
    "loss_metrics/global_max_loss",
    "batch_wise_load_balance_loss/mean",
    "microbatch_wise_load_balance_loss/mean",
]
for run in sys.argv[1:]:
    files = sorted(glob.glob(f"{run}/**/events.out.tfevents.*", recursive=True))
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        for t in TAGS:
            if t in ea.Tags()["scalars"]:
                print(
                    run.split("/")[-1],
                    t,
                    {e.step: repr(e.value) for e in ea.Scalars(t)},
                )
