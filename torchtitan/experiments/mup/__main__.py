# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI for the muP routine. `grid` prints the launch grid (the caller submits each line);
`report` collects from reporterv2 and writes the transferred-lr report."""

import sys

from .routine import build_report, collect, grid, hp_table
from .spec import SPECS


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("grid", "report"):
        sys.exit("usage: python -m torchtitan.experiments.mup {grid|report} <model>")
    verb, model = sys.argv[1], sys.argv[2]
    if model not in SPECS:
        sys.exit(f"unknown model {model!r}; known: {', '.join(SPECS)}")
    spec = SPECS[model]

    if verb == "grid":
        n = len(spec.widths) * len(spec.lrs) * 2
        print(
            f"# launch grid for {model}: {n} runs (standard + mup). submission is the caller's job."
        )
        print(
            "# wrap each line with your launcher, e.g. training/run.sh torchtitan/run_train.sh N=2 PARTITION=tbox2 ..."
        )
        for g in grid(spec):
            print(
                f"MODULE={spec.module} CONFIG={g['config']} "
                f"REPORTERV2_TRAINING_ID={g['training_id']} {g['cli']}"
            )
        return

    # verb == "report"
    results = collect(spec)
    url = build_report(spec, results=results)
    per_width, transferred = hp_table(spec, results)
    print(f"transferred muP lr = {transferred}")
    print(f"report -> {url}")


if __name__ == "__main__":
    main()
