#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Verify bitwise run-to-run determinism for Run 1.

For Run 1 (no default graph passes), the bitwise-vs-eager test does not apply.
Instead, this script extracts the last-step ``loss`` and ``grad_norm`` from
``run.log`` and compares them against a stored reference captured from the
baseline run.

Usage::

    python autoresearch/scripts/verify_numerics.py reference
        Save current run.log's last-step (loss, grad_norm) as the reference.

    python autoresearch/scripts/verify_numerics.py check
        Compare current run.log's last-step (loss, grad_norm) to the reference.
        Exits 0 on match, 1 on mismatch.

Reference file: ``autoresearch/baseline_numerics.tsv``.

Caveat: training logs print loss/grad_norm with ~5 significant digits. This
catches obvious drift but is not full-precision bitwise. For higher precision
checks, enable tensorboard logging and compare TB scalars directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_LOG = REPO_ROOT / "run.log"
REF_FILE = REPO_ROOT / "autoresearch" / "baseline_numerics.tsv"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STEP_RE = re.compile(
    r"step:\s*(\d+)\s+loss:\s*([\d.eE+-]+)\s+grad_norm:\s*([\d.eE+-]+)"
)


def extract_last_step(run_log: Path) -> tuple[str, str, str] | None:
    """Return (step, loss, grad_norm) for the last step logged, or None."""
    if not run_log.exists():
        print(f"ERROR: {run_log} does not exist")
        return None
    last: tuple[str, str, str] | None = None
    for line in run_log.read_text().splitlines():
        m = _STEP_RE.search(_ANSI_RE.sub("", line))
        if m:
            last = (m.group(1), m.group(2), m.group(3))
    if last is None:
        print(f"ERROR: no 'step: N loss: X grad_norm: Y' line found in {run_log}")
    return last


def cmd_reference() -> int:
    last = extract_last_step(RUN_LOG)
    if last is None:
        return 1
    step, loss, grad_norm = last
    REF_FILE.write_text(f"step\tloss\tgrad_norm\n{step}\t{loss}\t{grad_norm}\n")
    print(f"OK: reference set step={step} loss={loss} grad_norm={grad_norm}")
    return 0


def cmd_check() -> int:
    if not REF_FILE.exists():
        print(f"ERROR: reference file {REF_FILE} not found; run 'reference' first")
        return 1
    last = extract_last_step(RUN_LOG)
    if last is None:
        return 1
    step, loss, grad_norm = last

    lines = REF_FILE.read_text().splitlines()
    if len(lines) < 2:
        print(f"ERROR: reference file {REF_FILE} is malformed")
        return 1
    ref_step, ref_loss, ref_grad_norm = lines[1].split("\t")

    if step != ref_step:
        print(
            f"WARN: step mismatch (current={step} vs ref={ref_step}); "
            "comparing anyway"
        )

    if loss == ref_loss and grad_norm == ref_grad_norm:
        print(f"OK: loss={loss} grad_norm={grad_norm} matches reference")
        return 0
    print(
        f"FAIL: current  loss={loss} grad_norm={grad_norm}\n"
        f"      reference loss={ref_loss} grad_norm={ref_grad_norm}"
    )
    return 1


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("reference", "check"):
        print(__doc__)
        return 1
    return cmd_reference() if sys.argv[1] == "reference" else cmd_check()


if __name__ == "__main__":
    sys.exit(main())
