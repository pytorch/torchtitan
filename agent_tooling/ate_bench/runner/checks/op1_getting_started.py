"""OP1 check: the smoke training script reaches step 5 with a finite loss.

Paper (B.2.2): after the agent finishes installing deps, the harness re-runs the
read-only training script and parses the log; accepted iff step 5 prints a finite
loss value. Pass an existing log with --log, or --rerun to launch train.sh.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from checks import CheckResult, parse_loss_log
else:
    from . import CheckResult, parse_loss_log

TRAIN_SH = Path(__file__).resolve().parents[2] / "setup" / "train.sh"


def check(log_text: str, target_step: int = 5) -> CheckResult:
    pairs = parse_loss_log(log_text)
    if not pairs:
        return CheckResult(False, "no 'step: N  loss: ...' lines found in log")
    by_step = dict(pairs)
    if target_step not in by_step:
        return CheckResult(
            False,
            f"did not reach step {target_step} (max step seen: {max(by_step)})",
            {"steps_seen": sorted(by_step)},
        )
    loss = by_step[target_step]
    if not math.isfinite(loss):
        return CheckResult(False, f"step {target_step} loss is not finite: {loss}")
    return CheckResult(
        True, f"reached step {target_step} with finite loss {loss}", {"loss": loss}
    )


def rerun(steps: int = 5, timeout: int = 1800) -> str:
    proc = subprocess.run(
        ["bash", str(TRAIN_SH)],
        env={"STEPS": str(steps), **_env()},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.stdout + "\n" + proc.stderr


def _env() -> dict:
    import os

    return dict(os.environ)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--log", type=Path, help="path to an existing training log")
    g.add_argument("--rerun", action="store_true", help="launch setup/train.sh (read-only)")
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args(argv)

    text = args.log.read_text(errors="ignore") if args.log else rerun(args.step)
    res = check(text, args.step)
    print(("PASS " if res.passed else "FAIL ") + res.detail)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
