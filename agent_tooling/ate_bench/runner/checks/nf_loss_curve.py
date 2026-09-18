"""New-feature check (loss axis): CE loss decreases over the 64-step run and is finite.

Paper (B.3): correctness has two axes; this is the first — the modified pipeline
must produce a learnable model, i.e. cross-entropy loss decreases across the run
and stays finite (no NaN, no explosion). The second axis (the three per-feature
rules) is judged separately by judge.py.

"Decreases" is checked robustly against noise: the median loss of the last window
must be below the median of the first window, and every step's loss must be
finite.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from checks import CheckResult, parse_loss_log
else:
    from . import CheckResult, parse_loss_log


def check(
    log_text: str,
    min_steps: int = 64,
    window: int = 8,
) -> CheckResult:
    pairs = parse_loss_log(log_text)
    if not pairs:
        return CheckResult(False, "no 'step: N  loss: ...' lines found in log")
    pairs.sort(key=lambda p: p[0])
    losses = [loss for _, loss in pairs]

    non_finite = [s for s, loss in pairs if not math.isfinite(loss)]
    if non_finite:
        return CheckResult(
            False,
            f"loss not finite at steps {non_finite[:5]}",
            {"n_non_finite": len(non_finite)},
        )
    if len(losses) < min_steps:
        return CheckResult(
            False,
            f"only {len(losses)} steps logged, expected >= {min_steps}",
            {"steps": len(losses)},
        )
    w = min(window, len(losses) // 2)
    first = statistics.median(losses[:w])
    last = statistics.median(losses[-w:])
    if not (last < first):
        return CheckResult(
            False,
            f"loss did not decrease: median first-{w}={first:.4f} -> last-{w}={last:.4f}",
            {"first": first, "last": last},
        )
    return CheckResult(
        True,
        f"loss finite and decreasing: median first-{w}={first:.4f} -> last-{w}={last:.4f}",
        {"first": first, "last": last, "steps": len(losses)},
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="path to the 64-step training log")
    ap.add_argument("--min-steps", type=int, default=64)
    ap.add_argument("--window", type=int, default=8)
    args = ap.parse_args(argv)
    res = check(args.log.read_text(errors="ignore"), args.min_steps, args.window)
    print(("PASS " if res.passed else "FAIL ") + res.detail)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
