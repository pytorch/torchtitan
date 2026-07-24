"""OP2 check: the eval pipeline produced a finite HellaSwag score.

Paper (B.2.2): satisfied when evaluate.sh runs to completion and writes a finite
HellaSwag score; the score is not required to clear any quality threshold (25
steps from random init is expected near-random).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from checks import CheckResult
else:
    from . import CheckResult


def check(hellaswag_json: str | Path) -> CheckResult:
    p = Path(hellaswag_json)
    if not p.exists():
        return CheckResult(False, f"missing eval output: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        return CheckResult(False, f"eval output is not valid JSON: {exc}")
    score = data.get("hellaswag_acc")
    if score is None:
        return CheckResult(False, f"no 'hellaswag_acc' key in {p}", {"keys": list(data)})
    try:
        score = float(score)
    except (TypeError, ValueError):
        return CheckResult(False, f"hellaswag_acc is not a number: {score!r}")
    if not math.isfinite(score):
        return CheckResult(False, f"hellaswag_acc is not finite: {score}")
    return CheckResult(
        True, f"pipeline completed; finite HellaSwag acc={score:.4f}", {"acc": score}
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hellaswag_json", help="path to evaluate.sh's hellaswag.json")
    args = ap.parse_args(argv)
    res = check(args.hellaswag_json)
    print(("PASS " if res.passed else "FAIL ") + res.detail)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
