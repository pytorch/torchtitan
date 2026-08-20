"""OP4 check: top-kernels.csv matches the prescribed schema.

Paper (B.2.2): the CSV is checked programmatically against the schema (header,
exactly three rows sorted by total time descending); kernel *names* are validated
by a human against the Nsight GUI. We also confirm the raw profile.nsys-rep is
present so the result is reproducible.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from checks import CheckResult
else:
    from . import CheckResult

EXPECTED_HEADER = ["kernel_name", "total_time_ms", "instances", "mean_time_ms"]


def check(heavy_kernels_dir: str | Path, require_nsys_rep: bool = True) -> CheckResult:
    d = Path(heavy_kernels_dir)
    csv_path = d / "top-kernels.csv"
    if not csv_path.exists():
        return CheckResult(False, f"missing {csv_path}")
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return CheckResult(False, "top-kernels.csv is empty")
    header, *data = rows
    header = [h.strip() for h in header]
    if header != EXPECTED_HEADER:
        return CheckResult(False, f"header {header} != {EXPECTED_HEADER}")
    if len(data) != 3:
        return CheckResult(False, f"expected exactly 3 data rows, got {len(data)}")

    totals = []
    for i, row in enumerate(data, 1):
        if len(row) != 4:
            return CheckResult(False, f"row {i} has {len(row)} columns, expected 4")
        name, total, inst, mean = row
        if not name.strip():
            return CheckResult(False, f"row {i} has empty kernel_name")
        try:
            total_f = float(total)
            inst_i = int(float(inst))
            mean_f = float(mean)
        except ValueError as exc:
            return CheckResult(False, f"row {i} non-numeric value: {exc}")
        totals.append(total_f)
        # sanity: mean ~= total / instances (loose; allow rounding)
        if inst_i > 0 and total_f > 0:
            expected_mean = total_f / inst_i
            if abs(expected_mean - mean_f) > max(0.05 * expected_mean, 1e-3):
                return CheckResult(
                    False,
                    f"row {i}: mean_time_ms {mean_f} != total/instances {expected_mean:.4f}",
                )
    if totals != sorted(totals, reverse=True):
        return CheckResult(False, f"rows not sorted by total_time_ms descending: {totals}")

    if require_nsys_rep:
        reps = list(d.glob("*.nsys-rep"))
        if not reps:
            return CheckResult(False, f"no raw *.nsys-rep profile in {d}")

    return CheckResult(
        True, "top-kernels.csv schema + ordering valid; profile present", {"totals": totals}
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("heavy_kernels_dir", help="dir with top-kernels.csv + *.nsys-rep")
    ap.add_argument("--no-nsys-rep", action="store_true", help="skip the .nsys-rep presence check")
    args = ap.parse_args(argv)
    res = check(args.heavy_kernels_dir, require_nsys_rep=not args.no_nsys_rep)
    print(("PASS " if res.passed else "FAIL ") + res.detail)
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
