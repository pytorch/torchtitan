#!/usr/bin/env python3
"""Parse torchtitan / graph_trainer run logs into metric tables.

Per-step metrics come from the `step:` lines; memory metrics come from the
`Real Peak memory` / `Estimated Peak memory` / `Categories:` block that the
trainer emits immediately *before* each step line, so each block is attached
to the step that follows it.

  ./parse_run.py run.log                  # summary tables
  ./parse_run.py --steps run.log          # per-step table too
  ./parse_run.py cmp*.log                 # one summary row per config
  ./parse_run.py --warmup 10 run.log      # fixed warmup instead of auto
  ./parse_run.py --csv out.csv run.log    # per-step rows to CSV
  ./parse_run.py --json run.log           # machine-readable dump
"""
import argparse
import csv
import gzip
import json
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RANK = re.compile(r"^\[rank(\d+)\]:")

STEP = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-\d.naif]+)\s+"
    r"grad_norm:\s*(?P<grad_norm>[-\d.naif]+)\s+"
    r"memory:\s*(?P<reserved>[\d.]+)GiB\((?P<reserved_pct>[\d.]+)%\)\s+"
    r"tps:\s*(?P<tps>[\d,]+)\s+"
    r"tflops:\s*(?P<tflops>[\d.]+)\s+"
    r"mfu:\s*(?P<mfu>[\d.]+)%",
    re.I,
)
REAL_PEAK = re.compile(r"Real Peak memory:\s*([\d.]+)")
EST_PEAK = re.compile(r"Estimated Peak memory:\s*([\d.]+)")
CATEGORIES = re.compile(r"Categories:\s*(.*)")
CAT_KV = re.compile(r"(\w+):\s*([\d.]+)")
PASSES = re.compile(r"All (\d+) graph passes took ([\d.]+)s")

# Failure modes worth surfacing instead of a row of numbers.
FAILURES = [
    (re.compile(r"structure differs"), "RANK DIVERGENCE"),
    (re.compile(r"tightest plan measured is ([\d.]+) GiB"), "REFUSED floor={0} GiB"),
    (re.compile(r"no split reached a solvable plan"), "REFUSED infeasible"),
    (re.compile(r"CUDA out of memory|torch\.OutOfMemoryError"), "OOM"),
]

CAT_ORDER = ["Activation", "Grad", "INPUT", "PARAM", "TEMP", "OPT"]


def _open(path):
    if path == "-":
        return sys.stdin
    return (
        gzip.open(path, "rt", errors="ignore")
        if path.endswith(".gz")
        else open(path, "rt", errors="ignore")
    )


def parse_file(path, rank=None):
    """Stream the log, flushing a pending memory block onto each step line."""
    steps, pending, meta = [], {}, {"passes": None, "pass_time": None, "failure": None}
    trailing = None
    with _open(path) as fh:
        for raw in fh:
            line = ANSI.sub("", raw).rstrip("\n")
            m = RANK.match(line)
            if m:
                if rank is not None and int(m.group(1)) != rank:
                    continue
                line = line[m.end() :]
            elif rank not in (None, 0):
                continue

            if meta["failure"] is None:
                for pat, tmpl in FAILURES:
                    f = pat.search(line)
                    if f:
                        meta["failure"] = tmpl.format(*f.groups())
                        break

            m = PASSES.search(line)
            if m:
                meta["passes"], meta["pass_time"] = int(m.group(1)), float(m.group(2))
                continue
            m = REAL_PEAK.search(line)
            if m:
                pending["real_peak"] = float(m.group(1))
                continue
            m = EST_PEAK.search(line)
            if m:
                pending["est_peak"] = float(m.group(1))
                continue
            m = CATEGORIES.search(line)
            if m:
                pending["cat"] = {k: float(v) for k, v in CAT_KV.findall(m.group(1))}
                continue

            m = STEP.search(line)
            if m:
                d = m.groupdict()
                row = {
                    "step": int(d["step"]),
                    "loss": float(d["loss"]),
                    "grad_norm": float(d["grad_norm"]),
                    "reserved": float(d["reserved"]),
                    "reserved_pct": float(d["reserved_pct"]),
                    "tps": float(d["tps"].replace(",", "")),
                    "tflops": float(d["tflops"]),
                    "mfu": float(d["mfu"]),
                }
                row.update(pending)
                steps.append(row)
                pending = {}

    if pending:
        trailing = pending  # memory block after the last step
    return {"path": path, "steps": steps, "trailing": trailing, **meta}


def split_warmup(steps, warmup=None, frac=0.9):
    """Return (warm, steady). Auto mode drops leading steps well below median tps.

    Compile, allocator growth and the first collectives make the opening steps
    slow, so a relative-to-median cut separates them without hardcoding a step
    count that varies per model. Capped at a quarter of the run so a genuinely
    noisy config can never eat its own measurement window.
    """
    if not steps:
        return [], []
    if warmup is not None:
        return steps[:warmup], steps[warmup:]
    med = statistics.median(s["tps"] for s in steps)
    cap = max(1, len(steps) // 4)
    i = 0
    while i < cap and steps[i]["tps"] < frac * med:
        i += 1
    return steps[:i], steps[i:]


def summarize(vals):
    if not vals:
        return {}
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    sem = sd / n**0.5 if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(vals),
        "sd": sd,
        "cv": 100 * sd / mean if mean else 0.0,
        "sem": sem,
        "ci95": 1.96 * sem,
        "min": min(vals),
        "max": max(vals),
    }


def analyze(rec, warmup=None):
    warm, steady = split_warmup(rec["steps"], warmup)
    out = dict(rec)
    out["warmup_steps"] = len(warm)
    out["steady"] = steady
    out["tps"] = summarize([s["tps"] for s in steady])
    out["tflops"] = summarize([s["tflops"] for s in steady])
    out["mfu"] = summarize([s["mfu"] for s in steady])
    out["reserved"] = summarize([s["reserved"] for s in steady])

    peaks = [s["real_peak"] for s in steady if "real_peak" in s]
    ests = [s["est_peak"] for s in steady if "est_peak" in s]
    if rec["trailing"] and "real_peak" in rec["trailing"]:
        peaks.append(rec["trailing"]["real_peak"])
    if rec["trailing"] and "est_peak" in rec["trailing"]:
        ests.append(rec["trailing"]["est_peak"])
    out["real_peak"] = max(peaks) if peaks else None
    out["est_peak"] = max(ests) if ests else None
    out["peak_err"] = (
        100 * (out["est_peak"] - out["real_peak"]) / out["real_peak"]
        if peaks and ests and out["real_peak"]
        else None
    )
    cats = [s["cat"] for s in steady if "cat" in s]
    if not cats and rec["trailing"] and "cat" in rec["trailing"]:
        cats = [rec["trailing"]["cat"]]
    out["cat"] = cats[-1] if cats else None
    out["final_loss"] = steady[-1]["loss"] if steady else None
    return out


def tag_of(rec):
    return (
        os.path.basename(rec["path"]).rsplit(".log", 1)[0].rsplit(".", 1)[0]
        or rec["path"]
    )


def fmt(v, spec, dash="-"):
    return dash if v is None else format(v, spec)


def print_steps(rec):
    print(f"\n=== {tag_of(rec)} : per-step ===")
    hdr = (
        f"{'step':>5} {'loss':>9} {'gnorm':>8} {'tps':>8} {'tflops':>8} {'mfu%':>6} "
        f"{'resvd':>8} {'real_pk':>8} {'est_pk':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    warm, _ = split_warmup(rec["steps"], rec.get("_warmup"))
    warm_ids = {s["step"] for s in warm}
    for s in rec["steps"]:
        mark = "*" if s["step"] in warm_ids else " "
        print(
            f"{s['step']:>4}{mark} {s['loss']:>9.5f} {s['grad_norm']:>8.4f} "
            f"{s['tps']:>8,.0f} {s['tflops']:>8.2f} {s['mfu']:>6.2f} "
            f"{s['reserved']:>8.2f} {fmt(s.get('real_peak'), '>8.2f')} "
            f"{fmt(s.get('est_peak'), '>8.2f')}"
        )
    if warm_ids:
        print(f"  (* = warmup, excluded from summary: {len(warm_ids)} step(s))")


def print_throughput(recs, stat):
    print("\n=== throughput (steady-state steps only) ===")
    hdr = (
        f"{'config':<22} {'steps':>6} {'warm':>5} {f'tps({stat})':>11} {'±sd':>7} "
        f"{'±95%CI':>7} {'cv%':>5} {'min':>8} {'max':>8} {'tflops':>8} {'mfu%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        if r["failure"] or not r["tps"]:
            print(f"{tag_of(r):<22} {r['failure'] or 'no step data'}")
            continue
        t, f_, m = r["tps"], r["tflops"], r["mfu"]
        print(
            f"{tag_of(r):<22} {t['n']:>6} {r['warmup_steps']:>5} {t[stat]:>11,.0f} "
            f"{t['sd']:>7,.0f} {t['ci95']:>7,.0f} {t['cv']:>5.1f} {t['min']:>8,.0f} "
            f"{t['max']:>8,.0f} {f_[stat]:>8.1f} {m[stat]:>6.2f}"
        )


def print_memory(recs):
    print("\n=== memory (GiB) ===")
    hdr = (
        f"{'config':<22} {'real_peak':>10} {'est_peak':>10} {'est_err%':>9} "
        f"{'reserved':>9} {'frag':>7}  " + " ".join(f"{c[:6]:>7}" for c in CAT_ORDER)
    )
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        if r["failure"] or r["real_peak"] is None:
            print(f"{tag_of(r):<22} {r['failure'] or 'no memory data'}")
            continue
        resv = r["reserved"].get("max") if r["reserved"] else None
        frag = resv - r["real_peak"] if resv is not None else None
        cats = r["cat"] or {}
        print(
            f"{tag_of(r):<22} {r['real_peak']:>10.2f} {fmt(r['est_peak'], '>10.2f')} "
            f"{fmt(r['peak_err'], '>+9.1f')} {fmt(resv, '>9.2f')} {fmt(frag, '>7.2f')}  "
            + " ".join(f"{cats.get(c, float('nan')):>7.2f}" for c in CAT_ORDER)
        )
    print(
        "  real_peak = allocated peak (compare to est_peak); reserved = allocator "
        "high-water; frag = reserved - allocated"
    )


def print_compile(recs):
    rows = [r for r in recs if r["pass_time"] is not None]
    if not rows:
        return
    print("\n=== compile ===")
    for r in rows:
        print(f"{tag_of(r):<22} {r['passes']} graph passes, {r['pass_time']:.2f}s")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+", help="log files ('-' for stdin, .gz ok)")
    ap.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="drop this many leading steps (default: auto-detect)",
    )
    ap.add_argument(
        "--stat",
        choices=["mean", "median"],
        default="mean",
        help="point estimate for the summary tables (default: mean)",
    )
    ap.add_argument(
        "--steps", action="store_true", help="also print the per-step table"
    )
    ap.add_argument(
        "--rank", type=int, default=0, help="which [rankN] prefix to keep (default: 0)"
    )
    ap.add_argument("--csv", metavar="OUT", help="write per-step rows to a CSV file")
    ap.add_argument("--json", action="store_true", help="dump everything as JSON")
    args = ap.parse_args()

    recs = []
    for path in args.files:
        rec = analyze(parse_file(path, rank=args.rank), warmup=args.warmup)
        rec["_warmup"] = args.warmup
        recs.append(rec)

    if args.json:
        json.dump(
            [
                {k: v for k, v in r.items() if k not in ("steps", "steady")}
                | {"steps": r["steps"]}
                for r in recs
            ],
            sys.stdout,
            indent=2,
            default=str,
        )
        print()
        return

    if args.steps:
        for r in recs:
            print_steps(r)
    print_throughput(recs, args.stat)
    print_memory(recs)
    print_compile(recs)

    if args.csv:
        cols = [
            "config",
            "step",
            "loss",
            "grad_norm",
            "tps",
            "tflops",
            "mfu",
            "reserved",
            "reserved_pct",
            "real_peak",
            "est_peak",
            "warmup",
        ] + CAT_ORDER
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                warm_ids = {s["step"] for s in split_warmup(r["steps"], args.warmup)[0]}
                for s in r["steps"]:
                    w.writerow(
                        {
                            "config": tag_of(r),
                            "warmup": int(s["step"] in warm_ids),
                            **{k: v for k, v in s.items() if k != "cat"},
                            **(s.get("cat") or {}),
                        }
                    )
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
