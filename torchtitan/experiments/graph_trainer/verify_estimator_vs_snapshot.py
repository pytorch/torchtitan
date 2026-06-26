# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify the FX peak-memory estimator against a real CUDA memory snapshot.

The estimator predicts per-category memory (params / optimizer / activations /
gradients) from the FX graph. This tool reads a real
``torch.cuda.memory._dump_snapshot()`` pickle, categorizes every live allocation
by its allocation call-stack, and prints a side-by-side comparison so you can
see which categories the estimator gets right and where it is off.

Categorization (by allocation stack):
  - ``adam.py``                       -> optimizer  (Adam exp_avg / exp_avg_sq)
  - ``<eval_with_key>`` / graph_module -> working_set (activations+grads+temps)
  - everything else (init)            -> params

It can verify two things:
  * params and optimizer (constant, resident the whole step) from ``segments``
    (the resident state at dump time);
  * the working set (activations+grads+temps) from the device-trace *peak* live
    set -- this is meaningful only when cuda graphs are DISABLED, otherwise the
    working set lives in a private pool that ``memory_allocated`` under-counts.

Usage:
    # snapshot-only categorization
    python -m torchtitan.experiments.graph_trainer.verify_estimator_vs_snapshot \\
        outputs/profiling/memory_snapshot/step_000000000010/000000_step_10.pickle

    # with estimator numbers (GB) to print the comparison table
    python -m torchtitan.experiments.graph_trainer.verify_estimator_vs_snapshot \\
        <snapshot.pickle> --est-params 6.94 --est-optimizer 13.765 \\
        --est-working 29.37

IMPORTANT: capture the snapshot with cuda graphs disabled for a clean working-set
comparison, e.g. add ``--compile.disable_passes cudagraph_pass`` to run_train.sh.
"""

import argparse
import pickle
from collections import Counter

GB = 1e9


def _categorize(frames) -> str:
    """Map an allocation's call-stack to a memory category."""
    if not frames:
        return "params"
    s = " ".join((f.get("filename", "") + "|" + f.get("name", "")) for f in frames[:12])
    if "adam.py" in s:
        return "optimizer"
    if any(k in s for k in ("eval_with_key", "graph_module", "cudagraph", "_to_copy")):
        return "working_set"
    return "params"


def _resting_by_category(snap) -> dict:
    """Resident (active_allocated) bytes per category from ``segments`` -- the
    state at dump time. Best for the constant costs (params, optimizer)."""
    out = Counter()
    for seg in snap["segments"]:
        for b in seg["blocks"]:
            if b["state"] == "active_allocated":
                out[_categorize(b.get("frames"))] += b["size"]
    return dict(out)


def _trace_peak_by_category(snap) -> tuple[float, dict]:
    """Replay the device trace, find the peak live set, return (peak_bytes,
    per-category bytes at peak). Captures the full working set only when cuda
    graphs are disabled. Misses init-allocated params (outside the ring buffer);
    read those from ``segments`` instead."""
    ev = snap["device_traces"][0]
    live, cur, peak, peak_live = {}, 0, 0, {}
    for e in ev:
        if e["action"] == "alloc":
            live[e["addr"]] = (e["size"], _categorize(e.get("frames")))
            cur += e["size"]
            if cur > peak:
                peak, peak_live = cur, dict(live)
        elif e["action"] == "free_completed" and e["addr"] in live:
            cur -= live[e["addr"]][0]
            del live[e["addr"]]
    out = Counter()
    for sz, c in peak_live.values():
        out[c] += sz
    return peak, dict(out)


def _row(name, est, real):
    if est is None:
        return f"  {name:24s} {'-':>10}   {real / GB:8.3f} GB"
    ratio = est / real if real else float("inf")
    return (
        f"  {name:24s} {est / GB:8.3f} GB  {real / GB:8.3f} GB  " f"ratio={ratio:6.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", help="path to a memory_snapshot .pickle")
    p.add_argument("--est-params", type=float, default=None, help="estimator params GB")
    p.add_argument(
        "--est-optimizer", type=float, default=None, help="estimator optimizer GB"
    )
    p.add_argument(
        "--est-working",
        type=float,
        default=None,
        help="estimator working set (activation+gradient+temporary) GB",
    )
    args = p.parse_args()

    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)

    resting = _resting_by_category(snap)
    reserved = sum(s["total_size"] for s in snap["segments"])
    peak_alloc, peak_cat = _trace_peak_by_category(snap)

    print(f"\nsnapshot: {args.snapshot}")
    print(f"reserved (whole process): {reserved / GB:.3f} GB")
    print("\n-- snapshot categories --")
    print(f"  params (resting):    {resting.get('params', 0) / GB:.3f} GB")
    print(f"  optimizer (resting): {resting.get('optimizer', 0) / GB:.3f} GB")
    print(f"  working set (trace peak): {peak_cat.get('working_set', 0) / GB:.3f} GB")

    est_p = None if args.est_params is None else args.est_params * GB
    est_o = None if args.est_optimizer is None else args.est_optimizer * GB
    est_w = None if args.est_working is None else args.est_working * GB
    if any(v is not None for v in (est_p, est_o, est_w)):
        print("\n-- estimator vs snapshot --")
        print(f"  {'category':24s} {'estimator':>10}  {'snapshot':>11}")
        print(_row("params", est_p, resting.get("params", 0)))
        print(_row("optimizer", est_o, resting.get("optimizer", 0)))
        print(_row("working_set(act+grad+tmp)", est_w, peak_cat.get("working_set", 0)))
        if est_p is not None and est_o is not None and est_w is not None:
            true_peak = est_p + est_o + est_w
            real_total = (
                resting.get("params", 0)
                + resting.get("optimizer", 0)
                + peak_cat.get("working_set", 0)
            )
            print(_row("TRUE PEAK (sum)", true_peak, real_total))
            print(f"  (reserved high-water: {reserved / GB:.3f} GB)")


if __name__ == "__main__":
    main()
