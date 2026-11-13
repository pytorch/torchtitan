# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Spatio-temporal allocation plots: address (y) x time (x) for one train step.

Time is the allocation ordinal within the step (identical across runs because
the allocation sequence is deterministic): an alloc at ordinal k is drawn from
x=k; a free is placed at x = number of allocs that precede it.

Panels:
  caching   -- PyTorch caching allocator (from memory-history snapshot). The
               y axis concatenates every reserved address range (segments)
               with gaps squeezed out, so total height == reserved memory.
               Grey = reserved but unallocated (fragmentation / cache).
  planned   -- planned allocator. Bottom band: fallback pool (persistent and
               step-escaping tensors), height = measured pool reserved peak;
               top band: the arena with exact planned offsets.

Run with PYTHONPATH=/tmp/pdftools (matplotlib side install); does not import torch.
"""

import argparse
import bisect
import json
import os
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

G = 1024**3
COLORS = {
    "persistent": "#7f8c8d",
    "fwd": "#2e86de",
    "bwd": "#e67e22",
    "optim": "#27ae60",
    "other": "#8e44ad",
    "escaping": "#c0392b",
}


def _phase_of(ev, line_bwd):
    # The innermost frame in FX-generated code ("<eval_with_key>.N") gives the
    # graph line, which maps to the node and its forward/backward marker.
    for fr in ev.get("frames") or []:
        if fr["filename"].startswith("<eval_with_key>"):
            bwd = line_bwd.get(fr["line"])
            if bwd is not None:
                return "bwd" if bwd else "fwd"
            break
    md = ev.get("user_metadata") or ""
    if md == "optim":
        return "optim"
    return "other"


def load_caching(run_dir, graph_info_path):
    state = pickle.load(open(os.path.join(run_dir, "state_before.pkl"), "rb"))
    snap = pickle.load(open(os.path.join(run_dir, "trace.pkl"), "rb"))
    line_bwd = {}
    if graph_info_path and os.path.exists(graph_info_path):
        import re

        gi = json.load(open(graph_info_path))
        node_bwd = {n["name"]: n["bwd"] for n in gi["nodes"]}
        code_path = os.path.join(os.path.dirname(graph_info_path), "graph_code.py")
        for i, line in enumerate(open(code_path).read().split("\n"), start=1):
            m = re.match(r"\s*(\w+) = ", line)
            if m and m.group(1) in node_bwd:
                line_bwd[i] = node_bwd[m.group(1)]
    traces = [t for t in snap["device_traces"] if t]
    events = traces[0]

    blocks = []  # dict(start, end, addr, size, kind)
    open_by_addr = {}
    seg_ranges = []  # (addr, size)
    seg_events = []  # (t, +size/-size)
    for seg in state["segments"]:
        seg_ranges.append((seg["address"], seg["total_size"]))
        seg_events.append((0, seg["total_size"]))
        a = seg["address"]
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                b = {
                    "start": 0,
                    "end": None,
                    "addr": a,
                    "size": blk["size"],
                    "kind": "persistent",
                }
                blocks.append(b)
                open_by_addr[a] = b
            a += blk["size"]

    t = 0
    has_completed = any(e["action"] == "free_completed" for e in events)
    free_action = "free_completed" if has_completed else "free_requested"
    for ev in events:
        act = ev["action"]
        if act == "alloc":
            b = {
                "start": t,
                "end": None,
                "addr": ev["addr"],
                "size": ev["size"],
                "kind": _phase_of(ev, line_bwd),
            }
            blocks.append(b)
            open_by_addr[ev["addr"]] = b
            t += 1
        elif act == free_action:
            b = open_by_addr.pop(ev["addr"], None)
            if b is not None:
                b["end"] = t
        elif act in ("segment_alloc", "segment_map"):
            seg_ranges.append((ev["addr"], ev["size"]))
            seg_events.append((t, ev["size"]))
        elif act in ("segment_free", "segment_unmap"):
            seg_events.append((t, -ev["size"]))
    t_end = t
    for b in blocks:
        if b["end"] is None:
            b["end"] = t_end
            if b["kind"] != "persistent":
                b["kind"] = "escaping"

    # Squeeze gaps between reserved address ranges.
    seg_ranges.sort()
    merged = []
    for a, s in seg_ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], a + s)
        else:
            merged.append([a, a + s])
    starts = [m[0] for m in merged]
    # Stack ranges with the largest persistent share at the bottom so the
    # transient region is contiguous at the top (layout within a range is
    # exact; only the order of disjoint cudaMalloc ranges changes).
    persist = np.zeros(len(merged))
    for b in blocks:
        if b["kind"] == "persistent" and b["end"] == t_end:
            i = bisect.bisect_right(starts, b["addr"]) - 1
            persist[i] += b["size"]
    share = persist / np.array([m[1] - m[0] for m in merged])
    order = sorted(range(len(merged)), key=lambda i: -share[i])
    base_of = {}
    y = 0
    for i in order:
        base_of[i] = y
        y += merged[i][1] - merged[i][0]
    merged_sorted = [merged[i] for i in order]

    def y_of(addr):
        i = bisect.bisect_right(starts, addr) - 1
        return base_of[i] + (addr - merged[i][0])

    for b in blocks:
        b["y"] = y_of(b["addr"])
    # Reserved over time.
    reserved = np.zeros(t_end + 1, dtype=np.int64)
    for tt, d in seg_events:
        reserved[min(tt, t_end)] += d
    reserved = np.cumsum(reserved)
    return {
        "blocks": blocks,
        "t_end": t_end,
        "reserved_t": reserved,
        "segments": merged_sorted,
        "y_total": int(y),
    }


def load_planned(window_trace, planner_summary, planned_memstat, caching):
    """Arena blocks from the plan (exact offsets, wrap-around ranges); the
    fallback band is total live bytes minus arena live bytes."""
    summ = json.load(open(planner_summary))
    rec = None
    for line in open(planned_memstat):
        r = json.loads(line)
        if r.get("hits", 0) > 0:
            rec = r
    if rec is None:
        raise ValueError("planned memstat has no step with plan hits")
    pool_reserved = rec["pool_reserved_high"]
    arena = rec["arena_size"]
    window_kinds = [
        b["kind"]
        for b in caching["blocks"]
        if not (b["kind"] == "persistent" and b["start"] == 0)
    ]
    # Ordinal time axis from the traced windows A (this step) and B (next).
    events = [
        tuple(map(int, l.split())) for l in open(window_trace) if not l.startswith("#")
    ]
    wa = events[0][0]
    t_all = sum(1 for e in events if e[0] == wa and e[1] == 1 and e[3] >= 0)
    ord_start, ord_free = {}, {}
    seen = defaultdict(int)
    open_by_ptr = {}
    for w, kind, _, ordinal, size, ptr, _ in events:
        if kind == 1:
            if w == wa and ordinal >= 0:
                ord_start[ordinal] = seen[w]
                open_by_ptr[ptr] = ordinal
            else:
                open_by_ptr.pop(ptr, None)
            if ordinal >= 0:
                seen[w] += 1
        else:
            o = open_by_ptr.pop(ptr, None)
            if o is not None:
                ord_free[o] = (w, seen[w])
    t_end = caching["t_end"]
    arena_blocks = []
    arena_live = np.zeros(t_end + 2, dtype=np.int64)
    for o, size, _, off in summ["placements"]:
        if off < 0:
            continue
        kind = window_kinds[o] if o < len(window_kinds) else "other"
        w, k = ord_free[o]
        if w == wa:
            ranges = [(ord_start[o], k)]
        else:
            kind = "escaping"
            ranges = [(ord_start[o], t_all), (0, k)]
        for x0, x1 in ranges:
            if x1 <= x0:
                continue
            arena_blocks.append(
                {
                    "start": x0,
                    "end": x1,
                    "y": pool_reserved + off,
                    "size": size,
                    "kind": kind,
                }
            )
            arena_live[min(x0, t_end + 1)] += size
            arena_live[min(x1, t_end + 1)] -= size
    arena_live = np.cumsum(arena_live)[: t_end + 1]
    total_live = np.zeros(t_end + 2, dtype=np.int64)
    for b in caching["blocks"]:
        total_live[b["start"]] += b["size"]
        total_live[min(b["end"], t_end + 1)] -= b["size"]
    total_live = np.cumsum(total_live)[: t_end + 1]
    return {
        "arena_blocks": arena_blocks,
        "fallback_live_t": total_live - arena_live,
        "pool_reserved": pool_reserved,
        "arena": arena,
        "t_end": t_end,
        "rec": rec,
        "summary": summ,
    }


def _rects(ax, blocks, alpha=0.9):
    by_kind = defaultdict(list)
    for b in blocks:
        x0, x1 = b["start"], max(b["end"], b["start"] + 1)
        y0, y1 = b["y"] / G, (b["y"] + b["size"]) / G
        by_kind[b["kind"]].append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    for kind, verts in by_kind.items():
        pc = PolyCollection(
            np.array(verts),
            facecolors=COLORS[kind],
            edgecolors="none",
            alpha=alpha,
            rasterized=True,
        )
        ax.add_collection(pc)


def _live_curve(blocks, t_end):
    live = np.zeros(t_end + 2, dtype=np.int64)
    for b in blocks:
        live[b["start"]] += b["size"]
        live[min(b["end"], t_end + 1)] -= b["size"]
    return np.cumsum(live)[: t_end + 1]


def plot(caching, planned, out_png, title_c, title_p, ymax):
    fig, axes = plt.subplots(
        2, 2, figsize=(22, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    t_end = caching["t_end"]
    x = np.arange(t_end + 1)
    live = _live_curve(caching["blocks"], t_end)

    ax = axes[0, 0]
    y = 0
    for a, b in caching["segments"]:
        ax.add_patch(
            plt.Rectangle(
                (0, y / G), t_end, (b - a) / G, color="#dfe4ea", lw=0, zorder=0
            )
        )
        y += b - a
    _rects(ax, caching["blocks"])
    ax.axhline(caching["reserved_t"].max() / G, color="k", lw=1.2)
    ax.set_title(title_c)
    ax.set_ylabel(
        "address space (GiB; cudaMalloc ranges stacked,\npersistent-heavy ranges at the bottom)"
    )

    ax = axes[0, 1]
    pr = planned["pool_reserved"]
    ax.add_patch(plt.Rectangle((0, 0), t_end, pr / G, color="#dfe4ea", lw=0, zorder=0))
    ax.fill_between(
        x,
        0,
        planned["fallback_live_t"] / G,
        color=COLORS["persistent"],
        alpha=0.9,
        lw=0,
        rasterized=True,
    )
    ax.add_patch(
        plt.Rectangle(
            (0, pr / G), t_end, planned["arena"] / G, color="#dfe4ea", lw=0, zorder=0
        )
    )
    _rects(ax, planned["arena_blocks"])
    ax.axhline(pr / G, color="#57606f", lw=0.8, ls=":")
    ax.axhline((pr + planned["arena"]) / G, color="k", lw=1.2)
    ax.text(
        t_end * 0.01,
        pr / G * 0.5,
        "fallback pool: persistent tensors\n(cudaMallocAsync; drawn as a filled\n"
        "live-bytes area, not address-resolved)",
        fontsize=9,
        va="center",
    )
    ax.text(
        t_end * 0.01, pr / G, " planned arena (exact offsets)", fontsize=9, va="bottom"
    )
    ax.set_title(title_p)
    for a in axes[0]:
        a.set_ylim(0, ymax / G)
        a.set_xlim(0, t_end)
        a.grid(alpha=0.25)

    # Zoomed curves: live vs reserved.
    lb_total = (planned["fallback_live_t"] + 0).max()
    lo = max(0.0, (live[:t_end].min() - 3 * G) / G)
    hi = ymax / G
    ax = axes[1, 0]
    ax.fill_between(
        x,
        live / G,
        caching["reserved_t"] / G,
        color="#dfe4ea",
        label="reserved - live (fragmentation + cache)",
    )
    ax.plot(x, caching["reserved_t"] / G, color="k", lw=1.2, label="reserved")
    ax.plot(x, live / G, color="#c0392b", lw=1.0, label="live (allocated)")
    ax.set_ylabel("GiB")
    ax.legend(loc="lower right", fontsize=9)
    ax = axes[1, 1]
    planned_res = (pr + planned["arena"]) / G
    ax.fill_between(x, live / G, planned_res, color="#dfe4ea", label="reserved - live")
    ax.axhline(planned_res, color="k", lw=1.2, label="reserved (pool peak + arena)")
    ax.plot(x, live / G, color="#c0392b", lw=1.0, label="live (allocated)")
    ax.legend(loc="lower right", fontsize=9)
    for a in axes[1]:
        a.set_ylim(lo, hi)
        a.set_xlim(0, t_end)
        a.grid(alpha=0.25)
        a.set_xlabel(
            "allocation ordinal within one train step (same sequence in both runs)"
        )
    del lb_total
    handles = [Patch(color=c, label=k) for k, c in COLORS.items()] + [
        Patch(color="#dfe4ea", label="reserved, unallocated")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caching-dir", required=True)
    ap.add_argument("--graph-info", default=None)
    ap.add_argument("--window-trace", required=True)
    ap.add_argument("--planner-summary", required=True)
    ap.add_argument("--planned-memstat", required=True)
    ap.add_argument("--caching-label", default="caching allocator")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    c = load_caching(args.caching_dir, args.graph_info)
    p = load_planned(args.window_trace, args.planner_summary, args.planned_memstat, c)
    ymax = max(c["y_total"], p["pool_reserved"] + p["arena"]) * 1.03
    peak_c = c["reserved_t"].max()
    plot(
        c,
        p,
        args.out,
        f"{args.caching_label}: reserved peak {peak_c / G:.1f} GiB",
        f"planned ({p['summary']['planner']}): reserved peak "
        f"{(p['pool_reserved'] + p['arena']) / G:.1f} GiB",
        ymax,
    )


if __name__ == "__main__":
    main()
