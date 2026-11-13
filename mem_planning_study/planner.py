# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline planner for one allocator window (one train step), periodic.

Input: ``window_trace.txt`` written by ``memplan_alloc.so`` in trace mode,
covering two consecutive windows A and B. Blocks are the plannable
allocations of A. A block freed in A occupies one time range; a block freed
in B ("wraps": e.g. gradients freed by the next step's zero_grad) occupies
[alloc, T) of this step plus [0, free) of the next step, which by
periodicity is the same plan. Allocations not freed by the end of B stay on
the fallback allocator.

Output: a plan file for ``memplan_load_plan`` plus a JSON summary.

Planners:
  - ``stalloc``: port of the STAlloc plan synthesizer's default path
    (``allocate_group(None, blocks, group=False)`` in Synthesizer/allocator.py):
    exact-size clusters in descending size; each cluster first tries to fit
    into the free time/space of earlier (larger) layers using a per-layer
    skyline, after STAlloc's "failed-first" reordering trials; the rest form
    new layers with Algorithm 1 (append to the layer whose last block ends
    latest before the new block starts). Wrapping blocks start their own
    layer, which later blocks may fill between the two wrapped ranges; this
    plays the role of STAlloc's cross-iteration interval shifting.
  - ``greedy``: greedy-by-size best fit.

Time is compressed to alloc ranks among plannable allocations, which keeps
every time overlap of the original event sequence.
"""

import argparse
import bisect
import dataclasses
import json
import time
from collections import Counter, defaultdict

import numpy as np

ALIGN = 512


@dataclasses.dataclass
class Block:
    ordinal: int
    size: int  # bytes, 512-aligned
    ranges: list  # [(start, end)] half-open, compressed time in [0, T]
    offset: int = -1

    @property
    def start(self):
        return self.ranges[0][0]

    @property
    def wraps(self):
        return len(self.ranges) == 2


def parse_window_trace(path: str):
    events = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            events.append(tuple(map(int, line.split())))
    # (window, kind, on_thread, ordinal, size, ptr, stream)
    windows = sorted({e[0] for e in events})
    if len(windows) != 2:
        raise ValueError(f"expected 2 traced windows, got {windows}")
    wa, wb = windows
    allocs_a = [e for e in events if e[0] == wa and e[1] == 1 and e[3] >= 0]
    main_stream = Counter(e[6] for e in allocs_a).most_common(1)[0][0]
    num_ordinals = max(e[3] for e in allocs_a) + 1

    # ordinal -> [size, stream, freed_in, free_after_k_allocs_of_that_window]
    info_by_ord = {}
    open_by_ptr = {}
    allocs_seen = {wa: 0, wb: 0}
    for w, kind, _, ordinal, size, ptr, stream in events:
        if kind == 1:
            if ordinal >= 0:
                allocs_seen[w] += 1
            if w == wa and ordinal >= 0:
                info_by_ord[ordinal] = [size, stream, None, None]
                open_by_ptr[ptr] = ordinal
            else:
                open_by_ptr.pop(ptr, None)
        else:
            o = open_by_ptr.pop(ptr, None)
            if o is not None:
                info_by_ord[o][2] = w
                # Window ordinals allocated before this free (same window).
                info_by_ord[o][3] = allocs_seen[w]

    plannable = sorted(
        o
        for o, (_, st, fw, _) in info_by_ord.items()
        if fw is not None and st == main_stream
    )
    # Compressed time: rank among plannable ordinals. A free after k window
    # allocs maps to the number of plannable ordinals < k.
    T = len(plannable)
    blocks = []
    for rank, o in enumerate(plannable):
        size, _, fw, k = info_by_ord[o]
        end = bisect.bisect_left(plannable, k)
        if fw == wa:
            blocks.append(Block(o, size, [(rank, max(end, rank + 1))]))
        else:
            blocks.append(Block(o, size, [(rank, T), (0, end)]))
    for b in blocks:
        b.ranges = [r for r in b.ranges if r[1] > r[0]]
        b.ranges.sort(key=lambda r: -r[0])  # main (allocation) range first
    not_freed = [o for o, v in info_by_ord.items() if v[2] is None]
    other_stream = [
        o for o, v in info_by_ord.items() if v[2] is not None and v[1] != main_stream
    ]
    wraps = [b for b in blocks if b.wraps]
    info = {
        "num_events": len(events),
        "num_ordinals": num_ordinals,
        "num_plannable": len(blocks),
        "num_wrapping": len(wraps),
        "wrapping_bytes": int(sum(b.size for b in wraps)),
        "num_not_freed": len(not_freed),
        "not_freed_bytes": int(sum(info_by_ord[o][0] for o in not_freed)),
        "num_other_stream": len(other_stream),
        "num_distinct_sizes": len({b.size for b in blocks}),
        "num_non_window_thread_allocs": sum(
            1 for e in events if e[0] == wa and e[1] == 1 and e[3] < 0
        ),
    }
    return blocks, num_ordinals, T, info


def liveness_profile(blocks, num_t):
    delta = np.zeros(num_t + 1, dtype=np.int64)
    for b in blocks:
        for s, e in b.ranges:
            delta[s] += b.size
            delta[e] -= b.size
    return np.cumsum(delta)[:num_t]


def validate(blocks):
    """Assert no two blocks overlap in both time and address."""
    rows = sorted(
        (s, e, b.offset, b.size, i) for i, b in enumerate(blocks) for s, e in b.ranges
    )
    a = np.array(rows, dtype=np.int64)
    starts, ends, offs, sizes, ids = a.T
    assert (offs >= 0).all()
    for i in range(len(a)):
        j_end = np.searchsorted(starts, ends[i], side="left")
        if j_end <= i + 1:
            continue
        sl = slice(i + 1, j_end)
        bad = (
            (offs[sl] < offs[i] + sizes[i])
            & (offs[i] < offs[sl] + sizes[sl])
            & (ids[sl] != ids[i])
        )
        if bad.any():
            j = i + 1 + int(np.argmax(bad))
            raise AssertionError(f"overlap: {blocks[ids[i]]} vs {blocks[ids[j]]}")


# ----------------------------------------------------------------------------
# Greedy-by-size best fit
# ----------------------------------------------------------------------------


def plan_greedy(blocks):
    order = sorted(
        blocks,
        key=lambda b: (-b.size, -sum(e - s for s, e in b.ranges), b.start),
    )
    n = sum(len(b.ranges) for b in blocks)
    ps = np.empty(n, dtype=np.int64)
    pe = np.empty(n, dtype=np.int64)
    po = np.empty(n, dtype=np.int64)
    pz = np.empty(n, dtype=np.int64)
    k = 0
    top = 0
    for b in order:
        m = np.zeros(k, dtype=bool)
        for s, e in b.ranges:
            m |= (ps[:k] < e) & (s < pe[:k])
        offset = 0
        if m.any():
            o = po[:k][m]
            z = pz[:k][m]
            idx = np.argsort(o, kind="stable")
            o, z = o[idx], z[idx]
            ends = np.maximum.accumulate(o + z)
            gaps_start = np.concatenate(([0], ends[:-1]))
            fit = np.nonzero(o - gaps_start >= b.size)[0]
            offset = int(gaps_start[fit[0]]) if len(fit) else int(ends[-1])
        b.offset = offset
        for s, e in b.ranges:
            ps[k], pe[k], po[k], pz[k] = s, e, offset, b.size
            k += 1
        top = max(top, offset + b.size)
    return top


# ----------------------------------------------------------------------------
# STAlloc default path
# ----------------------------------------------------------------------------


def form_layer(cluster):
    """Algorithm 1 with periodic blocks.

    ``cluster`` is sorted by start. A wrapping block starts its own layer,
    which a non-wrapping block may join if it fits between the wrapped
    block's early range end (tail) and late range start (limit). Otherwise a
    block joins the normal layer whose last block ends latest but no later
    than the block's start.
    """
    layers = []
    tails = []  # normal layers: sorted (tail, layer_idx)
    wrap_layers = []  # [tail, limit, layer_idx]
    for b in cluster:
        if b.wraps:
            late, early = b.ranges[0], b.ranges[1] if len(b.ranges) > 1 else (0, 0)
            wrap_layers.append([early[1], late[0], len(layers)])
            layers.append([b])
            continue
        s, e = b.ranges[0]
        best_tail, best = -1, None
        i = bisect.bisect_right(tails, (s, float("inf"))) - 1
        if i >= 0:
            best_tail, best = tails[i][0], ("n", i)
        for wl in wrap_layers:
            if wl[0] <= s and e <= wl[1] and wl[0] > best_tail:
                best_tail, best = wl[0], ("w", wl)
        if best is None:
            li = len(layers)
            layers.append([b])
            bisect.insort(tails, (e, li))
        elif best[0] == "n":
            _, li = tails.pop(best[1])
            layers[li].append(b)
            bisect.insort(tails, (e, li))
        else:
            wl = best[1]
            layers[wl[2]].append(b)
            wl[0] = e
    return layers


class _Layer:
    __slots__ = ("offset", "size", "sky")

    def __init__(self, offset, size, num_t):
        self.offset = offset
        self.size = size
        self.sky = np.zeros(num_t, dtype=np.int32)  # ALIGN units


def _try_place(layers, blocks, size_u, *, commit):
    placed = np.zeros(len(blocks), dtype=bool)
    nr = max(len(b.ranges) for b in blocks)
    S = np.zeros((len(blocks), nr), dtype=np.int64)
    E = np.zeros((len(blocks), nr), dtype=np.int64)
    for j, b in enumerate(blocks):
        for r, (s, e) in enumerate(b.ranges):
            S[j, r], E[j, r] = s, e
    for layer in layers:
        if layer.size < size_u:
            continue
        remaining = np.nonzero(~placed)[0]
        if len(remaining) == 0:
            break
        bad = (layer.size - layer.sky) < size_u
        bad_prefix = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
        nbad = (bad_prefix[E[remaining]] - bad_prefix[S[remaining]]).sum(axis=1)
        cand = remaining[nbad == 0]
        if len(cand) == 0:
            continue
        sky = layer.sky if commit else layer.sky.copy()
        for j in cand:
            rs = blocks[j].ranges
            h = max(int(sky[s:e].max()) for s, e in rs)
            if h + size_u <= layer.size:
                for s, e in rs:
                    sky[s:e] = h + size_u
                placed[j] = True
                if commit:
                    blocks[j].offset = (layer.offset + h) * ALIGN
    return placed


def plan_stalloc(blocks, num_t, *, reorder_trials=6):
    clusters = defaultdict(list)
    for b in blocks:
        clusters[b.size].append(b)
    layers: list[_Layer] = []
    top_u = 0
    for size in sorted(clusters, reverse=True):
        size_u = size // ALIGN
        cluster = sorted(clusters[size], key=lambda b: b.start)
        order = [b for layer in form_layer(cluster) for b in layer]
        if layers and reorder_trials:
            for _ in range(reorder_trials):
                placed = _try_place(layers, order, size_u, commit=False)
                if placed.all() or not placed.any():
                    break
                new_order = [b for b, p in zip(order, placed) if not p] + [
                    b for b, p in zip(order, placed) if p
                ]
                if [b.ordinal for b in new_order] == [b.ordinal for b in order]:
                    break
                order = new_order
        placed = (
            _try_place(layers, order, size_u, commit=True)
            if layers
            else np.zeros(len(order), bool)
        )
        remaining = sorted(
            (b for b, p in zip(order, placed) if not p), key=lambda b: b.start
        )
        for group in form_layer(remaining):
            layer = _Layer(top_u, size_u, num_t)
            for b in group:
                b.offset = top_u * ALIGN
                for s, e in b.ranges:
                    layer.sky[s:e] = size_u
            layers.append(layer)
            top_u += size_u
    return top_u * ALIGN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--planner", choices=["stalloc", "greedy"], default="stalloc")
    ap.add_argument("--plan-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    blocks, num_ordinals, num_t, info = parse_window_trace(args.trace)
    live = liveness_profile(blocks, num_t)
    lower_bound = int(live.max())
    t1 = time.perf_counter()
    arena = (
        plan_stalloc(blocks, num_t)
        if args.planner == "stalloc"
        else plan_greedy(blocks)
    )
    t2 = time.perf_counter()
    validate(blocks)
    t3 = time.perf_counter()

    sizes_by_ord = {}
    with open(args.trace) as f:
        first_window = None
        for line in f:
            if line.startswith("#"):
                continue
            w, kind, _, ordinal, size, _, _ = map(int, line.split())
            first_window = w if first_window is None else first_window
            if w == first_window and kind == 1 and ordinal >= 0:
                sizes_by_ord[ordinal] = size
    plan = [[sizes_by_ord[o], -1] for o in range(num_ordinals)]
    for b in blocks:
        plan[b.ordinal][1] = b.offset
    with open(args.plan_out, "w") as f:
        f.write(f"{arena} {num_ordinals}\n")
        for size, off in plan:
            f.write(f"{size} {off}\n")

    summary = {
        **info,
        "planner": args.planner,
        "num_t": num_t,
        "lower_bound_bytes": lower_bound,
        "arena_bytes": int(arena),
        "packing_overhead": arena / lower_bound - 1,
        "plan_s": round(t2 - t1, 1),
        "validate_s": round(t3 - t2, 1),
        "placements": [[b.ordinal, b.size, b.ranges, b.offset] for b in blocks],
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f)
    G = 1024**3
    print(
        f"{args.planner}: blocks={len(blocks)} (wrapping {info['num_wrapping']}, "
        f"{info['wrapping_bytes'] / G:.2f}GiB) sizes={info['num_distinct_sizes']} "
        f"LB={lower_bound / G:.2f}GiB arena={arena / G:.2f}GiB "
        f"overhead={100 * (arena / lower_bound - 1):.2f}% plan={t2 - t1:.1f}s "
        f"not_freed={info['num_not_freed']} ({info['not_freed_bytes'] / G:.2f}GiB) "
        f"other_stream={info['num_other_stream']}"
    )


if __name__ == "__main__":
    main()
