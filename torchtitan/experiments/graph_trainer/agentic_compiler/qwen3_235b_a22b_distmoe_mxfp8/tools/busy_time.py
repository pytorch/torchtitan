# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU busy time of the profiled step: the union of kernel intervals on all streams.

    python busy_time.py <run dir> [<run dir> ...]

Reads <run>/analysis/{kernels,node_table}.pkl (fusion_finder parse_trace.py + build_node_table.py) and
prints busy ms with and without the kernels of fake_comm nodes (copies that only emulate collectives under the
fake process group), plus the plain sum of kernel durations, which double-counts PDL overlap.
"""

import sys

import pandas as pd


def union_ms(df):
    total, end = 0.0, None
    for ts, dur in sorted(zip(df.ts, df.dur)):
        if end is None or ts > end:
            total += dur
            end = ts + dur
        elif ts + dur > end:
            total += ts + dur - end
            end = ts + dur
    return total / 1000


for run in sys.argv[1:]:
    k = pd.read_pickle(f"{run}/analysis/kernels.pkl")
    t = pd.read_pickle(f"{run}/analysis/node_table.pkl")
    comm_nodes = set(t.name[t.category == "fake_comm"])
    is_comm = k.node.isin(comm_nodes)
    print(
        f"{run}: busy {union_ms(k):.1f} ms, busy excl fake comm {union_ms(k[~is_comm]):.1f} ms, "
        f"sum {k.dur.sum() / 1000:.1f} ms, sum excl fake comm {k.dur[~is_comm].sum() / 1000:.1f} ms"
    )
