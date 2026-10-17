# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A dynamic-shape pointwise kernel keeps the launch config autotuned at its first dynamic call, for every size.

SwiGLU on routed rows gate_up [R, 2, 512] -> silu(gate) * up. The first call compiles static, the second compiles
dynamic, and the dynamic kernel's XBLOCK/num_warps are autotuned at that second call. At decode sizes (272, 512
rows) all configs tie, so the pick is arbitrary, and it is kept at R=524,288 rows (65,536 tokens x top-8).
Each line is a fresh process with fresh Inductor/Triton cache dirs (the pick is also saved in .best_config files).

Expected on 1x GB300 (see ../README.md, ask 8), us at R = 131,072 / 524,288 rows:
decode first ~111 / ~422, large first ~64 / ~237, static ~64 / ~237.
"""
import os
import subprocess
import sys
import tempfile

import torch

LARGE_ROWS = [131072, 524288]


def time_after_warmup(warmup_rows, static):
    from _common import graph_us, HIDDEN

    def swiglu(gate_up):
        gate, up = gate_up.unbind(-2)
        return torch.nn.functional.silu(gate) * up

    compiled = torch.compile(swiglu, fullgraph=True, dynamic=False if static else None)
    make = lambda num_rows: torch.randn(
        num_rows, 2, HIDDEN, device="cuda", dtype=torch.bfloat16
    )  # noqa: E731
    with torch.no_grad():
        for num_rows in warmup_rows:
            compiled(make(num_rows))
        times = []
        for num_rows in LARGE_ROWS:
            gate_up = make(num_rows)
            times.append(graph_us(lambda: compiled(gate_up)))
    print("ROW " + " ".join(f"{us:7.1f}" for us in times), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["child"]:
        time_after_warmup(
            [int(r) for r in sys.argv[2].split(",") if r], sys.argv[3] == "static"
        )
        sys.exit()
    from _common import header

    header()
    trials = [("decode first (272, 512 rows)", "272,512", "dynamic")] * 3
    trials += [("large first (131072, 114688)", "131072,114688", "dynamic")] * 2
    trials += [("static compile at each size", "", "static")]
    print(
        f"us per call at R = {LARGE_ROWS} rows (CUDA graph), one fresh process per line:"
    )
    for name, warmup, kind in trials:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(
                os.environ,
                TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor",
                TRITON_CACHE_DIR=f"{cache}/triton",
            )
            out = subprocess.run(
                [sys.executable, __file__, "child", warmup, kind],
                env=env,
                capture_output=True,
                text=True,
            ).stdout
        rows = [line[4:] for line in out.splitlines() if line.startswith("ROW")]
        print(f"  {name:30s} {rows[0] if rows else out[-300:]}")
