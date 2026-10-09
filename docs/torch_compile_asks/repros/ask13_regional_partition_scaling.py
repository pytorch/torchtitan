# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""regional_inductor's region partitioning (no Inductor compile) scales superlinearly with tagged nodes.

A synthetic FX graph of L "layers": an untagged matmul, then 8 tagged pointwise ops, with a residual into
the next layer; tagged either with one `inductor_region` id per layer or with one shared id
(GraphTrainer's bridge uses the region name as the id, so production is close to the shared case).
Times only `_RegionScooper.scoop_regions` (CapabilityBasedPartitioner per region id +
fuse_by_partitions), the step that runs before any partition is compiled. CPU only.

Expected (CPU; see ../README.md, ask 13): with one region id per layer, 0.17 / 1.15 / 7.9 / 64 s at
400 / 800 / 1600 / 3200 tagged nodes (~N^2.9); with one shared id, 0.24 / 1.44 / 11 s at 400 / 800 / 1600.
"""
import time

import torch
import torch.fx as fx
from torch.fx.passes.regional_inductor import _RegionScooper

aten = torch.ops.aten
OPS_PER_REGION = 8


def build(layers: int, per_layer_ids: bool = True) -> fx.GraphModule:
    g = fx.Graph()
    x = g.placeholder("x")
    w = g.placeholder("w")
    for i in range(layers):
        h = g.call_function(aten.mm.default, (x, w))
        y = h
        for j in range(OPS_PER_REGION):
            op = (aten.sin.default, aten.cos.default)[j % 2]
            y = (
                g.call_function(op, (y,))
                if j < OPS_PER_REGION - 1
                else g.call_function(aten.add.Tensor, (y, x))
            )
            tag = {"inductor_region": f"layer{i}"} if per_layer_ids else {}
            y.meta["custom"] = {"compile_with_inductor": tag}
        x = y
    g.output(x)
    return fx.GraphModule(torch.nn.Module(), g)


if __name__ == "__main__":
    for per_layer_ids, sizes in (
        (True, (25, 50, 100, 200, 400)),
        (False, (25, 50, 100, 200)),
    ):
        print(
            "one inductor_region per layer"
            if per_layer_ids
            else "one shared region (no inductor_region key)"
        )
        prev = None
        for layers in sizes:
            gm = build(layers, per_layer_ids)
            t0 = time.perf_counter()
            _RegionScooper.scoop_regions(gm)
            dt = time.perf_counter() - t0
            ratio = f"  x{dt / prev:.1f} vs half" if prev else ""
            print(
                f"tagged nodes {layers * OPS_PER_REGION:5d}  "
                f"regions {layers if per_layer_ids else 1:4d}  scoop_regions {dt:8.2f} s{ratio}"
            )
            prev = dt
