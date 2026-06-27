# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Show ``estimate_transfertime`` vs real ``.copy_()`` across tensor sizes.

Builds fake FX nodes whose ``meta['val']`` tensors span 1 MB .. 256 MB, runs
``estimate_transfertime`` through its normal entry point, and prints the
predicted transfer time next to the time of a real ``.copy_()`` of the same
size, in both H2D and D2H directions. The numbers are printed for the reviewer
to interpret; the test only asserts the estimator ran and returned positive
values.

Run (no pytest in this venv -- use unittest):

    python -m unittest torchtitan.experiments.graph_trainer.tests.\\
        test_transfer_time.TestTransferTimeMeasured -v
"""

import time
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torchtitan.experiments.graph_trainer.transfertime_estimator import (
    estimate_transfertime,
    measure_transfer_bw,
)

DTYPE = torch.bfloat16
SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ITERS = 30


def _itemsize(dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def _numel_for_mb(mb: int, dtype) -> int:
    return mb * 1024 * 1024 // _itemsize(dtype)


def _build_fake_gm(numels, dtype=DTYPE, device="cuda"):
    """A linear chain of ``clone`` nodes whose outputs are fake tensors of the
    given sizes -- just enough graph for ``estimate_transfertime`` to walk.

    Returns (gm, producer_nodes) where producer_nodes[i] produces a 1-D tensor of
    numels[i] elements. The graph is never executed; only node.meta['val'] is
    read, so the op identity (clone) is irrelevant.
    """
    g = torch.fx.Graph()
    fake = FakeTensorMode()
    x = g.placeholder("x")
    with fake:
        x.meta["val"] = torch.empty(1, dtype=dtype, device=device)
    prev, nodes = x, []
    for n_elem in numels:
        node = g.call_function(torch.ops.aten.clone.default, (prev,))
        with fake:
            node.meta["val"] = torch.empty(n_elem, dtype=dtype, device=device)
        nodes.append(node)
        prev = node
    g.output((prev,))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    return gm, nodes


def _real_copy_ms(numel: int, direction: str, dtype=DTYPE, iters=ITERS) -> float:
    """Mean time (ms) of a single ``.copy_()`` of ``numel`` elements, pinned host
    memory, over ``iters`` iterations after a warmup."""
    gpu = torch.empty(numel, dtype=dtype, device="cuda")
    host = torch.empty(numel, dtype=dtype, pin_memory=True)
    dst, src = (gpu, host) if direction == "h2d" else (host, gpu)

    dst.copy_(src, non_blocking=True)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTransferTimeMeasured(unittest.TestCase):
    def _run_direction(self, direction: str):
        bw = measure_transfer_bw()  # real device bandwidth
        sizes_mb = SIZES_MB
        numels = [_numel_for_mb(mb, DTYPE) for mb in sizes_mb]
        gm, nodes = _build_fake_gm(numels)
        res = estimate_transfertime(gm, bw_h2d=bw["h2d"], bw_d2h=bw["d2h"])

        # inputs are RELOADED (H2D); outputs are OFFLOADED (D2H). For an offload we
        # read the node's own produced tensor (clean per-size mapping); for a
        # reload we read the predecessor tensor feeding the next node.
        if direction == "d2h":
            per_node = res.output_tensor_transfer_times_ms
            pred = [per_node[n.name][0].offload_time_ms for n in nodes]
        else:  # h2d
            per_node = res.input_tensor_transfer_times_ms
            pred = [
                per_node[nodes[i].name][0].offload_time_ms for i in range(1, len(nodes))
            ]
            sizes_mb = sizes_mb[:-1]  # predecessor sizes feeding nodes[1:]
            numels = numels[:-1]

        meas = [_real_copy_ms(n, direction) for n in numels]

        print(f"\n[{direction}] bw={bw[direction]:.1f} GB/s")
        print(f"  {'MB':>4} {'pred_ms':>9} {'meas_ms':>9} {'pred/meas':>10}")
        for mb, p, mss in zip(sizes_mb, pred, meas):
            r = p / mss if mss else float("nan")
            print(f"  {mb:>4} {p:>9.4f} {mss:>9.4f} {r:>10.3f}")

        # only sanity: the estimator ran and returned positive times.
        for p in pred:
            self.assertGreater(p, 0.0)
        for m in meas:
            self.assertGreater(m, 0.0)

    def test_h2d_predicted_vs_measured(self):
        self._run_direction("h2d")

    def test_d2h_predicted_vs_measured(self):
        self._run_direction("d2h")


if __name__ == "__main__":
    unittest.main()
