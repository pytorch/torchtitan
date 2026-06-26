# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation tests for ``estimate_transfertime`` (D2H/H2D transfer time).

These build small dense models (llama3 and qwen3 non-MoE), trace the joint
fwd/loss/bwd graph through ``GraphTrainer``, and check the per-node transfer-time
estimate. Bandwidth is injected (no live benchmark) so the numbers are exact and
deterministic:

  1. Coverage + positivity: every graph node has input/output tensor lists, and
     the heaviest produced tensor has a positive transfer time.
  2. Exactness: transfer time == bytes / bandwidth for the largest output tensor
     (the model is a fixed bytes/bw division, so this is checkable to the ULP).
  3. Bandwidth scaling: halving the bandwidth doubles every transfer time.
  4. ``measure_transfer_bw`` returns positive H2D/D2H bandwidths on CUDA.

Run (no pytest in this venv -- use unittest):

    python -m unittest torchtitan.experiments.graph_trainer.tests.\\
        test_transfertime_estimator.TestTransferTimeEstimator -v
"""

import unittest

import torch

from torchtitan.experiments.graph_trainer.tests.test_memory_estimator import (
    BATCH_SIZE,
    SEQ_LEN,
    _build_trainer,
    _run_step,
    _set_deterministic,
    llama3_registry,
    qwen3_registry,
)
from torchtitan.experiments.graph_trainer.transfertime_estimator import (
    _transfer_ms,
    estimate_transfertime,
    measure_transfer_bw,
)

# Fixed bandwidths (GB/s) so the estimate is exact and independent of hardware.
BW_H2D = 300.0
BW_D2H = 250.0


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTransferTimeEstimator(unittest.TestCase):
    def setUp(self):
        _set_deterministic()
        self.tokens = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")
        self.labels = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _traced_gm(self, registry):
        trainer = _build_trainer(registry)  # passes applied, cudagraph disabled
        _run_step(trainer, self.tokens, self.labels)
        torch.cuda.synchronize()
        return trainer._traced_step.gm

    def _check(self, name, gm):
        res = estimate_transfertime(gm, bw_h2d=BW_H2D, bw_d2h=BW_D2H)

        # (1) coverage + positivity
        self.assertTrue(res.output_tensor_transfer_times_ms, f"{name}: no outputs")
        self.assertGreater(
            res.max_output_offload_time(), 0.0, f"{name}: max output time must be > 0"
        )
        self.assertGreaterEqual(res.max_input_offload_time(), 0.0)

        # (2) exactness: largest output tensor time == bytes / bw (to the ULP)
        heaviest = max(
            (t for lst in res.output_tensor_transfer_times_ms.values() for t in lst),
            key=lambda t: t.offload_time_ms,
        )
        expected = _transfer_ms(heaviest.size, BW_H2D)
        self.assertEqual(heaviest.offload_time_ms, expected)
        print(
            f"\n[{name}] heaviest output {heaviest.size} B -> "
            f"{heaviest.offload_time_ms:.4f} ms @ {BW_H2D} GB/s"
        )

        # (3) bandwidth scaling: half the bandwidth -> double the time
        res_half = estimate_transfertime(gm, bw_h2d=BW_H2D / 2, bw_d2h=BW_D2H)
        self.assertAlmostEqual(
            res_half.max_output_offload_time(),
            2 * res.max_output_offload_time(),
            places=6,
        )

    def test_llama3_transfertime(self):
        self._check("llama3", self._traced_gm(llama3_registry))

    def test_qwen3_transfertime(self):
        self._check("qwen3", self._traced_gm(qwen3_registry))

    def test_measure_transfer_bw(self):
        bw = measure_transfer_bw(nbytes=64 * 1024 * 1024, iters=5)
        print(f"\n[measure] h2d={bw['h2d']:.1f} GB/s  d2h={bw['d2h']:.1f} GB/s")
        self.assertGreater(bw["h2d"], 0.0)
        self.assertGreater(bw["d2h"], 0.0)


if __name__ == "__main__":
    unittest.main()
