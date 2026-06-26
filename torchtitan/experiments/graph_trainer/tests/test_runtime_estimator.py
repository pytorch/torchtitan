# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation tests for ``estimate_runtime_original`` (roofline runtime).

These build small dense models (llama3 and qwen3 non-MoE), trace the joint
fwd/loss/bwd graph through ``GraphTrainer``, and check the static roofline
runtime estimate for internal consistency. The roofline is an analytical
lower-bound-ish model (no launch overhead, occupancy, or overlap), so we do NOT
assert a tight match to wall-clock -- we assert structure:

  1. Positivity + fwd/bwd split: total > 0, fwd > 0, bwd > 0, and total == fwd +
     bwd (every costed op is attributed to exactly one region).
  2. Attention is costed: the debug models use the sdpa backend, so the graph
     contains aten SDPA ops (regional_inductor does not compile these away), and
     at least one must carry a positive estimate -- this guards the SDPA
     multi-dtype and flex-HOP costing fixes.
  3. Determinism: two estimates of the same graph are identical.

Run (no pytest in this venv -- use unittest):

    python -m unittest torchtitan.experiments.graph_trainer.tests.\\
        test_runtime_estimator.TestRuntimeEstimator -v
"""

import unittest

import torch

from torchtitan.experiments.graph_trainer.runtime_estimator import (
    estimate_runtime_original,
)
from torchtitan.experiments.graph_trainer.tests.test_memory_estimator import (
    BATCH_SIZE,
    SEQ_LEN,
    _build_trainer,
    _run_step,
    _set_deterministic,
    llama3_registry,
    qwen3_registry,
)

# total == fwd + bwd up to float summation order.
SPLIT_TOL_MS = 1e-6
ATTENTION_MARKERS = ("scaled_dot_product", "flex_attention")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRuntimeEstimator(unittest.TestCase):
    def setUp(self):
        _set_deterministic()
        self.tokens = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")
        self.labels = torch.randint(0, 2048, (BATCH_SIZE, SEQ_LEN), device="cuda")

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _traced_gm(self, registry):
        # passes applied (cudagraph disabled); the sdpa-backend attention ops
        # survive regional_inductor, so they remain costable.
        trainer = _build_trainer(registry)
        _run_step(trainer, self.tokens, self.labels)
        torch.cuda.synchronize()
        return trainer._traced_step.gm

    def _check(self, name, gm):
        res = estimate_runtime_original(gm)
        print(f"\n[{name}] " + res.summary().splitlines()[0])

        # (1) positivity + fwd/bwd split
        self.assertGreater(res.total_runtime_ms, 0.0, f"{name}: total must be > 0")
        self.assertGreater(res.fwd_runtime_ms, 0.0, f"{name}: fwd must be > 0")
        self.assertGreater(res.bwd_runtime_ms, 0.0, f"{name}: bwd must be > 0")
        self.assertAlmostEqual(
            res.total_runtime_ms,
            res.fwd_runtime_ms + res.bwd_runtime_ms,
            delta=SPLIT_TOL_MS,
            msg=f"{name}: total must equal fwd + bwd",
        )

        # (2) attention is costed (guards the SDPA/flex costing fixes)
        attn = {
            k: v
            for k, v in res.node_runtimes_ms.items()
            if any(m in k for m in ATTENTION_MARKERS)
        }
        self.assertTrue(attn, f"{name}: no attention ops were costed")
        self.assertTrue(
            all(v > 0.0 for v in attn.values()),
            f"{name}: an attention op was costed at 0 ({attn})",
        )

        # (3) determinism
        res2 = estimate_runtime_original(gm)
        self.assertEqual(res.total_runtime_ms, res2.total_runtime_ms)

    def test_llama3_runtime(self):
        self._check("llama3", self._traced_gm(llama3_registry))

    def test_qwen3_runtime(self):
        self._check("qwen3", self._traced_gm(qwen3_registry))


if __name__ == "__main__":
    unittest.main()
