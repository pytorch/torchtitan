# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint interop tests for the FusedSwiGLU override (issue #3569).

FusedSwiGLU stores a single fused ``w13`` parameter but checkpoints in the stock
``FeedForward`` layout (``w1.weight`` / ``w3.weight``) via state_dict hooks, so
its checkpoints round-trip with the non-fused module. These run on CPU.
"""

import unittest

import torch

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.overrides.fused_swiglu import FusedSwiGLU

_DIM = 16
_HIDDEN = 32


def _build_fused() -> FusedSwiGLU:
    fused = FusedSwiGLU.Config(
        dim=_DIM,
        hidden_dim=_HIDDEN,
        w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
    ).build()
    with torch.no_grad():
        fused.w13.copy_(torch.randn(_HIDDEN, 2, _DIM))
        fused.w2.weight.copy_(torch.randn(_DIM, _HIDDEN))
    return fused


def _build_stock() -> FeedForward:
    stock = FeedForward.Config(
        w1=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
        w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
        w3=Linear.Config(in_features=_DIM, out_features=_HIDDEN),
    ).build()
    with torch.no_grad():
        for p in stock.parameters():
            p.copy_(torch.randn_like(p))
    return stock


class TestFusedSwiGLUCheckpointInterop(unittest.TestCase):
    def test_saves_in_stock_layout(self):
        """state_dict() emits the stock w1/w3 layout, not the fused w13."""
        fused = _build_fused()
        sd = fused.state_dict()
        self.assertEqual(set(sd), {"w1.weight", "w3.weight", "w2.weight"})
        self.assertTrue(torch.equal(sd["w1.weight"], fused.w13[:, 0]))
        self.assertTrue(torch.equal(sd["w3.weight"], fused.w13[:, 1]))

    def test_fused_checkpoint_loads_into_stock(self):
        """A fused checkpoint loads into the stock FeedForward, weights + output."""
        fused = _build_fused()
        stock = _build_stock()
        stock.load_state_dict(fused.state_dict())
        self.assertTrue(torch.equal(stock.w1.weight, fused.w13[:, 0]))
        self.assertTrue(torch.equal(stock.w3.weight, fused.w13[:, 1]))
        self.assertTrue(torch.equal(stock.w2.weight, fused.w2.weight))
        x = torch.randn(4, _DIM)
        self.assertTrue(torch.allclose(fused(x), stock(x), atol=1e-5, rtol=1e-5))

    def test_stock_checkpoint_loads_into_fused(self):
        """A stock checkpoint loads into FusedSwiGLU, weights + output."""
        stock = _build_stock()
        fused = _build_fused()
        fused.load_state_dict(stock.state_dict())
        self.assertTrue(torch.equal(fused.w13[:, 0], stock.w1.weight))
        self.assertTrue(torch.equal(fused.w13[:, 1], stock.w3.weight))
        self.assertTrue(torch.equal(fused.w2.weight, stock.w2.weight))
        x = torch.randn(4, _DIM)
        self.assertTrue(torch.allclose(fused(x), stock(x), atol=1e-5, rtol=1e-5))

    def test_fused_roundtrip(self):
        """fused -> save -> load into a fresh fused preserves w13 exactly."""
        src = _build_fused()
        dst = _build_fused()
        dst.load_state_dict(src.state_dict())
        self.assertTrue(torch.equal(dst.w13, src.w13))
        self.assertTrue(torch.equal(dst.w2.weight, src.w2.weight))

    def test_loads_native_w13(self):
        """A legacy checkpoint keyed by the native w13 still loads (back-compat)."""
        src = _build_fused()
        native = {
            "w13": src.w13.detach().clone(),
            "w2.weight": src.w2.weight.detach().clone(),
        }
        dst = _build_fused()
        dst.load_state_dict(native)
        self.assertTrue(torch.equal(dst.w13, src.w13))

    def test_strict_load_reports_missing(self):
        """strict load still flags a genuinely incomplete checkpoint."""
        fused = _build_fused()
        with self.assertRaises(RuntimeError):
            fused.load_state_dict({"w2.weight": fused.w2.weight.detach().clone()})


if __name__ == "__main__":
    unittest.main()
