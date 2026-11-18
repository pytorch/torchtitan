# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Accelerator checkpoint interoperability tests for fused SwiGLU."""

import unittest

import torch

from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.tools.utils import device_type
from torchtitan_recipes.overrides.fused_swiglu import fused_swiglu

_DIM = 16
_HIDDEN = 32
_DEVICE = torch.accelerator.current_accelerator()
# The ops register for torchtitan's detected device type, which can differ from
# torch.accelerator (e.g. MPS), so skip unless the two agree.
_HAS_OP_DEVICE = _DEVICE is not None and _DEVICE.type == device_type


def _build_fused() -> DenseFeedForward:
    fused = _feed_forward_config()
    fused.activation_fn = fused_swiglu(fused.activation_fn)
    fused = fused.build()
    with torch.no_grad():
        fused.w13.weight.copy_(torch.randn(2, _HIDDEN, _DIM))
        fused.w2.weight.copy_(torch.randn(_DIM, _HIDDEN))
    return fused


def _logical_w13(feed_forward: DenseFeedForward) -> torch.Tensor:
    return feed_forward.w13.weight


def _feed_forward_config() -> DenseFeedForward.Config:
    return DenseFeedForward.Config(
        w13=Linear.Config(in_features=_DIM, out_features=_HIDDEN, num_linears=2),
        w2=Linear.Config(in_features=_HIDDEN, out_features=_DIM),
    )


def _build_native() -> DenseFeedForward:
    native = _feed_forward_config().build()
    with torch.no_grad():
        for p in native.parameters():
            p.copy_(torch.randn_like(p))
    return native


class TestFusedSwiGLU(unittest.TestCase):
    @unittest.skipUnless(_HAS_OP_DEVICE, f"requires a {device_type} accelerator")
    def test_triton_checkpoint_loads_into_native(self):
        fused = _build_fused().to(_DEVICE)
        native = _build_native().to(_DEVICE)
        native.load_state_dict(fused.state_dict())
        self.assertTrue(torch.equal(_logical_w13(native)[0], _logical_w13(fused)[0]))
        self.assertTrue(torch.equal(_logical_w13(native)[1], _logical_w13(fused)[1]))
        self.assertTrue(torch.equal(native.w2.weight, fused.w2.weight))
        x = torch.randn(4, _DIM, device=_DEVICE)
        self.assertTrue(torch.allclose(fused(x), native(x), atol=1e-4, rtol=1e-5))

    @unittest.skipUnless(_HAS_OP_DEVICE, f"requires a {device_type} accelerator")
    def test_native_checkpoint_loads_into_triton(self):
        native = _build_native().to(_DEVICE)
        fused = _build_fused().to(_DEVICE)
        fused.load_state_dict(native.state_dict())
        self.assertTrue(torch.equal(_logical_w13(fused)[0], _logical_w13(native)[0]))
        self.assertTrue(torch.equal(_logical_w13(fused)[1], _logical_w13(native)[1]))
        self.assertTrue(torch.equal(fused.w2.weight, native.w2.weight))
        x = torch.randn(4, _DIM, device=_DEVICE)
        self.assertTrue(torch.allclose(fused(x), native(x), atol=1e-4, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
