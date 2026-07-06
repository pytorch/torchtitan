# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that in-place writes to FusedQKVLinear's state_dict entries propagate
back to the fused parameter only after load_state_dict is called.

This validates the fix for the TorchStore weight-sync bug: TorchStore's
get_state_dict writes updated weights in-place into the user_state_dict, but
FusedQKVLinear's state_dict post-hook returns .contiguous() copies that don't
share storage with the fused wqkv parameter. Without a subsequent
load_state_dict call, the model's live weights are never updated.

All tests run on CPU.
"""

import unittest

import torch
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.common.nn_modules import Linear

_DIM = 16
_N_HEADS = 4
_N_KV_HEADS = 2
_HEAD_DIM = 8
_HPK = _N_HEADS // _N_KV_HEADS
_R_DIM = _HPK + 2
_WQKV_OUT = (_N_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM


def _build_fused() -> FusedQKVLinear:
    fused = FusedQKVLinear.Config(
        head_dim=_HEAD_DIM,
        n_heads=_N_HEADS,
        n_kv_heads=_N_KV_HEADS,
        wqkv=Linear.Config(in_features=_DIM, out_features=_WQKV_OUT, bias=False),
    ).build()
    with torch.no_grad():
        fused.wqkv.weight.copy_(torch.randn(_WQKV_OUT, _DIM))
    return fused


class TestFusedQKVInplaceWrite(unittest.TestCase):
    """Simulate TorchStore's in-place weight update pattern."""

    def test_state_dict_entries_do_not_share_storage_with_fused_param(self):
        """state_dict() post-hook produces detached copies (the bug precondition)."""
        fused = _build_fused()
        sd = fused.state_dict()

        # The split hook produces wq/wk/wv keys, not wqkv
        self.assertIn("wq.weight", sd)
        self.assertIn("wk.weight", sd)
        self.assertIn("wv.weight", sd)
        self.assertNotIn("wqkv.weight", sd)

        # None of the split tensors share storage with the fused parameter
        fused_ptr = fused.wqkv.weight.data_ptr()
        fused_end = fused_ptr + fused.wqkv.weight.nelement() * fused.wqkv.weight.element_size()
        for key in ("wq.weight", "wk.weight", "wv.weight"):
            t = sd[key]
            t_ptr = t.data_ptr()
            t_end = t_ptr + t.nelement() * t.element_size()
            overlaps = not (t_end <= fused_ptr or t_ptr >= fused_end)
            self.assertFalse(
                overlaps,
                f"{key} should NOT share storage with wqkv (post-hook returns .contiguous() copies)",
            )

    def test_inplace_write_without_load_does_not_update_model(self):
        """Simulates TorchStore bug: in-place copy into state_dict without
        load_state_dict leaves the fused parameter unchanged."""
        fused = _build_fused()
        original_wqkv = fused.wqkv.weight.clone()

        sd = fused.state_dict()

        # Simulate TorchStore's in-place write (copy_ new data into sd tensors)
        new_wq = torch.randn_like(sd["wq.weight"])
        new_wk = torch.randn_like(sd["wk.weight"])
        new_wv = torch.randn_like(sd["wv.weight"])
        sd["wq.weight"].copy_(new_wq)
        sd["wk.weight"].copy_(new_wk)
        sd["wv.weight"].copy_(new_wv)

        # BUG: the fused parameter is unchanged because sd entries are detached
        self.assertTrue(
            torch.equal(fused.wqkv.weight, original_wqkv),
            "In-place write to state_dict should NOT affect fused param (no shared storage)",
        )

    def test_inplace_write_with_load_updates_model(self):
        """The fix: load_state_dict after in-place write triggers merge hooks."""
        fused = _build_fused()
        original_wqkv = fused.wqkv.weight.clone()

        sd = fused.state_dict()

        # Simulate TorchStore's in-place write
        new_wq = torch.randn_like(sd["wq.weight"])
        new_wk = torch.randn_like(sd["wk.weight"])
        new_wv = torch.randn_like(sd["wv.weight"])
        sd["wq.weight"].copy_(new_wq)
        sd["wk.weight"].copy_(new_wk)
        sd["wv.weight"].copy_(new_wv)

        # FIX: load_state_dict triggers _merge_qkv_on_load
        fused.load_state_dict(sd, strict=False)

        # The fused parameter should now reflect the new weights
        self.assertFalse(
            torch.equal(fused.wqkv.weight, original_wqkv),
            "After load_state_dict, fused param must be updated",
        )

        # Verify the merge is correct: re-extract and compare
        sd_after = fused.state_dict()
        torch.testing.assert_close(sd_after["wq.weight"], new_wq)
        torch.testing.assert_close(sd_after["wk.weight"], new_wk)
        torch.testing.assert_close(sd_after["wv.weight"], new_wv)

    def test_forward_reflects_updated_weights(self):
        """After the fix, forward() uses the new weights."""
        fused = _build_fused()
        x = torch.randn(1, 2, _DIM)

        # Get output with original weights
        xq_before, xk_before, xv_before = fused(x)

        # Simulate TorchStore update + fix
        sd = fused.state_dict()
        sd["wq.weight"].copy_(torch.randn_like(sd["wq.weight"]))
        sd["wk.weight"].copy_(torch.randn_like(sd["wk.weight"]))
        sd["wv.weight"].copy_(torch.randn_like(sd["wv.weight"]))
        fused.load_state_dict(sd, strict=False)

        # Forward must produce different output
        xq_after, xk_after, xv_after = fused(x)
        self.assertFalse(torch.equal(xq_before, xq_after))
        self.assertFalse(torch.equal(xk_before, xk_after))
        self.assertFalse(torch.equal(xv_before, xv_after))

        # And the output must match a fresh matmul with the new weights
        ref_q = (x @ sd["wq.weight"].T).view(1, 2, _N_HEADS, _HEAD_DIM)
        ref_k = (x @ sd["wk.weight"].T).view(1, 2, _N_KV_HEADS, _HEAD_DIM)
        ref_v = (x @ sd["wv.weight"].T).view(1, 2, _N_KV_HEADS, _HEAD_DIM)
        torch.testing.assert_close(xq_after, ref_q)
        torch.testing.assert_close(xk_after, ref_k)
        torch.testing.assert_close(xv_after, ref_v)


if __name__ == "__main__":
    unittest.main()
