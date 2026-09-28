# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import torch
import torch.nn.functional as F

from torchtitan.models.common.rope import apply_rotary_emb_cos_sin


_HAS_HELION = importlib.util.find_spec("helion") is not None

if _HAS_HELION:
    from torchtitan.experiments.helion_kernels.rope import (
        apply_qk_rmsnorm_rotary_emb_cos_sin_helion,
    )


def _reference(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, eps)
    return apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)


@unittest.skipUnless(_HAS_HELION, "requires Helion")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQKRMSNormRoPEHelion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _make_inputs(
        self,
        *,
        batch_size: int = 1,
        seq_len: int = 16,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        head_dim: int = 128,
    ) -> tuple[torch.Tensor, ...]:
        xq = torch.randn(
            batch_size,
            seq_len,
            n_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        xk = torch.randn(
            batch_size,
            seq_len,
            n_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
        k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
        rope_cache = torch.randn(
            seq_len * 2,
            head_dim * 2,
            device="cuda",
            dtype=torch.float32,
        )
        positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).repeat(
            batch_size, 1
        )
        return xq, xk, q_weight, k_weight, rope_cache, positions

    def test_forward_matches_pytorch_reference(self):
        for seq_len in (1, 16, 192):
            with self.subTest(seq_len=seq_len):
                args = self._make_inputs(seq_len=seq_len)
                with torch.inference_mode():
                    actual = apply_qk_rmsnorm_rotary_emb_cos_sin_helion(*args)
                    expected = _reference(*args, eps=1e-5)

                torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=7e-2)
                torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=7e-2)

    def test_training_falls_back_to_autograd_safe_pytorch_path(self):
        args = list(self._make_inputs(seq_len=8))
        for tensor in args[:4]:
            tensor.requires_grad_(True)

        actual = apply_qk_rmsnorm_rotary_emb_cos_sin_helion(*args)
        expected = _reference(*args, eps=1e-5)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=7e-2)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=7e-2)

        (actual[0].sum() + actual[1].sum()).backward()
        for tensor in args[:4]:
            self.assertIsNotNone(tensor.grad)


if __name__ == "__main__":
    unittest.main()
