# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import torch


_HAS_HELION = importlib.util.find_spec("helion") is not None

if _HAS_HELION:
    from torchtitan.experiments.helion_kernels.rope import (
        apply_rotary_emb_cos_sin_helion,
    )


def _apply_rotary_emb_cos_sin_reference(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    rope_cache = rope_cache[None, :, None, :].expand(xq.shape[0], -1, -1, -1)
    rope_cache = torch.gather(
        rope_cache,
        dim=1,
        index=positions.view(xq.shape[0], xq.shape[1], 1, 1).expand(
            xq.shape[0], xq.shape[1], 1, head_dim * 2
        ),
    )
    cos = rope_cache[..., :head_dim]
    sin = rope_cache[..., head_dim:]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    xq_out = (xq.float() * cos) + (rotate_half(xq.float()) * sin)
    xk_out = (xk.float() * cos) + (rotate_half(xk.float()) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@unittest.skipUnless(_HAS_HELION, "requires Helion")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestApplyRotaryEmbCosSinHelion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bsz = 2
        self.seqlen = 16
        self.n_heads = 4
        self.n_kv_heads = 2
        self.head_dim = 64
        self.xq = torch.randn(
            self.bsz,
            self.seqlen,
            self.n_heads,
            self.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.xk = torch.randn(
            self.bsz,
            self.seqlen,
            self.n_kv_heads,
            self.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.rope_cache = torch.randn(
            self.seqlen * 2,
            self.head_dim * 2,
            device="cuda",
            dtype=torch.float32,
        )
        self.positions = torch.arange(
            self.seqlen, device="cuda", dtype=torch.int32
        ).repeat(self.bsz, 1)

    def test_forward_matches_pytorch_reference(self):
        xq_out, xk_out = apply_rotary_emb_cos_sin_helion(
            self.xq, self.xk, self.rope_cache, self.positions
        )
        xq_ref, xk_ref = _apply_rotary_emb_cos_sin_reference(
            self.xq, self.xk, self.rope_cache, self.positions
        )

        torch.testing.assert_close(xq_out, xq_ref, rtol=0, atol=1e-5)
        torch.testing.assert_close(xk_out, xk_ref, rtol=0, atol=1e-5)

    def test_backward_matches_pytorch_reference(self):
        xq = self.xq.detach().clone().requires_grad_(True)
        xk = self.xk.detach().clone().requires_grad_(True)
        xq_ref = self.xq.detach().clone().requires_grad_(True)
        xk_ref = self.xk.detach().clone().requires_grad_(True)

        grad_xq = torch.randn_like(xq)
        grad_xk = torch.randn_like(xk)

        xq_out, xk_out = apply_rotary_emb_cos_sin_helion(
            xq, xk, self.rope_cache, self.positions
        )
        torch.autograd.backward((xq_out, xk_out), (grad_xq, grad_xk))

        xq_ref_out, xk_ref_out = _apply_rotary_emb_cos_sin_reference(
            xq_ref, xk_ref, self.rope_cache, self.positions
        )
        torch.autograd.backward((xq_ref_out, xk_ref_out), (grad_xq, grad_xk))

        torch.testing.assert_close(xq.grad, xq_ref.grad, rtol=0, atol=1e-5)
        torch.testing.assert_close(xk.grad, xk_ref.grad, rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
