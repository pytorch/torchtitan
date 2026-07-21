# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torchtitan.models.common.attention import create_varlen_metadata_for_document

# Tensor shape suffixes: B batch, L seq len, N heads, K key head dim,
# V value head dim.


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm using rsqrt(sum(x^2) + eps), not x/max(norm, eps) like F.normalize, to match FLA kernel."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_native_gated_delta(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    g_BLN: torch.Tensor,
    beta_BLN: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop -- far too slow for training; kept here as the
    numerical baseline for the FLA kernels.

    Args:
        q_BLNK, k_BLNK: (batch, seq, n_heads, key_head_dim)
        v_BLNV: (batch, seq, n_heads, value_head_dim)
        g_BLN: (batch, seq, n_heads) -- log-space decay, always negative
        beta_BLN: (batch, seq, n_heads) -- update gate in (0, 1)

    Returns:
        output: (batch, seq, n_heads, value_head_dim)
    """
    B, L, N, K = q_BLNK.shape
    V = v_BLNV.shape[-1]
    dtype = q_BLNK.dtype

    # Upcast to float32 -- recurrence accumulates over seqlen steps
    q_BLNK = _l2norm(q_BLNK.float(), dim=-1) * (K**-0.5)
    k_BLNK = _l2norm(k_BLNK.float(), dim=-1)
    v_BLNV, g_BLN, beta_BLN = v_BLNV.float(), g_BLN.float(), beta_BLN.float()

    out_BLNV = torch.zeros(B, L, N, V, dtype=torch.float32, device=q_BLNK.device)
    state_BNKV = torch.zeros(B, N, K, V, dtype=torch.float32, device=q_BLNK.device)

    for t in range(L):
        q_BNK = q_BLNK[:, t]
        k_BNK = k_BLNK[:, t]
        v_BNV = v_BLNV[:, t]
        g_BN11 = g_BLN[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_BN1 = beta_BLN[:, t].unsqueeze(-1)

        state_BNKV = state_BNKV * g_BN11
        kv_mem_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, k_BNK)
        delta_BNV = (v_BNV - kv_mem_BNV) * beta_BN1
        state_BNKV = state_BNKV + torch.einsum("bnk,bnv->bnkv", k_BNK, delta_BNV)
        out_BLNV[:, t] = torch.einsum("bnkv,bnk->bnv", state_BNKV, q_BNK)

    return out_BLNV.to(dtype)


def _torch_native_gated_delta_varlen(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    g_BLN: torch.Tensor,
    beta_BLN: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    """Varlen reference: run each packed document through the batched reference."""
    out_segments_BLNV: list[torch.Tensor] = []
    cu_seqlens_list = cu_seqlens_cpu.tolist()
    for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:], strict=False):
        out_segments_BLNV.append(
            _torch_native_gated_delta(
                q_BLNK[:, start:end],
                k_BLNK[:, start:end],
                v_BLNV[:, start:end],
                g_BLN[:, start:end],
                beta_BLN[:, start:end],
            )
        )
    return torch.cat(out_segments_BLNV, dim=1)


class ReferenceGatedDeltaKernel(nn.Module):
    """Drop-in replacement for GatedDeltaKernel backed by the reference math.

    Mirrors GatedDeltaKernel.forward's interface, including the grouped-query
    Q/K head expansion, so tests can swap it onto a built GatedDeltaNet and
    exercise the full varlen plumbing (flattening, conv resets, host-offset
    contract) on CPU.
    """

    def forward(
        self,
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xq_BLNK.shape[2] != xv_BLNV.shape[2]:
            assert xv_BLNV.shape[2] % xq_BLNK.shape[2] == 0
            repeat = xv_BLNV.shape[2] // xq_BLNK.shape[2]
            xq_BLNK = xq_BLNK.repeat_interleave(repeat, dim=2)
            xk_BLNK = xk_BLNK.repeat_interleave(repeat, dim=2)

        if cu_seqlens is None:
            return _torch_native_gated_delta(xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN)
        assert cu_seqlens_cpu is not None
        return _torch_native_gated_delta_varlen(
            xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, cu_seqlens_cpu
        )


class TestQwen35DeltaNetVarlen(unittest.TestCase):
    def _make_deltanet(
        self,
        *,
        # None builds the model with the default FLA kernel config, then swaps
        # in ReferenceGatedDeltaKernel so the model runs on CPU without FLA
        # triton kernels.
        backend: str | None = None,
        dim: int = 4,
        key_head_dim: int = 2,
        value_head_dim: int = 2,
        num_key_heads: int = 1,
        num_value_heads: int = 1,
        conv_kernel_size: int = 3,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        try:
            from torchtitan.models.common import Conv1d, Linear
            from torchtitan.models.qwen3_5.model import (
                GatedDeltaKernel,
                GatedDeltaNet,
                RMSNormGated,
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"Qwen3.5 optional dependency unavailable: {exc.name}"
            ) from exc

        key_dim = num_key_heads * key_head_dim
        value_dim = num_value_heads * value_head_dim

        def linear(out_features: int) -> Linear.Config:
            return Linear.Config(
                in_features=dim,
                out_features=out_features,
                bias=False,
            )

        def conv(channels: int) -> Conv1d.Config:
            return Conv1d.Config(
                in_channels=channels,
                out_channels=channels,
                kernel_size=conv_kernel_size,
                groups=channels,
                bias=False,
            )

        model = GatedDeltaNet.Config(
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel_size=conv_kernel_size,
            in_proj_q=linear(key_dim),
            in_proj_k=linear(key_dim),
            in_proj_v=linear(value_dim),
            in_proj_z=linear(value_dim),
            in_proj_a=linear(num_value_heads),
            in_proj_b=linear(num_value_heads),
            conv_q=conv(key_dim),
            conv_k=conv(key_dim),
            conv_v=conv(value_dim),
            kernel=(
                GatedDeltaKernel.Config()
                if backend is None
                else GatedDeltaKernel.Config(backend=backend)
            ),
            norm=RMSNormGated.Config(dim=value_head_dim),
            out_proj=Linear.Config(
                in_features=value_dim,
                out_features=dim,
                bias=False,
            ),
        ).build()
        if backend is None:
            model.kernel = ReferenceGatedDeltaKernel()

        model = model.to(device=device, dtype=dtype)
        with torch.no_grad():
            for param in model.parameters():
                values = torch.linspace(
                    -0.2,
                    0.2,
                    param.numel(),
                    dtype=param.dtype,
                    device=param.device,
                )
                param.copy_(values.reshape_as(param))
            model.A_log.fill_(0.0)
            model.dt_bias.zero_()
            model.norm.weight.fill_(1.0)
        return model

    def test_varlen_matches_independent_document_forwards(self):
        torch.manual_seed(42)
        model = self._make_deltanet()
        x = torch.randn(2, 5, 4)
        positions = torch.tensor(
            [
                [0, 1, 0, 1, 2],
                [0, 1, 2, 0, 1],
            ],
            dtype=torch.int32,
        )

        attention_masks = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )
        actual = model(x, attention_masks)

        expected = torch.empty_like(actual)
        for batch_idx in range(positions.shape[0]):
            doc_starts = (positions[batch_idx] == 0).nonzero(as_tuple=True)[0]
            starts = doc_starts.tolist()
            ends = starts[1:] + [positions.shape[1]]
            for start, end in zip(starts, ends, strict=False):
                expected[batch_idx : batch_idx + 1, start:end] = model(
                    x[batch_idx : batch_idx + 1, start:end]
                )

        self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=1e-6))

    def _assert_fla_varlen_matches_per_document(
        self, backend: str, *, atol: float, rtol: float
    ) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(42)
        # Mirror the debug model's GatedDeltaNet dims so the FLA Triton kernels
        # accept the shapes; n_value_heads > n_key_heads also exercises the
        # grouped-query head expansion inside the kernel.
        model = self._make_deltanet(
            backend=backend,
            dim=256,
            key_head_dim=64,
            value_head_dim=64,
            num_key_heads=2,
            num_value_heads=4,
            conv_kernel_size=4,
            device=device,
            dtype=dtype,
        )

        # Each row packs several documents; positions reset to 0 at every
        # document boundary, so the packed cu_seqlens is [0, 5, 12, 20, 24].
        positions = torch.tensor(
            [
                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
            ],
            dtype=torch.int32,
            device=device,
        )
        bs, seqlen = positions.shape
        x = torch.randn(bs, seqlen, 256, device=device, dtype=dtype)

        attention_masks = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )
        actual = model(x, attention_masks)

        # Reference: run each document on its own (non-varlen path) and stitch
        # the outputs back. Matching this proves the FLA varlen kernels reset
        # recurrent state at document boundaries instead of bleeding across them.
        expected = torch.empty_like(actual)
        for batch_idx in range(bs):
            doc_starts = (positions[batch_idx] == 0).nonzero(as_tuple=True)[0].tolist()
            ends = doc_starts[1:] + [seqlen]
            for start, end in zip(doc_starts, ends, strict=False):
                expected[batch_idx : batch_idx + 1, start:end] = model(
                    x[batch_idx : batch_idx + 1, start:end]
                )

        max_diff = (actual.float() - expected.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=rtol, atol=atol),
            msg=(
                f"{backend}: varlen output diverged from per-document forwards "
                f"(max abs diff {max_diff:.3e}, atol {atol}, rtol {rtol}). "
                "Cross-document state bleed produces diffs on the order of the "
                "output magnitude, far larger than bf16 kernel noise."
            ),
        )

    def test_fla_chunked_varlen_matches_independent_document_forwards(self):
        # bf16 tolerance absorbs the differing chunk boundaries between the
        # packed varlen run and the per-document runs; tighten once confirmed on
        # GPU (the failure message reports the observed max diff).
        self._assert_fla_varlen_matches_per_document(
            "fla_chunked", atol=2e-2, rtol=2e-2
        )

    def test_fla_fused_recurrent_varlen_matches_independent_document_forwards(self):
        self._assert_fla_varlen_matches_per_document(
            "fla_fused_recurrent", atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
