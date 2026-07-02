# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtitan.models.common.attention import create_varlen_metadata_for_document


class TestQwen35DeltaNetVarlen(unittest.TestCase):
    def _make_deltanet(
        self,
        *,
        backend: str = "torch_native",
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
            kernel=GatedDeltaKernel.Config(backend=backend),
            norm=RMSNormGated.Config(dim=value_head_dim),
            out_proj=Linear.Config(
                in_features=value_dim,
                out_features=dim,
                bias=False,
            ),
        ).build()

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
