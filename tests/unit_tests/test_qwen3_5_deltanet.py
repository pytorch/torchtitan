# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenMetadata,
)

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


def _reference_causal_conv1d_varlen(
    x_TD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    """Per-document depthwise causal conv + silu, matching the model's FLA
    varlen conv (which is triton/CUDA-only). Patched over
    ``gdn._causal_conv1d_varlen`` for CPU runs.
    """
    conv_kernel_size = weight.shape[-1]
    out_segments_BTD: list[torch.Tensor] = []
    cu_seqlens_list = cu_seqlens_cpu.tolist()
    for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:], strict=False):
        x_segment_BDT = F.pad(
            x_TD[start:end].transpose(0, 1).unsqueeze(0),
            [conv_kernel_size - 1, 0],
        )
        out_segment_BTD = F.conv1d(
            x_segment_BDT,
            weight,
            None,
            groups=weight.size(0),
        ).transpose(1, 2)
        out_segments_BTD.append(out_segment_BTD)
    return F.silu(torch.cat(out_segments_BTD, dim=1)).squeeze(0)


class ReferenceGatedDeltaKernel(nn.Module):
    """Drop-in replacement for GatedDeltaKernel backed by the reference math.

    Mirrors GatedDeltaKernel.forward's interface, including the grouped-query
    Q/K head expansion, so tests can swap it onto a built GatedDeltaNet and
    exercise the full varlen plumbing (flattening, conv resets, host-offset
    contract) on CPU.
    """

    def forward(
        self,
        xq_TNK: torch.Tensor,
        xk_TNK: torch.Tensor,
        xv_TNV: torch.Tensor,
        g_TN: torch.Tensor,
        beta_TN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xq_TNK.shape[1] != xv_TNV.shape[1]:
            assert xv_TNV.shape[1] % xq_TNK.shape[1] == 0
            repeat = xv_TNV.shape[1] // xq_TNK.shape[1]
            xq_TNK = xq_TNK.repeat_interleave(repeat, dim=1)
            xk_TNK = xk_TNK.repeat_interleave(repeat, dim=1)

        xq_BLNK = xq_TNK.unsqueeze(0)
        xk_BLNK = xk_TNK.unsqueeze(0)
        xv_BLNV = xv_TNV.unsqueeze(0)
        g_BLN = g_TN.unsqueeze(0)
        beta_BLN = beta_TN.unsqueeze(0)

        if cu_seqlens is None:
            return _torch_native_gated_delta(
                xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN
            ).squeeze(0)
        assert cu_seqlens_cpu is not None
        return _torch_native_gated_delta_varlen(
            xq_BLNK, xk_BLNK, xv_BLNV, g_BLN, beta_BLN, cu_seqlens_cpu
        ).squeeze(0)


class TestQwen35DeltaNetVarlen(unittest.TestCase):
    def test_flex_masks_ignore_padding_position_resets(self):
        try:
            from torchtitan.models.common.decoder import Decoder
            from torchtitan.models.qwen3_5 import qwen3_5_configs
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"Qwen3.5 optional dependency unavailable: {exc.name}"
            ) from exc

        with torch.device("meta"):
            model = qwen3_5_configs["debugmodel"]("flex").build()
        positions = torch.tensor([0, 1, 2, 0, 0], dtype=torch.int32)

        with mock.patch.object(Decoder, "get_attention_masks", return_value=None):
            attention_masks = model.get_attention_masks(positions)

        self.assertIsNone(attention_masks["deltanet"])

    def test_flex_masks_include_delta_net_varlen_metadata(self):
        try:
            from torchtitan.models.common.decoder import Decoder
            from torchtitan.models.qwen3_5 import qwen3_5_configs
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"Qwen3.5 optional dependency unavailable: {exc.name}"
            ) from exc

        with torch.device("meta"):
            model = qwen3_5_configs["debugmodel"]("flex").build()
        positions = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32)
        full_attention_mask = mock.sentinel.full_attention_mask

        with mock.patch.object(
            Decoder,
            "get_attention_masks",
            return_value=full_attention_mask,
        ):
            attention_masks = model.get_attention_masks(positions)

        self.assertIs(attention_masks["quadratic_attention"], full_attention_mask)
        torch.testing.assert_close(
            attention_masks["deltanet"].cu_seq_q,
            torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        self.assertEqual(attention_masks["deltanet"].cu_seq_q_host, (0, 2, 5))

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
            from torchtitan.models.qwen3_5.gdn import (
                GatedDeltaKernel,
                GatedDeltaNet,
                InnerGatedDeltaNet,
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
            inner_gated_delta_net=InnerGatedDeltaNet.Config(
                kernel=(
                    GatedDeltaKernel.Config()
                    if backend is None
                    else GatedDeltaKernel.Config(backend=backend)
                ),
            ),
            norm=RMSNormGated.Config(dim=value_head_dim),
            out_proj=Linear.Config(
                in_features=value_dim,
                out_features=dim,
                bias=False,
            ),
        ).build()
        if backend is None:
            model.inner_gated_delta_net.kernel = ReferenceGatedDeltaKernel()

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

    def _main_forward_reference(self, model, x_TD, attention_masks=None):
        """Run the current main-branch GatedDeltaNet forward structure."""
        num_tokens = x_TD.shape[0]
        cu_seqlens = None
        cu_seqlens_cpu = None
        if attention_masks is not None:
            cu_seqlens = attention_masks.cu_seq_q.clone()
            cu_seqlens_cpu = torch.tensor(
                attention_masks.cu_seq_q_host,
                dtype=cu_seqlens.dtype,
                device="cpu",
            )

        def causal_conv(tensor, conv):
            if cu_seqlens is not None:
                return _reference_causal_conv1d_varlen(
                    tensor,
                    conv.weight,
                    cu_seqlens,
                    cu_seqlens_cpu,
                )
            tensor = F.pad(
                tensor.transpose(0, 1).unsqueeze(0),
                [conv.weight.shape[-1] - 1, 0],
            )
            return (
                F.silu(
                    F.conv1d(
                        tensor,
                        conv.weight,
                        None,
                        groups=conv.weight.size(0),
                    )
                )
                .squeeze(0)
                .transpose(0, 1)
            )

        query_TNK = causal_conv(model.in_proj_q(x_TD), model.conv_q).reshape(
            num_tokens, -1, model.key_head_dim
        )
        key_TNK = causal_conv(model.in_proj_k(x_TD), model.conv_k).reshape(
            num_tokens, -1, model.key_head_dim
        )
        value_TNV = causal_conv(model.in_proj_v(x_TD), model.conv_v).reshape(
            num_tokens, -1, model.value_head_dim
        )
        gate_TNV = model.in_proj_z(x_TD).reshape(num_tokens, -1, model.value_head_dim)
        a_TN = model.in_proj_a(x_TD)
        b_TN = model.in_proj_b(x_TD)
        decay_TN = -torch.exp(model.A_log.float()) * F.softplus(
            a_TN.float() + model.dt_bias
        )
        update_gate_TN = torch.sigmoid(b_TN)
        output_TNV = model.inner_gated_delta_net.kernel(
            query_TNK,
            key_TNK,
            value_TNV,
            decay_TN,
            update_gate_TN,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        output_TNV = model.norm(output_TNV, gate_TNV)
        return model.out_proj(output_TNV.reshape(num_tokens, -1))

    def test_extracted_forward_matches_main(self):
        torch.manual_seed(42)
        model = self._make_deltanet()
        x_TD = torch.randn(10, 4)
        positions = torch.tensor(
            [0, 1, 0, 1, 2, 0, 1, 2, 0, 1],
            dtype=torch.int32,
        )
        attention_masks = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )

        for masks in (None, attention_masks):
            with mock.patch(
                "torchtitan.models.qwen3_5.gdn._causal_conv1d_varlen",
                _reference_causal_conv1d_varlen,
            ):
                actual = model(x_TD, masks)
            expected = self._main_forward_reference(model, x_TD, masks)
            self.assertTrue(torch.equal(actual, expected))

    def _assert_packed_run_matches_per_document(self, model, x, positions, masks):
        """Packed forward under ``masks`` must equal stitched per-doc forwards.

        The model's varlen conv is FLA (triton/CUDA-only); substitute the
        per-document torch reference for these CPU runs. The per-document
        forwards below take the non-varlen conv path, which runs on CPU.
        """
        with mock.patch(
            "torchtitan.models.qwen3_5.gdn._causal_conv1d_varlen",
            _reference_causal_conv1d_varlen,
        ):
            actual = model(x, masks)

        expected = torch.empty_like(actual)
        starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
        ends = starts[1:] + [positions.shape[0]]
        for start, end in zip(starts, ends, strict=False):
            expected[start:end] = model(x[start:end])

        self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=1e-6))

    def test_varlen_matches_independent_document_forwards(self):
        torch.manual_seed(42)
        model = self._make_deltanet()
        x_TD = torch.randn(10, 4)
        positions = torch.tensor(
            [0, 1, 0, 1, 2, 0, 1, 2, 0, 1],
            dtype=torch.int32,
        )

        attention_masks = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )
        self._assert_packed_run_matches_per_document(
            model, x_TD, positions, attention_masks
        )

    def test_get_attention_masks_pairs_flex_mask_with_deltanet_offsets(self):
        """Qwen35Model.get_attention_masks must return the per-consumer mask
        dict: under flex, a BlockMask ("quadratic_attention") paired with the
        document offsets ("deltanet"); under varlen, one VarlenMetadata shared
        by both keys. Each transformer block picks its entry by attn_mask_key.
        """
        from torch.nn.attention.flex_attention import BlockMask

        # torchtitan.models.qwen3_5 imports the FLA (flash-linear-attention)
        # kernels at module scope. FLA is a triton/CUDA-only optional
        # dependency, so skip instead of erroring on environments without it.
        try:
            from torchtitan.models.qwen3_5 import model_registry
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"Qwen3.5 optional dependency unavailable: {exc.name}"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        positions = torch.tensor(
            [0, 1, 2, 0, 1, 0, 1, 2, 3, 4],
            dtype=torch.int32,
            device=device,
        )

        flex_model = model_registry("debugmodel").model.build()
        masks = flex_model.get_attention_masks(positions)
        self.assertIsInstance(masks, dict)
        self.assertEqual(set(masks.keys()), {"quadratic_attention", "deltanet"})
        self.assertIsInstance(masks["quadratic_attention"], BlockMask)
        self.assertIsInstance(masks["deltanet"], VarlenMetadata)
        # Three packed documents have lengths 3, 2, and 5.
        self.assertEqual(masks["deltanet"].cu_seq_q_host, (0, 3, 5, 10))

        # Each block picks the entry matching its layer type.
        mask_keys = {layer.attn_mask_key for layer in flex_model.layers.values()}
        self.assertEqual(mask_keys, {"quadratic_attention", "deltanet"})
        for layer in flex_model.layers.values():
            self.assertEqual(
                layer.attn_mask_key,
                "quadratic_attention" if layer.full_attn else "deltanet",
            )

        varlen_model = model_registry("debugmodel", attn_backend="varlen").model.build()
        varlen_masks = varlen_model.get_attention_masks(positions)
        self.assertIsInstance(varlen_masks, dict)
        self.assertIs(varlen_masks["quadratic_attention"], varlen_masks["deltanet"])
        self.assertIsInstance(varlen_masks["deltanet"], VarlenMetadata)
        self.assertEqual(varlen_masks["deltanet"].cu_seq_q_host, (0, 3, 5, 10))

        deltanet_only_config = model_registry("debugmodel").model
        deltanet_only_config.layers = [
            layer
            for layer in deltanet_only_config.layers
            if layer.delta_net is not None
        ]
        deltanet_only_model = deltanet_only_config.build()
        deltanet_only_masks = deltanet_only_model.get_attention_masks(positions)
        self.assertEqual(
            set(deltanet_only_masks.keys()),
            {"quadratic_attention", "deltanet"},
        )
        self.assertIsNone(deltanet_only_masks["quadratic_attention"])
        self.assertIsInstance(deltanet_only_masks["deltanet"], VarlenMetadata)
        self.assertEqual(
            deltanet_only_masks["deltanet"].cu_seq_q_host,
            (0, 3, 5, 10),
        )

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

        # Positions reset at every document boundary, so the packed cu_seqlens
        # is [0, 5, 12, 20, 24].
        positions = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                0,
                1,
                2,
                3,
            ],
            dtype=torch.int32,
            device=device,
        )
        x_TD = torch.randn(positions.shape[0], 256, device=device, dtype=dtype)

        attention_masks = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )
        actual = model(x_TD, attention_masks)

        # Reference: run each document on its own (non-varlen path) and stitch
        # the outputs back. Matching this proves the FLA varlen kernels reset
        # recurrent state at document boundaries instead of bleeding across them.
        expected = torch.empty_like(actual)
        doc_starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
        ends = doc_starts[1:] + [positions.shape[0]]
        for start, end in zip(doc_starts, ends, strict=False):
            expected[start:end] = model(x_TD[start:end])

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

    def test_varlen_offsets_are_fresh_per_deltanet_invocation(self):
        """Successive DeltaNet invocations must not share FLA's cache key."""
        torch.manual_seed(42)
        model = self._make_deltanet()
        x_TD = torch.randn(8, 4)
        positions = torch.tensor(
            [0, 1, 2, 0, 1, 2, 3, 4],
            dtype=torch.int32,
        )
        attention_masks = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )
        captured_cu_seqlens = []

        def record_cu_seqlens(x_TD, weight, cu_seqlens, cu_seqlens_cpu):
            captured_cu_seqlens.append(cu_seqlens)
            return _reference_causal_conv1d_varlen(
                x_TD,
                weight,
                cu_seqlens,
                cu_seqlens_cpu,
            )

        with mock.patch(
            "torchtitan.models.qwen3_5.gdn._causal_conv1d_varlen",
            side_effect=record_cu_seqlens,
        ):
            model(x_TD, attention_masks)
            model(x_TD, attention_masks)

        # Main runs separate Q/K/V convolutions, so each invocation uses the
        # same cloned offsets three times.
        self.assertEqual(len(captured_cu_seqlens), 6)
        first_invocation = captured_cu_seqlens[0]
        second_invocation = captured_cu_seqlens[3]
        self.assertTrue(all(x is first_invocation for x in captured_cu_seqlens[:3]))
        self.assertTrue(all(x is second_invocation for x in captured_cu_seqlens[3:]))
        self.assertIsNot(first_invocation, attention_masks.cu_seq_q)
        self.assertIsNot(second_invocation, first_invocation)


if __name__ == "__main__":
    unittest.main()
