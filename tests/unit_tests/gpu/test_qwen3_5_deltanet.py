# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from attn_gym.linear import l2norm, recurrent_gdn
from torch import nn

from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenMetadata,
)

# Tensor shape suffixes: B batch, L seq len, H heads, K key head dim,
# V value head dim.


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Match Attention Gym's rsqrt-based L2 normalization."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_native_gated_delta(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    g_BLH: torch.Tensor,
    beta_BLH: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop -- far too slow for training; kept here as the
    numerical baseline for the fused kernels.

    Args:
        q_BLHK, k_BLHK: (batch, seq, num_heads, key_head_dim)
        v_BLHV: (batch, seq, num_heads, value_head_dim)
        g_BLH: (batch, seq, num_heads) -- log-space decay, always negative
        beta_BLH: (batch, seq, num_heads) -- update gate in (0, 1)

    Returns:
        output: (batch, seq, num_heads, value_head_dim)
    """
    B, L, H, K = q_BLHK.shape
    V = v_BLHV.shape[-1]
    dtype = q_BLHK.dtype

    # Upcast to float32 -- recurrence accumulates over seqlen steps
    q_BLHK = _l2norm(q_BLHK.float(), dim=-1) * (K**-0.5)
    k_BLHK = _l2norm(k_BLHK.float(), dim=-1)
    v_BLHV, g_BLH, beta_BLH = v_BLHV.float(), g_BLH.float(), beta_BLH.float()

    out_BLHV = torch.zeros(B, L, H, V, dtype=torch.float32, device=q_BLHK.device)
    state_BHKV = torch.zeros(B, H, K, V, dtype=torch.float32, device=q_BLHK.device)

    for t in range(L):
        q_BHK = q_BLHK[:, t]
        k_BHK = k_BLHK[:, t]
        v_BHV = v_BLHV[:, t]
        g_BH11 = g_BLH[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_BH1 = beta_BLH[:, t].unsqueeze(-1)

        state_BHKV = state_BHKV * g_BH11
        kv_mem_BHV = torch.einsum("bhkv,bhk->bhv", state_BHKV, k_BHK)
        delta_BHV = (v_BHV - kv_mem_BHV) * beta_BH1
        state_BHKV = state_BHKV + torch.einsum("bhk,bhv->bhkv", k_BHK, delta_BHV)
        out_BLHV[:, t] = torch.einsum("bhkv,bhk->bhv", state_BHKV, q_BHK)

    return out_BLHV.to(dtype)


def _torch_native_gated_delta_varlen(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    g_BLH: torch.Tensor,
    beta_BLH: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    """Varlen reference: run each packed document through the batched reference."""
    out_segments_BLHV: list[torch.Tensor] = []
    cu_seqlens_list = cu_seqlens_cpu.tolist()
    for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:], strict=False):
        out_segments_BLHV.append(
            _torch_native_gated_delta(
                q_BLHK[:, start:end],
                k_BLHK[:, start:end],
                v_BLHV[:, start:end],
                g_BLH[:, start:end],
                beta_BLH[:, start:end],
            )
        )
    return torch.cat(out_segments_BLHV, dim=1)


def _reference_causal_conv1d_varlen(
    x_TD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Per-document depthwise causal conv + silu, matching the model's Attention
    Gym varlen conv (which is CUDA-only). Patched over
    ``gdn._causal_conv1d_varlen`` for CPU runs.
    """
    conv_kernel_size = weight.shape[-1]
    out_segments_BTD: list[torch.Tensor] = []
    cu_seqlens_list = cu_seqlens.tolist()
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
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xq_THK.shape[1] != xv_THV.shape[1]:
            assert xv_THV.shape[1] % xq_THK.shape[1] == 0
            repeat = xv_THV.shape[1] // xq_THK.shape[1]
            xq_THK = xq_THK.repeat_interleave(repeat, dim=1)
            xk_THK = xk_THK.repeat_interleave(repeat, dim=1)

        xq_BLHK = xq_THK.unsqueeze(0)
        xk_BLHK = xk_THK.unsqueeze(0)
        xv_BLHV = xv_THV.unsqueeze(0)
        g_BLH = g_TH.unsqueeze(0)
        beta_BLH = beta_TH.unsqueeze(0)

        if cu_seqlens is None:
            return _torch_native_gated_delta(
                xq_BLHK, xk_BLHK, xv_BLHV, g_BLH, beta_BLH
            ).squeeze(0)
        return _torch_native_gated_delta_varlen(
            xq_BLHK, xk_BLHK, xv_BLHV, g_BLH, beta_BLH, cu_seqlens.cpu()
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
            build_config, max_context_length = qwen3_5_configs["debugmodel"]
            model = build_config("flex", seq_len=max_context_length).build()
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
            build_config, max_context_length = qwen3_5_configs["debugmodel"]
            model = build_config("flex", seq_len=max_context_length).build()
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

    def _make_deltanet(
        self,
        *,
        use_fused: bool = False,
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
                kernel=GatedDeltaKernel.Config(),
            ),
            norm=RMSNormGated.Config(dim=value_head_dim),
            out_proj=Linear.Config(
                in_features=value_dim,
                out_features=dim,
                bias=False,
            ),
        ).build()
        if not use_fused:
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
        if attention_masks is not None:
            cu_seqlens = attention_masks.cu_seq_q.clone()

        def causal_conv(tensor, conv):
            if cu_seqlens is not None:
                return _reference_causal_conv1d_varlen(
                    tensor,
                    conv.weight,
                    cu_seqlens,
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

        query_THK = causal_conv(model.in_proj_q(x_TD), model.conv_q).reshape(
            num_tokens, -1, model.key_head_dim
        )
        key_THK = causal_conv(model.in_proj_k(x_TD), model.conv_k).reshape(
            num_tokens, -1, model.key_head_dim
        )
        value_THV = causal_conv(model.in_proj_v(x_TD), model.conv_v).reshape(
            num_tokens, -1, model.value_head_dim
        )
        gate_THV = model.in_proj_z(x_TD).reshape(num_tokens, -1, model.value_head_dim)
        a_TH = model.in_proj_a(x_TD)
        b_TH = model.in_proj_b(x_TD)
        decay_TH = -torch.exp(model.A_log.float()) * F.softplus(
            a_TH.float() + model.dt_bias
        )
        update_gate_TH = torch.sigmoid(b_TH)
        output_THV = model.inner_gated_delta_net.kernel(
            query_THK,
            key_THK,
            value_THV,
            decay_TH,
            update_gate_TH,
            cu_seqlens=cu_seqlens,
        )
        output_THV = model.norm(output_THV, gate_THV)
        return model.out_proj(output_THV.reshape(num_tokens, -1))

    def test_extracted_forward_matches_main(self):
        torch.manual_seed(42)
        model = self._make_deltanet()
        x_TD = torch.randn(10, 4)
        positions = torch.tensor(
            [0, 1, 0, 1, 2, 0, 1, 2, 0, 1],
            dtype=torch.int32,
        )
        attention_masks = create_varlen_metadata_for_document(positions)

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

        The model's varlen conv is Attention Gym (CUDA-only); substitute the
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

        attention_masks = create_varlen_metadata_for_document(positions)
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
        torch.testing.assert_close(
            masks["deltanet"].cu_seq_q,
            torch.tensor([0, 3, 5, 10], dtype=torch.int32, device=device),
        )

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
        torch.testing.assert_close(
            varlen_masks["deltanet"].cu_seq_q,
            torch.tensor([0, 3, 5, 10], dtype=torch.int32, device=device),
        )

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
        torch.testing.assert_close(
            deltanet_only_masks["deltanet"].cu_seq_q,
            torch.tensor([0, 3, 5, 10], dtype=torch.int32, device=device),
        )

    def _assert_fused_varlen_matches_per_document(
        self, *, atol: float, rtol: float
    ) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(42)
        # The fused chunk kernel requires 128-wide heads. Unequal key and value
        # head counts also exercise grouped-head execution.
        model = self._make_deltanet(
            use_fused=True,
            dim=256,
            key_head_dim=128,
            value_head_dim=128,
            num_key_heads=1,
            num_value_heads=2,
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
        x_TD = torch.randn(
            positions.shape[0],
            256,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        attention_masks = create_varlen_metadata_for_document(positions)
        actual = model(x_TD, attention_masks)

        # Reference: run each document on its own and stitch the outputs back.
        # Matching proves packed execution resets state at document boundaries.
        expected = torch.empty_like(actual)
        doc_starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
        ends = doc_starts[1:] + [positions.shape[0]]
        for start, end in zip(doc_starts, ends, strict=False):
            expected[start:end] = model(x_TD[start:end])

        max_diff = (actual.float() - expected.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=rtol, atol=atol),
            msg=(
                "varlen output diverged from per-document forwards "
                f"(max abs diff {max_diff:.3e}, atol {atol}, rtol {rtol}). "
                "Cross-document state bleed produces diffs on the order of the "
                "output magnitude, far larger than bf16 kernel noise."
            ),
        )

        actual.float().square().mean().backward()
        self.assertIsNotNone(x_TD.grad)
        self.assertTrue(torch.isfinite(x_TD.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fused_varlen_matches_independent_document_forwards(self):
        # BF16 tolerance absorbs differing packed and per-document chunk boundaries.
        self._assert_fused_varlen_matches_per_document(atol=2e-2, rtol=2e-2)

    def test_batch_invariant_recurrent_matches_paged_attention_gym(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

        from torchtitan.models.qwen3_5.gdn import _recurrent_gdn_fwd

        torch.manual_seed(42)
        num_tokens, num_key_heads, num_value_heads, key_dim, value_dim = (
            12,
            1,
            2,
            128,
            128,
        )
        q = torch.randn(
            1,
            num_tokens,
            num_key_heads,
            key_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            1,
            num_tokens,
            num_value_heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        decay = -torch.rand(
            1,
            num_tokens,
            num_value_heads,
            device="cuda",
            dtype=torch.float32,
        )
        update_gate = torch.rand(
            1,
            num_tokens,
            num_value_heads,
            device="cuda",
            dtype=torch.float32,
        )
        for tensor in (q, k, v, decay, update_gate):
            tensor.requires_grad_()
        cu_seqlens = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)

        actual = torch.compile(_recurrent_gdn_fwd, fullgraph=True)(
            q,
            k,
            v,
            decay,
            update_gate,
            cu_seqlens,
        )

        normalized_q = l2norm(q, cu_seqlens=cu_seqlens)
        normalized_k = l2norm(k, cu_seqlens=cu_seqlens)
        state_cache = torch.randn(
            5,
            num_value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
        )
        prefix_end = 2
        prefix_cu_seqlens = torch.tensor(
            [0, prefix_end], device="cuda", dtype=torch.int32
        )
        with torch.no_grad():
            prefix_output, _ = recurrent_gdn(
                normalized_q[:, :prefix_end],
                normalized_k[:, :prefix_end],
                v[:, :prefix_end],
                decay[:, :prefix_end],
                update_gate[:, :prefix_end],
                state_cache,
                cu_seqlens=prefix_cu_seqlens,
                scale=key_dim**-0.5,
                state_indices=torch.tensor([3], device="cuda", dtype=torch.int32),
                has_initial_state=torch.tensor([False], device="cuda"),
            )

        state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        has_initial_state = torch.tensor([True, False], device="cuda")
        remaining_cu_seqlens = torch.tensor(
            [0, 5 - prefix_end, num_tokens - prefix_end],
            device="cuda",
            dtype=torch.int32,
        )
        with torch.no_grad():
            remaining_output, _ = recurrent_gdn(
                normalized_q[:, prefix_end:],
                normalized_k[:, prefix_end:],
                v[:, prefix_end:],
                decay[:, prefix_end:],
                update_gate[:, prefix_end:],
                state_cache,
                cu_seqlens=remaining_cu_seqlens,
                scale=key_dim**-0.5,
                state_indices=state_indices,
                has_initial_state=has_initial_state,
            )
        expected = torch.cat(
            (prefix_output, remaining_output),
            dim=1,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        actual.float().square().mean().backward()
        self.assertIsNotNone(decay.grad)
        self.assertTrue(torch.isfinite(decay.grad).all())


if __name__ == "__main__":
    unittest.main()
