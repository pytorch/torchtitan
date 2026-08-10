# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common import Embedding
from torchtitan.protocols.module import Module

# torchtitan.models.kimi_k3 imports FLA at module scope for the KDA kernel.
# FLA is a per-model dependency (kimi_k3/requirements.txt), not part of the
# core requirements, so skip the Kimi suites instead of failing collection
# when it is absent. Modules importing this one inherit the skip.
try:
    from torchtitan.models.kimi_k3 import (
        _feed_forward_config,
        _kda_config,
        _latent_moe_config,
        _linear,
        _mla_config,
        _norm,
        _vision_encoder_config,
        kimi_k3_configs,
    )
    from torchtitan.models.kimi_k3.model import (
        KimiK3Model,
        KimiK3TransformerBlock,
        KimiKDAKernel,
    )
    from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
    from torchtitan.models.kimi_k3.vision_encoder import KimiExactGELU
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        f"Kimi K3 optional dependency unavailable: {exc.name}"
    ) from exc


class ReferenceKimiKDAKernel(Module):
    """Pure-PyTorch stand-in for KimiKDAKernel backed by an explicit recurrence.

    Mirrors ``KimiKDAKernel.forward``'s interface so tests can build a model
    with it in place of the FLA kernel and exercise the surrounding eager model
    on CPU. The loop is O(seqlen) and far too slow for training; it exists to
    pin the kernel's math.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        lower_bound: float | None = -5.0

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.lower_bound = config.lower_bound

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> torch.Tensor:
        return _kda_recurrent_reference(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            gate_BLHK,
            beta_BLH,
            A_log_H,
            dt_bias_HK,
            lower_bound=self.lower_bound,
        )


def _use_reference_kda_kernel(config: KimiK3Model.Config) -> KimiK3Model.Config:
    """Point every KDA layer at the recurrent reference kernel.

    Test configurations use head dimensions far below what FLA's chunked KDA
    kernel can compile, and the CPU suite has no Triton runtime at all.
    """
    for layer in config.layers:
        if layer.delta_attention is None:
            continue
        kernel = layer.delta_attention.kernel
        assert isinstance(kernel, KimiKDAKernel.Config)
        layer.delta_attention.kernel = ReferenceKimiKDAKernel.Config(
            head_dim=kernel.head_dim,
            lower_bound=kernel.lower_bound,
        )
    return config


def _small_model_config(*, attn_res_block_size: int = 1) -> KimiK3Model.Config:
    """Build the reduced two-layer model used across the Kimi K3 tests.

    ``attn_res_block_size`` defaults to 1, which makes every layer extend the
    attention residual. Pass 2 to make the second layer pass the residual
    through instead, which is the shape the released cadence uses and which
    routes the residual back out through the FSDP module boundary. Callers
    comparing against frozen reference values must keep the default, since the
    parameter shapes and ordering feed those values.
    """
    dim = 16

    def block(
        layer_id: int,
        *,
        use_mla: bool,
        use_moe: bool,
    ) -> KimiK3TransformerBlock.Config:
        return KimiK3TransformerBlock.Config(
            layer_id=layer_id,
            attn_res_block_size=attn_res_block_size,
            attention=(
                _mla_config(
                    dim=dim,
                    num_heads=2,
                    q_lora_rank=8,
                    kv_lora_rank=8,
                    qk_nope_head_dim=4,
                    qk_rope_head_dim=4,
                    v_head_dim=4,
                )
                if use_mla
                else None
            ),
            delta_attention=(
                None
                if use_mla
                else _kda_config(
                    dim=dim,
                    num_heads=2,
                    head_dim=4,
                    conv_kernel_size=3,
                )
            ),
            feed_forward=(
                None if use_moe else _feed_forward_config(dim=dim, hidden_dim=32)
            ),
            moe=(
                _latent_moe_config(
                    dim=dim,
                    latent_dim=8,
                    expert_hidden_dim=8,
                    num_experts=2,
                    top_k=1,
                    num_shared_experts=1,
                )
                if use_moe
                else None
            ),
            attention_norm=_norm(dim),
            ffn_norm=_norm(dim),
            attention_res_norm=_norm(dim),
            attention_res_proj=_linear(dim, 1),
            ffn_res_norm=_norm(dim),
            ffn_res_proj=_linear(dim, 1),
        )

    return _use_reference_kda_kernel(
        KimiK3Model.Config(
            dim=dim,
            vocab_size=32,
            tok_embeddings=Embedding.Config(
                num_embeddings=32,
                embedding_dim=dim,
                param_init={
                    "weight": lambda parameter: nn.init.normal_(parameter, std=0.02)
                },
            ),
            layers=[
                block(0, use_mla=False, use_moe=False),
                block(1, use_mla=True, use_moe=True),
            ],
            norm=_norm(dim),
            lm_head=_linear(dim, 32),
            output_res_norm=_norm(dim),
            output_res_proj=_linear(dim, 1),
            vision_encoder=_vision_encoder_config(
                text_dim=dim,
                dim=16,
                qkv_dim=24,
                hidden_dim=32,
                num_layers=1,
                num_heads=3,
                patch_size=2,
                merge_kernel_size=(2, 2),
                init_pos_emb_height=2,
                init_pos_emb_width=2,
                max_num_frames=1,
            ),
            spatial_merge_size=2,
        )
    )


def _kda_recurrent_reference(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    gate_BLHK: torch.Tensor,
    beta_BLH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    *,
    lower_bound: float | None,
) -> torch.Tensor:
    """Explicit KDA recurrence in FP32, matching the released Kimi K3 math.

    ``lower_bound`` selects the same two gate activations FLA exposes through
    ``safe_gate``: the bounded ``lower_bound * sigmoid(...)`` form when set,
    and ``-exp(A_log) * softplus(...)`` when ``None``.
    """
    input_dtype = q_BLHK.dtype
    q_BLHK = q_BLHK.float()
    k_BLHK = k_BLHK.float()
    q_BLHK = q_BLHK * torch.rsqrt(q_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    k_BLHK = k_BLHK * torch.rsqrt(k_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    v_BLHV = v_BLHV.float()
    if lower_bound is None:
        log_decay_BLHK = -torch.exp(A_log_H.float()).view(1, 1, -1, 1) * F.softplus(
            gate_BLHK.float() + dt_bias_HK.float()
        )
    else:
        log_decay_BLHK = lower_bound * torch.sigmoid(
            torch.exp(A_log_H.float()).view(1, 1, -1, 1)
            * (gate_BLHK.float() + dt_bias_HK.float())
        )
    decay_BLHK = torch.exp(log_decay_BLHK)
    beta_BLH = torch.sigmoid(beta_BLH.float())

    B, L, H, K = q_BLHK.shape
    V = v_BLHV.shape[-1]
    state_BHKV = torch.zeros(B, H, K, V, device=q_BLHK.device)
    outputs_BHV = []
    for token_idx in range(L):
        state_BHKV = state_BHKV * decay_BLHK[:, token_idx].unsqueeze(-1)
        old_value_BHV = torch.matmul(
            k_BLHK[:, token_idx].unsqueeze(-2),
            state_BHKV,
        ).squeeze(-2)
        delta_BHV = (v_BLHV[:, token_idx] - old_value_BHV) * beta_BLH[
            :, token_idx
        ].unsqueeze(-1)
        state_BHKV = state_BHKV + (
            k_BLHK[:, token_idx].unsqueeze(-1) * delta_BHV.unsqueeze(-2)
        )
        outputs_BHV.append(
            torch.matmul(
                q_BLHK[:, token_idx].unsqueeze(-2),
                state_BHKV,
            ).squeeze(-2)
            * (K**-0.5)
        )
    return torch.stack(outputs_BHV, dim=1).to(input_dtype)


class TestKimiK3(unittest.TestCase):
    def test_exact_gelu_matches_pytorch_reference(self):
        x = torch.linspace(-4.0, 4.0, 257)
        actual = KimiExactGELU.Config().build()(x)
        expected = F.gelu(x, approximate="none")

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        x_bf16 = x.bfloat16()
        self.assertEqual(KimiExactGELU.Config().build()(x_bf16).dtype, x_bf16.dtype)

    def test_debugmodel_preserves_reduced_k3_topology(self):
        config = kimi_k3_configs["debugmodel"]("eager")

        self.assertEqual(config.vocab_size, 163840)
        self.assertEqual(len(config.layers), 13)
        self.assertEqual(
            [
                layer_idx + 1
                for layer_idx, layer in enumerate(config.layers)
                if layer.attention is not None
            ],
            [4, 8, 12, 13],
        )
        self.assertIsNotNone(config.layers[0].feed_forward)
        self.assertTrue(all(layer.moe is not None for layer in config.layers[1:]))
        moe_config = config.layers[1].moe
        assert moe_config is not None
        self.assertEqual(moe_config.num_experts, 8)
        self.assertEqual(moe_config.router.top_k, 2)
        self.assertEqual(moe_config.routed_experts.hidden_dim, 128)
        vision_config = config.vision_encoder
        assert vision_config is not None
        self.assertEqual(vision_config.dim, 256)
        self.assertEqual(vision_config.num_layers, 4)

    @unittest.skipIf(not torch.cuda.is_available(), "FLA KDA kernel requires CUDA.")
    def test_fla_kda_kernel_matches_recurrent_reference(self):
        torch.manual_seed(1)
        head_dim = 32
        num_heads = 3

        def parameter(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, device="cuda", requires_grad=True)

        for lower_bound in (-5.0, None):
            with self.subTest(lower_bound=lower_bound):
                q_BLHK = parameter(2, 64, num_heads, head_dim)
                k_BLHK = parameter(2, 64, num_heads, head_dim)
                v_BLHV = parameter(2, 64, num_heads, head_dim)
                gate_BLHK = parameter(2, 64, num_heads, head_dim)
                beta_BLH = parameter(2, 64, num_heads)
                A_log_H = torch.rand(num_heads, device="cuda")
                A_log_H = A_log_H.uniform_(1.0, 16.0).log().requires_grad_()
                dt_bias_HK = parameter(num_heads, head_dim)

                kernel = KimiKDAKernel.Config(
                    head_dim=head_dim,
                    lower_bound=lower_bound,
                ).build()
                actual_BLHV = kernel(
                    q_BLHK,
                    k_BLHK,
                    v_BLHV,
                    gate_BLHK,
                    beta_BLH,
                    A_log_H,
                    dt_bias_HK,
                )
                expected_BLHV = _kda_recurrent_reference(
                    q_BLHK,
                    k_BLHK,
                    v_BLHV,
                    gate_BLHK,
                    beta_BLH,
                    A_log_H,
                    dt_bias_HK,
                    lower_bound=lower_bound,
                )

                # The chunked kernel accumulates over chunk boundaries and uses
                # reduced-precision matmuls internally, so it does not reproduce
                # the sequential FP32 recurrence bit for bit.
                torch.testing.assert_close(
                    actual_BLHV,
                    expected_BLHV,
                    atol=2e-3,
                    rtol=2e-3,
                )
                actual_BLHV.square().mean().backward()
                for tensor in (
                    q_BLHK,
                    k_BLHK,
                    v_BLHV,
                    gate_BLHK,
                    beta_BLH,
                    A_log_H,
                    dt_bias_HK,
                ):
                    self.assertIsNotNone(tensor.grad)
                    assert tensor.grad is not None
                    self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_unused_moe_experts_receive_zero_gradients(self):
        torch.manual_seed(2)
        model = _small_model_config().build()
        model.init_states()
        moe = model.layers["1"].moe
        assert moe is not None
        with torch.no_grad():
            moe.router.gate.weight.zero_()

        inputs = torch.randn(2, 4, 16, requires_grad=True)
        _, expert_ids, _ = moe.router(inputs, moe.expert_bias_E)
        selected_experts = set(expert_ids.flatten().tolist())
        unused_experts = set(range(moe.num_experts)) - selected_experts
        self.assertTrue(unused_experts)

        moe(inputs).float().sum().backward()
        for parameter in (
            moe.routed_experts.w1_EFD,
            moe.routed_experts.w2_EDF,
            moe.routed_experts.w3_EFD,
        ):
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            for expert_idx in unused_experts:
                torch.testing.assert_close(
                    parameter.grad[expert_idx],
                    torch.zeros_like(parameter.grad[expert_idx]),
                )

    def test_small_multimodal_model_forward_backward_and_adapter(self):
        torch.manual_seed(2)
        config = _small_model_config()
        model = config.build()
        model.verify_module_protocol()
        model.init_states()

        image_token_id = 7
        tokens_BL = torch.tensor(
            [
                [1, 2, image_token_id, 3, 4, 5],
                [6, image_token_id, image_token_id, 8, 9, 10],
            ]
        )
        pixel_values_NPK = torch.randn(2, 8, 3 * 2 * 2)
        grid_thw_N3 = torch.tensor([[1, 2, 2], [1, 4, 2]])
        logits_BLV = model(
            tokens_BL,
            pixel_values=pixel_values_NPK,
            grid_thw=grid_thw_N3,
            special_tokens={"image_id": image_token_id},
        )

        self.assertEqual(logits_BLV.shape, (2, 6, config.vocab_size))
        moe = model.layers["1"].moe
        assert moe is not None
        self.assertIs(
            moe._buffers["tokens_per_expert_E"],
            moe.tokens_per_expert_E,
        )
        self.assertEqual(moe.tokens_per_expert_E.sum().item(), tokens_BL.numel())
        logits_BLV.float().square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

        state_dict = model.state_dict()
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
        hf_state_dict = adapter.to_hf(state_dict)
        roundtrip_state_dict = adapter.from_hf(hf_state_dict)
        self.assertEqual(state_dict.keys(), roundtrip_state_dict.keys())
        for key, value in state_dict.items():
            torch.testing.assert_close(value, roundtrip_state_dict[key])


if __name__ == "__main__":
    unittest.main()
