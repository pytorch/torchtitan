# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common import Embedding
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


def _small_model_config() -> KimiK3Model.Config:
    dim = 16

    def block(
        layer_id: int,
        *,
        use_mla: bool,
        use_moe: bool,
    ) -> KimiK3TransformerBlock.Config:
        return KimiK3TransformerBlock.Config(
            layer_id=layer_id,
            attn_res_block_size=1,
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

    return KimiK3Model.Config(
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


def _kda_recurrent_reference(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    gate_BLHK: torch.Tensor,
    beta_BLH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    *,
    lower_bound: float,
) -> torch.Tensor:
    q_BLHK = q_BLHK.float()
    k_BLHK = k_BLHK.float()
    q_BLHK = q_BLHK * torch.rsqrt(q_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    k_BLHK = k_BLHK * torch.rsqrt(k_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    log_decay_BLHK = lower_bound * torch.sigmoid(
        torch.exp(A_log_H.float()).view(1, 1, -1, 1)
        * (gate_BLHK.float() + dt_bias_HK.float())
    )
    decay_BLHK = torch.exp(log_decay_BLHK)
    beta_BLH = torch.sigmoid(beta_BLH.float())

    B, L, H, K = q_BLHK.shape
    V = v_BLHV.shape[-1]
    state_BHKV = torch.zeros(B, H, K, V)
    output_BLHV = torch.empty(B, L, H, V)
    for token_idx in range(L):
        state_BHKV = state_BHKV * decay_BLHK[:, token_idx].unsqueeze(-1)
        old_value_BHV = torch.matmul(
            k_BLHK[:, token_idx].unsqueeze(-2),
            state_BHKV,
        ).squeeze(-2)
        delta_BHV = (v_BLHV[:, token_idx].float() - old_value_BHV) * beta_BLH[
            :, token_idx
        ].unsqueeze(-1)
        state_BHKV = state_BHKV + (
            k_BLHK[:, token_idx].unsqueeze(-1) * delta_BHV.unsqueeze(-2)
        )
        output_BLHV[:, token_idx] = torch.matmul(
            q_BLHK[:, token_idx].unsqueeze(-2),
            state_BHKV,
        ).squeeze(-2) * (K**-0.5)
    return output_BLHV


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

        self.assertEqual(len(config.layers), 13)
        self.assertEqual(
            [
                layer_idx + 1
                for layer_idx, layer in enumerate(config.layers)
                if layer.attention is not None
            ],
            [4, 8, 12],
        )
        self.assertIsNotNone(config.layers[0].feed_forward)
        self.assertTrue(all(layer.moe is not None for layer in config.layers[1:]))
        moe_config = config.layers[1].moe
        assert moe_config is not None
        self.assertEqual(moe_config.num_experts, 8)
        self.assertEqual(moe_config.router.top_k, 2)

    def test_kda_kernel_matches_recurrent_reference(self):
        torch.manual_seed(1)
        q_BLHK = torch.randn(2, 5, 3, 4, requires_grad=True)
        k_BLHK = torch.randn(2, 5, 3, 4, requires_grad=True)
        v_BLHV = torch.randn(2, 5, 3, 4, requires_grad=True)
        gate_BLHK = torch.randn(2, 5, 3, 4, requires_grad=True)
        beta_BLH = torch.randn(2, 5, 3, requires_grad=True)
        A_log_H = torch.randn(3, requires_grad=True)
        dt_bias_HK = torch.randn(3, 4, requires_grad=True)

        kernel = KimiKDAKernel.Config(
            head_dim=4,
            lower_bound=-5.0,
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
            lower_bound=-5.0,
        )

        torch.testing.assert_close(
            actual_BLHV,
            expected_BLHV,
            atol=1e-6,
            rtol=1e-6,
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
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_small_multimodal_model_forward_backward_and_adapter(self):
        torch.manual_seed(2)
        config = _small_model_config()
        model = config.build()
        model.verify_module_protocol()
        model.init_states()

        tokens_BL = torch.randint(0, config.vocab_size, (2, 6))
        image_token_id = 7
        tokens_BL[0, 2] = image_token_id
        pixel_values_NPK = torch.randn(1, 4, 3 * 2 * 2)
        grid_thw_N3 = torch.tensor([[1, 2, 2]])
        logits_BLV = model(
            tokens_BL,
            pixel_values=pixel_values_NPK,
            grid_thw=grid_thw_N3,
            special_tokens={"image_id": image_token_id},
        )

        self.assertEqual(logits_BLV.shape, (2, 6, config.vocab_size))
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
