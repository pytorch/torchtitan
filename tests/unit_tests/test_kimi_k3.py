# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config
from torchtitan.models.kimi_k3.kda import KDAKernel
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def _small_model_config() -> KimiK3Model.Config:
    """Build a reduced KDA+MLA, dense+MoE, multimodal Kimi K3 config."""
    dim = 64
    return _kimi_k3_config(
        dim=dim,
        vocab_size=32,
        num_layers=2,
        full_attention_layers={1},
        attn_res_block_size=1,
        num_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kda_head_dim=128,
        conv_kernel_size=3,
        dense_hidden_dim=128,
        latent_dim=32,
        expert_hidden_dim=32,
        num_experts=2,
        top_k=1,
        num_shared_experts=1,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=48,
            qkv_dim=48,
            hidden_dim=96,
            num_layers=1,
            num_heads=3,
            patch_size=2,
            merge_kernel_size=(2, 2),
            init_pos_emb_height=2,
            init_pos_emb_width=2,
            max_num_frames=1,
        ),
        attn_backend="flex",
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
    """Explicit bounded KDA recurrence in FP32."""
    input_dtype = q_BLHK.dtype
    q_BLHK = q_BLHK.float()
    k_BLHK = k_BLHK.float()
    q_BLHK = q_BLHK * torch.rsqrt(q_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    k_BLHK = k_BLHK * torch.rsqrt(k_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    v_BLHV = v_BLHV.float()
    log_decay_BLHK = lower_bound * torch.sigmoid(
        torch.exp(A_log_H).view(1, 1, -1, 1) * (gate_BLHK.float() + dt_bias_HK.float())
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
    def test_flex_attention_mask(self):
        config = _small_model_config()
        model = config.build()
        positions = torch.arange(4, dtype=torch.int32)
        attention_masks = model.get_attention_masks(positions)
        self.assertIsInstance(attention_masks, BlockMask)

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in {(10, 0), (10, 3)},
        "Attention Gym KDA requires CUDA capability 10.0 or 10.3.",
    )
    def test_attention_gym_kda_kernel_matches_recurrent_reference(self):
        torch.manual_seed(1)
        head_dim = 128
        num_heads = 3

        def parameter(*shape: int) -> torch.Tensor:
            return torch.randn(
                *shape,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )

        lower_bound = -5.0
        A_log_H = (
            torch.empty(num_heads, device="cuda")
            .uniform_(1.0, 16.0)
            .log_()
            .requires_grad_()
        )
        actual_inputs = (
            parameter(2, 64, num_heads, head_dim),
            parameter(2, 64, num_heads, head_dim),
            parameter(2, 64, num_heads, head_dim),
            parameter(2, 64, num_heads, head_dim),
            parameter(2, 64, num_heads),
            A_log_H,
            parameter(num_heads, head_dim),
        )
        expected_inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in actual_inputs
        )

        kernel = KDAKernel.Config(lower_bound=lower_bound).build()
        actual_BLHV = kernel(*actual_inputs)
        expected_BLHV = _kda_recurrent_reference(
            *expected_inputs,
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
        output_grad_BLHV = torch.randn_like(actual_BLHV)
        actual_grads = torch.autograd.grad(
            actual_BLHV,
            actual_inputs,
            grad_outputs=output_grad_BLHV,
        )
        expected_grads = torch.autograd.grad(
            expected_BLHV,
            expected_inputs,
            grad_outputs=output_grad_BLHV,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                atol=2e-2,
                rtol=2e-2,
            )

    def test_state_dict_round_trips_through_hf_adapter(self):
        torch.manual_seed(2)
        config = _small_model_config()
        model = config.build()
        model.init_states()

        state_dict = model.state_dict()
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
        hf_state_dict = adapter.to_hf(state_dict)
        self.assertIn(
            "layers.1.moe.routed_experts.inner_experts.w1_EFD",
            state_dict,
        )
        self.assertIn(
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight",
            hf_state_dict,
        )
        roundtrip_state_dict = adapter.from_hf(hf_state_dict)
        self.assertEqual(state_dict.keys(), roundtrip_state_dict.keys())
        for key, value in state_dict.items():
            torch.testing.assert_close(value, roundtrip_state_dict[key])


if __name__ == "__main__":
    unittest.main()
