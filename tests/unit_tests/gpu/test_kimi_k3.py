# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from attn_gym.linear.kda import bound_gate, chunk_kda
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.kimi_k3 import _kimi_k3_config, _vision_encoder_config
from torchtitan.models.kimi_k3.kda import KDAKernel
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def _small_model_config() -> KimiK3Model.Config:
    """Build a reduced KDA+MLA, dense+MoE, multimodal Kimi K3 config."""
    dim = 64
    return _kimi_k3_config(
        max_context_length=128,
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


_KDA_FUDGE_FACTOR = 2.0
_KDA_FORWARD_ATOL = 2e-3
_KDA_GRAD_ATOL = 2e-2
_KDA_REFERENCE_RELATIVE_LIMIT = 0.1


def _kda_recurrent_oracle(
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
    """Explicit bounded KDA recurrence in the inputs' high precision."""
    q_BLHK = q_BLHK * torch.rsqrt(q_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    k_BLHK = k_BLHK * torch.rsqrt(k_BLHK.square().sum(dim=-1, keepdim=True) + 1e-6)
    log_decay_BLHK = lower_bound * torch.sigmoid(
        torch.exp(A_log_H).view(1, 1, -1, 1) * (gate_BLHK + dt_bias_HK)
    )
    decay_BLHK = torch.exp(log_decay_BLHK)
    beta_BLH = torch.sigmoid(beta_BLH)

    B, L, H, K = q_BLHK.shape
    V = v_BLHV.shape[-1]
    state_BHKV = torch.zeros(
        B,
        H,
        K,
        V,
        device=q_BLHK.device,
        dtype=q_BLHK.dtype,
    )
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
    return torch.stack(outputs_BHV, dim=1)


def _kda_eager_reference(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    raw_gate_BLHK: torch.Tensor,
    raw_beta_BLH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    *,
    lower_bound: float,
) -> torch.Tensor:
    """Run the public Attention Gym eager path with production dtypes."""

    def l2norm_reference(x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        return (
            x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + 1e-6)
        ).to(x.dtype)

    gate_BLHK = bound_gate(
        raw_gate_BLHK,
        A_log_H,
        dt_bias_HK,
        lower_bound=lower_bound,
        impl="reference",
    )
    output_BLHV, _ = chunk_kda(
        l2norm_reference(q_BLHK),
        l2norm_reference(k_BLHK),
        v_BLHV,
        gate_BLHK,
        raw_beta_BLH.float().sigmoid(),
        impl="reference",
    )
    return output_BLHV


class TestKimiK3(unittest.TestCase):
    def test_flex_attention_mask(self):
        config = _small_model_config()
        model = config.build()
        positions = torch.arange(4, dtype=torch.int32)
        attention_masks = model.get_attention_masks(positions)
        # MLA layers read the BlockMask; KDA layers read document offsets.
        self.assertIsInstance(attention_masks["quadratic_attention"], BlockMask)
        torch.testing.assert_close(
            attention_masks["kda"].cu_seq_q, torch.tensor([0, 4], dtype=torch.int32)
        )

    def test_padded_tail_is_one_kda_segment(self):
        config = _small_model_config()
        model = config.build()
        # Two documents (3 and 4 tokens), then padding numbered the way the
        # multimodal packer and collator emit it.
        positions = torch.cat([torch.arange(3), torch.arange(4), torch.arange(5)])
        padding_mask = torch.zeros(12, dtype=torch.bool)
        padding_mask[7:] = True
        masks = model.get_attention_masks(positions, padding_mask=padding_mask)
        torch.testing.assert_close(
            masks["kda"].cu_seq_q, torch.tensor([0, 3, 7, 12], dtype=torch.int32)
        )

    def _assert_kda_error_bound(
        self,
        name: str,
        reference: torch.Tensor,
        target: torch.Tensor,
        golden: torch.Tensor,
        *,
        project_atol: float,
    ) -> None:
        self.assertEqual(reference.shape, target.shape, name)
        self.assertEqual(reference.shape, golden.shape, name)
        self.assertEqual(reference.dtype, target.dtype, name)
        for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
            self.assertTrue(
                torch.equal(predicate(reference), predicate(target)),
                name,
            )
            self.assertTrue(
                torch.equal(predicate(reference), predicate(golden)),
                name,
            )

        reference_64 = reference.double()
        target_64 = target.double()
        golden_64 = golden.double()
        reference_error = (reference_64 - golden_64).abs().max().item()
        target_error = (target_64 - golden_64).abs().max().item()
        rounding_floor = (
            (golden_64.to(target.dtype).to(golden.dtype) - golden_64).abs().max().item()
        )
        absolute_floor = max(project_atol, rounding_floor)
        golden_scale = golden_64.abs().max().item()
        reference_relative_error = reference_error / max(
            golden_scale,
            absolute_floor,
        )
        self.assertLessEqual(
            reference_relative_error,
            _KDA_REFERENCE_RELATIVE_LIMIT,
            f"{name}: eager reference is too inaccurate to gate the kernel",
        )
        threshold = _KDA_FUDGE_FACTOR * reference_error + absolute_floor
        self.assertLessEqual(
            target_error,
            threshold,
            f"{name}: target_error={target_error:.8e}, "
            f"reference_error={reference_error:.8e}, "
            f"rounding_floor={rounding_floor:.8e}, threshold={threshold:.8e}",
        )

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in {(10, 0), (10, 3)},
        "Attention Gym KDA requires CUDA capability 10.0 or 10.3.",
    )
    def test_attention_gym_kda_kernel_matches_recurrent_reference(self):
        head_dim = 128
        num_heads = 2
        lower_bound = -5.0
        input_names = (
            "q",
            "k",
            "v",
            "raw_gate",
            "raw_beta",
            "A_log",
            "dt_bias",
        )

        for seed, num_tokens in ((1, 1), (2, 63), (3, 64), (4, 65)):
            with self.subTest(seed=seed, num_tokens=num_tokens):
                torch.manual_seed(seed)

                def parameter(
                    *shape: int,
                    dtype: torch.dtype = torch.bfloat16,
                ) -> torch.Tensor:
                    return torch.randn(
                        *shape,
                        device="cuda",
                        dtype=dtype,
                        requires_grad=True,
                    )

                A_log_H = (
                    torch.empty(num_heads, device="cuda", dtype=torch.float32)
                    .uniform_(1.0, 16.0)
                    .log_()
                    .requires_grad_()
                )
                inputs = (
                    parameter(1, num_tokens, num_heads, head_dim),
                    parameter(1, num_tokens, num_heads, head_dim),
                    parameter(1, num_tokens, num_heads, head_dim),
                    parameter(1, num_tokens, num_heads, head_dim),
                    parameter(1, num_tokens, num_heads),
                    A_log_H,
                    torch.zeros(
                        num_heads,
                        head_dim,
                        device="cuda",
                        dtype=torch.float32,
                        requires_grad=True,
                    ),
                )
                target_inputs = tuple(
                    tensor.detach().clone().requires_grad_() for tensor in inputs
                )
                reference_inputs = tuple(
                    tensor.detach().clone().requires_grad_() for tensor in inputs
                )
                golden_inputs = tuple(
                    tensor.detach().double().requires_grad_() for tensor in inputs
                )

                kernel = KDAKernel.Config(lower_bound=lower_bound).build()
                target_BLHV = kernel(*target_inputs)
                reference_BLHV = _kda_eager_reference(
                    *reference_inputs,
                    lower_bound=lower_bound,
                )
                golden_BLHV = _kda_recurrent_oracle(
                    *golden_inputs,
                    lower_bound=lower_bound,
                )
                self._assert_kda_error_bound(
                    "output",
                    reference_BLHV,
                    target_BLHV,
                    golden_BLHV,
                    project_atol=_KDA_FORWARD_ATOL,
                )

                output_grad_BLHV = torch.randn_like(target_BLHV)
                target_grads = torch.autograd.grad(
                    target_BLHV,
                    target_inputs,
                    grad_outputs=output_grad_BLHV,
                )
                reference_grads = torch.autograd.grad(
                    reference_BLHV,
                    reference_inputs,
                    grad_outputs=output_grad_BLHV,
                )
                golden_grads = torch.autograd.grad(
                    golden_BLHV,
                    golden_inputs,
                    grad_outputs=output_grad_BLHV.double(),
                )
                for name, reference_grad, target_grad, golden_grad in zip(
                    input_names,
                    reference_grads,
                    target_grads,
                    golden_grads,
                    strict=True,
                ):
                    self._assert_kda_error_bound(
                        f"grad_{name}",
                        reference_grad,
                        target_grad,
                        golden_grad,
                        project_atol=_KDA_GRAD_ATOL,
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
