# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.token_dispatcher import MinimalAsyncEPTokenDispatcher
from torchtitan.models.kimi_k3 import (
    _kimi_k3_config,
    _vision_encoder_config,
    model_registry,
)
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.models.kimi_k3.data import KimiK3MultiModalCollator
from torchtitan.models.kimi_k3.kda import (
    _compiled_rms_norm_gated,
    _rms_norm_gated,
    KimiDeltaAttention,
    KimiKDAKernel,
)
from torchtitan.models.kimi_k3.model import (
    _attention_residual,
    _compiled_attention_residual,
    KimiK3Model,
    KimiMLAAttention,
)
from torchtitan.models.kimi_k3.moe import _compiled_situ_glu, _situ_glu
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
        kda_head_dim=64,
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
    @unittest.skipIf(not torch.cuda.is_available(), "Gated norm test requires CUDA.")
    def test_compiled_rms_norm_gated_matches_eager(self):
        torch.manual_seed(3)
        actual_inputs = (
            torch.randn(
                8,
                128,
                64,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            torch.randn(
                8,
                128,
                64,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            torch.randn(64, device="cuda", requires_grad=True),
        )
        expected_inputs = tuple(
            value.detach().clone().requires_grad_() for value in actual_inputs
        )

        actual = _compiled_rms_norm_gated(*actual_inputs, 1e-5)
        expected = _rms_norm_gated(*expected_inputs, 1e-5)
        torch.testing.assert_close(actual, expected)

        output_grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual,
            actual_inputs,
            grad_outputs=output_grad,
        )
        expected_grads = torch.autograd.grad(
            expected,
            expected_inputs,
            grad_outputs=output_grad,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
            strict=True,
        ):
            torch.testing.assert_close(actual_grad, expected_grad)

    @unittest.skipIf(not torch.cuda.is_available(), "Residual test requires CUDA.")
    def test_compiled_attention_residual_matches_eager(self):
        torch.manual_seed(4)
        actual_inputs = (
            torch.randn(
                128,
                256,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            torch.randn(
                128,
                5,
                256,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            torch.randn(256, device="cuda", requires_grad=True),
            torch.randn(256, device="cuda", requires_grad=True),
        )
        expected_inputs = tuple(
            value.detach().clone().requires_grad_() for value in actual_inputs
        )

        actual_TD = _compiled_attention_residual(*actual_inputs, 1e-5)
        expected_TD = _attention_residual(*expected_inputs, 1e-5)
        torch.testing.assert_close(
            actual_TD,
            expected_TD,
            rtol=2e-2,
            atol=4e-3,
        )

        output_grad_TD = torch.randn_like(actual_TD)
        actual_grads = torch.autograd.grad(
            actual_TD,
            actual_inputs,
            grad_outputs=output_grad_TD,
        )
        expected_grads = torch.autograd.grad(
            expected_TD,
            expected_inputs,
            grad_outputs=output_grad_TD,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
            strict=True,
        ):
            tolerance = 2e-2 if actual_grad.dtype == torch.bfloat16 else 2e-4
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=tolerance,
                atol=tolerance,
            )

    @unittest.skipIf(not torch.cuda.is_available(), "SiTU test requires CUDA.")
    def test_compiled_situ_glu_matches_eager(self):
        torch.manual_seed(5)
        gate_RF = torch.randn(
            128,
            256,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up_RF = torch.randn_like(gate_RF, requires_grad=True)
        expected_gate_RF = gate_RF.detach().clone().requires_grad_()
        expected_up_RF = up_RF.detach().clone().requires_grad_()

        actual_RF = _compiled_situ_glu(gate_RF, up_RF, 4.0, 25.0)
        expected_RF = _situ_glu(expected_gate_RF, expected_up_RF, 4.0, 25.0)
        torch.testing.assert_close(actual_RF, expected_RF)

        output_grad_RF = torch.randn_like(actual_RF)
        actual_grads = torch.autograd.grad(
            actual_RF,
            (gate_RF, up_RF),
            grad_outputs=output_grad_RF,
        )
        expected_grads = torch.autograd.grad(
            expected_RF,
            (expected_gate_RF, expected_up_RF),
            grad_outputs=output_grad_RF,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
            strict=True,
        ):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_collator_batches_equal_length_rows(self):
        tokenizer = type("Tokenizer", (), {"pad_id": 99})()
        collator = KimiK3MultiModalCollator.Config().build(
            context=SimpleNamespace(
                tokenizer=tokenizer,
                num_tokens_per_batch=8,
                max_context_length=4,
            )
        )
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([2, 3, 4]),
                "positions": torch.tensor([0, 1, 2]),
            },
            {
                "input_ids": torch.tensor([5, 6]),
                "labels": torch.tensor([6, 7]),
                "positions": torch.tensor([0, 1]),
            },
        ]

        input_ids, labels, positions = collator.collate_text(batch)

        torch.testing.assert_close(
            input_ids,
            torch.tensor([1, 2, 3, 99, 5, 6, 99, 99]),
        )
        torch.testing.assert_close(
            labels,
            torch.tensor([2, 3, 4, -100, 6, 7, -100, -100]),
        )
        torch.testing.assert_close(
            positions,
            torch.tensor([0, 1, 2, 0, 0, 1, 0, 1]),
        )

    def test_minimal_async_ep_uses_latent_expert_dim(self):
        trainer_config = kimi_k3_debugmodel()
        trainer_config.model_spec = model_registry(
            "debugmodel",
            moe_comm_backend="minimal_async_ep",
        )
        trainer_config.parallelism = ParallelismConfig(
            data_parallel_shard_degree=4,
            expert_parallel_degree=4,
            spmd_backend="partial_dtensor",
        )
        trainer_config.activation_checkpoint = FullAC.Config()
        trainer_config.debug.moe_force_load_balance = True

        model_config = trainer_config.model_spec.model
        model_config.update_from_config(config=trainer_config)

        for layer_config in model_config.layers:
            if layer_config.moe is None:
                continue
            routed_experts = layer_config.moe.routed_experts
            dispatcher = routed_experts.token_dispatcher
            self.assertIsInstance(dispatcher, MinimalAsyncEPTokenDispatcher.Config)
            self.assertEqual(dispatcher.hidden_dim, routed_experts.inner_experts.dim)
            self.assertEqual(dispatcher.hidden_dim, 512)
            self.assertEqual(dispatcher.receive_capacity, 1024)
            self.assertIsNotNone(routed_experts.inner_experts.sharding_config)

    def test_flex_attention_mask(self):
        config = _small_model_config()
        model = config.build()
        positions = torch.arange(4, dtype=torch.int32)
        attention_masks = model.get_attention_masks(positions)
        self.assertIsInstance(attention_masks, BlockMask)

    @unittest.skipIf(not torch.cuda.is_available(), "FLA KDA kernel requires CUDA.")
    def test_fla_causal_conv_matches_torch(self):
        torch.manual_seed(1)
        config = _small_model_config().layers[0].delta_attention
        assert config is not None
        attention = KimiDeltaAttention(config).cuda().bfloat16()
        attention.q_conv.weight.data.normal_(std=0.02)
        input_TC = torch.randn(
            64,
            attention.q_conv.in_channels,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        expected_input_TC = input_TC.detach().clone().requires_grad_()

        actual_TC = attention._causal_conv(
            input_TC,
            attention.q_conv,
            num_sequences=1,
        )
        expected_1CT = F.pad(
            expected_input_TC.T.unsqueeze(0),
            (attention.conv_kernel_size - 1, 0),
        )
        expected_TC = F.silu(attention.q_conv(expected_1CT)).squeeze(0).T
        torch.testing.assert_close(actual_TC, expected_TC, atol=2e-3, rtol=2e-3)

        output_grad_TC = torch.randn_like(actual_TC)
        actual_grad_TC = torch.autograd.grad(
            actual_TC,
            input_TC,
            grad_outputs=output_grad_TC,
        )[0]
        expected_grad_TC = torch.autograd.grad(
            expected_TC,
            expected_input_TC,
            grad_outputs=output_grad_TC,
        )[0]
        torch.testing.assert_close(
            actual_grad_TC,
            expected_grad_TC,
            atol=2e-2,
            rtol=2e-2,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "FLA KDA kernel requires CUDA.")
    def test_kda_batch_matches_independent_sequences(self):
        torch.manual_seed(3)
        config = _small_model_config().layers[0].delta_attention
        assert config is not None
        attention = KimiDeltaAttention(config).cuda().bfloat16()
        for parameter in attention.parameters():
            parameter.data.normal_(std=0.02)
        input_TD = torch.randn(
            128,
            config.dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        actual_TD = attention(
            input_TD,
            sequence_offsets=torch.tensor([0, 64, 128], device="cuda"),
        )
        expected_TD = torch.cat([attention(input_TD[:64]), attention(input_TD[64:])])

        torch.testing.assert_close(actual_TD, expected_TD, atol=2e-3, rtol=2e-3)

    @unittest.skipIf(not torch.cuda.is_available(), "Attention test requires CUDA.")
    def test_batched_sdpa_matches_document_masked_flex_attention(self):
        torch.manual_seed(4)
        model_config = _small_model_config()
        flex_config = model_config.layers[1].attention
        assert flex_config is not None
        sdpa_config = replace(
            flex_config,
            inner_attention=ScaledDotProductAttention.Config(),
        )
        flex_attention = KimiMLAAttention(flex_config).cuda().bfloat16()
        sdpa_attention = KimiMLAAttention(sdpa_config).cuda().bfloat16()
        for parameter in flex_attention.parameters():
            parameter.data.normal_(std=0.02)
        sdpa_attention.load_state_dict(flex_attention.state_dict())

        input_TD = torch.randn(
            128,
            flex_config.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        expected_input_TD = input_TD.detach().clone().requires_grad_()
        positions = torch.arange(64, device="cuda").repeat(2)
        attention_masks = model_config.build().get_attention_masks(positions)

        actual_TD = sdpa_attention(
            input_TD,
            sequence_offsets=torch.tensor([0, 64, 128], device="cuda"),
        )
        expected_TD = flex_attention(
            expected_input_TD,
            attention_masks=attention_masks,
        )
        torch.testing.assert_close(actual_TD, expected_TD, atol=2e-2, rtol=2e-2)

        output_grad_TD = torch.randn_like(actual_TD)
        actual_grad_TD = torch.autograd.grad(
            actual_TD,
            input_TD,
            grad_outputs=output_grad_TD,
        )[0]
        expected_grad_TD = torch.autograd.grad(
            expected_TD,
            expected_input_TD,
            grad_outputs=output_grad_TD,
        )[0]
        torch.testing.assert_close(
            actual_grad_TD,
            expected_grad_TD,
            atol=2e-2,
            rtol=2e-2,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "FLA KDA kernel requires CUDA.")
    def test_fla_kda_kernel_matches_recurrent_reference(self):
        torch.manual_seed(1)
        head_dim = 64
        num_heads = 3

        for lower_bound in (-5.0, None):
            for disable_recompute in (False, True):
                with self.subTest(
                    lower_bound=lower_bound,
                    disable_recompute=disable_recompute,
                ):
                    self._check_fla_kda_kernel(
                        head_dim,
                        num_heads,
                        lower_bound=lower_bound,
                        disable_recompute=disable_recompute,
                    )

    def _check_fla_kda_kernel(
        self,
        head_dim: int,
        num_heads: int,
        *,
        lower_bound: float | None,
        disable_recompute: bool,
    ) -> None:
        def parameter(*shape: int) -> torch.Tensor:
            return torch.randn(
                *shape,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )

        A_log_H = torch.rand(num_heads, device="cuda")
        A_log_H = A_log_H.uniform_(1.0, 16.0).log().requires_grad_()
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

        kernel = KimiKDAKernel.Config(
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
        ).build()
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
