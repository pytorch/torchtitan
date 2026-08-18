# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import BlockMask
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims

# FLA is a per-model dependency (kimi_k3/requirements.txt) imported at module
# scope for the KDA kernel, so skip rather than fail collection without it.
try:
    from torchtitan.models.kimi_k3 import (
        _kimi_k3_config,
        _vision_encoder_config,
        parallelize_kimi_k3,
    )
    from torchtitan.models.kimi_k3.model import KimiK3Model, KimiKDAKernel
    from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        f"Kimi K3 optional dependency unavailable: {exc.name}"
    ) from exc


def _small_model_config(
    *,
    attn_res_block_size: int = 1,
    full_attention_layers: set[int] | None = None,
) -> KimiK3Model.Config:
    """Build a reduced KDA+MLA, dense+MoE, multimodal Kimi K3 config."""
    if full_attention_layers is None:
        full_attention_layers = {1}

    dim = 64
    return _kimi_k3_config(
        dim=dim,
        vocab_size=32,
        num_layers=2,
        full_attention_layers=full_attention_layers,
        attn_res_block_size=attn_res_block_size,
        num_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kda_head_dim=32,
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
    def test_multimodal_forward(self):
        # All-MLA so the forward runs without the CUDA-only KDA kernel; the
        # KDA path is covered by the FSDP parity test below.
        config = _small_model_config(full_attention_layers={0, 1})
        model = config.build()
        model.init_states()
        positions = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        attention_masks = model.get_attention_masks(positions)
        self.assertIsInstance(attention_masks, BlockMask)
        with torch.no_grad():
            logits = model(
                torch.tensor([[1, 7, 2, 3, 4, 5]]),
                pixel_values=torch.randn(1, 4, 3 * 2 * 2),
                grid_thw=torch.tensor([[1, 2, 2]]),
                special_tokens={"image_id": 7},
                positions=positions,
                attention_masks=attention_masks,
            )
        self.assertEqual(logits.shape, (1, 6, config.vocab_size))

    def test_update_from_config_propagates_moe_force_load_balance(self):
        from torchtitan.config import DebugConfig
        from torchtitan.trainer import Trainer

        model_config = _small_model_config()
        runtime_config = Trainer.Config(
            debug=DebugConfig(moe_force_load_balance=True),
            activation_checkpoint=None,
        )
        model_config.update_from_config(config=runtime_config)

        router_configs = [
            layer.moe.router for layer in model_config.layers if layer.moe is not None
        ]
        self.assertGreater(len(router_configs), 0)
        self.assertTrue(
            all(router._debug_force_load_balance for router in router_configs)
        )

    @unittest.skipIf(not torch.cuda.is_available(), "FLA KDA kernel requires CUDA.")
    def test_fla_kda_kernel_matches_recurrent_reference(self):
        torch.manual_seed(1)
        head_dim = 32
        num_heads = 3

        def parameter(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, device="cuda", requires_grad=True)

        for lower_bound in (-5.0, None):
            with self.subTest(lower_bound=lower_bound):
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
                    head_dim=head_dim,
                    lower_bound=lower_bound,
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


class TestKimiK3FSDP(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @unittest.skipIf(not torch.cuda.is_available(), "Kimi K3 FSDP requires CUDA.")
    @with_comms
    def test_fsdp_matches_non_distributed_forward_backward(self):
        torch.manual_seed(3)
        # Layer 0 is KDA, layer 1 is MLA, so one run covers both attentions.
        config = _small_model_config(attn_res_block_size=2)
        with torch.device("meta"):
            model = config.build()
        model.to_empty(device=self.device_type)
        model.init_states()
        with torch.no_grad():
            for transformer_block in model.layers.values():
                if transformer_block.moe is not None:
                    transformer_block.moe.router.gate.weight.zero_()

        reference = copy.deepcopy(model)
        for parameter in reference.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)

        parallelism = ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
            spmd_backend="partial_dtensor",
        )
        parallel_dims = ParallelDims.from_config(parallelism, world_size=1)
        with patch(
            "torchtitan.distributed.parallel_dims.device_type",
            self.device_type,
        ):
            parallel_dims.build_mesh()
        model = parallelize_kimi_k3(
            model,
            parallel_dims=parallel_dims,
            training=TrainingConfig(
                local_batch_size=1,
                seq_len=6,
                steps=1,
                dtype="bfloat16",
            ),
            parallelism=parallelism,
            compile_config=CompileConfig(),
            ac_config=None,
            dump_folder="",
        )

        assert isinstance(model, KimiK3Model)
        self.assertIsInstance(model, FSDPModule)
        self.assertIsInstance(model.vision_encoder, FSDPModule)

        positions_BL = torch.arange(
            6,
            dtype=torch.int32,
            device=self.device_type,
        ).unsqueeze(0)
        attention_masks = reference.get_attention_masks(positions_BL)
        inputs = {
            "tokens": torch.tensor(
                [[1, 7, 2, 3, 4, 5]],
                dtype=torch.long,
                device=self.device_type,
            ),
            "pixel_values": torch.randn(
                1,
                4,
                3 * 2 * 2,
                device=self.device_type,
            ),
            "grid_thw": torch.tensor(
                [[1, 2, 2]],
                dtype=torch.long,
                device=self.device_type,
            ),
            "special_tokens": {"image_id": 7},
            "positions": positions_BL,
            "attention_masks": attention_masks,
        }

        actual_BLV = model(**inputs)  # pyrefly: ignore [not-callable]
        expected_BLV = reference(**inputs)
        torch.testing.assert_close(actual_BLV, expected_BLV, atol=0.0, rtol=0.0)

        actual_BLV.float().square().mean().backward()
        expected_BLV.float().square().mean().backward()

        reference_parameters = dict(reference.named_parameters())
        compared_gradients = 0
        for name, parameter in model.named_parameters():
            actual_grad = parameter.grad
            expected_grad = reference_parameters[name].grad
            self.assertEqual(actual_grad is None, expected_grad is None)
            if actual_grad is None:
                continue
            if isinstance(actual_grad, DTensor):
                actual_grad = actual_grad.to_local()
            assert expected_grad is not None
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad.float(),
                atol=0.0,
                rtol=0.0,
            )
            compared_gradients += 1
        self.assertGreater(compared_gradients, 0)


if __name__ == "__main__":
    unittest.main()
