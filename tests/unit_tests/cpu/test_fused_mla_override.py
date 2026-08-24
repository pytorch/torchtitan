# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import cast

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.models.deepseek_v3.model import Attention, DeepSeekV3Model
from torchtitan.overrides.fused_mla import fused_mla_kv, fused_mla_q, FusedMLAAttention


class TestFusedMLAOverrideConfig(unittest.TestCase):
    def test_override_replaces_all_debug_attention_configs(self):
        config = deepseek_v3_debugmodel()
        model_spec = config.model_spec
        self.assertIsNotNone(model_spec)
        assert model_spec is not None
        model_config = cast(DeepSeekV3Model.Config, model_spec.model)
        stock_attention_config = copy.deepcopy(model_config.layers[0].attention)

        replacements = apply_overrides(
            OverrideConfig(
                imports=["torchtitan.overrides.fused_mla.fused_mla"],
            ),
            config,
        )

        self.assertEqual(len(replacements), len(model_config.layers))
        self.assertTrue(
            all(
                isinstance(layer.attention, FusedMLAAttention.Config)
                for layer in model_config.layers
            )
        )

        stock_attention = stock_attention_config.build()
        fused_attention = model_config.layers[0].attention.build()
        self.assertEqual(
            list(stock_attention.state_dict()),
            list(fused_attention.state_dict()),
        )


@unittest.skipUnless(torch.cuda.is_available(), "Fused MLA requires CUDA")
class TestFusedMLANumerics(unittest.TestCase):
    batch = 2
    seq_len = 16
    n_heads = 128
    q_nope_dim = 128
    rope_dim = 64
    value_dim = 128

    def setUp(self):
        torch.manual_seed(42)
        self._inductor_config_backup = dict(FlexAttention.inductor_configs)
        FlexAttention.inductor_configs["max_autotune"] = False
        FlexAttention.inductor_configs["coordinate_descent_tuning"] = False
        cuda = torch.device("cuda")
        self.rope = ComplexRoPE.Config(
            dim=self.rope_dim,
            max_seq_len=128,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ).build()
        self.rope = self.rope.to(cuda)
        self.positions = torch.stack(
            [
                torch.arange(self.seq_len, device=cuda),
                torch.arange(
                    self.seq_len - 1,
                    -1,
                    -1,
                    device=cuda,
                ),
            ]
        )

    def tearDown(self):
        FlexAttention.inductor_configs.clear()
        FlexAttention.inductor_configs.update(self._inductor_config_backup)
        torch._dynamo.reset()

    def assert_dtype_close(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        dtype: torch.dtype,
        *,
        reduction: bool = False,
        exact: bool = False,
        msg: str | None = None,
    ) -> None:
        if exact:
            rtol = atol = 0.0
        elif dtype == torch.bfloat16:
            rtol = atol = 2e-2 if reduction else 1e-2
        elif reduction:
            # The fused K-position backward changes the order of the FP32 head
            # reduction. Its observed error is ~5e-6 absolute / 2e-4 relative.
            rtol, atol = 2e-4, 1e-5
        else:
            rtol, atol = 1e-5, 1e-6
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            msg=msg,
        )

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_q_forward_backward_and_storage_match_eager(self, dtype: torch.dtype):
        self._check_q_forward_backward_and_storage(dtype)

    def _check_q_forward_backward_and_storage(self, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        q_source = torch.randn(
            self.batch,
            self.seq_len,
            self.n_heads,
            self.q_nope_dim + self.rope_dim,
            device=self.positions.device,
            dtype=dtype,
            requires_grad=True,
        )
        q_storage = q_source.clone()
        fused_q = fused_mla_q(
            q_storage,
            self.rope.cache,
            self.positions,
            self.q_nope_dim,
        )

        q_reference_source = q_source.detach().clone().requires_grad_()
        q_nope, q_pos = torch.split(
            q_reference_source,
            [self.q_nope_dim, self.rope_dim],
            dim=-1,
        )
        cache = self.rope._reshape_cache(q_pos, self.positions)
        q_pos, _ = self.rope.apply_rotary_emb(
            q_pos,
            q_pos[:, :, :1],
            cache,
        )
        reference_q = torch.cat([q_nope, q_pos], dim=-1)

        self.assertEqual(fused_q.data_ptr(), q_storage.data_ptr())
        self.assert_dtype_close(
            fused_q[..., : self.q_nope_dim],
            reference_q[..., : self.q_nope_dim],
            dtype,
            exact=True,
        )
        self.assert_dtype_close(fused_q, reference_q, dtype)

        grad_q = torch.randn_like(fused_q)
        (fused_grad,) = torch.autograd.grad(fused_q, q_source, grad_q.clone())
        (reference_grad,) = torch.autograd.grad(
            reference_q,
            q_reference_source,
            grad_q.clone(),
        )
        self.assert_dtype_close(
            fused_grad[..., : self.q_nope_dim],
            reference_grad[..., : self.q_nope_dim],
            dtype,
            exact=True,
        )
        self.assert_dtype_close(fused_grad, reference_grad, dtype)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_q_sum_backward_matches_eager(self, dtype: torch.dtype):
        q_source = torch.randn(
            self.batch,
            self.seq_len,
            self.n_heads,
            self.q_nope_dim + self.rope_dim,
            device=self.positions.device,
            dtype=dtype,
            requires_grad=True,
        )
        fused_q = fused_mla_q(
            q_source.clone(),
            self.rope.cache,
            self.positions,
            self.q_nope_dim,
        )

        q_reference_source = q_source.detach().clone().requires_grad_()
        q_nope, q_pos = torch.split(
            q_reference_source,
            [self.q_nope_dim, self.rope_dim],
            dim=-1,
        )
        cache = self.rope._reshape_cache(q_pos, self.positions)
        q_pos, _ = self.rope.apply_rotary_emb(
            q_pos,
            q_pos[:, :, :1],
            cache,
        )
        reference_q = torch.cat([q_nope, q_pos], dim=-1)

        (fused_grad,) = torch.autograd.grad(fused_q.sum(), q_source)
        (reference_grad,) = torch.autograd.grad(
            reference_q.sum(),
            q_reference_source,
        )
        self.assert_dtype_close(fused_grad, reference_grad, dtype)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_singleton_positions_broadcast_matches_eager(self, dtype: torch.dtype):
        self._check_singleton_positions_broadcast(dtype)

    def _check_singleton_positions_broadcast(self, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.n_heads,
            self.q_nope_dim + self.rope_dim,
            device=self.positions.device,
            dtype=dtype,
        )
        singleton_positions = self.positions[:1]
        q_nope, q_pos = torch.split(
            q,
            [self.q_nope_dim, self.rope_dim],
            dim=-1,
        )
        cache = self.rope._reshape_cache(q_pos, singleton_positions)
        q_pos, _ = self.rope.apply_rotary_emb(
            q_pos,
            q_pos[:, :, :1],
            cache,
        )
        reference_q = torch.cat([q_nope, q_pos], dim=-1)

        fused_q = fused_mla_q(
            q.clone(),
            self.rope.cache,
            singleton_positions,
            self.q_nope_dim,
        )
        self.assert_dtype_close(fused_q, reference_q, dtype)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_kv_forward_backward_and_storage_match_eager(self, dtype: torch.dtype):
        self._check_kv_forward_backward_and_storage(dtype)

    def _check_kv_forward_backward_and_storage(self, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        kv = torch.randn(
            self.batch,
            self.seq_len,
            self.n_heads,
            self.q_nope_dim + self.value_dim,
            device=self.positions.device,
            dtype=dtype,
            requires_grad=True,
        )
        k_pos = torch.randn(
            self.batch,
            self.seq_len,
            self.rope_dim,
            device=self.positions.device,
            dtype=dtype,
            requires_grad=True,
        )
        fused_k, fused_v = fused_mla_kv(
            kv,
            k_pos,
            self.rope.cache,
            self.positions,
            self.q_nope_dim,
        )

        reference_kv = kv.detach().clone().requires_grad_()
        reference_k_pos = k_pos.detach().clone().requires_grad_()
        k_nope, reference_v = torch.split(
            reference_kv,
            [self.q_nope_dim, self.value_dim],
            dim=-1,
        )
        k_pos_view = reference_k_pos.unsqueeze(2)
        cache = self.rope._reshape_cache(k_pos_view, self.positions)
        _, rotated_k_pos = self.rope.apply_rotary_emb(
            k_pos_view,
            k_pos_view,
            cache,
        )
        reference_k = torch.cat(
            [
                k_nope,
                rotated_k_pos.expand(-1, -1, self.n_heads, -1),
            ],
            dim=-1,
        )

        self.assertEqual(
            fused_v.untyped_storage().data_ptr(),
            kv.untyped_storage().data_ptr(),
        )
        self.assert_dtype_close(
            fused_k[..., : self.q_nope_dim],
            reference_k[..., : self.q_nope_dim],
            dtype,
            exact=True,
        )
        self.assert_dtype_close(fused_k, reference_k, dtype)
        self.assert_dtype_close(fused_v, reference_v, dtype, exact=True)

        grad_k = torch.randn_like(fused_k)
        grad_v = torch.randn_like(fused_v)
        fused_grad_kv, fused_grad_k_pos = torch.autograd.grad(
            (fused_k, fused_v),
            (kv, k_pos),
            (grad_k.clone(), grad_v.clone()),
        )
        reference_grad_kv, reference_grad_k_pos = torch.autograd.grad(
            (reference_k, reference_v),
            (reference_kv, reference_k_pos),
            (grad_k.clone(), grad_v.clone()),
        )
        self.assert_dtype_close(
            fused_grad_kv,
            reference_grad_kv,
            dtype,
            exact=True,
        )
        self.assert_dtype_close(
            fused_grad_k_pos,
            reference_grad_k_pos,
            dtype,
            reduction=True,
        )

    def test_make_fx_keeps_forward_and_backward_custom_ops(self):
        """GraphTrainer fake tracing sees stable fused MLA operator nodes."""
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.n_heads,
            self.q_nope_dim + self.rope_dim,
            device=self.positions.device,
            requires_grad=True,
        )
        kv = torch.randn(
            self.batch,
            self.seq_len,
            self.n_heads,
            self.q_nope_dim + self.value_dim,
            device=self.positions.device,
            requires_grad=True,
        )
        k_pos = torch.randn(
            self.batch,
            self.seq_len,
            self.rope_dim,
            device=self.positions.device,
            requires_grad=True,
        )
        grad_q = torch.randn_like(q)
        grad_k = torch.randn(
            *q.shape[:-1],
            self.q_nope_dim + self.rope_dim,
            device=q.device,
        )
        grad_v = torch.randn(
            *q.shape[:-1],
            self.value_dim,
            device=q.device,
        )

        def forward_backward(q, kv, k_pos, cache, positions, grad_q, grad_k, grad_v):
            q_out = fused_mla_q(
                q.clone(),
                cache,
                positions,
                self.q_nope_dim,
            )
            k, v = fused_mla_kv(
                kv,
                k_pos,
                cache,
                positions,
                self.q_nope_dim,
            )
            return torch.autograd.grad(
                (q_out, k, v),
                (q, kv, k_pos),
                (grad_q, grad_k, grad_v),
            )

        graph = make_fx(forward_backward, tracing_mode="fake")(
            q,
            kv,
            k_pos,
            self.rope.cache,
            self.positions,
            grad_q,
            grad_k,
            grad_v,
        )
        targets = [node.target for node in graph.graph.nodes]
        self.assertEqual(
            targets.count(torch.ops.torchtitan.fused_mla_q_rope_.default),
            2,
        )
        self.assertIn(torch.ops.torchtitan.fused_mla_k_rope.default, targets)
        self.assertIn(torch.ops.torchtitan.fused_mla_kv_backward.default, targets)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_attention_module_forward_backward_matches_eager(self, dtype: torch.dtype):
        """The override is numerically a drop-in replacement for Attention."""
        self._check_attention_module_forward_backward(dtype)

    def _check_attention_module_forward_backward(self, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        config = deepseek_v3_debugmodel()
        model_spec = config.model_spec
        self.assertIsNotNone(model_spec)
        assert model_spec is not None
        model_config = cast(DeepSeekV3Model.Config, model_spec.model)
        stock_config = copy.deepcopy(model_config.layers[0].attention)

        apply_overrides(
            OverrideConfig(
                imports=["torchtitan.overrides.fused_mla.fused_mla"],
            ),
            config,
        )
        fused_config = model_config.layers[0].attention
        self.assertIsInstance(fused_config, FusedMLAAttention.Config)

        stock = stock_config.build().to(self.positions.device)
        fused = fused_config.build().to(self.positions.device)
        self.assertIsInstance(stock, Attention)
        self.assertIsInstance(fused, FusedMLAAttention)

        # Convert parameters without casting the non-persistent complex RoPE
        # cache to a real dtype.
        for module in (stock, fused):
            for parameter in module.parameters():
                parameter.data = parameter.data.to(dtype)

        with torch.no_grad():
            for name, parameter in stock.named_parameters():
                if "norm.weight" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.02)
        fused.load_state_dict(stock.state_dict(), strict=True)

        batch = 2
        seq_len = 16
        hidden_dim = cast(Attention.Config, stock_config).dim
        x = torch.randn(
            batch,
            seq_len,
            hidden_dim,
            device=self.positions.device,
            dtype=dtype,
        )
        stock_x = x.detach().clone().requires_grad_()
        fused_x = x.detach().clone().requires_grad_()
        positions = self.positions[:, :seq_len]
        attention_mask = create_block_mask(
            lambda batch_idx, head_idx, query_idx, key_value_idx: (
                query_idx >= key_value_idx
            ),
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=self.positions.device,
        )

        stock_out = stock(
            stock_x,
            attention_masks=attention_mask,
            positions=positions,
        )
        fused_out = fused(
            fused_x,
            attention_masks=attention_mask,
            positions=positions,
        )
        self.assert_dtype_close(stock_out, fused_out, dtype)

        grad_out = torch.randn_like(stock_out)
        stock_out.backward(grad_out.clone())
        fused_out.backward(grad_out.clone())
        stock_x_grad = stock_x.grad
        fused_x_grad = fused_x.grad
        self.assertIsNotNone(stock_x_grad)
        self.assertIsNotNone(fused_x_grad)
        assert stock_x_grad is not None and fused_x_grad is not None
        self.assert_dtype_close(
            stock_x_grad,
            fused_x_grad,
            dtype,
            reduction=True,
        )

        stock_parameters = dict(stock.named_parameters())
        fused_parameters = dict(fused.named_parameters())
        self.assertEqual(stock_parameters.keys(), fused_parameters.keys())
        for name in stock_parameters:
            stock_grad = stock_parameters[name].grad
            fused_grad = fused_parameters[name].grad
            self.assertIsNotNone(stock_grad)
            self.assertIsNotNone(fused_grad)
            assert stock_grad is not None and fused_grad is not None
            self.assert_dtype_close(
                stock_grad,
                fused_grad,
                dtype,
                reduction=True,
                msg=f"parameter gradient differs: {name}",
            )


instantiate_parametrized_tests(TestFusedMLANumerics)


if __name__ == "__main__":
    unittest.main()
