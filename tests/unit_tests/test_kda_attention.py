# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.kimi_k3.kda import InnerKDA, KDA, KDAKernel, KimiRMSNormGated

_HAS_BLACKWELL = (
    importlib.util.find_spec("attn_gym") is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(10, 0), (10, 3)}
)

_HAS_VLLM = importlib.util.find_spec("vllm") is not None


def _kda_config() -> KDA.Config:
    def linear(in_features: int, out_features: int) -> Linear.Config:
        return Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

    projection_dim = 256

    def conv() -> Conv1d.Config:
        return Conv1d.Config(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=4,
            groups=projection_dim,
            bias=False,
        )

    return KDA.Config(
        num_heads=2,
        head_dim=128,
        conv_kernel_size=4,
        q_proj=linear(32, projection_dim),
        k_proj=linear(32, projection_dim),
        v_proj=linear(32, projection_dim),
        q_conv=conv(),
        k_conv=conv(),
        v_conv=conv(),
        forget_a=linear(32, 128),
        forget_b=linear(128, projection_dim),
        beta=linear(32, 2),
        output_gate=linear(32, projection_dim),
        inner_kda=InnerKDA.Config(
            head_dim=128,
            kernel=KDAKernel.Config(),
        ),
        output_norm=KimiRMSNormGated.Config(dim=128),
        output_proj=linear(projection_dim, 32),
    )


@unittest.skipUnless(
    _HAS_BLACKWELL, "KDA requires Attention Gym on CUDA capability 10.0 or 10.3"
)
class TestKDA(unittest.TestCase):
    def _make_kda(self):
        model = _kda_config().build()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(1)
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(mean=0.0, std=0.02)
            model.A_log.uniform_(1.0, 16.0).log_()
            model.dt_bias.zero_()
            model.output_norm.weight.fill_(1.0)
        return model

    def _inputs(self, seed: int, tokens: int = 128) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(tokens, 32, device="cuda", dtype=torch.bfloat16)

    def test_varlen_matches_independent_documents(self):
        lengths = (37, 64, 91)
        x_TD = self._inputs(seed=2, tokens=sum(lengths)).requires_grad_()
        positions_T = torch.tensor(
            [index for length in lengths for index in range(length)],
            device="cuda",
            dtype=torch.int32,
        )
        masks = create_varlen_metadata_for_document(
            positions_T,
            include_host_offsets=True,
        )
        self.assertEqual(masks.cu_seq_q_host, (0, 37, 101, 192))

        model = self._make_kda()
        packed_TD = model(x_TD, masks)
        independent_TD = torch.cat(
            [model(document_TD, None) for document_TD in x_TD.split(lengths)]
        )
        torch.testing.assert_close(
            packed_TD.float(),
            independent_TD.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        output_grad_TD = torch.randn_like(packed_TD)
        parameters = tuple(model.parameters())
        packed_grads = torch.autograd.grad(
            packed_TD,
            (x_TD, *parameters),
            output_grad_TD,
        )
        independent_grads = torch.autograd.grad(
            independent_TD,
            (x_TD, *parameters),
            output_grad_TD,
        )
        torch.testing.assert_close(
            packed_grads,
            independent_grads,
            rtol=2e-2,
            atol=2e-2,
        )


@unittest.skipUnless(
    _HAS_BLACKWELL and _HAS_VLLM,
    "paged KDA requires attention-gym, vLLM, and CUDA capability 10.0 or newer",
)
class TestVLLMInnerKDA(unittest.TestCase):
    def _wrapper(self):
        from torchtitan.experiments.rl.models.kda_attention import VLLMInnerKDA

        wrapper = object.__new__(VLLMInnerKDA)
        torch.nn.Module.__init__(wrapper)
        wrapper.local_num_heads = 2
        wrapper.head_dim = 128
        wrapper.lower_bound = -5.0
        return wrapper

    def test_prefill_advances_pool_without_gather_scatter(self):
        from attn_gym.linear.kda import bounded_gate_cumsum, chunk_kda, l2norm

        torch.manual_seed(5)
        wrapper = self._wrapper()
        tokens, heads, head_dim = 128, wrapper.local_num_heads, wrapper.head_dim
        mixed_qkv_TC = torch.randn(
            tokens, heads * 3 * head_dim, device="cuda", dtype=torch.bfloat16
        )
        raw_gate = torch.randn(
            1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        raw_beta = torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
        query_start_loc = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
        slots = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
        has_initial_state = torch.tensor([True, False], device="cuda")
        metadata = SimpleNamespace(
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=slots,
            has_initial_state=has_initial_state,
        )
        initial_pool = torch.randn(6, heads, head_dim, head_dim, device="cuda")
        actual_pool = initial_pool.clone()
        expected_pool = initial_pool.clone()
        conv_state = mixed_qkv_TC.new_empty(6, 3, mixed_qkv_TC.shape[1])
        conv_weight = mixed_qkv_TC.new_empty(mixed_qkv_TC.shape[1], 4)

        query, key, value = (
            tensor.reshape(1, tokens, heads, head_dim)
            for tensor in mixed_qkv_TC.unflatten(-1, (heads, 3, head_dim)).unbind(-2)
        )
        cumulative_gate = bounded_gate_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            chunk_size=64,
            lower_bound=wrapper.lower_bound,
            cu_seqlens=query_start_loc,
        )
        expected_initial = torch.stack(
            (expected_pool[slots[0]], torch.zeros_like(expected_pool[slots[1]]))
        ).transpose(-1, -2)
        expected, expected_state = chunk_kda(
            l2norm(query),
            l2norm(key),
            value,
            cumulative_gate,
            raw_beta.float().sigmoid(),
            expected_initial,
            cu_seqlens=query_start_loc,
            output_final_state=True,
        )
        self.assertIsNotNone(expected_state)

        def identity_conv(x, *_args, **_kwargs):
            return x

        with (
            torch.no_grad(),
            patch(
                "torchtitan.experiments.rl.models.kda_attention.causal_conv1d_fn",
                identity_conv,
            ),
        ):
            actual = wrapper._kda_prefill(
                mixed_qkv_TC,
                raw_gate,
                raw_beta,
                A_log,
                dt_bias,
                metadata,
                conv_state,
                conv_weight,
                actual_pool,
            )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_pool[slots], expected_state.transpose(-1, -2))
        torch.testing.assert_close(actual_pool[0], initial_pool[0], rtol=0, atol=0)

    def test_decode_uses_vllm_sd_cache_layout_and_ignores_padding(self):
        from attn_gym.linear.kda import recurrent_kda_decode
        from attn_gym.linear.kda.short_conv import causal_conv1d_decode

        torch.manual_seed(7)
        wrapper = self._wrapper()
        sequences, heads, head_dim = 2, wrapper.local_num_heads, wrapper.head_dim
        channels = heads * 3 * head_dim
        mixed_qkv_TC = torch.randn(
            sequences, channels, device="cuda", dtype=torch.bfloat16
        )
        raw_gate = torch.randn(
            1, sequences, heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        raw_beta = torch.randn(1, sequences, heads, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
        conv_weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
        metadata = SimpleNamespace(non_spec_state_indices_tensor=slots)
        conv_state = torch.randn(5, 3, channels, device="cuda", dtype=torch.bfloat16)
        recurrent_state = torch.randn(5, heads, head_dim, head_dim, device="cuda")
        expected_conv_state = conv_state.clone()
        expected_recurrent_state = recurrent_state.clone()

        with torch.no_grad():
            expected_conv = causal_conv1d_decode(
                mixed_qkv_TC,
                conv_weight,
                expected_conv_state,
                activation="silu",
                state_indices=slots,
            )
            expected = recurrent_kda_decode(
                expected_conv,
                raw_gate,
                raw_beta,
                A_log,
                dt_bias,
                expected_recurrent_state,
                slots,
                lower_bound=wrapper.lower_bound,
            )
            actual = wrapper._kda_decode(
                mixed_qkv_TC,
                raw_gate,
                raw_beta,
                A_log,
                dt_bias,
                metadata,
                conv_state,
                conv_weight,
                recurrent_state,
            )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(conv_state, expected_conv_state, rtol=0, atol=0)
        torch.testing.assert_close(
            recurrent_state, expected_recurrent_state, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
