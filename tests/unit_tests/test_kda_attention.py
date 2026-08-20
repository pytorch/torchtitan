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

from torchtitan.models.common.attention import create_varlen_metadata_for_document

_HAS_BLACKWELL = (
    importlib.util.find_spec("attn_gym") is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (10, 0)
)
_HAS_VLLM = importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(
    _HAS_BLACKWELL, "KDA requires attention-gym and CUDA capability 10.0 or newer"
)
class TestKDA(unittest.TestCase):
    def _make_kda(self, *, backend: str = "chunked"):
        """Build a KDA layer with deterministic weights."""
        from torchtitan.models.common import Conv1d, Linear, RMSNorm
        from torchtitan.models.common.attention import (
            KDA,
            KDAAttention,
            KDAInnerAttention,
        )

        def linear(in_features: int, out_features: int) -> Linear.Config:
            return Linear.Config(
                in_features=in_features, out_features=out_features, bias=False
            )

        model = KDA.Config(
            num_heads=2,
            head_dim=128,
            in_proj_qkv=linear(32, 768),
            conv_qkv=Conv1d.Config(
                in_channels=768,
                out_channels=768,
                kernel_size=4,
                groups=768,
                bias=False,
            ),
            gate_proj_a=linear(32, 128),
            gate_proj_b=linear(128, 256),
            beta_proj=linear(32, 2),
            out_gate_proj_a=linear(32, 128),
            out_gate_proj_b=linear(128, 256),
            out_norm=RMSNorm.Config(normalized_shape=128),
            out_proj=linear(256, 32),
            attention=KDAAttention.Config(
                head_dim=128,
                inner_attention=KDAInnerAttention.Config(backend=backend),
            ),
        ).build()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for param in model.parameters():
                values = torch.linspace(
                    -0.2, 0.2, param.numel(), dtype=param.dtype, device=param.device
                )
                param.copy_(values.reshape_as(param))
            model.A_log.fill_(0.0)
            model.dt_bias.zero_()
            model.out_norm.weight.fill_(1.0)
        return model

    def _inputs(self, seed: int, tokens: int = 128) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(1, tokens, 32, device="cuda", dtype=torch.bfloat16)

    def test_chunked_and_recurrent_backends_agree(self):
        chunked = self._make_kda(backend="chunked")
        recurrent = self._make_kda(backend="recurrent")
        recurrent.load_state_dict(chunked.state_dict())

        x_BLD = self._inputs(seed=0)
        torch.testing.assert_close(
            chunked(x_BLD).float(), recurrent(x_BLD).float(), rtol=5e-2, atol=5e-2
        )

    def test_varlen_matches_independent_document_forwards(self):
        lengths = (37, 64, 91)
        x_BLD = self._inputs(seed=2, tokens=sum(lengths))
        positions = torch.tensor(
            [[index for length in lengths for index in range(length)]],
            device="cuda",
            dtype=torch.int32,
        )
        masks = create_varlen_metadata_for_document(
            positions, include_host_offsets=True
        )

        for backend in ("chunked", "recurrent"):
            model = self._make_kda(backend=backend)
            packed = model(x_BLD, masks)
            start = 0
            for document, length in enumerate(lengths):
                with self.subTest(backend=backend, document=document):
                    document_slice = slice(start, start + length)
                    torch.testing.assert_close(
                        packed[:, document_slice].float(),
                        model(x_BLD[:, document_slice], None).float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )
                start += length


@unittest.skipUnless(
    _HAS_BLACKWELL and _HAS_VLLM,
    "paged KDA requires attention-gym, vLLM, and CUDA capability 10.0 or newer",
)
class TestVLLMKDA(unittest.TestCase):
    def _wrapper(self):
        from torchtitan.experiments.rl.models.kda_attention import VLLMKDAWrapper

        wrapper = object.__new__(VLLMKDAWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.local_num_heads = 2
        wrapper.head_dim = 128
        wrapper.gate_lower_bound = -5.0
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
            tensor.reshape(1, tokens, heads, head_dim).contiguous()
            for tensor in mixed_qkv_TC.unflatten(-1, (heads, 3, head_dim)).unbind(-2)
        )
        cumulative_gate = bounded_gate_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            chunk_size=64,
            lower_bound=wrapper.gate_lower_bound,
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
                lower_bound=wrapper.gate_lower_bound,
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
