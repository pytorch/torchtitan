# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_mla import UlyssesMLAAttention
from torchtitan.models.common.attention.mla import materialize_mla_kv
from torchtitan.models.common.linear import Linear
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.attention import KimiMLAAttention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMLAKVLocalCompile(unittest.TestCase):
    def setUp(self):
        previous = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    def test_compiled_matches_eager_bitwise(self):
        torch.manual_seed(42)
        layers = build_model_config(
            "debugmodel", attn_backend="flex", parallelism=ParallelismConfig()
        ).layers
        attention_config = next(
            layer.attention
            for layer in layers
            if isinstance(layer.attention, KimiMLAAttention.Config)
        )
        kv_width = attention_config.qk_nope_head_dim + attention_config.v_head_dim
        q_THK = torch.empty(
            64,
            attention_config.n_heads,
            attention_config.qk_nope_head_dim + attention_config.qk_rope_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        kv_THC = torch.randn(
            64,
            attention_config.n_heads,
            kv_width,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_rope_TK = torch.randn(
            64,
            attention_config.qk_rope_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def run():
            inputs = [
                kv_THC.clone().requires_grad_(),
                k_rope_TK.clone().requires_grad_(),
            ]
            k_THK, v_THV = materialize_mla_kv(q_THK, *inputs)
            generator = torch.Generator(device="cuda").manual_seed(42)
            grad_k_THK = torch.randn(
                k_THK.shape,
                dtype=k_THK.dtype,
                device="cuda",
                generator=generator,
            )
            grad_v_THV = torch.randn(
                v_THV.shape,
                dtype=v_THV.dtype,
                device="cuda",
                generator=generator,
            )
            grads = torch.autograd.grad(
                (k_THK, v_THV), inputs, (grad_k_THK, grad_v_THV)
            )
            return (k_THK, v_THV, *grads)

        LocalCompileConfig().apply()
        eager = run()
        LocalCompileConfig(regions=["mla_kv"]).apply()
        compiled = run()

        # This region only moves data and sums the repeated rope gradients.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _MathAttention(torch.nn.Module):
    """Use deterministic SDPA math while exercising the real compact CP path."""

    def forward(self, q_THK, k_THK, v_THV, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return (
                F.scaled_dot_product_attention(
                    q_THK.transpose(0, 1).unsqueeze(0),
                    k_THK.transpose(0, 1).unsqueeze(0),
                    v_THV.transpose(0, 1).unsqueeze(0),
                )
                .squeeze(0)
                .transpose(0, 1)
            )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestUlyssesMLAKVLocalCompile(DTensorTestBase):
    """Preserve every gradient through real CP2 communication and materialization."""

    @property
    def world_size(self):
        return 2

    @pytest.mark.multi_gpu
    @with_comms
    def test_compiled_compact_cp_preserves_shared_key_gradient(self):
        torch.manual_seed(42)
        previous = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        torch.use_deterministic_algorithms(True)
        self.addCleanup(LocalCompileConfig().apply)
        device = torch.device(self.device_type, self.rank)
        config = UlyssesMLAAttention.Config(
            wkv_b=Linear.Config(in_features=5, out_features=28),
            packed_kv_head_dim=7,
            inner_attention=FlexInnerAttention.Config(),
        )
        reference = config.build().to(device)
        compiled = config.build().to(device)
        compiled.load_state_dict(reference.state_dict())
        reference.inner_attention = _MathAttention()
        compiled.inner_attention = _MathAttention()
        generator = torch.Generator(device=device).manual_seed(42 + self.rank)
        inputs = [
            torch.randn(shape, generator=generator, device=device)
            for shape in ((4, 4, 6), (4, 5), (4, 2))
        ]

        def run(module):
            actual_inputs = [
                tensor.detach().clone().requires_grad_() for tensor in inputs
            ]
            with patch(
                "torchtitan.models.common.attention.cp_mla.spmd_mesh_group",
                return_value=dist.group.WORLD,
            ):
                output_THV = module(*actual_inputs)
                loss = output_THV.square().mean()
                gradients = torch.autograd.grad(
                    loss, (*actual_inputs, *module.parameters())
                )
            return output_THV, loss, *gradients

        LocalCompileConfig().apply()
        expected = run(reference)
        assert torch.count_nonzero(expected[4]) > 0
        LocalCompileConfig(regions=["mla_kv"]).apply()
        actual = run(compiled)
        for candidate, reference_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(candidate, reference_tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
