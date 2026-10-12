# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense SDPA output/gradient validation against the independent math backend."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan_recipes.overrides.dense_causal_sdpa import (
    dense_causal_bf16_sdpa,
    dense_causal_fused_mla,
    DenseCausalBf16SDPA,
)


def test_dense_causal_fused_mla_matches_unfused_assembly_with_native_wrapper():
    torch.manual_seed(42)
    config = (
        build_model_config(
            "debugmodel",
            seq_len=128,
            attn_backend="varlen",
            parallelism=ParallelismConfig(),
        )
        .layers[0]
        .attention
    )
    fused_config = dense_causal_fused_mla(config)
    config.inner_attention = derive(
        config.inner_attention,
        type(config.inner_attention),
        inner_attention=dense_causal_bf16_sdpa(config.inner_attention.inner_attention),
    )
    reference = config.build().to(device="cuda").bfloat16()
    fused = fused_config.build().to(device="cuda").bfloat16()
    reference.init_states(buffer_device=torch.device("cuda"))
    fused.init_states(buffer_device=torch.device("cuda"))
    fused.load_state_dict(reference.state_dict())
    positions = torch.arange(128, device="cuda")
    metadata = config.inner_attention.build_attention_metadata(positions)
    x = torch.randn(128, config.dim, device="cuda", dtype=torch.bfloat16)
    inputs = [x.clone().requires_grad_() for _ in range(2)]
    outputs = [
        module(value, attention_metadata=metadata, positions=positions)
        for module, value in zip((reference, fused), inputs, strict=True)
    ]
    gradient = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(gradient)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.02, atol=0.02)
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=0.03, atol=0.03)
    for expected, actual in zip(
        reference.parameters(), fused.parameters(), strict=True
    ):
        assert expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(expected.grad, actual.grad, rtol=0.03, atol=0.03)


@pytest.mark.parametrize(
    "head_dim,value_dim,kv_heads", [(64, 64, 4), (192, 128, 4), (64, 64, 2)]
)
def test_dense_causal_sdpa_matches_math_outputs_and_gradients(
    head_dim, value_dim, kv_heads
):
    torch.manual_seed(42)
    inputs = (
        torch.randn(
            128, 4, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        ),
        torch.randn(
            128,
            kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        torch.randn(
            128,
            kv_heads,
            value_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in inputs
    )
    attention = DenseCausalBf16SDPA.Config().build()
    metadata = attention.Config().build_attention_metadata(
        torch.arange(128, device="cuda")
    )
    actual = attention(*inputs, attention_metadata=metadata, enable_gqa=kv_heads != 4)
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            F.scaled_dot_product_attention(
                *(value.transpose(0, 1).unsqueeze(0) for value in reference_inputs),
                is_causal=True,
                enable_gqa=kv_heads != 4,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    for value, reference in zip(inputs, reference_inputs, strict=True):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0.03, atol=0.03)
