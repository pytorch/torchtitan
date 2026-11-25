# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn import functional as F

from torchtitan.models.common import Linear
from torchtitan.models.glm5.dsa import (
    DsaMetadata,
    Glm5MLAAttention,
    Glm5SparseAttention,
)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="requires NVIDIA fused gather attention")
def test_absorbed_fused_matches_expanded_fp32_oracle():
    """Exercise fused single-head pool, K absorption and V projection backward."""
    torch.manual_seed(42)
    heads, latent, nope, rope, value, tokens = 4, 64, 32, 64, 32, 128
    model = Glm5MLAAttention.Config(
        wkv_b=Linear.Config(in_features=latent, out_features=heads * (nope + value)),
        packed_kv_head_dim=nope + value,
        inner_attention=Glm5SparseAttention.Config(),
    ).build().to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        model.wkv_b.weight.normal_(std=0.02)
    inputs = tuple(torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for shape in (
        (tokens, heads, nope + rope), (tokens, latent), (tokens, rope),
    ))
    positions = torch.arange(tokens, device="cuda") % 64
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    padding[[63, 127]] = True
    metadata = DsaMetadata.from_positions(positions, padding)
    rows = torch.arange(tokens, device="cuda", dtype=torch.int32)
    indices = torch.stack((rows, rows - 1), dim=-1)
    indices[(positions == 0), 1] = -1
    indices[padding] = -1
    scale = (nope + rope) ** -0.5
    actual = model(*inputs, attention_metadata=metadata, topk_indices_QS=indices, scale=scale)
    reference = tuple(x.detach().float().requires_grad_() for x in (*inputs, model.wkv_b.weight))
    q, latent_k, rope_k, weight = reference
    kv = F.linear(latent_k, weight).view(tokens, heads, nope + value)
    k_nope, v = kv.split((nope, value), dim=-1)
    k = torch.cat((k_nope, rope_k[:, None].expand(-1, heads, -1)), dim=-1)
    allowed = (indices[:, :, None] == rows).any(1)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
        attn_mask=allowed, scale=scale,
    ).transpose(0, 1)
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.05)
    cotangent = torch.randn_like(actual)
    gradients = torch.autograd.grad((actual * cotangent).sum(), (*inputs, model.wkv_b.weight))
    expected_gradients = torch.autograd.grad((expected * cotangent.float()).sum(), reference)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient.float(), expected_gradient, atol=0.05, rtol=0.05)
