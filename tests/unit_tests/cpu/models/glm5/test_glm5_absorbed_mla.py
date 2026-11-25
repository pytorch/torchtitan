# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from torchtitan.models.common import Linear
from torchtitan.models.glm5.dsa import (
    DsaMetadata,
    gather_attn,
    Glm5CPMLAAttention,
    Glm5MLAAttention,
    Glm5SparseAttention,
)


@pytest.mark.parametrize("cp", [False, True])
def test_absorbed_matches_expanded_output_and_all_gradients(cp):
    """Guard absorption, original scale, shared pool and checkpoint weight layout."""
    torch.manual_seed(42)
    # Unequal latent/nope/value widths catch accidental shape equivalences.
    heads, latent, nope, rope, value = 2, 12, 4, 4, 6
    config = (Glm5CPMLAAttention if cp else Glm5MLAAttention).Config(
        wkv_b=Linear.Config(in_features=latent, out_features=heads * (nope + value)),
        packed_kv_head_dim=nope + value,
        inner_attention=Glm5SparseAttention.Config(),
    )
    model = config.build()
    with torch.no_grad():
        model.wkv_b.weight.normal_(std=0.2)
    inputs = (
        torch.randn(4, heads, nope + rope, requires_grad=True),
        torch.randn(8, latent, requires_grad=True),
        torch.randn(8, rope, requires_grad=True),
    )
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    padding = torch.tensor([False, False, False, True] * 2)
    metadata = DsaMetadata.from_positions(positions, padding)
    queries = torch.arange(4, 8)
    metadata = DsaMetadata(positions, metadata.documents_T, ~padding, queries)
    indices = torch.tensor([[4, -1], [5, 4], [6, 4], [-1, -1]], dtype=torch.int32)
    # Exercise the CP gather boundary without requiring a distributed CPU job.
    with (
        patch("torchtitan.models.glm5.dsa.cp_all_gather_tensors", side_effect=lambda tensors, **_: tensors),
        patch("torchtitan.models.glm5.dsa.gather_attn", wraps=gather_attn) as kernel,
    ):
        actual = model(*inputs, attention_metadata=metadata, topk_indices_QS=indices, scale=0.37)
    assert kernel.call_args.args[2].shape == (1, 1, 8, latent + rope)
    assert kernel.call_args.kwargs["scale"] == 0.37
    assert set(model.state_dict()) == {"wkv_b.weight"}
    reference = [x.detach().clone().requires_grad_() for x in (*inputs, model.wkv_b.weight)]
    q, latent_k, rope_k, weight = reference
    kv = F.linear(latent_k, weight).view(8, heads, nope + value)
    k_nope, v = kv.split((nope, value), dim=-1)
    k = torch.cat((k_nope, rope_k[:, None].expand(-1, heads, -1)), dim=-1)
    allowed = (indices[:, :, None] == torch.arange(8)).any(1)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
        attn_mask=allowed, scale=0.37,
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    cotangent = torch.randn_like(actual)
    gradients = torch.autograd.grad((actual * cotangent).sum(), (*inputs, model.wkv_b.weight))
    expected_gradients = torch.autograd.grad((expected * cotangent).sum(), reference)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-6, rtol=3e-5)
