# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sparse geometry, alignment guards and compressor numerical equivalence."""

import pytest
import torch

from torchtitan.models.deepseek_v4.attention import HeavilyCompressedAttention
from torchtitan_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel


@pytest.mark.parametrize("lengths", [(133, 7, 116), (1, 2, 3), (256, 128)])
@pytest.mark.parametrize("ratio", [4, 128])
def test_cached_compression_matches_legacy_forward_and_backward(lengths, ratio):
    cfg = next(
        layer.attention
        for layer in deepseek_v4_debugmodel(seq_len=512).model.layers
        if layer.attention.compress_ratio == ratio
    )
    module = (cfg.compressor if ratio == 4 else cfg.compressor_128).build()
    torch.manual_seed(42)
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.1)
    x = torch.randn(sum(lengths), cfg.dim, requires_grad=True)
    positions = torch.cat([torch.arange(n) for n in lengths])
    metadata = cfg.inner_attention.build_attention_metadata(
        positions, max_num_documents=8, max_context_length=512
    )
    expected = module(x, positions, metadata.cu_seq_q)
    actual = module(x, positions, attention_metadata=metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if actual.requires_grad:
        inputs = (x, *module.parameters())
        expected_grad = torch.autograd.grad(expected.sum(), inputs, allow_unused=True)
        actual_grad = torch.autograd.grad(actual.sum(), inputs, allow_unused=True)
        for a, b in zip(actual_grad, expected_grad):
            if a is not None:
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [4, 128])
def test_aligned_reshape_matches_general_compressor(ratio):
    cfg = next(
        layer.attention
        for layer in deepseek_v4_debugmodel(seq_len=512).model.layers
        if layer.attention.compress_ratio == ratio
    )
    compressor = (cfg.compressor if ratio == 4 else cfg.compressor_128).build()
    torch.manual_seed(42)
    for param in compressor.parameters():
        torch.nn.init.normal_(param, std=0.1)
    positions = torch.cat([torch.arange(256), torch.arange(128)])
    x = torch.randn(384, cfg.dim, requires_grad=True)
    cfg.inner_attention.document_alignment = 128
    metadata = cfg.inner_attention.build_attention_metadata(positions)
    expected = compressor(x, positions, metadata.cu_seq_q)
    actual = compressor(x, positions, attention_metadata=metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (x, *compressor.parameters())
    a = torch.autograd.grad(actual.sum(), inputs)
    b = torch.autograd.grad(expected.sum(), inputs)
    for actual_grad, expected_grad in zip(a, b):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_alignment_rejects_unaligned_documents():
    cfg = deepseek_v4_debugmodel(seq_len=256, align_documents=True).model
    inner = next(
        layer.attention.inner_attention
        for layer in cfg.layers
        if layer.attention.compress_ratio == 128
    )
    with pytest.raises(RuntimeError, match="document boundaries"):
        inner.build_attention_metadata(
            torch.cat([torch.arange(129), torch.arange(127)])
        )


def test_variant_geometry_must_be_uniform():
    cfg = deepseek_v4_debugmodel(seq_len=256).model
    # SWA 0 and 1 are equivalent, including the MTP convention.
    cfg.layers[1].attention.compress_ratio = 1
    cfg.layers[1].attention.inner_attention.compress_ratio = 1
    cfg.__post_init__()
    cfg.layers[1].attention.inner_attention.window_size += 1
    with pytest.raises(ValueError, match="same compress_ratio and window_size"):
        cfg.__post_init__()


def test_hca_width_is_bounded_by_document_length():
    cfg = HeavilyCompressedAttention.Config(
        window_size=4, compress_ratio=128, softmax_scale=0.1, index_topk=2
    )
    metadata = cfg.build_attention_metadata(torch.arange(256).repeat(4))
    assert metadata.causal_cmp.shape == (1024, 2)
    assert metadata.cu_seq_cmp.tolist() == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("ratio", [4, 128])
def test_padding_does_not_change_real_token_attention(ratio):
    cfg = next(
        layer.attention
        for layer in deepseek_v4_debugmodel(seq_len=256).model.layers
        if layer.attention.compress_ratio == ratio
    )
    torch.manual_seed(42)
    module = cfg.build().eval()
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.02)
    lengths = [133, 7, 116]
    parts = [torch.randn(n, cfg.dim) for n in lengths]
    positions = torch.cat([torch.arange(n) for n in lengths])
    original = module(
        torch.cat(parts),
        positions=positions,
        attention_metadata=cfg.inner_attention.build_attention_metadata(positions),
    )
    padded = torch.cat(
        [torch.nn.functional.pad(x, (0, 0, 0, (-len(x)) % 128)) for x in parts]
    )
    padded_positions = torch.cat([torch.arange(n + (-n) % 128) for n in lengths])
    valid = torch.cat([torch.arange(n + (-n) % 128) < n for n in lengths])
    cfg.inner_attention.document_alignment = 128
    metadata = cfg.inner_attention.build_attention_metadata(
        padded_positions, padding_mask=~valid
    )
    aligned = module(padded, positions=padded_positions, attention_metadata=metadata)
    torch.testing.assert_close(aligned[valid], original)
