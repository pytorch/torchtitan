# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepSeek-V4 KV compressor."""

import pytest
import torch

from torchtitan_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel


def _compressor(ratio: int):
    """The debug model's compressor for ``ratio`` (4: CSA, 128: HCA), randomly initialized."""
    for layer in deepseek_v4_debugmodel(seq_len=1024).model.layers:
        attn = layer.attention
        if attn.compress_ratio == ratio:
            cfg = attn.compressor if ratio == 4 else attn.compressor_128
            torch.manual_seed(0)
            with torch.device("cpu"):
                compressor = cfg.build()
                compressor.init_states(buffer_device=torch.device("cpu"))
            return compressor.double()
    raise AssertionError(f"debug model has no compress_ratio={ratio} layer")


@pytest.mark.parametrize("ratio", [4, 128])
def test_default_positions_match_explicit_positions(ratio):
    """Without ``positions``, compressed entry j is placed at token j * ratio.

    Regression: RoPE fell back to position j (the compressed index), so a model
    called without ``positions`` disagreed with the training path, which always
    passes them.
    """
    compressor = _compressor(ratio)
    seqlen = 4 * ratio
    x = torch.randn(seqlen, compressor.wkv.in_features, dtype=torch.float64)
    implicit = compressor(x, positions=None)
    explicit = compressor(x, positions=torch.arange(seqlen))
    torch.testing.assert_close(implicit, explicit)


@pytest.mark.parametrize("ratio", [4, 128])
def test_aligned_documents_match_per_document_compression(ratio):
    """Aligned packing lets the compressor regroup the whole stream.

    Regression: regrouping must be equivalent to compressing each document on
    its own, including the overlap candidate at a document start.
    """
    compressor = _compressor(ratio)
    lengths = (2 * ratio, 3 * ratio)
    x = torch.randn(sum(lengths), compressor.wkv.in_features, dtype=torch.float64)
    packed = compressor(x, positions=torch.cat([torch.arange(n) for n in lengths]))
    per_document = torch.cat(
        [
            compressor(x_doc, positions=torch.arange(n))
            for x_doc, n in zip(x.split(lengths), lengths)
        ]
    )
    torch.testing.assert_close(packed, per_document)
