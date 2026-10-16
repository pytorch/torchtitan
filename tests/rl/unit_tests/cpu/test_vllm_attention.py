# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest

import torchtitan.rl.generator.vllm.attention as vllm_attention


@pytest.mark.parametrize(
    "backend",
    [
        vllm_attention.TorchTitanVarlenInnerAttentionBackend,
        vllm_attention.TorchTitanVarlenInnerAttentionDiffKVBackend,
    ],
)
@pytest.mark.parametrize("flash_impl, page_size", [("FA4", 128), ("FA3", 16)])
def test_kv_page_size_follows_flash_impl(backend, flash_impl, page_size):
    """FA4 gets 128-token pages (its TMA path); other impls keep vLLM's default of 16."""
    with patch.object(
        vllm_attention, "get_cuda_flash_attention_impl", return_value=flash_impl
    ):
        assert backend.get_preferred_block_size(16) == page_size
