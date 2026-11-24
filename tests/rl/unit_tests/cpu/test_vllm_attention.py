# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch

from torchtitan.rl.generator.vllm.attention import attention as vllm_attention
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import select_common_block_size


@pytest.mark.parametrize(
    "backend",
    [
        vllm_attention.TorchTitanVarlenInnerAttentionBackend,
        vllm_attention.TorchTitanVarlenInnerAttentionDiffKVBackend,
    ],
)
@pytest.mark.parametrize(
    "flash_impl, capability, batch_invariant, page_size",
    [
        ("FA4", (10, 0), False, 128),
        ("FA3", (9, 0), False, 16),
        ("FA4", (10, 0), True, 16),
        ("FA4", (12, 0), False, 16),
    ],
)
def test_kv_page_size_follows_flash_impl(
    backend, flash_impl, capability, batch_invariant, page_size
):
    """FA4 on SM100/SM103 gets 128-token pages, except in batch-invariant mode; the rest keep 16."""
    with (
        patch.object(
            vllm_attention, "get_cuda_flash_attention_impl", return_value=flash_impl
        ),
        patch.object(
            vllm_attention.torch.cuda, "get_device_capability", return_value=capability
        ),
        patch.object(
            vllm_attention, "is_in_batch_invariant_mode", return_value=batch_invariant
        ),
    ):
        assert backend.get_preferred_block_size(16) == page_size


@pytest.mark.parametrize(
    "flash_impl, capability, batch_invariant, page_size",
    [
        ("FA4", (10, 0), False, 128),
        ("FA4", (10, 0), True, 16),
        ("FA3", (9, 0), False, 1152),
        ("FA3", (9, 0), True, 16),
    ],
)
def test_hybrid_kv_blocks_split_into_flash_attention_pages(
    flash_impl, capability, batch_invariant, page_size
):
    """A hybrid model's 1152-token KV block splits into kernel pages, or stays one page.

    Batch-invariant mode needs 16-token pages: one 1152-token page would put FA4's decode
    on SwapAB, or give FA3's long prefills (not its decode) 128-key TMA tiles.
    """
    backend = vllm_attention.TorchTitanVarlenInnerAttentionBackend
    with (
        patch.object(
            vllm_attention, "get_cuda_flash_attention_impl", return_value=flash_impl
        ),
        patch.object(
            vllm_attention.torch.cuda, "get_device_capability", return_value=capability
        ),
        patch.object(
            vllm_attention, "is_in_batch_invariant_mode", return_value=batch_invariant
        ),
    ):
        assert select_common_block_size(1152, [backend]) == page_size


def test_fa4_page_size_with_kv_cache_spec():
    """vLLM 0.33 and later call the hook with the layer's KV cache spec."""
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    backend = vllm_attention.TorchTitanVarlenInnerAttentionBackend
    with (
        patch.object(
            vllm_attention, "get_cuda_flash_attention_impl", return_value="FA4"
        ),
        patch.object(
            vllm_attention.torch.cuda, "get_device_capability", return_value=(10, 0)
        ),
    ):
        assert backend.get_supported_kernel_block_sizes(spec) == [128]
