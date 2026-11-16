# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.rl.generator.vllm.attention.mla import TorchTitanMLAImpl
from torchtitan.tools.utils import has_cuda_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="FA4 absorbed MLA requires SM100",
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
SCALE = 0.1


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )


def test_mla_decode_attends_to_latent_cache():
    """forward_mqa reads vLLM's [kv_c_normed | k_pe] cache as latent keys/values and rope keys."""
    query_lengths, key_lengths = [1, 3, 1, 2], [40, 300, 1, 129]
    num_heads, block_size = 16, 64
    device = torch.device("cuda")
    blocks = [-(-length // block_size) for length in key_lengths]
    cache = torch.randn(
        sum(blocks) + 2,
        block_size,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    order = torch.randperm(cache.shape[0])
    block_table = torch.zeros(len(blocks), max(blocks), dtype=torch.int32)
    start = 0
    for sequence, count in enumerate(blocks):
        block_table[sequence, :count] = order[start : start + count]
        start += count
    q = torch.randn(
        sum(query_lengths),
        num_heads,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=block_table.to(device),
            query_start_loc=_offsets(query_lengths).to(device),
            seq_lens=torch.tensor(key_lengths, dtype=torch.int32, device=device),
            max_query_len=max(query_lengths),
        )
    )
    impl = TorchTitanMLAImpl.__new__(TorchTitanMLAImpl)
    impl.kv_lora_rank = KV_LORA_RANK
    impl.scale = SCALE

    out, _ = impl.forward_mqa(
        (q[..., :KV_LORA_RANK], q[..., KV_LORA_RANK:]), cache, metadata, None
    )

    start = 0
    for sequence, (num_queries, num_keys) in enumerate(zip(query_lengths, key_lengths)):
        keys = cache[block_table[sequence, : blocks[sequence]].long()].flatten(0, 1)
        keys = keys[:num_keys]
        mask = torch.ones(num_queries, num_keys, dtype=torch.bool, device=device)
        logits = SCALE * q[start : start + num_queries].transpose(0, 1).float()
        logits = logits @ keys.float().T
        logits = logits.masked_fill(~mask.tril(num_keys - num_queries), float("-inf"))
        expected = torch.softmax(logits, dim=-1) @ keys[:, :KV_LORA_RANK].float()
        expected = expected.transpose(0, 1)
        torch.testing.assert_close(
            out[start : start + num_queries].float(), expected, atol=2e-2, rtol=0
        )
        start += num_queries
