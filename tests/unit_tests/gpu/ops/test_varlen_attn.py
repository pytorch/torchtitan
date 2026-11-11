# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_utils import DeterministicGuard

from torchtitan.models.common.attention import (
    DeterministicVarlenInnerAttention,
    VarlenAttentionMetadata,
)


def test_backward_is_deterministic_without_global_flag():
    # Building the module activates the installed FA3/FA4, as in training.
    attention = DeterministicVarlenInnerAttention.Config().build()
    torch.manual_seed(0)
    # NaViT-like segments: one per image, several image sizes.
    lengths = [361, 1369, 64, 100, 144] * 3
    max_len = max(lengths)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    q, k, v = (
        torch.randn(
            sum(lengths), 8, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    grad_out = torch.randn_like(q)
    metadata = VarlenAttentionMetadata(
        cu_seq_q=cu_seqlens,
        cu_seq_k=cu_seqlens,
        max_q=max_len,
        max_k=max_len,
        window_size=(-1, -1),
    )

    def ours():
        return attention(q, k, v, attention_metadata=metadata)

    def torch_flash():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return varlen_attn(
                q, k, v, cu_seqlens, cu_seqlens, max_len, max_len, window_size=(-1, -1)
            )

    def run(fn):
        q.grad = k.grad = v.grad = None
        out = fn()
        out.backward(grad_out)
        return [out.detach().clone(), q.grad.clone(), k.grad.clone(), v.grad.clone()]

    with DeterministicGuard(False):
        expected = run(ours)
        for fn in [ours] * 9 + [torch.compile(ours, fullgraph=True)]:
            for actual, reference in zip(run(fn), expected):
                assert torch.equal(actual, reference)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.utils.deterministic.fill_uninitialized_memory
    with DeterministicGuard(True):
        for actual, reference in zip(run(torch_flash), expected):
            assert torch.equal(actual, reference)
