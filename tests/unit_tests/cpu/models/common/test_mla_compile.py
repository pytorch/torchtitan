# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.attention import materialize_mla_kv

# T tokens, H heads, N per-head key dims, R shared key dims, V value dims
T, H, N, R, V = 16, 4, 8, 4, 6


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    torch._dynamo.reset()
    yield
    apply_local_compile([])


@pytest.mark.parametrize("layout", ["projected", "ulysses"])
def test_compiled_mla_kv_matches_eager(layout: str) -> None:
    """``projected``: kv viewed from the wkv_b output (MLAAttention).
    ``ulysses``: q and kv split from one packed tensor (UlyssesMLAAttention)."""
    torch.manual_seed(0)
    if layout == "projected":
        bases = [torch.randn(T, H, N + R), torch.randn(T, H * (N + V))]
    else:
        bases = [torch.randn(T, H, N + R + N + V)]
    k_shared = torch.randn(T, R)
    grad_k, grad_v = torch.randn(T, H, N + R), torch.randn(T, H, V)

    results = []
    for regions in ([], ["mla_kv"]):
        apply_local_compile(regions)
        leaves = [t.clone().requires_grad_() for t in (*bases, k_shared)]
        if layout == "projected":
            q, kv = leaves[0], leaves[1].view(T, H, N + V)
        else:
            q, kv = leaves[0].split([N + R, N + V], dim=-1)
        k, v = materialize_mla_kv(q, kv, leaves[-1])
        torch.autograd.backward((k, v), (grad_k, grad_v))
        results.append((k, v, *(t.grad for t in leaves)))

        assert v.is_contiguous()  # attention saves v; a view would keep kv alive
        with torch.no_grad():  # nothing is saved, so v stays a view
            assert not materialize_mla_kv(q, kv, leaves[-1])[1].is_contiguous()

    for eager, compiled in zip(*results, strict=True):
        torch.testing.assert_close(compiled, eager)
