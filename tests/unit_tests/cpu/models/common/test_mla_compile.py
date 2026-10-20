# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.attention import (
    cp_mla,
    FlexInnerAttention,
    materialize_mla_kv,
)
from torchtitan.models.common.linear import Linear

# T tokens, H heads, N per-head key dims, R shared key dims, V value dims, L latent dims
T, H, N, R, V, L = 16, 4, 8, 4, 6, 8


@pytest.fixture(autouse=True)
def reset_local_compile():
    LocalCompileConfig().apply()
    torch._dynamo.reset()
    yield
    LocalCompileConfig().apply()


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
        LocalCompileConfig(regions=regions).apply()
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


class _CausalSDPA(torch.nn.Module):
    """Stands in for FlexInnerAttention, whose compiled flex kernel doesn't build on every CPU."""

    def forward(self, q, k, v, **kwargs):
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(0, 1)


def test_ulysses_mla_grads_match_eager() -> None:
    """UlyssesMLAAttention on a 1-rank gloo CP group: every grad under ``mla_kv``
    matches eager. An async k_shared all-gather reaches the region unwaited and
    detached, so k_shared's grad would be zero."""
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        attention = cp_mla.UlyssesMLAAttention.Config(
            wkv_b=Linear.Config(in_features=L, out_features=H * (N + V)),
            packed_kv_head_dim=N + V,
            inner_attention=FlexInnerAttention.Config(),
        ).build()
        attention.inner_attention = _CausalSDPA()
        inputs = (torch.randn(T, H, N + R), torch.randn(T, L), torch.randn(T, R))
        results = []
        with (
            patch.object(cp_mla, "spmd_mesh_group", return_value=dist.group.WORLD),
            spmd.no_typecheck(),
        ):
            for regions in ([], ["mla_kv"]):
                LocalCompileConfig(regions=regions).apply()
                attention.zero_grad(set_to_none=True)
                q, kv_c, k_shared = (x.clone().requires_grad_() for x in inputs)
                out = attention(q, kv_c, k_shared, attention_metadata=None)
                out.backward(torch.ones_like(out))
                results.append(
                    (out, q.grad, kv_c.grad, k_shared.grad, attention.wkv_b.weight.grad)
                )
    finally:
        dist.destroy_process_group()

    for eager, compiled in zip(*results, strict=True):
        torch.testing.assert_close(compiled, eager)
