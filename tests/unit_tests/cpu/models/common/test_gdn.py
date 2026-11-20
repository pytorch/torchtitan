# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

pytest.importorskip("attn_gym")

from torchtitan.models.common.attention.gdn import _split_in_proj


@pytest.mark.parametrize("heads_per_key", [1, 3])
def test_gdn_split_in_proj(heads_per_key: int) -> None:
    # 5 tokens, 2 key heads; at one value head per key head (0.8B, 2B) a, b and gate
    # would be strided views without the split's copies.
    r, T, G = heads_per_key, 5, 2
    q, k = torch.randn(T, G, 4), torch.randn(T, G, 4)
    v, z = torch.randn(T, G, 2 * r), torch.randn(T, G, 2 * r)
    a, b = torch.randn(T, G, r), torch.randn(T, G, r)
    projected_TC = torch.cat([q, k, v, z, a, b], dim=-1).flatten(1)
    outputs = _split_in_proj(projected_TC, [4, 4, 2 * r, 2 * r, r, r], 2)
    mixed_qkv_TC, a_TH, b_TH, gate_THV = outputs
    qkv = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1)
    assert torch.equal(mixed_qkv_TC, qkv)
    assert torch.equal(a_TH, a.flatten(1)) and torch.equal(b_TH, b.flatten(1))
    assert torch.equal(gate_THV, z.reshape(T, G * r, 2))
    assert all(output.is_contiguous() for output in outputs)
