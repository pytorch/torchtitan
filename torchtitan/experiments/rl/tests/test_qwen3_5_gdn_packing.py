# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GatedDeltaNet packed-sequence correctness (the cu_seqlens boundary fix).

The RL batcher packs several training samples into one row (positions restart at 0
per sample). Linear-attention (GatedDeltaNet) must reset its recurrent state AND
causal conv at those boundaries; otherwise state/conv bleed across samples and the
trainer computes wrong logprobs for every non-first packed sample (full attention
avoids this via a block-diagonal mask). This test pins:

1. cu_seqlens derivation (``_gdn_cu_seqlens``, which reuses the shared
   ``create_varlen_metadata_for_document``): a single sequence -> None (unchanged
   path, no regression); a packed sequence -> the boundary offsets.
2. Packed + cu_seqlens matches processing each sample separately (the fix).
3. Packed WITHOUT cu_seqlens is contaminated at the non-first sample (the bug the
   fix removes) -- guards against silently dropping the boundary handling.

Test 1 (cu_seqlens derivation) runs on CPU; test 2 needs a GPU because the FLA
gated-delta kernels are Triton/CUDA only.
"""

import pytest
import torch

from torchtitan.models.qwen3_5 import _qwen35_deltanet_config
from torchtitan.models.qwen3_5.model import _gdn_cu_seqlens


def _max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x - y).abs().max().item()


def test_gdn_cu_seqlens():
    # Single sequence (positions [0..L-1]) has one boundary -> None -> old path.
    single = torch.arange(50).unsqueeze(0)
    assert _gdn_cu_seqlens(single) is None

    # Packed [0..29, 0..39] -> boundaries at 0, 30 plus the total length 70.
    packed = torch.cat([torch.arange(30), torch.arange(40)]).unsqueeze(0)
    cu = _gdn_cu_seqlens(packed)
    assert cu.tolist() == [0, 30, 70]

    # 3D MRoPE positions [B, L, 3]: boundary is the temporal-component reset.
    mrope = packed.unsqueeze(-1).repeat(1, 1, 3)
    cu_mrope = _gdn_cu_seqlens(mrope)
    assert cu_mrope.tolist() == [0, 30, 70]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA GDN kernels need CUDA")
def test_gated_delta_net_packing_matches_separate():
    torch.manual_seed(0)
    dev, dt, dim = "cuda", torch.bfloat16, 256
    cfg = _qwen35_deltanet_config(
        dim=dim,
        n_key_heads=2,
        n_value_heads=4,
        key_head_dim=64,
        value_head_dim=64,
        layer_id=0,
    )
    gdn = cfg.build().to(dev)
    with torch.no_grad():
        for name, param in gdn.named_parameters():
            if "A_log" in name or "dt_bias" in name:
                param.zero_()
            elif "norm" in name and "weight" in name:
                param.fill_(1.0)
            else:
                param.normal_(0, 0.02)
    gdn = gdn.to(dt)

    len_a, len_b = 30, 40
    x_a = torch.randn(1, len_a, dim, device=dev, dtype=dt)
    x_b = torch.randn(1, len_b, dim, device=dev, dtype=dt)
    x_packed = torch.cat([x_a, x_b], dim=1)
    pos_packed = torch.cat(  # positions restart at 0 for the second sample
        [torch.arange(len_a, device=dev), torch.arange(len_b, device=dev)]
    ).unsqueeze(0)
    cu_seqlens = _gdn_cu_seqlens(pos_packed)

    with torch.no_grad():
        out_a = gdn(x_a)  # single sequence -> cu_seqlens None
        out_b = gdn(x_b)
        out_fix = gdn(x_packed, cu_seqlens)  # packed WITH cu_seqlens (the fix)
        out_bug = gdn(x_packed)  # packed WITHOUT cu_seqlens (contaminated)

    # The fix: the packed second sample matches its standalone output to bf16.
    fix_b = _max_abs_diff(out_fix[:, len_a:], out_b)
    bug_b = _max_abs_diff(out_bug[:, len_a:], out_b)
    assert fix_b < 1e-3, f"packed+cu_seqlens should match separate, got {fix_b}"
    # Without cu_seqlens the recurrent state / conv bleed -> clearly contaminated.
    assert bug_b > 10 * fix_b, f"expected contamination without cu_seqlens, got {bug_b}"

    # The first packed sample is unaffected either way (nothing precedes it).
    assert _max_abs_diff(out_bug[:, :len_a], out_a) == 0.0
