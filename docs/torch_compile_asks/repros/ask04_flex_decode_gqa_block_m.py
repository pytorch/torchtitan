# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ask 4: FlexAttention's decode kernel rejects its own default BLOCK_M under GQA.

Q length 64 (< 128, so Inductor picks the decode kernel), 128 query / 16 KV heads (GQA 8),
default 128 x 128 BlockMask, cold compile.

Usage (GPU): python ask04_flex_decode_gqa_block_m.py

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130 and 2.16.0.dev20261014+cu132; see
../README.md, ask 4): InductorError ... Invalid FlexAttention decode kernel options ...
SPARSE_Q_BLOCK_SIZE=128, SPARSE_KV_BLOCK_SIZE=128, and BLOCK_M=512, BLOCK_N=64.
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

T, HQ, HKV, D = 64, 128, 16, 64
q = torch.randn(1, HQ, T, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, HKV, T, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, HKV, T, D, device="cuda", dtype=torch.bfloat16)
block_mask = create_block_mask(
    lambda b, h, qi, ki: qi >= ki, 1, None, T, T, device="cuda"
)
with torch.no_grad():
    out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask, enable_gqa=True)
print("ok", tuple(out.shape))
