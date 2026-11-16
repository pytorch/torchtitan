# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Same MoE GEMMs, two CUTLASS 4.8 kernels. Needs out-of-tree CUTLASS builds (not pure torch):
set GMM_EXT_DIR to a directory with `gmm_patched_v3` (GroupMM.cu with small-group tiles, ptr-array kernel) and
`gmm_mixed` (CUTLASS examples/92_blackwell_moe_gemm_grouped).

  ptr-array : torch._grouped_mm's kernel. One TMA descriptor is rewritten per group, plus a setup kernel
              (prepare_grouped_gemm_data) per call.
  moe       : CUTLASS's MoE kernel. One batched TMA descriptor for the weights, cp.async for the rows, counts read
              on device, no setup kernel. Rows padded per expert to (E, max_rows, K) (the padding is not timed).

Expected on 1x GB300 (see ../README.md, ask 2): w2 T=34 ptr-array ~69 us vs moe ~66 us; T=8 ~28 vs ~24 us; bitwise equal.
"""
import os
import sys

import torch

sys.path.insert(0, os.environ["GMM_EXT_DIR"])
import gmm_mixed  # noqa: E402
import gmm_patched_v3  # noqa: E402
from _common import (  # noqa: E402
    DIM,
    graph_us,
    header,
    HIDDEN,
    NUM_EXPERTS,
    routed_inputs,
    weight_roofline_us,
)

header()
for name, (out_features, in_features) in (
    ("w13", (2 * HIDDEN, DIM)),
    ("w2", (DIM, HIDDEN)),
):
    for num_tokens in (8, 34, 128):
        weight, rows, offs, counts = routed_inputs(
            num_tokens, out_features, in_features
        )
        out = torch.empty(
            rows.shape[0], out_features, device="cuda", dtype=torch.bfloat16
        )
        # Padded layout for the MoE kernel: row r of expert e goes to rows_padded[e, r].
        expert_of_row = torch.repeat_interleave(
            torch.arange(NUM_EXPERTS, device="cuda"), counts
        )
        row_in_expert = (
            torch.arange(rows.shape[0], device="cuda") - (offs - counts)[expert_of_row]
        )
        max_rows = (int(counts.max()) + 7) // 8 * 8
        rows_padded = torch.zeros(
            NUM_EXPERTS, max_rows, in_features, device="cuda", dtype=torch.bfloat16
        )
        rows_padded[expert_of_row, row_in_expert] = rows
        out_padded = torch.empty(
            NUM_EXPERTS, max_rows, out_features, device="cuda", dtype=torch.bfloat16
        )
        ptr_array = graph_us(
            lambda: gmm_patched_v3.grouped_mm(rows, weight.transpose(-2, -1), offs, out)
        )
        moe = graph_us(
            lambda: gmm_mixed.m128x16x128_c1(
                weight, rows_padded, counts.to(torch.int32), out_padded
            )
        )
        floor = weight_roofline_us(counts, out_features, in_features)
        print(
            f"{name} T={num_tokens:4d}: ptr-array {ptr_array:6.1f} us ({100 * floor / ptr_array:3.0f}%)   "
            f"moe {moe:6.1f} us ({100 * floor / moe:3.0f}%)   "
            f"bitwise equal {torch.equal(out_padded[expert_of_row, row_in_expert], out)}"
        )
