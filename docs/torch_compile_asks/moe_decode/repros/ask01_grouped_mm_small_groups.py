# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`torch._grouped_mm` (2D rows x 3D expert weights) at decode, as % of the weight-read roofline.

At decode each expert gets ~1-8 rows. ATen's sm100 grouped GEMM picks one of two fixed tiles (128 or 256 rows,
2x1 cluster), so most of each tile is padding and the second CTA of the cluster idles.

Expected on 1x GB300 (see ../README.md, ask 1): w13 T=34 ~152 us (~60% of roofline), w2 ~94 us (~49%);
FlashInfer's equivalent GEMMs run at 88% / 84%.
"""
import torch
from _common import DIM, graph_us, header, HIDDEN, routed_inputs, weight_roofline_us

header()
for name, (out_features, in_features) in (
    ("w13", (2 * HIDDEN, DIM)),
    ("w2", (DIM, HIDDEN)),
):
    for num_tokens in (1, 8, 34, 128):
        weight, rows, offs, counts = routed_inputs(
            num_tokens, out_features, in_features
        )
        us = graph_us(
            lambda: torch._grouped_mm(rows, weight.transpose(-2, -1), offs=offs)
        )
        floor = weight_roofline_us(counts, out_features, in_features)
        print(
            f"{name} T={num_tokens:4d} rows/expert={rows.shape[0] / 256:5.2f}: {us:7.1f} us, "
            f"roofline {floor:6.1f} us ({100 * floor / us:3.0f}%)"
        )
