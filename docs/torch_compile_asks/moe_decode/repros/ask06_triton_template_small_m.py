# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's Triton grouped_mm template (max-autotune, TRITON backend only) vs ATen at decode, Qwen3.5 w13.

Expected on 1x GB300 (see ../README.md, ask 6): T=1 template ~33 us vs ATen ~24 (slower); T=34 ~136 vs ~151;
T=128 ~187 vs ~210. ATen with small-group tiles (ask 1) is ~107-112 us at T=34.
"""
import torch
import torch._inductor.config as inductor_config
from _common import DIM, graph_us, header, HIDDEN, routed_inputs

inductor_config.max_autotune_gemm_backends = "TRITON"
header()
for num_tokens in (1, 34, 128):
    weight, rows, offs, _ = routed_inputs(num_tokens, 2 * HIDDEN, DIM)
    weight_EKN = weight.transpose(-2, -1)
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda a, b, o: torch._grouped_mm(a, b, offs=o),
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
    )
    compiled(rows, weight_EKN, offs)
    aten = graph_us(lambda: torch._grouped_mm(rows, weight_EKN, offs=offs))
    template = graph_us(lambda: compiled(rows, weight_EKN, offs))
    print(
        f"w13 T={num_tokens:4d}: ATen {aten:6.1f} us   Inductor Triton template {template:6.1f} us"
    )
