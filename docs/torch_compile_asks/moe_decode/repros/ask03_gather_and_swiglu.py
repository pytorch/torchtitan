# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What `torch._grouped_mm` can't take in today: the token gather into expert order (A rows by index) and the
SwiGLU on w13's output. Each runs as its own kernel next to the GEMM (compiled, CUDA graph, kernel time from the
profiler; ATen's GEMM setup kernel is counted with the GEMM).

Expected on 1x GB300 (see ../README.md, ask 3): T=34 gather ~2 us + w13 ~144 us + SwiGLU ~1 us;
T=4096 gather ~35 us + SwiGLU ~15 us. FlashInfer does both inside its first GEMM.
"""
import torch
import torch.nn.functional as F
from _common import DIM, graph_kernels, header, HIDDEN, routed_inputs, TOP_K


@torch.compile(fullgraph=True, dynamic=False)
def gather_rows(x_TD, token_of_row):
    return x_TD[token_of_row]


@torch.compile(fullgraph=True, dynamic=False)
def swiglu(gate_up):
    return F.silu(gate_up[:, :HIDDEN]) * gate_up[:, HIDDEN:]


header()
for num_tokens in (8, 34, 256, 4096):
    weight, rows, offs, counts = routed_inputs(num_tokens, 2 * HIDDEN, DIM)
    x_TD = torch.randn(num_tokens, DIM, device="cuda").bfloat16()
    token_of_row = torch.randperm(rows.shape[0], device="cuda") // TOP_K
    gate_up = torch._grouped_mm(rows, weight.transpose(-2, -1), offs=offs)
    kernel_us = lambda fn: sum(us for us, _ in graph_kernels(fn))  # noqa: E731
    gather_us = kernel_us(lambda: gather_rows(x_TD, token_of_row))
    gemm_us = kernel_us(
        lambda: torch._grouped_mm(rows, weight.transpose(-2, -1), offs=offs)
    )
    swiglu_us = kernel_us(lambda: swiglu(gate_up))
    print(
        f"T={num_tokens:5d}: gather {gather_us:5.1f} us + w13 {gemm_us:6.1f} us + SwiGLU {swiglu_us:5.1f} us"
    )
