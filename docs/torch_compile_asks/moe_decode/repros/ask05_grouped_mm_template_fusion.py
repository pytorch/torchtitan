# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The whole MoE routed-expert computation in ONE compiled function. Does Inductor fuse the glue into the GEMMs?

    counting-sort dispatch -> x[token_of_row] -> _grouped_mm(w13) -> SwiGLU -> _grouped_mm(w2) -> weighted gather-sum

Modes (argv[1]):
  default : _grouped_mm lowers to the extern ATen CUTLASS kernel (no fusion into extern kernels).
  triton  : mode="max-autotune-no-cudagraphs" with max_autotune_gemm_backends="TRITON", forcing Inductor's own
            grouped_mm template, which is eligible for prologue/epilogue fusion.
Prints every kernel of one call and, for the triton mode (cold compile), Inductor's reasons for each rejected
fusion that involves a GEMM template ("op5", "op8" below are the w13 and w2 templates).

Expected on 1x GB300 (see ../README.md, ask 5): default 10 kernels, triton 8; in both the gathered rows and the
SwiGLU output are separate kernels and buffers.
"""
import collections
import logging
import sys

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from _common import (
    counting_sort_dispatch,
    DIM,
    graph_kernels,
    graph_us,
    header,
    HIDDEN,
    NUM_EXPERTS,
    router_inputs,
    TOP_K,
)


def routed_experts(x_TD, scores_TK, expert_ids_TK, w13_EOI, w2_EOI):
    num_tokens = x_TD.shape[0]
    rows_ND, _, _, row_of_route_N = counting_sort_dispatch(
        x_TD, scores_TK, expert_ids_TK
    )
    experts_E = torch.arange(NUM_EXPERTS, device=x_TD.device)
    offs_E = (
        (expert_ids_TK.reshape(-1, 1) == experts_E).sum(0).cumsum(0).to(torch.int32)
    )
    gate_up_N2F = torch._grouped_mm(rows_ND, w13_EOI.transpose(-2, -1), offs=offs_E)
    hidden_NF = (
        F.silu(gate_up_N2F[:, :HIDDEN].float()) * gate_up_N2F[:, HIDDEN:].float()
    ).bfloat16()
    out_ND = torch._grouped_mm(hidden_NF, w2_EOI.transpose(-2, -1), offs=offs_E)
    rows_TKD = out_ND[row_of_route_N].view(num_tokens, TOP_K, DIM).float()
    return (rows_TKD * scores_TK[..., None]).sum(1).bfloat16()


def is_gemm(name):
    """GEMM kernels themselves; Inductor pointwise/reduction kernels and ATen's GEMM setup kernel count as glue."""
    if (
        name.startswith(("triton_poi", "triton_per", "triton_red"))
        or "prepare_grouped_gemm" in name
    ):
        return False
    return any(s in name for s in ("GemmUniversal", "triton_tem", "cutlass"))


class FusionReasons(logging.Handler):
    """Collect Inductor's "cannot fuse opA with opB: reason" fusion-log lines."""

    def __init__(self):
        super().__init__()
        self.reasons = collections.Counter()

    def emit(self, record):
        message = record.getMessage()
        if "cannot fuse" in message:
            self.reasons[message.strip()] += 1


mode = sys.argv[1] if len(sys.argv) > 1 else "default"
header()
fusion_reasons = FusionReasons()
if mode == "triton":
    inductor_config.force_disable_caches = True
    torch._logging.set_logs(fusion=True)
    for logger_name in ("torch._inductor.scheduler", "torch._inductor"):
        logging.getLogger(logger_name).addHandler(fusion_reasons)
torch.manual_seed(0)
w13 = (torch.randn(NUM_EXPERTS, 2 * HIDDEN, DIM, device="cuda") * 0.02).bfloat16()
w2 = (torch.randn(NUM_EXPERTS, DIM, HIDDEN, device="cuda") * 0.02).bfloat16()
if mode == "triton":
    inductor_config.max_autotune_gemm_backends = "TRITON"
    compiled = torch.compile(
        routed_experts, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
    )
else:
    compiled = torch.compile(routed_experts, fullgraph=True, dynamic=False)
with torch.no_grad():
    for num_tokens in (34, 256):
        inputs = (*router_inputs(num_tokens), w13, w2)
        reference = routed_experts(*inputs)
        rel = (
            (compiled(*inputs).float() - reference.float()).norm()
            / reference.float().norm()
        ).item()
        wall = graph_us(lambda: compiled(*inputs))
        kernels = graph_kernels(lambda: compiled(*inputs))
        gemm = sum(us for us, name in kernels if is_gemm(name))
        glue = sum(us for us, name in kernels if not is_gemm(name))
        print(
            f"mode={mode} T={num_tokens}: graph {wall:6.1f} us, {len(kernels)} kernels, GEMM {gemm:6.1f} us, "
            f"glue {glue:5.1f} us, rel L2 vs eager {rel:.1e}"
        )
        for us, name in kernels:
            print(
                f"    {us:7.2f} us  {'GEMM' if is_gemm(name) else '    '} {name[:100]}"
            )
for message in sorted(fusion_reasons.reasons):
    if "template" in message or "prologue" in message or "memory deps" in message:
        print(f"fusion log: {message}")
