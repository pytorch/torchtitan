# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's CuteDSL grouped_mm template at decode (w13, T=34). Needs cutlass-dsl 4.8 on PYTHONPATH (4.7.1 fails).

Prints the GEMM kernel alone, all device work per call, kernels per call, device-to-host syncs per call
(aten::_local_scalar_dense), and whether the call can be captured in a CUDA graph. Profiled eagerly.

Expected on 1x GB300 (see ../README.md, ask 7): GEMM kernel ~143 us (ATen level), ~1540 kernels and ~2.8 ms of
device work per call, 1 D2H sync per call, CUDA graph capture fails.
"""
import torch
import torch._inductor.config as inductor_config
from _common import DIM, header, HIDDEN, routed_inputs
from torch.profiler import profile, ProfilerActivity

inductor_config.max_autotune_gemm_backends = "CUTEDSL"
inductor_config.compile_threads = 1
inductor_config.autotune_in_subproc = False
header()
weight, rows, offs, _ = routed_inputs(34, 2 * HIDDEN, DIM)
weight_EKN = weight.transpose(-2, -1)
compiled = torch.compile(
    lambda a, b, o: torch._grouped_mm(a, b, offs=o),
    mode="max-autotune-no-cudagraphs",
    dynamic=False,
)
for _ in range(3):
    compiled(rows, weight_EKN, offs)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        compiled(rows, weight_EKN, offs)
    torch.cuda.synchronize()
events = prof.key_averages()
kernels = [
    e
    for e in events
    if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0
]
gemm_us = (
    sum(e.self_device_time_total for e in events if "GroupedGemmKernel" in e.key) / 5
)
device_us = sum(e.self_device_time_total for e in kernels) / 5
syncs = sum(e.count for e in events if e.key == "aten::_local_scalar_dense") / 5
print(
    f"w13 T=34: GEMM kernel {gemm_us:.1f} us, all device work per call {device_us:.1f} us, "
    f"kernels per call {sum(e.count for e in kernels) / 5:.0f}, D2H syncs per call {syncs:.0f}"
)
try:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(rows, weight_EKN, offs)
    print("CUDA graph capture: OK")
except Exception as error:
    print("CUDA graph capture fails:", type(error).__name__, str(error)[:120])
