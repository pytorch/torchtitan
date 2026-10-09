# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone timing of the two Triton kernels Inductor generated for ask 13's weighted sum
(T static vs T dynamic). The sources (ask12_kernel_static.py, ask12_kernel_masked.py) are identical
except for `xmask = xindex < xnumel` on every load; both carry tt.divisibility=16 on xnumel.
Swept over XBLOCK / num_warps, so the slowdown is not an autotuning artifact. For Triton triage.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 12):
unmasked 308-563 us, masked 597-832 us across XBLOCK / num_warps.
"""
import importlib.util
import os
import tempfile

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
T, N, D = 16384, 7, 7168
xnumel = T * D
kernels = {}
for label, filename in (
    ("unmasked (T static)", "ask12_kernel_static.py"),
    ("masked (T dynamic)", "ask12_kernel_masked.py"),
):
    src = (
        "import triton\nimport triton.language as tl\n"
        "from torch._inductor.runtime.triton_helpers import libdevice\n"
        + open(os.path.join(HERE, filename)).read()
    )
    path = os.path.join(tempfile.gettempdir(), f"compile_asks_{filename}")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernels[
        label
    ] = module.triton_poi_fused__softmax__to_copy_add_mul_slice_sum_unsqueeze_3

scores = torch.randn(T, N + 1, device="cuda")
row_max = torch.randn(T, device="cuda")
row_sum = torch.rand(T, device="cuda") + 1
stack = torch.randn(T, N, D, device="cuda", dtype=torch.bfloat16)
partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
out = torch.empty(T, D, device="cuda", dtype=torch.bfloat16)
for label, kernel in kernels.items():
    row = []
    for xblock in (512, 1024, 2048):
        for num_warps in (4, 8):
            grid = (triton.cdiv(xnumel, xblock),)
            fn = lambda: kernel[grid](
                scores,
                row_max,
                row_sum,
                stack,
                partial,
                out,
                xnumel,
                XBLOCK=xblock,
                num_warps=num_warps,
            )  # noqa: E731
            fn()
            torch.cuda.synchronize()
            row.append(
                f"{xblock}/{num_warps}: {triton.testing.do_bench(fn, warmup=5, rep=50) * 1000:.0f}"
            )
    print(f"{label:20s} us by XBLOCK/num_warps: " + ", ".join(row))
