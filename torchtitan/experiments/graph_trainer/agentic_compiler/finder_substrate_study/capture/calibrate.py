# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared roofline numbers for every finder run: HBM bandwidth and GEMM peaks, random data.

    CUDA_VISIBLE_DEVICES=3 python calibrate.py <out.json>
"""

import json
import sys

import torch


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    n = 4 * 1024**3 // 2  # 4 GiB of bf16
    src = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    ms = _time_ms(lambda: dst.copy_(src))
    out["hbm_copy_gbps"] = 2 * src.numel() * 2 / (ms * 1e-3) / 1e9
    ms = _time_ms(lambda: src.sum(dtype=torch.float32))
    out["hbm_read_gbps"] = src.numel() * 2 / (ms * 1e-3) / 1e9
    del src, dst
    gemm = {}
    for m, k, n_ in [(8192, 7168, 7168), (16384, 4096, 4096), (32768, 2048, 7168)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n_, device="cuda", dtype=torch.bfloat16)
        ms = _time_ms(lambda: a @ b)
        gemm[f"{m}x{k}x{n_}"] = 2 * m * k * n_ / (ms * 1e-3) / 1e12
    out["bf16_gemm_tflops"] = gemm
    out["bf16_gemm_tflops_max"] = max(gemm.values())
    try:
        from torchao.prototype.mx_formats.mx_tensor import to_mx
        from torchao.prototype.mx_formats.utils import to_blocked

        mx = {}
        for m, k, n_ in [(8192, 7168, 7168), (16384, 4096, 4096), (32768, 2048, 7168)]:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(n_, k, device="cuda", dtype=torch.bfloat16)
            sa, qa = to_mx(a, torch.float8_e4m3fn, 32)
            sb, qb = to_mx(b, torch.float8_e4m3fn, 32)
            sa_b, sb_b = to_blocked(sa.view(m, k // 32)), to_blocked(
                sb.view(n_, k // 32)
            )
            fn = lambda: torch._scaled_mm(  # noqa: E731
                qa, qb.t(), sa_b, sb_b, out_dtype=torch.bfloat16
            )
            ms = _time_ms(fn)
            mx[f"{m}x{k}x{n_}"] = 2 * m * k * n_ / (ms * 1e-3) / 1e12
        out["mxfp8_gemm_tflops"] = mx
        out["mxfp8_gemm_tflops_max"] = max(mx.values())
    except Exception as e:  # report, don't guess a number
        out["mxfp8_gemm_error"] = repr(e)[:300]
    out["note"] = (
        "Random-data rates. Ideal time = max(bytes / hbm_copy_gbps, flops / GEMM peak). "
        "Low-entropy data (fake-PG runs) can make GEMMs faster than these peaks."
    )
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
