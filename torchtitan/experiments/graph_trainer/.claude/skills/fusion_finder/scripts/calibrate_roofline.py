# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the roofline used for ideal times: HBM bandwidth and bf16 GEMM peak.

Usage: ``CUDA_VISIBLE_DEVICES=<idle gpu> python calibrate_roofline.py <capture dir>
[--gemm MxKxN ...]``. Writes ``<capture dir>/roofline.json``.

GEMM throughput depends on operand values (power/clock): near-zero operands run
much faster than random ones. Under the fake process group, activations and
weights have very low entropy and GEMMs run in that fast regime, so the
zero-operand rate is the peak that keeps ideal times a lower bound. With real
data, use ``--gemm-peak random``.
"""

import argparse
import json
import os
import subprocess

import torch


def _time_ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


def _mxfp8_gemm_tflops(shapes, dev):
    """MXFP8 GEMM throughput: rowwise 1x32 E8M0 scales, swizzled, BF16 output."""
    import torch.nn.functional as F
    from torchao.prototype.mx_formats.kernels import (
        mxfp8_quantize_cuda,
        triton_mx_block_rearrange,
    )

    def quantize(x):
        q, _, scale, _ = mxfp8_quantize_cuda(
            x, rowwise=True, colwise=False, scaling_mode="rceil"
        )
        return q, triton_mx_block_rearrange(scale)

    out = {}
    for shape in shapes:
        m, k, n = (int(x) for x in shape.split("x"))
        for fill in ("random", "zero"):
            a = torch.empty(m, k, dtype=torch.bfloat16, device=dev)
            w = torch.empty(n, k, dtype=torch.bfloat16, device=dev)
            if fill == "random":
                a.normal_()
                w.normal_()
            else:
                a.zero_()
                w.zero_()
            (aq, sa), (wq, sw) = quantize(a), quantize(w)

            def mm():
                return F.scaled_mm(
                    aq,
                    wq.t(),
                    scale_a=sa,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=sw,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )

            out[f"{shape}_{fill}"] = (
                2 * m * k * n / _time_ms(mm, iters=10, warmup=3) / 1e9
            )
            del a, w, aq, wq
            torch.cuda.empty_cache()
    return out


def _smi():
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="capture directory; roofline.json is written there")
    p.add_argument(
        "--gemm",
        action="append",
        default=[],
        help="bf16 GEMM shape MxKxN at a representative model shape (repeatable); default 16384x8192x8192",
    )
    p.add_argument("--gemm-peak", choices=["zero", "random"], default="zero")
    p.add_argument(
        "--mxfp8",
        action="store_true",
        help="also measure MXFP8 GEMMs (1x32 block scales, as MXFP8Linear runs them); needs torchao",
    )
    args = p.parse_args()
    dev = torch.device("cuda")
    res = {
        "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
        "nvidia_smi_before": _smi(),
    }

    bw = {}
    for gib in (1, 4, 8):
        a = torch.empty(
            gib * (1 << 30) // 2, dtype=torch.bfloat16, device=dev
        ).normal_()
        b = torch.empty_like(a)
        bw[f"d2d_copy_{gib}GiB"] = 2 * a.nbytes / _time_ms(lambda: b.copy_(a)) / 1e6
        bw[f"sum_bf16_{gib}GiB"] = (
            a.nbytes / _time_ms(lambda: a.sum(dtype=torch.float32)) / 1e6
        )
        bw[f"mul_bf16_{gib}GiB"] = (
            2 * a.nbytes / _time_ms(lambda: torch.mul(a, 2.0, out=b)) / 1e6
        )
        del a, b
        torch.cuda.empty_cache()
    res["bandwidth_gbps"] = bw

    gemm = {}
    for shape in args.gemm or ["16384x8192x8192"]:
        m, k, n = (int(x) for x in shape.split("x"))
        for fill in ("random", "zero"):
            a = torch.empty(m, k, dtype=torch.bfloat16, device=dev)
            b = torch.empty(k, n, dtype=torch.bfloat16, device=dev)
            if fill == "random":
                a.normal_()
                b.normal_()
            else:
                a.zero_()
                b.zero_()
            gemm[f"{shape}_{fill}"] = (
                2
                * m
                * k
                * n
                / _time_ms(lambda: torch.mm(a, b), iters=10, warmup=3)
                / 1e9
            )
            del a, b
            torch.cuda.empty_cache()
    res["bf16_gemm_tflops"] = gemm
    mx = _mxfp8_gemm_tflops(args.gemm or ["16384x8192x8192"], dev) if args.mxfp8 else {}
    if mx:
        res["mxfp8_gemm_tflops"] = mx
    res["nvidia_smi_after"] = _smi()
    res["roofline"] = {
        "hbm_gbps": round(max(v for k, v in bw.items() if k.startswith("d2d")), 1),
        "hbm_read_only_gbps": round(
            max(v for k, v in bw.items() if k.startswith("sum")), 1
        ),
        "hbm_pointwise_rw_gbps": round(
            max(v for k, v in bw.items() if k.startswith("mul")), 1
        ),
        "bf16_gemm_tflops": round(
            max(v for k, v in gemm.items() if k.endswith(args.gemm_peak)), 1
        ),
        "bf16_gemm_tflops_random_operands": round(
            max(v for k, v in gemm.items() if k.endswith("random")), 1
        ),
        "gemm_peak": args.gemm_peak,
    }
    if mx:
        res["roofline"]["mxfp8_gemm_tflops"] = round(
            max(v for k, v in mx.items() if k.endswith(args.gemm_peak)), 1
        )
        res["roofline"]["mxfp8_gemm_tflops_random_operands"] = round(
            max(v for k, v in mx.items() if k.endswith("random")), 1
        )
    print(json.dumps(res, indent=1))
    os.makedirs(args.root, exist_ok=True)
    with open(os.path.join(args.root, "roofline.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
