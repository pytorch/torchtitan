# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Independent bandwidth, FP32, and libdevice.exp controls on the assigned GPU."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl

from benchmarks._dsv3_mtp_measurement import benchmark_cuda
from triton.language.extra.cuda import libdevice


@triton.jit
def _copy(source, output, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, offsets < ELEMENTS, other=0)
    tl.store(output + offsets, values, offsets < ELEMENTS)


@triton.jit
def _triad(first, second, output, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    left = tl.load(first + offsets, offsets < ELEMENTS, other=0)
    right = tl.load(second + offsets, offsets < ELEMENTS, other=0)
    tl.store(output + offsets, left + 0.375 * right, offsets < ELEMENTS)


@triton.jit
def _register_chains(
    source, output, STEPS: tl.constexpr, BLOCK: tl.constexpr, EXP: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + offsets)
    first, second, third, fourth = value - 8.0, value - 8.5, value - 9.0, value - 9.5
    fifth, sixth, seventh, eighth = (
        value - 10.0,
        value - 10.5,
        value - 11.0,
        value - 11.5,
    )
    for _ in range(STEPS):
        if EXP:
            first = libdevice.exp(first) * 0.125 - 8.0
            second = libdevice.exp(second) * 0.250 - 8.5
            third = libdevice.exp(third) * 0.375 - 9.0
            fourth = libdevice.exp(fourth) * 0.500 - 9.5
            fifth = libdevice.exp(fifth) * 0.625 - 10.0
            sixth = libdevice.exp(sixth) * 0.750 - 10.5
            seventh = libdevice.exp(seventh) * 0.875 - 11.0
            eighth = libdevice.exp(eighth) * 1.000 - 11.5
        else:
            first = tl.fma(first, 1.0001, 0.0001)
            second = tl.fma(second, 1.0002, 0.0002)
            third = tl.fma(third, 1.0003, 0.0003)
            fourth = tl.fma(fourth, 1.0004, 0.0004)
            fifth = tl.fma(fifth, 1.0005, 0.0005)
            sixth = tl.fma(sixth, 1.0006, 0.0006)
            seventh = tl.fma(seventh, 1.0007, 0.0007)
            eighth = tl.fma(eighth, 1.0008, 0.0008)
    tl.store(
        output + offsets,
        first + second + third + fourth + fifth + sixth + seventh + eighth,
    )


def calibrate(output_path):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": {
            "name": properties.name,
            "uuid": str(properties.uuid),
            "sm_count": properties.multi_processor_count,
            "l2_bytes": properties.L2_cache_size,
            "total_memory_bytes": properties.total_memory,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "controls": [],
        "assumptions": [
            "Highest measured throughput across independent controls; "
            "no candidate kernel is a calibration.",
            "Copy counts one read and one write. FP32 STREAM triad counts two reads and one write.",
            "Streaming buffers exceed twice L2; warm buffers fit within half L2.",
            "Eight independent exp register chains use normal negative values "
            "and one FMA per exp to keep values bounded.",
            "No reduction, gather, or mixed-resource scheduling ceiling "
            "is certified by these controls.",
            "Clocks are not modified or locked. Device UUID and every timing sample are retained.",
        ],
    }

    def record(
        name, function, tensors, regime, *, count_bytes=0, count_ops=0, extra=None
    ):
        timing = benchmark_cuda(
            function,
            warmup=10,
            batches=30,
            iterations=16 if regime == "streaming" else 64,
            regime=regime,
            buffer_tensors=tensors,
        )
        row: dict[str, Any] = dict(
            name=name,
            timing=timing,
            minimum_bytes=count_bytes,
            operations=count_ops,
            bytes_per_second=count_bytes / timing["median_ms"] * 1000,
            operations_per_second=count_ops / timing["median_ms"] * 1000,
            **(extra or {}),
        )
        result["controls"].append(row)
        print(
            name,
            extra,
            "us",
            timing["median_ms"] * 1000,
            "TB/s",
            row["bytes_per_second"] / 1e12,
            "Tops/s",
            row["operations_per_second"] / 1e12,
            flush=True,
        )
        output_path.with_suffix(".partial.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return row

    source = torch.ones(1, device="cuda")
    output = torch.empty_like(source)
    launch = record(
        "launch",
        lambda: cast(Any, _copy)[(1,)](source, output, 1, 32, num_warps=1),
        [source, output],
        "warm",
        count_bytes=8,
    )
    del source, output
    for array_bytes in (16 * 1024**2, 256 * 1024**2, 1024**3):
        regime = (
            "warm" if 3 * array_bytes < properties.L2_cache_size // 2 else "streaming"
        )
        for dtype in (torch.float32, torch.bfloat16):
            elements = array_bytes // torch.empty((), dtype=dtype).element_size()
            first = torch.full((elements,), 0.625, device="cuda", dtype=dtype)
            second = torch.full_like(first, 0.375)
            output = torch.empty_like(first)
            for block, warps in ((1024, 4), (4096, 4), (8192, 4), (8192, 8)):
                grid = (triton.cdiv(elements, block),)
                extra = dict(
                    array_bytes=array_bytes, dtype=str(dtype), block=block, warps=warps
                )
                record(
                    "copy",
                    lambda: cast(Any, _copy)[grid](
                        first, output, elements, block, num_warps=warps
                    ),
                    [first, output],
                    regime,
                    count_bytes=2 * array_bytes,
                    extra=extra,
                )
                if dtype == torch.float32:
                    record(
                        "triad",
                        lambda: cast(Any, _triad)[grid](
                            first, second, output, elements, block, num_warps=warps
                        ),
                        [first, second, output],
                        regime,
                        count_bytes=3 * array_bytes,
                        extra=extra,
                    )
            del first, second, output
            torch.cuda.empty_cache()
    for operation in ("fp32", "exp"):
        for blocks_per_sm, block, warps in ((4, 256, 4), (8, 256, 4), (8, 512, 8)):
            blocks, steps = properties.multi_processor_count * blocks_per_sm, 256
            elements = blocks * block
            source = torch.full((elements,), 0.25, device="cuda")
            output = torch.empty_like(source)
            function = lambda: cast(Any, _register_chains)[(blocks,)](
                source, output, steps, block, operation == "exp", num_warps=warps
            )
            compiled = function()
            ptx_path = (
                output_path.parent / f"{operation}-{blocks_per_sm}-{block}-{warps}.ptx"
            )
            ptx_path.write_text(compiled.asm["ptx"])
            record(
                operation,
                function,
                [source, output],
                "warm",
                count_ops=elements * steps * 8 * (2 if operation == "fp32" else 1),
                extra=dict(
                    blocks_per_sm=blocks_per_sm, block=block, warps=warps, steps=steps
                ),
            )
            del source, output, function
    stable = [
        r for r in result["controls"] if r["timing"]["coefficient_of_variation"] <= 0.05
    ]
    memory = [r for r in stable if r["name"] in ("copy", "triad")]
    result["roofs"] = {
        "launch_ms": launch["timing"]["p10_ms"],
        "hbm_bytes_per_second": max(
            r["bytes_per_second"]
            for r in memory
            if r["timing"]["regime"] == "streaming"
        ),
        "l2_bytes_per_second": max(
            r["bytes_per_second"] for r in memory if r["timing"]["regime"] == "warm"
        ),
        "fp32_ops_per_second": max(
            r["operations_per_second"] for r in stable if r["name"] == "fp32"
        ),
        "special_functions_per_second": {
            "exp": max(r["operations_per_second"] for r in stable if r["name"] == "exp")
        },
    }
    result["telemetry"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pstate,temperature.gpu,"
            "power.draw,power.limit,clocks.sm,clocks.mem,memory.used",
            "--format=csv",
            "--id=GPU-" + str(properties.uuid).removeprefix("GPU-"),
        ],
        text=True,
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["roofs"], indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fused_dsv3_mtp_loss/roofline/calibration.json"),
    )
    calibrate(parser.parse_args().output)
