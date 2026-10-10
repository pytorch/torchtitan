# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Production-shape CE timing, exact parity, peak memory, and roofline evidence."""

import argparse
import datetime
import gc
import hashlib
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from benchmarks._dsv3_mtp_measurement import (
    assess_roofline,
    benchmark_cuda,
    compare_bits,
)

from torchtitan_recipes.overrides._dsv3_mtp_cross_entropy import kernels
from torchtitan_recipes.overrides._dsv3_mtp_cross_entropy.ops import (
    backward_op,
    cross_entropy_sum,
)


def native(logits, labels):
    return F.cross_entropy(logits.float(), labels, reduction="sum", ignore_index=-100)


def peak_memory(function):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "base_allocated_bytes": base_allocated,
        "base_reserved_bytes": base_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "live_allocated_bytes": torch.cuda.memory_allocated(),
    }
    result["incremental_peak_allocated_bytes"] = (
        result["peak_allocated_bytes"] - base_allocated
    )
    result["incremental_peak_reserved_bytes"] = (
        result["peak_reserved_bytes"] - base_reserved
    )
    del output
    gc.collect()
    torch.cuda.synchronize()
    return result


def launches(function, output_path):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profile:
        output = function()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(output_path))
    events = json.loads(output_path.read_text())["traceEvents"]
    names = [event["name"] for event in events if event.get("cat") == "kernel"]
    del output
    return {"count": len(names), "names": names}


def benchmark(args):
    warnings.filterwarnings("error", message=".*AccumulateGrad.*")
    if args.calibration is None:
        raise ValueError("Independent calibration is required")
    calibration = json.loads(args.calibration.read_text())
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if calibration["device"]["uuid"] != str(properties.uuid):
        raise ValueError("Calibration and candidate must use the same GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(821)
    tokens, vocab = 4096, 129280
    logits = torch.randn(
        tokens, vocab, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    labels = torch.randint(vocab, (tokens,), device="cuda")
    labels[::41] = -100
    grad_loss = torch.tensor(
        0.1 / (32 * 4095 * 128), device="cuda", dtype=torch.float32
    )
    functions = {"native": native, "fused": cross_entropy_sum}

    def forward_backward(function):
        current_logits = logits.detach().requires_grad_()
        loss = function(current_logits, labels)
        gradient = torch.autograd.grad(loss, current_logits, grad_loss)[0]
        return loss, gradient

    reference = forward_backward(native)
    candidate = forward_backward(cross_entropy_sum)
    parity = compare_bits(reference, candidate)
    assert parity["passed"], parity
    del reference, candidate
    kernel_metadata = {"forward": {}, "backward": {}}
    with torch.no_grad():
        _, stats = kernels.forward(logits, labels, metadata=kernel_metadata["forward"])
        kernels.backward(
            logits, labels, stats, grad_loss, metadata=kernel_metadata["backward"]
        )
    del stats
    result = {
        "kernel_metadata": kernel_metadata,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "node": 747954,
        "shape": [tokens, vocab],
        "logits_dtype": str(logits.dtype),
        "labels_dtype": str(labels.dtype),
        "grad_loss": grad_loss.item(),
        "device_uuid": str(properties.uuid),
        "device_name": properties.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "allocator_configuration": os.environ.get("PYTORCH_ALLOC_CONF"),
        "parity": parity,
        "timings": [],
        "peak_memory": {},
        "launches": {},
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "torchtitan_recipes/overrides/_dsv3_mtp_cross_entropy/kernels.py",
                "torchtitan_recipes/overrides/_dsv3_mtp_cross_entropy/ce_exp.py",
                "torchtitan_recipes/overrides/_dsv3_mtp_cross_entropy/ops.py",
                "tests/unit_tests/gpu/test_fused_dsv3_mtp_loss.py",
                "benchmarks/fused_dsv3_mtp_loss.py",
                "benchmarks/calibrate_dsv3_mtp_loss.py",
                "benchmarks/_dsv3_mtp_measurement.py",
                "torchtitan_recipes/overrides/fused_dsv3_mtp_loss.py",
            )
        },
        "method": [
            "CUDA graph replay: 8 operations per graph, 10 warmups, 40 samples "
            "per round, alternating native/fused order.",
            "Backward-only timings use exactly the native ATen backward "
            "operators versus the fused custom operator; preparation is excluded.",
            "Forward+backward timings use the complete public autograd operation "
            "with explicit scalar upstream and fresh gradients.",
            "Inputs alone exceed twice L2; no output copies, parity checks, "
            "setup, or profiling are in the timings.",
            "Peak memory includes original BF16 logits, labels, scalar upstream, "
            "temporaries, saved state, and the returned gradient.",
            "This is the complete loss region, not a full-model training-step benchmark.",
        ],
    }
    for name, function in functions.items():
        result["peak_memory"][name] = {
            "forward": peak_memory(lambda: function(logits, labels)),
            "forward_backward": peak_memory(lambda: forward_backward(function)),
        }
        result["launches"][name] = {
            "forward": launches(
                lambda: function(logits, labels),
                args.output.parent / f"{name}-forward-profile.json",
            ),
            "forward_backward": launches(
                lambda: forward_backward(function),
                args.output.parent / f"{name}-full-profile.json",
            ),
        }
    for round_index in range(2):
        for name in ("native", "fused") if round_index == 0 else ("fused", "native"):
            function = functions[name]
            for phase in ("forward", "backward", "forward_backward"):
                gc.collect()
                torch.cuda.empty_cache()
                state = None
                if phase == "forward":
                    operation = lambda: function(logits, labels)
                    buffers = [logits, labels]
                elif phase == "forward_backward":
                    operation = lambda: forward_backward(function)
                    buffers = [logits, labels, grad_loss]
                elif name == "native":
                    with torch.no_grad():
                        log_probabilities = F.log_softmax(logits.float(), dim=-1)
                        _, total_weight = torch.ops.aten.nll_loss_forward.default(
                            log_probabilities, labels, None, 2, -100
                        )
                    state = (log_probabilities, total_weight)

                    def operation():
                        nll_gradient = torch.ops.aten.nll_loss_backward.default(
                            grad_loss,
                            log_probabilities,
                            labels,
                            None,
                            2,
                            -100,
                            total_weight,
                        )
                        gradient = torch.ops.aten._log_softmax_backward_data.default(
                            nll_gradient, log_probabilities, 1, torch.float32
                        )
                        return gradient.to(torch.bfloat16)

                    buffers = [log_probabilities, labels, total_weight, grad_loss]
                else:
                    with torch.no_grad():
                        _, stats = kernels.forward(logits, labels)
                    state = stats
                    operation = lambda: backward_op(
                        logits, labels, stats, grad_loss, -100
                    )
                    buffers = [logits, labels, stats, grad_loss]
                timing = benchmark_cuda(
                    operation,
                    warmup=10,
                    batches=40,
                    iterations=8,
                    regime="streaming",
                    buffer_tensors=buffers,
                )
                row: dict[str, Any] = dict(
                    implementation=name, phase=phase, round=round_index, timing=timing
                )
                if name == "fused":
                    state_bytes = tokens * 2 * 4
                    minimum_bytes = logits.nbytes + labels.nbytes + state_bytes + 4
                    if phase == "backward":
                        minimum_bytes = (
                            2 * logits.nbytes
                            + labels.nbytes
                            + state_bytes
                            + grad_loss.nbytes
                        )
                    elif phase == "forward_backward":
                        minimum_bytes = (
                            3 * logits.nbytes + labels.nbytes + grad_loss.nbytes + 4
                        )
                    row["roofline"] = assess_roofline(
                        timing,
                        minimum_bytes=minimum_bytes,
                        calibration=calibration,
                        memory_level="hbm",
                        floating_ops=tokens
                        * vocab
                        * (6 if phase == "forward_backward" else 4),
                        special_function_ops={
                            "exp": tokens
                            * vocab
                            * (2 if phase == "forward_backward" else 1)
                        },
                    )
                result["timings"].append(row)
                print(
                    name,
                    phase,
                    round_index,
                    "median_us",
                    timing["median_ms"] * 1000,
                    "p90_us",
                    timing["p90_ms"] * 1000,
                    "roofline",
                    row.get("roofline", {}).get("p90_efficiency"),
                    flush=True,
                )
                args.output.with_suffix(".partial.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                del operation, buffers, state
                log_probabilities = total_weight = stats = None
    result["memory_passed"] = all(
        result["peak_memory"]["fused"][phase][field]
        <= result["peak_memory"]["native"][phase][field]
        for phase in ("forward", "forward_backward")
        for field in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "incremental_peak_allocated_bytes",
            "incremental_peak_reserved_bytes",
        )
    )
    graph_pool_bytes = {
        (row["implementation"], row["phase"], row["round"]): row["timing"][
            "capture_memory"
        ]["private_pool_reserved_bytes"]
        for row in result["timings"]
    }
    result["graph_pool_memory_passed"] = all(
        graph_pool_bytes["fused", phase, round_index]
        <= graph_pool_bytes["native", phase, round_index]
        for phase in ("forward", "backward", "forward_backward")
        for round_index in range(2)
    )
    result["roofline_passed"] = all(
        row["roofline"]["passed"]
        for row in result["timings"]
        if row["implementation"] == "fused"
    )
    result["all_gates_passed"] = all(
        (
            result["parity"]["passed"],
            result["memory_passed"],
            result["graph_pool_memory_passed"],
            result["roofline_passed"],
        )
    )
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
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "MEMORY",
        result["memory_passed"],
        "GRAPH POOL",
        result["graph_pool_memory_passed"],
        "ROOFLINE",
        result["roofline_passed"],
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/fused_dsv3_mtp_loss/roofline/calibration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fused_dsv3_mtp_loss/benchmark.json"),
    )
    benchmark(parser.parse_args())
