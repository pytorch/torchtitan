# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Strict correctness and independently calibrated CUDA fusion measurements.

This module deliberately contains no candidate kernels. Bandwidth is counted from
the fused operation's required inputs/outputs, never from removed intermediates.
"""

from __future__ import annotations

import json
import math
import statistics
import struct
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils import _pytree


def _raw_bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("Bitwise comparison requires unquantized strided tensors")
    tensor = tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
    integer_dtype = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    if tensor.element_size() in integer_dtype:
        return tensor.view(integer_dtype[tensor.element_size()])
    return tensor.view(torch.uint8).reshape(tensor.numel(), tensor.element_size())


def compare_bits(reference: Any, candidate: Any) -> dict[str, Any]:
    """Compare pytree values exactly, including signed zero and NaN payloads.

    ``None`` is not equivalent to a zero tensor. Tensor shape, dtype, and layout
    must match; strides are reported separately because an operator may legally
    return the same values in a different layout.
    """
    reference_leaves, reference_spec = _pytree.tree_flatten_with_path(reference)
    candidate_leaves, candidate_spec = _pytree.tree_flatten_with_path(candidate)
    if reference_spec != candidate_spec:
        return {
            "passed": False,
            "reason": "pytree_structure",
            "reference_structure": str(reference_spec),
            "candidate_structure": str(candidate_spec),
        }
    comparisons = []
    for (path, expected), (_, actual) in zip(reference_leaves, candidate_leaves):
        result: dict[str, Any] = {"path": _pytree.keystr(path)}
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            result.update(
                shape=list(expected.shape),
                dtype=str(expected.dtype),
                reference_stride=list(expected.stride()),
                candidate_stride=list(actual.stride()),
            )
            if (
                expected.shape != actual.shape
                or expected.dtype != actual.dtype
                or expected.layout != actual.layout
                or expected.device != actual.device
            ):
                result.update(passed=False, reason="tensor_metadata")
            else:
                expected_bits, actual_bits = _raw_bits(expected), _raw_bits(actual)
                unequal = expected_bits != actual_bits
                if unequal.ndim == 2:
                    unequal = unequal.any(dim=1)
                different = int(unequal.sum().item())
                result.update(passed=different == 0, different_elements=different)
                if different:
                    first = int(unequal.nonzero()[0].item())
                    result["first_different_element"] = first
                    result["reference_bits"] = expected_bits[first].cpu().tolist()
                    result["candidate_bits"] = actual_bits[first].cpu().tolist()
                    if expected.is_floating_point() or expected.is_complex():
                        expected_values = expected.detach().reshape(-1)
                        actual_values = actual.detach().reshape(-1)
                        result["reference_value"] = str(expected_values[first].item())
                        result["candidate_value"] = str(actual_values[first].item())
        elif type(expected) is not type(actual):
            result.update(passed=False, reason="value_type")
        elif isinstance(expected, float):
            result["passed"] = struct.pack("d", expected) == struct.pack("d", actual)
        else:
            result["passed"] = expected == actual
        comparisons.append(result)
    return {
        "passed": all(item["passed"] for item in comparisons),
        "leaves": comparisons,
    }


def tensor_storage_bytes(tensors: Sequence[torch.Tensor]) -> int:
    """Count distinct allocated storages once, including aliased tensor views."""
    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = fraction * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def benchmark_cuda(
    functions: Callable[[], Any] | Sequence[Callable[[], Any]],
    *,
    warmup: int = 10,
    batches: int = 20,
    iterations: int = 100,
    regime: str = "warm",
    unique_storage_bytes: int | None = None,
    buffer_tensors: Sequence[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Time one captured sequence per batch, excluding Python launch overhead.

    A sequence of closures cycles through independent buffer sets. For streaming
    measurements their actual allocated footprint must exceed twice device L2;
    each closure must touch its declared inputs/outputs once per cycle. Merely
    allocating a large unused tensor does not establish a streaming regime.
    ``iterations`` is rounded up to complete cycles. Callables must be repeatable
    and CUDA-graph safe, with no accumulation or state growth across invocations.
    """
    if warmup < 10 or batches < 10:
        raise ValueError("At least ten warmup and timed batches are required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if regime not in {"warm", "streaming"}:
        raise ValueError("regime must be 'warm' or 'streaming'")
    functions = [functions] if callable(functions) else list(functions)
    if not functions:
        raise ValueError("At least one callable is required")
    if buffer_tensors is not None:
        measured_storage_bytes = tensor_storage_bytes(buffer_tensors)
        if (
            unique_storage_bytes is not None
            and unique_storage_bytes != measured_storage_bytes
        ):
            raise ValueError("Declared footprint differs from supplied tensor storages")
        unique_storage_bytes = measured_storage_bytes
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    l2_bytes = properties.L2_cache_size
    if regime == "streaming" and (
        unique_storage_bytes is None or unique_storage_bytes <= 2 * l2_bytes
    ):
        raise ValueError(
            "Streaming measurement needs a touched footprint greater than 2 * L2"
        )
    operation_count = math.ceil(iterations / len(functions)) * len(functions)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for function in functions:
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    allocated_before_capture = torch.cuda.memory_allocated(device)
    reserved_before_capture = torch.cuda.memory_reserved(device)
    with torch.cuda.graph(graph, stream=stream):
        for index in range(operation_count):
            functions[index % len(functions)]()
    allocated_after_capture = torch.cuda.memory_allocated(device)
    reserved_after_capture = torch.cuda.memory_reserved(device)
    private_pool_id = graph.pool()
    private_pool_segments = (
        torch.cuda.memory_snapshot(mempool_id=private_pool_id, include_traces=False)
        if any(private_pool_id)
        else []
    )
    private_pool_reserved_bytes = sum(
        segment["total_size"] for segment in private_pool_segments
    )
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    elapsed = []
    for _ in range(batches):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) / operation_count)
    median = statistics.median(elapsed)
    result = {
        "median_ms": median,
        "mean_ms": statistics.mean(elapsed),
        "minimum_ms": min(elapsed),
        "maximum_ms": max(elapsed),
        "p10_ms": _percentile(elapsed, 0.10),
        "p90_ms": _percentile(elapsed, 0.90),
        "coefficient_of_variation": statistics.pstdev(elapsed)
        / statistics.mean(elapsed),
        "samples_ms": elapsed,
        "warmup_batches": warmup,
        "timed_batches": batches,
        "operations_per_graph": operation_count,
        "buffer_sets": len(functions),
        "regime": regime,
        "unique_storage_bytes": unique_storage_bytes,
        "working_set_upper_bound_bytes": (
            unique_storage_bytes + private_pool_reserved_bytes
            if unique_storage_bytes is not None
            else None
        ),
        "footprint_source": "tensor_storages"
        if buffer_tensors is not None
        else "caller_declared",
        "footprint_caveat": (
            "Supplied tensors exclude internal outputs/scratch; warm L2 claims "
            "need a complete working-set accounting."
        ),
        "capture_memory": {
            "allocated_before_bytes": allocated_before_capture,
            "allocated_after_bytes": allocated_after_capture,
            "reserved_before_bytes": reserved_before_capture,
            "reserved_after_bytes": reserved_after_capture,
            "reserved_increase_bytes": max(
                reserved_after_capture - reserved_before_capture, 0
            ),
            "private_pool_reserved_bytes": private_pool_reserved_bytes,
        },
        "device_l2_bytes": l2_bytes,
        "device_name": properties.name,
        "device_uuid": str(properties.uuid),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    del graph
    return result


def load_roofline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def assess_roofline(
    timing: dict[str, Any],
    *,
    minimum_bytes: int,
    calibration: dict[str, Any],
    memory_level: str,
    floating_ops: int = 0,
    special_function_ops: dict[str, int] | None = None,
    threshold: float = 0.90,
) -> dict[str, Any]:
    """Compare against independent bandwidth/SIMT throughput and launch floors.

    ``minimum_bytes`` counts only required fused reads and writes at the named
    memory level. ``floating_ops`` counts FP32 SIMT add/multiply operations (FMA
    counts two). ``special_function_ops`` counts calls to independently measured
    named special functions. Individual special functions sharing an execution
    pipeline require a combined resource analysis, not independent maxima.
    Passing this necessary bound alone is insufficient for reductions, gathers,
    unmeasured special functions, or multi-launch composites.
    """
    if memory_level not in {"hbm", "l2"}:
        raise ValueError("memory_level must be 'hbm' or 'l2'")
    if minimum_bytes < 0 or floating_ops < 0 or not 0 < threshold <= 1:
        raise ValueError("Invalid roofline count or threshold")
    special_function_ops = {} if special_function_ops is None else special_function_ops
    if any(count < 0 for count in special_function_ops.values()):
        raise ValueError("Special-function counts must be nonnegative")
    if (
        not minimum_bytes
        and not floating_ops
        and not any(special_function_ops.values())
    ):
        raise ValueError(
            "A zero-device-work backward has no standalone kernel roofline"
        )
    if timing["device_name"] != calibration["device"]["name"]:
        raise ValueError("Calibration and candidate device models differ")
    l2_bytes = timing["device_l2_bytes"]
    footprint = timing.get("unique_storage_bytes")
    working_set_upper_bound = timing.get("working_set_upper_bound_bytes", footprint)
    if memory_level == "l2" and (
        working_set_upper_bound is None
        or working_set_upper_bound > l2_bytes
        or timing["regime"] != "warm"
    ):
        raise ValueError(
            "An L2 comparison requires a measured warm footprint that fits L2"
        )
    if memory_level == "hbm" and (footprint is None or footprint <= l2_bytes):
        raise ValueError("A warm L2-sized working set cannot use the HBM roofline")
    bandwidth = calibration["roofs"][memory_level + "_bytes_per_second"]
    memory_ms = minimum_bytes / bandwidth * 1000
    compute_ms = floating_ops / calibration["roofs"]["fp32_ops_per_second"] * 1000
    special_function_bounds = {
        name: count / calibration["roofs"]["special_functions_per_second"][name] * 1000
        for name, count in special_function_ops.items()
    }
    launch_ms = calibration["roofs"]["launch_ms"]
    bound_ms = max(memory_ms, compute_ms, launch_ms, *special_function_bounds.values())
    efficiency = bound_ms / timing["median_ms"]
    conservative_efficiency = bound_ms / timing["p90_ms"]
    stable = timing["coefficient_of_variation"] <= 0.05
    calibration_consistent = efficiency <= 1.10
    return {
        "passed": conservative_efficiency >= threshold
        and stable
        and calibration_consistent,
        "threshold": threshold,
        "efficiency": efficiency,
        "p90_efficiency": conservative_efficiency,
        "stable": stable,
        "calibration_consistent": calibration_consistent,
        "calibration_consistency_note": (
            "A result over 110% of the calibrated roof requires a stronger "
            "independent control or corrected accounting before acceptance."
        ),
        "requires_streaming_confirmation": memory_level == "hbm"
        and footprint <= 2 * l2_bytes,
        "minimum_bytes": minimum_bytes,
        "floating_ops": floating_ops,
        "special_function_ops": special_function_ops,
        "special_function_bound_ms": special_function_bounds,
        "memory_level": memory_level,
        "bandwidth_bytes_per_second": bandwidth,
        "memory_bound_ms": memory_ms,
        "compute_bound_ms": compute_ms,
        "launch_bound_ms": launch_ms,
        "bound_ms": bound_ms,
        "observed_bytes_per_second": minimum_bytes / timing["median_ms"] * 1000,
        "model_limitations": "No reduction, gather transaction, contention, or combined shared-SFU resource roof is modeled.",
        "calibration_device_uuid": calibration["device"]["uuid"],
        "candidate_device_uuid": timing["device_uuid"],
    }
