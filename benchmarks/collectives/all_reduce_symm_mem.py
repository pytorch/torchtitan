# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark NCCL all-reduce on an NCCL symmetric-memory pool.

This benchmark compares four out-of-place, end-to-end paths. Each path starts
with a regular CUDA input, preserves it, and returns a regular CUDA output:

1. ``nccl_regular_e2e``: clone the input, then run regular NCCL in-place on the
   clone. This matches TorchTitan's NCCL fallback.
2. ``nccl_symk_e2e``: copy into an NCCL-pool staging buffer registered with
   ``symm=True``, run NCCL SymK, then copy into the regular CUDA output.
3. ``custom_e2e``: TorchTitan's out-of-place ``_custom_all_reduce``
   implementation. It preserves the original input and returns a new tensor.
   The measurement includes any copies needed to stage data through the
   persistent symmetric-memory buffer.
4. ``vllm_custom_e2e``: vLLM's ``CustomAllreduce`` with
   ``registered=False``, matching the direct custom-AR leg used by TorchTitan's
   RL integration. The vLLM op stages the regular input through its
   pre-registered IPC buffer and writes directly to a new output. Sizes rejected
   by vLLM's production eligibility check are reported as ``nan`` rather than
   silently measuring a fallback. This isolates the custom kernel; vLLM's full
   dispatcher may select NCCL symmetric memory before reaching it.

Pass ``--force-vllm-all-sizes`` to bypass vLLM's production size cap and size
its IPC workspaces to cover every requested message. This is an experimental
kernel characterization mode, not vLLM's production dispatch policy.

Example:

    PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
        benchmarks/collectives/all_reduce_symm_mem.py \
        --sizes 64k 1m 4m 8m 16m 32m --dtype bfloat16

The reported time is the median of repeated measurements. Each measurement is
the maximum per-call CUDA-event latency across all ranks.
"""

import argparse
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)

from torchtitan.distributed import comms


BenchmarkFn = Callable[[], torch.Tensor]

_SIZE_SUFFIXES = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
_DEFAULT_SIZES = ("4k", "16k", "64k", "128k", "1m", "4m", "8m", "16m", "32m")
_RESULT_HEADER = (
    "bytes,custom_algo,vllm_custom_eligible,nccl_regular_e2e_us,"
    "nccl_symk_e2e_us,custom_e2e_us,vllm_custom_e2e_us,"
    "symk/regular_e2e,custom/regular_e2e,custom/symk_e2e,"
    "vllm/regular_e2e,vllm/custom_e2e"
)


def _parse_size(value: str) -> int:
    value = value.strip().lower()
    suffix = value[-1:]
    if suffix in _SIZE_SUFFIXES:
        return int(value[:-1]) * _SIZE_SUFFIXES[suffix]
    return int(value)


def _iterations_for_size(nbytes: int) -> int:
    if nbytes <= 256 * 1024:
        return 2_000
    if nbytes <= 4 * 1024 * 1024:
        return 1_000
    if nbytes <= 16 * 1024 * 1024:
        return 500
    if nbytes <= 32 * 1024 * 1024:
        return 250
    return 150


def _max_across_ranks(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def _synchronize_all_ranks(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    dist.barrier()


def _time_cuda(
    fn: BenchmarkFn,
    *,
    iterations: int,
    repeats: int,
    device: torch.device,
) -> float:
    samples = []
    for _ in range(repeats):
        _synchronize_all_ranks(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()

        elapsed_us = start.elapsed_time(end) * 1_000.0 / iterations
        samples.append(_max_across_ranks(elapsed_us, device))

    return statistics.median(samples)


def _warm_up(
    methods: dict[str, BenchmarkFn], warmup: int, device: torch.device
) -> None:
    for fn in methods.values():
        for _ in range(warmup):
            fn()
        _synchronize_all_ranks(device)


def _measure_methods(
    methods: dict[str, BenchmarkFn],
    *,
    iterations: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    return {
        name: _time_cuda(
            fn,
            iterations=iterations,
            repeats=repeats,
            device=device,
        )
        for name, fn in methods.items()
    }


def _check_results(results: dict[str, torch.Tensor], expected: float) -> None:
    for name, tensor in results.items():
        if not torch.all(tensor == expected):
            raise AssertionError(f"{name} correctness check failed")


def _format_timing(timing: float | None) -> str:
    return "nan" if timing is None else f"{timing:.3f}"


def _format_ratio(numerator: float | None, denominator: float) -> str:
    return "nan" if numerator is None else f"{numerator / denominator:.3f}"


def _print_result(
    nbytes: int,
    algo: comms._Algo,
    timings: dict[str, float],
    *,
    vllm_custom_eligible: bool,
) -> None:
    vllm_timing = timings.get("vllm_custom_e2e")
    print(
        f"{nbytes},{algo.name},{str(vllm_custom_eligible).lower()},"
        f"{timings['nccl_regular_e2e']:.3f},"
        f"{timings['nccl_symk_e2e']:.3f},"
        f"{timings['custom_e2e']:.3f},"
        f"{_format_timing(vllm_timing)},"
        f"{timings['nccl_symk_e2e'] / timings['nccl_regular_e2e']:.3f},"
        f"{timings['custom_e2e'] / timings['nccl_regular_e2e']:.3f},"
        f"{timings['custom_e2e'] / timings['nccl_symk_e2e']:.3f},"
        f"{_format_ratio(vllm_timing, timings['nccl_regular_e2e'])},"
        f"{_format_ratio(vllm_timing, timings['custom_e2e'])}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=_DEFAULT_SIZES,
        help="Message sizes in bytes; k/m/g suffixes use powers of two.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the adaptive number of iterations per measurement.",
    )
    parser.add_argument(
        "--force-vllm-all-sizes",
        action="store_true",
        help=(
            "Bypass vLLM's production size cap and allocate IPC workspaces "
            "large enough for every requested size. This consumes roughly "
            "twice the largest message size per GPU."
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("iterations must be positive")

    sizes = [_parse_size(value) for value in args.sizes]
    dtype = getattr(torch, args.dtype)
    if any(nbytes <= 0 or nbytes % dtype.itemsize for nbytes in sizes):
        raise ValueError("every size must be positive and divisible by dtype.itemsize")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD
    group_name = group.group_name
    nccl_backend = group._get_backend(device)

    # Preallocate the SymK staging buffer. Every rank allocates from the NCCL
    # pool in the same order. Merely using
    # torch.distributed._symmetric_memory.empty() does not register this
    # ProcessGroupNCCL communicator for SymK.
    max_numel = max(sizes) // dtype.itemsize
    nccl_pool = torch.cuda.MemPool(
        nccl_backend.mem_allocator,
        use_on_oom=False,
        no_split=True,
    )
    with torch.cuda.use_mem_pool(nccl_pool):
        symk_staging_storage = torch.empty(max_numel, dtype=dtype, device=device)
    nccl_backend.register_mem_pool(nccl_pool, symm=True)

    # vLLM's custom all-reduce exchanges CUDA IPC handles over a CPU process
    # group. Production mode enables the architecture/world-size cap used when
    # vLLM's symmetric-memory backend is available. Force-all mode deliberately
    # bypasses that cap and makes the strict ``size < max_size`` check accept the
    # largest requested message. TorchTitan's RL integration forces this
    # communicator onto registered=False, which includes one staging copy.
    vllm_cpu_group = dist.new_group(backend="gloo")
    vllm_max_size = (
        max(sizes) + 16 if args.force_vllm_all_sizes else 8 * 1024 * 1024
    )
    vllm_custom_ar = CustomAllreduce(
        group=vllm_cpu_group,
        device=device,
        max_size=vllm_max_size,
        symm_mem_enabled=not args.force_vllm_all_sizes,
    )
    if vllm_custom_ar.disabled:
        raise RuntimeError("vLLM CustomAllreduce is disabled on this topology")

    if rank == 0:
        # Output columns:
        # - nccl_regular_e2e_us: clone + in-place regular NCCL.
        # - nccl_symk_e2e_us: copy-in + in-place NCCL SymK + copy-out.
        # - custom_e2e_us: TorchTitan's custom out-of-place all-reduce.
        # - vllm_custom_e2e_us: vLLM custom AR with internal copy-in and direct
        #   output, or nan when vLLM rejects the message size.
        print(
            f"# world_size={world_size} gpu={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} nccl={torch.cuda.nccl.version()} "
            f"dtype={dtype} multicast={comms._has_multicast(local_rank)} "
            f"vllm_custom_mode="
            f"{'forced' if args.force_vllm_all_sizes else 'production'} "
            f"vllm_custom_max_bytes={vllm_custom_ar.max_size}",
            flush=True,
        )
        print(_RESULT_HEADER, flush=True)

    try:
        for nbytes in sizes:
            numel = nbytes // dtype.itemsize
            input_tensor = torch.full(
                (numel,), rank + 1, dtype=dtype, device=device
            )
            symk_staging_buffer = symk_staging_storage[:numel]
            algo = comms._select_algo(input_tensor, "sum", group_name)

            def nccl_regular_e2e() -> torch.Tensor:
                output = input_tensor.clone()
                dist.all_reduce(output, group=group)
                return output

            def nccl_symk_e2e() -> torch.Tensor:
                symk_staging_buffer.copy_(input_tensor)
                dist.all_reduce(symk_staging_buffer, group=group)
                output = torch.empty_like(input_tensor)
                output.copy_(symk_staging_buffer)
                return output

            def custom_e2e() -> torch.Tensor:
                return comms._custom_all_reduce(input_tensor, "sum", group_name)

            def vllm_custom_e2e() -> torch.Tensor:
                return vllm_custom_ar.all_reduce(input_tensor, registered=False)

            methods = {
                "nccl_regular_e2e": nccl_regular_e2e,
                "nccl_symk_e2e": nccl_symk_e2e,
                "custom_e2e": custom_e2e,
            }
            vllm_custom_eligible = vllm_custom_ar.should_custom_ar(input_tensor)
            if vllm_custom_eligible:
                methods["vllm_custom_e2e"] = vllm_custom_e2e

            # Verify every path once before measuring it.
            expected = float(world_size * (world_size + 1) // 2)
            correctness_results = {name: fn() for name, fn in methods.items()}
            _check_results(correctness_results, expected)
            _synchronize_all_ranks(device)

            _warm_up(methods, args.warmup, device)

            iterations = args.iterations or _iterations_for_size(nbytes)
            timings = _measure_methods(
                methods,
                iterations=iterations,
                repeats=args.repeats,
                device=device,
            )

            if rank == 0:
                _print_result(
                    nbytes,
                    algo,
                    timings,
                    vllm_custom_eligible=vllm_custom_eligible,
                )

            del input_tensor, symk_staging_buffer
            del correctness_results
            dist.barrier()
    finally:
        try:
            vllm_custom_ar.close()
            dist.destroy_process_group(vllm_cpu_group)
        finally:
            try:
                nccl_backend.deregister_mem_pool(nccl_pool)
                del symk_staging_storage, nccl_pool
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
