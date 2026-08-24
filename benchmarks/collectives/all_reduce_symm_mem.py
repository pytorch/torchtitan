# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark NCCL all-reduce on an NCCL symmetric-memory pool.

This benchmark compares three out-of-place, end-to-end paths. Each path starts
with a regular CUDA input, preserves it, and returns a regular CUDA output:

1. ``nccl_regular_e2e``: clone the input, then run regular NCCL in-place on the
   clone. This matches TorchTitan's NCCL fallback.
2. ``nccl_symk_e2e``: copy into an NCCL-pool staging buffer registered with
   ``symm=True``, run NCCL SymK, then copy into the regular CUDA output.
3. ``custom_e2e``: TorchTitan's out-of-place ``_custom_all_reduce``
   implementation. It preserves the original input and returns a new tensor.
   The measurement includes any copies needed to stage data through the
   persistent symmetric-memory buffer.

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

from torchtitan.distributed import comms


BenchmarkFn = Callable[[], torch.Tensor]

_SIZE_SUFFIXES = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
_DEFAULT_SIZES = ("4k", "16k", "64k", "128k", "1m", "4m", "8m", "16m", "32m")
_RESULT_HEADER = (
    "bytes,custom_algo,nccl_regular_e2e_us,nccl_symk_e2e_us,custom_e2e_us,"
    "symk/regular_e2e,custom/regular_e2e,custom/symk_e2e"
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


def _print_result(nbytes: int, algo: comms._Algo, timings: dict[str, float]) -> None:
    print(
        f"{nbytes},{algo.name},"
        f"{timings['nccl_regular_e2e']:.3f},"
        f"{timings['nccl_symk_e2e']:.3f},"
        f"{timings['custom_e2e']:.3f},"
        f"{timings['nccl_symk_e2e'] / timings['nccl_regular_e2e']:.3f},"
        f"{timings['custom_e2e'] / timings['nccl_regular_e2e']:.3f},"
        f"{timings['custom_e2e'] / timings['nccl_symk_e2e']:.3f}",
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

    if rank == 0:
        # Output columns:
        # - nccl_regular_e2e_us: clone + in-place regular NCCL.
        # - nccl_symk_e2e_us: copy-in + in-place NCCL SymK + copy-out.
        # - custom_e2e_us: TorchTitan's custom out-of-place all-reduce.
        print(
            f"# world_size={world_size} gpu={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} nccl={torch.cuda.nccl.version()} "
            f"dtype={dtype} multicast={comms._has_multicast(local_rank)}",
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

            methods = {
                "nccl_regular_e2e": nccl_regular_e2e,
                "nccl_symk_e2e": nccl_symk_e2e,
                "custom_e2e": custom_e2e,
            }

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
                _print_result(nbytes, algo, timings)

            del input_tensor, symk_staging_buffer
            del correctness_results
            dist.barrier()
    finally:
        try:
            nccl_backend.deregister_mem_pool(nccl_pool)
            del symk_staging_storage, nccl_pool
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
