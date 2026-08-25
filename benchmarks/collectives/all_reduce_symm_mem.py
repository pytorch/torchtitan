# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark NCCL all-reduce on an NCCL symmetric-memory pool.

This benchmark compares five out-of-place paths. Each path starts with a
regular CUDA input and preserves it. Output placement follows the underlying
implementation; for example, vLLM NCCL SymK returns a symmetric-pool tensor.

1. ``nccl_regular_e2e``: clone the input, then run regular NCCL in-place on the
   clone. This matches TorchTitan's NCCL fallback.
2. ``nccl_symk_e2e``: copy the regular input into a ProcessGroupNCCL symmetric
   buffer, run NCCL in-place on that buffer, and return it directly. This is
   out-of-place relative to the original input and has no copy-out.
3. ``custom_e2e``: TorchTitan's out-of-place ``_custom_all_reduce``
   implementation. It preserves the original input and returns a new tensor.
   The measurement includes any copies needed to stage data through the
   persistent symmetric-memory buffer.
4. ``vllm_tp_e2e``: vLLM's production
   ``tensor_model_parallel_all_reduce`` entry point. Its dispatcher selects
   NCCL symmetric memory, vLLM CustomAllreduce, PyTorch symmetric memory,
   PyNCCL, or the PyTorch NCCL fallback according to the active configuration
   and input size. The benchmark reports separate columns with
   ``VLLM_USE_NCCL_SYMM_MEM=1`` and ``VLLM_USE_NCCL_SYMM_MEM=0``.
5. ``custom_no_multimem_e2e``: TorchTitan's custom all-reduce while simulating
   a platform without multicast support. It selects P2P one-shot through
   128 KiB, P2P two-shot through 16 MiB, and NCCL fallback above 16 MiB.

By default, ten operations from every path are captured in a separate CUDA
Graph and replay latency is normalized per all-reduce. Capture-time allocation
and setup are excluded consistently for all paths. Pass ``--no-cuda-graph``
to measure eager execution instead.

Example:

    PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
        benchmarks/collectives/all_reduce_symm_mem.py \
        --sizes 64k 1m 4m 8m 16m 32m --dtype bfloat16

The reported time is the median of repeated measurements. Each measurement is
the maximum per-call CUDA-event latency across all ranks.
"""

import argparse
import contextlib
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_allreduce,
)
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id

from torchtitan.distributed import comms


BenchmarkFn = Callable[[], torch.Tensor]

_SIZE_SUFFIXES = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
_DEFAULT_SIZES = (
    "4k",
    "16k",
    "64k",
    "128k",
    "1m",
    "4m",
    "8m",
    "16m",
    "32m",
)
_GRAPH_CAPTURE_CYCLES = 10
_RESULT_HEADER = (
    "bytes,custom_algo,vllm_tp_backend,vllm_tp_no_nccl_symm_mem_backend,"
    "nccl_regular_e2e_us,nccl_symk_e2e_us,custom_e2e_us,vllm_tp_e2e_us,"
    "vllm_tp_no_nccl_symm_mem_e2e_us,"
    "symk/regular_e2e,custom/regular_e2e,custom/symk_e2e,"
    "vllm_tp/regular_e2e,vllm_tp/custom_e2e,"
    "vllm_tp_no_nccl_symm_mem/custom_e2e,custom_no_multimem_algo,"
    "custom_no_multimem_e2e_us,custom_no_multimem/custom_e2e"
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
    work_per_call: int = 1,
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

        elapsed_us = (
            start.elapsed_time(end) * 1_000.0 / iterations / work_per_call
        )
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
    work_per_call: dict[str, int] | None = None,
) -> dict[str, float]:
    work_per_call = work_per_call or {}
    return {
        name: _time_cuda(
            fn,
            iterations=iterations,
            repeats=repeats,
            device=device,
            work_per_call=work_per_call.get(name, 1),
        )
        for name, fn in methods.items()
    }


def _check_results(results: dict[str, torch.Tensor], expected: float) -> None:
    for name, tensor in results.items():
        if not torch.all(tensor == expected):
            raise AssertionError(f"{name} correctness check failed")


def _select_algo_without_multimem(
    input: torch.Tensor, reduce_op: str, group_name: str
) -> comms._Algo:
    """Apply TorchTitan's selector as if multicast were unavailable."""
    if reduce_op != "sum":
        return comms._Algo.NCCL
    if input.dtype not in comms._SUPPORTED_DTYPES or not input.is_contiguous():
        return comms._Algo.NCCL
    if torch.are_deterministic_algorithms_enabled():
        return comms._Algo.NCCL

    world_size = comms._group_world_size(group_name)
    if (
        world_size not in comms._SUPPORTED_WORLD_SIZES
        or not comms._is_intra_node(group_name)
    ):
        return comms._Algo.NCCL

    numel = input.numel()
    if numel == 0:
        return comms._Algo.NCCL
    vec = 16 // input.element_size()
    if numel % (world_size * vec) != 0:
        return comms._Algo.NCCL

    nbytes = input.numel() * input.element_size()
    if nbytes > comms._TWO_SHOT_MAX_BYTES:
        return comms._Algo.NCCL
    if nbytes <= comms._ONE_SHOT_MAX_BYTES:
        return comms._Algo.ONE_SHOT
    return comms._Algo.TWO_SHOT


def _custom_all_reduce_without_multimem(
    input: torch.Tensor, reduce_op: str, group_name: str
) -> torch.Tensor:
    """Run TorchTitan custom AR while forcing its non-multimem algorithm tree."""
    algo = _select_algo_without_multimem(input, reduce_op, group_name)
    if algo is comms._Algo.NCCL:
        return comms._nccl_fallback(input, reduce_op, group_name)

    symm_buffer = comms._get_symm_buffer(group_name, input.dtype)
    view = symm_buffer[: input.numel()].view_as(input)
    output = torch.empty_like(input)
    if algo is comms._Algo.ONE_SHOT:
        torch.ops.symm_mem.one_shot_all_reduce_copy_out(
            view, input, "sum", group_name, output
        )
    else:
        view.copy_(input)
        torch.ops.symm_mem.two_shot_all_reduce_out(view, "sum", group_name, output)
    return output


@contextlib.contextmanager
def _vllm_nccl_symm_mem_enabled(enabled: bool):
    """Temporarily select whether vLLM may dispatch to NCCL SymK."""
    name = "VLLM_USE_NCCL_SYMM_MEM"
    previous = os.environ.get(name)
    os.environ[name] = str(int(enabled))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _vllm_tp_backend(
    input: torch.Tensor, *, use_nccl_symm_mem: bool
) -> str:
    """Mirror vLLM's dispatch predicates to report the selected TP backend."""
    with _vllm_nccl_symm_mem_enabled(use_nccl_symm_mem):
        device_comm = get_tp_group().device_communicator
        if device_comm is None:
            return "TORCH_NCCL"

        pynccl_comm = getattr(device_comm, "pynccl_comm", None)
        if (
            pynccl_comm is not None
            and not pynccl_comm.disabled
            and should_nccl_symm_mem_allreduce(pynccl_comm.world_size, input)
        ):
            return "NCCL_SYMM_MEM"

        qr_comm = getattr(device_comm, "qr_comm", None)
        if (
            qr_comm is not None
            and not qr_comm.disabled
            and qr_comm.should_quick_allreduce(input)
        ):
            return "QUICK_REDUCE"

        fi_ar_comm = getattr(device_comm, "fi_ar_comm", None)
        if (
            fi_ar_comm is not None
            and not fi_ar_comm.disabled
            and fi_ar_comm.should_use_fi_ar(input)
        ):
            return "FLASHINFER"

        aiter_ar_comm = getattr(device_comm, "aiter_ar_comm", None)
        if (
            aiter_ar_comm is not None
            and not aiter_ar_comm.disabled
            and aiter_ar_comm.should_custom_ar(input)
        ):
            return "AITER_CUSTOM"

        ca_comm = getattr(device_comm, "ca_comm", None)
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input)
        ):
            return "VLLM_CUSTOM"

        symm_mem_comm = getattr(device_comm, "symm_mem_comm", None)
        if (
            symm_mem_comm is not None
            and not symm_mem_comm.disabled
            and symm_mem_comm.should_use_symm_mem(input)
        ):
            return "PYTORCH_SYMM_MEM"

        if pynccl_comm is not None and not pynccl_comm.disabled:
            return "PYNCCL"
        return "TORCH_NCCL"


def _capture_cuda_graph(
    fn: BenchmarkFn,
    device: torch.device,
    *,
    capture_context: contextlib.AbstractContextManager | None = None,
    prepare_graph_pool: Callable[[tuple[int, int]], None] | None = None,
) -> BenchmarkFn:
    """Capture repeated calls to ``fn`` and return a graph replay function."""
    capture_context = capture_context or contextlib.nullcontext()

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()

        graph = torch.cuda.CUDAGraph()
        graph_pool = torch.cuda.graph_pool_handle()
        if prepare_graph_pool is not None:
            prepare_graph_pool(graph_pool)

        graph_output = None
        with capture_context:
            with torch.cuda.graph(graph, pool=graph_pool):
                for _ in range(_GRAPH_CAPTURE_CYCLES):
                    graph_output = fn()

    capture_stream.synchronize()
    if graph_output is None:
        raise RuntimeError("CUDA Graph capture produced no output")

    keepalive = (graph, graph_output)

    def replay() -> torch.Tensor:
        keepalive[0].replay()
        return keepalive[1]

    return replay


def _capture_vllm_tp_graph(
    input: torch.Tensor,
    backend: str,
    device: torch.device,
    *,
    use_nccl_symm_mem: bool,
) -> BenchmarkFn:
    """Capture vLLM TP all-reduce with its required graph setup."""
    device_comm = get_tp_group().device_communicator
    if device_comm is None:
        raise RuntimeError("vLLM TP device communicator is not initialized")

    capture_context = contextlib.nullcontext()
    if backend == "VLLM_CUSTOM":
        custom_comm = getattr(device_comm, "ca_comm", None)
        if custom_comm is None or custom_comm.disabled:
            raise RuntimeError("vLLM CustomAllreduce is unavailable for graph capture")
        capture_context = custom_comm.capture()

    graph_input = input.clone()

    def vllm_tp_e2e() -> torch.Tensor:
        with _vllm_nccl_symm_mem_enabled(use_nccl_symm_mem):
            return tensor_model_parallel_all_reduce(graph_input)

    replay = _capture_cuda_graph(
        vllm_tp_e2e,
        device,
        capture_context=capture_context,
        prepare_graph_pool=set_graph_pool_id,
    )

    # Keep graph_input alive for every replay.
    keepalive = (graph_input, replay)

    def replay_with_input() -> torch.Tensor:
        return keepalive[1]()

    return replay_with_input


def _print_result(
    nbytes: int,
    algo: comms._Algo,
    vllm_tp_backend: str,
    vllm_tp_no_nccl_symm_mem_backend: str,
    no_multimem_algo: comms._Algo,
    timings: dict[str, float],
) -> None:
    vllm_timing = timings["vllm_tp_e2e"]
    vllm_no_symm_timing = timings["vllm_tp_no_nccl_symm_mem_e2e"]
    print(
        f"{nbytes},{algo.name},{vllm_tp_backend},"
        f"{vllm_tp_no_nccl_symm_mem_backend},"
        f"{timings['nccl_regular_e2e']:.3f},"
        f"{timings['nccl_symk_e2e']:.3f},"
        f"{timings['custom_e2e']:.3f},"
        f"{vllm_timing:.3f},"
        f"{vllm_no_symm_timing:.3f},"
        f"{timings['nccl_symk_e2e'] / timings['nccl_regular_e2e']:.3f},"
        f"{timings['custom_e2e'] / timings['nccl_regular_e2e']:.3f},"
        f"{timings['custom_e2e'] / timings['nccl_symk_e2e']:.3f},"
        f"{vllm_timing / timings['nccl_regular_e2e']:.3f},"
        f"{vllm_timing / timings['custom_e2e']:.3f},"
        f"{vllm_no_symm_timing / timings['custom_e2e']:.3f},"
        f"{no_multimem_algo.name},"
        f"{timings['custom_no_multimem_e2e']:.3f},"
        f"{timings['custom_no_multimem_e2e'] / timings['custom_e2e']:.3f}",
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
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Measure every path through CUDA Graph replay. Each path gets a "
            f"separate graph containing {_GRAPH_CAPTURE_CYCLES} all-reduces, "
            "and latency is normalized per all-reduce (enabled by default)."
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

    # Allocate one persistent buffer from ProcessGroupNCCL's allocator and
    # register its pool as symmetric. NCCL operates in-place on a prefix view;
    # returning that view preserves the regular input without a copy-out.
    max_numel = max(sizes) // dtype.itemsize
    nccl_pool = torch.cuda.MemPool(
        nccl_backend.mem_allocator,
        use_on_oom=False,
        no_split=True,
    )
    with torch.cuda.use_mem_pool(nccl_pool):
        symk_output_storage = torch.empty(max_numel, dtype=dtype, device=device)
    nccl_backend.register_mem_pool(nccl_pool, symm=True)

    # Initialize vLLM with NCCL symmetric memory available. The benchmark then
    # captures separate vLLM graphs with its dispatch flag enabled and disabled.
    os.environ["VLLM_USE_NCCL_SYMM_MEM"] = "1"
    os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
    set_custom_all_reduce(True)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    if rank == 0:
        # Output columns:
        # - nccl_regular_e2e_us: clone + in-place regular NCCL.
        # - nccl_symk_e2e_us: copy into a symmetric-pool output, run
        #   ProcessGroupNCCL in-place on it, and return it directly.
        # - custom_e2e_us: TorchTitan's custom out-of-place all-reduce.
        # - vllm_tp_e2e_us: vLLM's full tensor-parallel all-reduce dispatcher.
        #   vllm_tp_backend reports the backend selected for that message size.
        # - vllm_tp_no_nccl_symm_mem_e2e_us: the same vLLM dispatcher captured
        #   with VLLM_USE_NCCL_SYMM_MEM=0.
        # - custom_no_multimem_e2e_us: TorchTitan custom AR using only its P2P
        #   one-shot/two-shot kernels, with NCCL fallback above 16 MiB.
        # With --cuda-graph, every column is graph replay time normalized per
        # captured all-reduce; graph capture and setup are excluded.
        print(
            f"# world_size={world_size} gpu={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} nccl={torch.cuda.nccl.version()} "
            f"dtype={dtype} multicast={comms._has_multicast(local_rank)} "
            "vllm_nccl_symm_mem_variants=1,0 "
            f"cuda_graph={args.cuda_graph} "
            f"graph_ops={_GRAPH_CAPTURE_CYCLES if args.cuda_graph else 1} "
            f"vllm={vllm.__version__} vllm_source={vllm.__file__}",
            flush=True,
        )
        print(_RESULT_HEADER, flush=True)

    try:
        for nbytes in sizes:
            numel = nbytes // dtype.itemsize
            input_tensor = torch.full(
                (numel,), rank + 1, dtype=dtype, device=device
            )
            symk_output = symk_output_storage[:numel]
            algo = comms._select_algo(input_tensor, "sum", group_name)

            def nccl_regular_e2e() -> torch.Tensor:
                output = input_tensor.clone()
                dist.all_reduce(output, group=group)
                return output

            def nccl_symk_e2e() -> torch.Tensor:
                symk_output.copy_(input_tensor)
                dist.all_reduce(symk_output, group=group)
                return symk_output

            def custom_e2e() -> torch.Tensor:
                return comms._custom_all_reduce(input_tensor, "sum", group_name)

            vllm_tp_backend = _vllm_tp_backend(
                input_tensor, use_nccl_symm_mem=True
            )
            vllm_tp_no_nccl_symm_mem_backend = _vllm_tp_backend(
                input_tensor, use_nccl_symm_mem=False
            )

            def vllm_tp_e2e() -> torch.Tensor:
                with _vllm_nccl_symm_mem_enabled(True):
                    return tensor_model_parallel_all_reduce(input_tensor)

            def vllm_tp_no_nccl_symm_mem_e2e() -> torch.Tensor:
                with _vllm_nccl_symm_mem_enabled(False):
                    return tensor_model_parallel_all_reduce(input_tensor)

            no_multimem_algo = _select_algo_without_multimem(
                input_tensor, "sum", group_name
            )

            def custom_no_multimem_e2e() -> torch.Tensor:
                return _custom_all_reduce_without_multimem(
                    input_tensor, "sum", group_name
                )

            eager_methods = {
                "nccl_regular_e2e": nccl_regular_e2e,
                "nccl_symk_e2e": nccl_symk_e2e,
                "custom_e2e": custom_e2e,
                "custom_no_multimem_e2e": custom_no_multimem_e2e,
                "vllm_tp_e2e": vllm_tp_e2e,
                "vllm_tp_no_nccl_symm_mem_e2e": (
                    vllm_tp_no_nccl_symm_mem_e2e
                ),
            }

            # Verify every eager path before capture or measurement.
            expected = float(world_size * (world_size + 1) // 2)
            correctness_results = {
                name: fn() for name, fn in eager_methods.items()
            }
            _check_results(correctness_results, expected)
            del correctness_results
            _synchronize_all_ranks(device)

            if args.cuda_graph:
                methods = {}
                for name, fn in eager_methods.items():
                    _synchronize_all_ranks(device)
                    if name == "vllm_tp_e2e":
                        methods[name] = _capture_vllm_tp_graph(
                            input_tensor,
                            vllm_tp_backend,
                            device,
                            use_nccl_symm_mem=True,
                        )
                    elif name == "vllm_tp_no_nccl_symm_mem_e2e":
                        methods[name] = _capture_vllm_tp_graph(
                            input_tensor,
                            vllm_tp_no_nccl_symm_mem_backend,
                            device,
                            use_nccl_symm_mem=False,
                        )
                    else:
                        methods[name] = _capture_cuda_graph(fn, device)

                correctness_results = {
                    name: fn() for name, fn in methods.items()
                }
                _check_results(correctness_results, expected)
                del correctness_results
                _synchronize_all_ranks(device)
            else:
                methods = eager_methods

            _warm_up(methods, args.warmup, device)

            iterations = args.iterations or _iterations_for_size(nbytes)
            timings = _measure_methods(
                methods,
                iterations=iterations,
                repeats=args.repeats,
                device=device,
                work_per_call=(
                    {name: _GRAPH_CAPTURE_CYCLES for name in methods}
                    if args.cuda_graph
                    else None
                ),
            )

            if rank == 0:
                _print_result(
                    nbytes,
                    algo,
                    vllm_tp_backend,
                    vllm_tp_no_nccl_symm_mem_backend,
                    no_multimem_algo,
                    timings,
                )

            del methods, eager_methods
            del input_tensor, symk_output
            dist.barrier()
    finally:
        try:
            destroy_model_parallel()
        finally:
            try:
                nccl_backend.deregister_mem_pool(nccl_pool)
                del symk_output_storage, nccl_pool
            finally:
                destroy_distributed_environment()


if __name__ == "__main__":
    main()
