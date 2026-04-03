# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark forward_backward_step for graph_trainer (eager vs aot_fx_trace).

Measures per-step wall-clock time, peak GPU memory, TFLOPS, and MFU.
Uses the full Trainer infrastructure (distributed init, real model, real data,
parallelization). No optimizer.step() is called.

Usage:
    # Eager baseline (8 GPU, Llama3 8B, FSDP+TP)
    torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
      -m torchtitan.experiments.graph_trainer.benchmark \
      --module graph_trainer.llama3 --config graph_trainer_llama3_8b \
      --compile.no-enable \
      --parallelism.data_parallel_shard_degree=4 \
      --parallelism.tensor_parallel_degree=2

    # aot_fx_trace (8 GPU, Llama3 8B, FSDP+TP)
    torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
      -m torchtitan.experiments.graph_trainer.benchmark \
      --module graph_trainer.llama3 --config graph_trainer_llama3_8b \
      --compile.mode aot_fx_trace \
      --parallelism.data_parallel_shard_degree=4 \
      --parallelism.tensor_parallel_degree=2

    Add --warmup_steps N (default 3) and --benchmark_steps N (default 10).
    Add --torch_profiler to capture a single-step chrome trace after benchmarking.
    All benchmark batches are preloaded onto GPU upfront (no data loading
    in the timed region).
"""

import os
import sys
import warnings

import torch
import torch.distributed as dist
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer


def _extract_benchmark_args(args: list[str]) -> tuple[int, int, bool, list[str]]:
    """Extract benchmark-specific args before passing to ConfigManager.

    Returns (warmup_steps, benchmark_steps, torch_profiler, remaining_args).
    Injects --training.steps = warmup + benchmark into remaining_args
    so the Trainer allocates enough data.
    """
    warmup_steps = 3
    benchmark_steps = 10
    torch_profiler = False
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--warmup_steps":
            if i + 1 < len(args):
                warmup_steps = int(args[i + 1])
                i += 2
                continue
            else:
                raise ValueError("--warmup_steps requires a value")
        elif args[i].startswith("--warmup_steps="):
            warmup_steps = int(args[i].split("=", 1)[1])
            i += 1
            continue
        elif args[i] == "--benchmark_steps":
            if i + 1 < len(args):
                benchmark_steps = int(args[i + 1])
                i += 2
                continue
            else:
                raise ValueError("--benchmark_steps requires a value")
        elif args[i].startswith("--benchmark_steps="):
            benchmark_steps = int(args[i].split("=", 1)[1])
            i += 1
            continue
        elif args[i] == "--torch_profiler":
            torch_profiler = True
            i += 1
            continue
        # Strip user-supplied --training.steps (we override it below)
        if args[i] == "--training.steps":
            i += 2 if (i + 1 < len(args)) else 1
            continue
        elif args[i].startswith("--training.steps="):
            i += 1
            continue
        filtered.append(args[i])
        i += 1

    # We need warmup_steps + benchmark_steps batches from the dataloader:
    # warmup uses fresh data per step, then we preload benchmark_steps batches
    # onto GPU upfront so data loading is excluded from the timed region.
    # Add 1 extra for the profiler step if enabled.
    total_steps = warmup_steps + benchmark_steps + (1 if torch_profiler else 0)
    filtered.extend(["--training.steps", str(total_steps)])

    return warmup_steps, benchmark_steps, torch_profiler, filtered


def _preload_batch(trainer: Trainer, data_iterator):
    """Load one batch from the iterator and move it to GPU.

    Returns (microbatches, global_valid_tokens) ready for repeated use
    so that data loading / H2D transfer is excluded from timed regions.
    """
    parallel_dims = trainer.parallel_dims

    microbatches = []
    local_valid_tokens = torch.tensor(0, dtype=torch.int64)
    for _ in range(trainer.gradient_accumulation_steps):
        input_dict, labels = next(data_iterator)
        local_valid_tokens += (labels != IGNORE_INDEX).sum()
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.to(trainer.device)
        labels = labels.to(trainer.device)
        microbatches.append((input_dict, labels))

    local_valid_tokens = local_valid_tokens.to(trainer.device)
    if parallel_dims.dp_enabled:
        batch_mesh = parallel_dims.get_mesh("batch")
        global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
    else:
        global_valid_tokens = local_valid_tokens.float()

    return microbatches, global_valid_tokens


def _run_step(trainer: Trainer, microbatches, global_valid_tokens) -> None:
    """Run one forward-backward step with preloaded data."""
    for input_dict, labels in microbatches:
        trainer.forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )


def run_benchmark() -> None:
    init_logger()

    warmup_steps, benchmark_steps, torch_profiler, remaining_args = (
        _extract_benchmark_args(sys.argv[1:])
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args(remaining_args)

    trainer: Trainer = config.build()

    # Compute FLOPS info
    model = trainer.model_parts[0]
    model_config = trainer.model_config
    _, num_flops_per_token = model_config.get_nparams_and_flops(
        model, config.training.seq_len
    )
    tokens_per_step = (
        config.training.local_batch_size
        * config.training.seq_len
        * trainer.gradient_accumulation_steps
    )
    gpu_peak_flops = utils.get_peak_flops(
        trainer.metrics_processor.device_memory_monitor.device_name
    )

    compile_mode = getattr(config.compile, "mode", "N/A")
    compile_enabled = config.compile.enable

    rank = dist.get_rank()
    if rank == 0:
        logger.info(
            f"Benchmark: mode={'eager' if not compile_enabled else compile_mode}, "
            f"warmup={warmup_steps}, benchmark={benchmark_steps}, "
            f"tokens/step={tokens_per_step}"
        )

    data_iterator = trainer.batch_generator(trainer.dataloader)

    # Warmup (fresh data each step)
    if rank == 0:
        logger.info(f"Running {warmup_steps} warmup steps...")
    for step in range(warmup_steps):
        trainer.optimizers.zero_grad()
        mb, gvt = _preload_batch(trainer, data_iterator)
        _run_step(trainer, mb, gvt)
        torch.cuda.synchronize()
        if rank == 0:
            logger.info(f"  warmup step {step + 1}/{warmup_steps} done")

    # Preload all benchmark (+ profiler) batches onto GPU upfront.
    # Each entry is a distinct batch so inputs vary realistically,
    # but data loading / H2D transfer is excluded from the timed region.
    num_preload = benchmark_steps + (1 if torch_profiler else 0)
    preloaded_batches = [
        _preload_batch(trainer, data_iterator) for _ in range(num_preload)
    ]

    # Reset memory stats so peak reflects only the benchmark region.
    # Preloaded batches are already on GPU but their allocations happened
    # before this reset, so they don't inflate the reported peak.
    torch.cuda.reset_peak_memory_stats()

    # Benchmark using CUDA events for accurate GPU timing
    if rank == 0:
        logger.info(f"Running {benchmark_steps} benchmark steps...")

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    # Synchronize all ranks and flush GPU pipeline before timing
    dist.barrier()
    torch.cuda.synchronize()

    ev_start.record()
    for step in range(benchmark_steps):
        trainer.optimizers.zero_grad()
        microbatches, global_valid_tokens = preloaded_batches[step]
        _run_step(trainer, microbatches, global_valid_tokens)
    ev_end.record()
    torch.cuda.synchronize()

    # elapsed_time() returns ms
    total_ms = ev_start.elapsed_time(ev_end)
    mean_ms = total_ms / benchmark_steps

    peak_mem_bytes = torch.cuda.max_memory_reserved()
    peak_mem_gib = peak_mem_bytes / (1024**3)

    # tokens-per-second per device: divide by non_data_parallel_size (tp * pp * cp)
    # because those dimensions share compute on the same tokens.
    # This matches the MFU calculation in MetricsProcessor.
    mean_s = mean_ms / 1e3
    tps = tokens_per_step / mean_s / trainer.parallel_dims.non_data_parallel_size
    tflops = num_flops_per_token * tps / 1e12
    mfu = 100 * num_flops_per_token * tps / gpu_peak_flops

    mode_str = "eager" if not compile_enabled else compile_mode
    if rank == 0:
        logger.info(
            f"\n{'=' * 50}\n"
            f"Benchmark Results (mode={mode_str})\n"
            f"{'=' * 50}\n"
            f"Steps:           {benchmark_steps} (after {warmup_steps} warmup)\n"
            f"Mean step time:  {mean_ms:.2f} ms\n"
            f"Tokens/step:     {tokens_per_step}\n"
            f"Peak memory:     {peak_mem_gib:.2f} GiB (reserved)\n"
            f"TFLOPS:          {tflops:.2f}\n"
            f"MFU:             {mfu:.2f}%\n"
            f"{'=' * 50}"
        )

    # Optional: capture a single-step torch profiler trace
    if torch_profiler:
        trace_dir = os.path.join(config.dump_folder, "benchmark_traces")
        os.makedirs(trace_dir, exist_ok=True)
        trace_file = os.path.join(trace_dir, f"benchmark_{mode_str}.{rank}.json.gz")

        def _make_profiler():
            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            )

        # Warm up CUPTI (suppress "clears events" warning from reuse)
        torch.cuda.synchronize()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Profiler clears events.*")
            with _make_profiler():
                pass
            # Profile a single step
            profiler_batch = preloaded_batches[benchmark_steps]
            profiler = _make_profiler()
            dist.barrier()
            with profiler:
                trainer.optimizers.zero_grad()
                _run_step(trainer, *profiler_batch)
                torch.cuda.synchronize()
        profiler.export_chrome_trace(trace_file)
        if rank == 0:
            logger.info(f"Profiler trace saved to {trace_dir}/")

    trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_benchmark()
