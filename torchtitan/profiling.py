# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import os

import torch
from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger


@contextlib.contextmanager
def maybe_run_profiler(config: JobConfig, *pos_args, **kwargs):
    # get user defined profiler settings
    run_profiler = config.profiling.enable_profiling

    if run_profiler:
        dump_dir = config.job.dump_folder
        save_trace_dir = config.profiling.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        iter_frequency = config.profiling.profile_freq

        _global_iter_count = 0

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            nonlocal _global_iter_count
            _global_iter_count += iter_frequency
            curr_trace_dir_name = "iteration_" + str(_global_iter_count)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=iter_frequency - 2,
                warmup=1,
                active=1,
                repeat=0,
            ),
            on_trace_ready=trace_handler,
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None
