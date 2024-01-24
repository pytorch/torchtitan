# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import contextlib
import os
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtrain.logging_utils import rank0_log

_config_file = "./torchtrain/train_config/train_config.toml"


def get_config_from_toml(config_path: str = _config_file) -> dict:
    """
    Reads a config file in TOML format and returns a dictionary.
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


@contextlib.contextmanager
def maybe_run_profiler(*pos_args, **kwargs):
    config = get_config_from_toml()

    # get user defined profiler settings
    run_profiler = config["profiling"].get("run_profiler", False)

    if run_profiler:
        dump_dir = config["global"]["dump_folder"]
        save_trace_dir = config["profiling"]["save_traces_folder"]
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        iter_frequency = config["profiling"]["profile_every_x_iter"]

        _global_iter_count = 0

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            nonlocal _global_iter_count
            _global_iter_count += iter_frequency
            curr_trace_dir_name = "iteration_" + str(_global_iter_count)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir)
            rank0_log(f"exporting profile traces to {curr_trace_dir}")

            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")

        rank0_log(f"Profiling active.  Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

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
