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

_config_file = "./torchtrain/train_config.toml"


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
    dump_dir = config["global"]["dump_folder"]
    run_profiler = config["profiling"].get("run_profiler", False)
    save_trace_dir = config["profiling"]["save_traces_folder"]
    trace_dir = os.path.join(dump_dir, save_trace_dir)

    num_iters_to_profile = config["profiling"]["num_iters_to_profile"]
    iter_to_start_profiling = config["profiling"]["iter_to_start_profiling"]
    # profiler wants a warmup, so we reduce when to start by 1
    iter_to_start_profiling -= 1

    rank = torch.distributed.get_rank()

    def trace_handler(prof):
        rank0_log(f"exporting profile traces to {trace_dir}")
        prof.export_chrome_trace(f"{trace_dir}/rank{rank}_trace.json")

    if run_profiler:
        rank0_log(f"Profiling active.  Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=iter_to_start_profiling,
                warmup=1,
                active=num_iters_to_profile,
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
