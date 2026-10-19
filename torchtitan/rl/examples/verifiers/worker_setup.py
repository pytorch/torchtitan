# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Setup that the Verifiers env server and each of its pool workers run first.

Torch-free on purpose: every pool worker imports this module to unpickle
`setup_env_server_worker` (`test_env_server_workers_skip_the_trainer_imports` checks it).
"""

import gc
import importlib
import os
import sys
import time

from verifiers.v1.utils.logging import setup_logging


def setup_env_server_worker(level: str, local_taskset_module: str | None) -> None:
    """Block torch, route Verifiers logs to stderr at `level`, and register the local taskset.

    Verifiers runs it first in the env server and in each pool worker it spawns:
        env server   torch already imported through torchtitan -> left as is
        pool worker  `import torch` raises ImportError, so the renderer's tokenizer loads
                     without torch (Qwen3.5 renderer: 776 -> 341 MB anonymous memory)
        pool worker  import_taskset(alias) -> the local taskset module
                     (without this hook: ModuleNotFoundError, and the worker exits)
    A taskset that imports torch fails in pool workers: give it num_workers=1, so the env
    server serves its rollouts itself.
    """
    if "torch" not in sys.modules:
        sys.modules["torch"] = None
    setup_logging(level)
    _log_gc_time()
    if local_taskset_module is not None:
        register_local_taskset_alias(local_taskset_module)


def register_local_taskset_alias(taskset_id: str) -> str:
    """Register a dotted local taskset under an importable Verifiers plugin ID.

    Example:
        "torchtitan.rl.examples.verifiers.terminal_bench.taskset"
        -> "torchtitan_rl_examples_verifiers_terminal_bench_taskset", now in sys.modules
    """
    if "." not in taskset_id or "/" in taskset_id:
        return taskset_id

    module = importlib.import_module(taskset_id)
    alias = taskset_id.replace(".", "_").lower()
    existing = sys.modules.get(alias)
    if existing is not None and existing is not module:
        raise ValueError(f"taskset alias {alias!r} is already registered")
    sys.modules[alias] = module
    return alias


def _log_gc_time(*, pause_s: float = 0.5, summary_every_s: float = 600.0) -> None:
    """Log this process's GC time: each pause over `pause_s`, and a total every `summary_every_s`.

    Env-server processes have no metrics channel to W&B (Verifiers drops extra fields), so their
    GC time goes to the job's stderr (print: `setup_logging` routes only Verifiers' loggers), e.g.
        env-server pid 4242: GC gen 2 paused 0.81 s (collected 1032)
        env-server pid 4242: GC 3.2 s in 1840 collections over the last 600 s
    """
    start = time.perf_counter()
    total_s, count, window_start = 0.0, 0, time.monotonic()

    def on_gc(phase: str, info: dict) -> None:
        nonlocal start, total_s, count, window_start
        if phase == "start":
            start = time.perf_counter()
            return
        seconds = time.perf_counter() - start
        total_s += seconds
        count += 1
        pid = os.getpid()
        if seconds > pause_s:
            print(
                f"env-server pid {pid}: GC gen {info['generation']} paused {seconds:.2f} s "
                f"(collected {info['collected']})",
                file=sys.stderr,
                flush=True,
            )
        window_s = time.monotonic() - window_start
        if window_s >= summary_every_s:
            print(
                f"env-server pid {pid}: GC {total_s:.1f} s in {count} collections over "
                f"the last {window_s:.0f} s",
                file=sys.stderr,
                flush=True,
            )
            total_s, count, window_start = 0.0, 0, time.monotonic()

    gc.callbacks.append(on_gc)
