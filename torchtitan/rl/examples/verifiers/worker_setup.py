# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Setup that the Verifiers env server and each of its pool workers run first.

Torch-free on purpose: every pool worker imports this module to unpickle
`setup_env_server_worker` (`test_env_server_workers_skip_the_trainer_imports` checks it).
"""

import importlib
import sys

from verifiers.v1.utils.logging import setup_logging


def setup_env_server_worker(level: str, local_taskset_module: str | None) -> None:
    """Route Verifiers logs to stderr at `level`, and make the local taskset importable.

    Verifiers runs it in the env server and in each pool worker it spawns:
        pool worker without it:  import_taskset(alias) -> ModuleNotFoundError, worker exits
        pool worker with it:     import_taskset(alias) -> the local taskset module
    """
    setup_logging(level)
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
