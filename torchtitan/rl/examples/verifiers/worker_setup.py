# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Make a local taskset module importable under a Verifiers taskset id."""

import importlib
import sys


# TODO: open a Verifiers issue, then delete this alias: Verifiers can't import a dotted module path
# as a taskset id, because `find_spec("verifiers.v1.tasksets.<dotted id>")` raises first:
# https://github.com/PrimeIntellect-ai/verifiers/blob/v0.4.0/verifiers/v1/utils/loaders.py#L76
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
