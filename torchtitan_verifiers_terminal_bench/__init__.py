# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers plugin of the Terminal-Bench example.

See "Taskset ids" in torchtitan/rl/examples/verifiers/README.md.
"""

from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarness,
)
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import TerminalTaskset

# Verifiers 0.4.0 runs a taskset with the Env its package exports.
# TODO: drop this export with the Verifiers 0.4.1 pin.
from verifiers.v1.tasksets.harbor import HarborEnv

__all__ = ["HarborEnv", "TerminalBenchTerminusHarness", "TerminalTaskset"]
