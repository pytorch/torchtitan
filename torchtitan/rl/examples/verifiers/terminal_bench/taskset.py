# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers' Harbor taskset, declared locally so the env-server worker imports us.

Nothing is overridden. The env-server worker is a spawned process, and it only
imports the module behind a locally registered taskset alias (see
``_local_taskset_module`` in ``torchtitan/rl/examples/verifiers/rollouter.py``).
Importing this module registers the harness alias in that process too, which
Verifiers needs to resolve the harness id.
"""

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import (
    HarborConfig,
    HarborEnv,
    HarborTask,
    HarborTaskset,
)

from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    register_harness_alias,
)

register_harness_alias()


class TerminalTasksetConfig(HarborConfig):
    """Harbor taskset config; a local class so its module is the worker's entry."""


class TerminalTaskset(HarborTaskset, vf.Taskset[HarborTask, TerminalTasksetConfig]):
    """Verifiers' Harbor taskset, unchanged."""

    config: TerminalTasksetConfig


# Verifiers discovers the Harbor environment from this taskset plugin.
__all__ = ["TerminalTaskset", "HarborEnv"]
