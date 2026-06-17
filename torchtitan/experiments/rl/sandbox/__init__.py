# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task-agnostic sandbox layer for RL rollouts (shared across envs).

The ``Sandbox`` contract + ``SandboxFactory`` backend seam live here so any
"run commands in an isolated environment" task (SWE, terminal-style shells, ...)
reuses them. ``base`` holds the contract; ``docker`` holds the podman/docker
backend.
"""

from torchtitan.experiments.rl.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxFactory,
    SweProvisionError,
    SweProvisionUserError,
)
from torchtitan.experiments.rl.sandbox.docker import (
    DockerSandbox,
    DockerSandboxFactory,
    reap_orphans,
    SANDBOX_LABEL,
)

__all__ = [
    "DockerSandbox",
    "DockerSandboxFactory",
    "ExecResult",
    "SANDBOX_LABEL",
    "Sandbox",
    "SandboxFactory",
    "SweProvisionError",
    "SweProvisionUserError",
    "reap_orphans",
]
