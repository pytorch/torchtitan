# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provider-agnostic sandbox/container resource for agent rollouts.

``base`` defines the ``Sandbox`` contract + ``make_sandbox`` factory; ``daytona``
is the backend; ``bridge`` is the Daytona-specific HTTP file-relay that lets an
in-sandbox agent reach the on-box adapter.
"""

from torchtitan.experiments.rl.harness.sandbox.base import (
    ExecResult,
    FileContent,
    make_sandbox,
    Sandbox,
)
from torchtitan.experiments.rl.harness.sandbox.bridge import DaytonaBridge, start_bridge
from torchtitan.experiments.rl.harness.sandbox.daytona import DaytonaSandbox

__all__ = [
    "DaytonaBridge",
    "DaytonaSandbox",
    "ExecResult",
    "FileContent",
    "Sandbox",
    "make_sandbox",
    "start_bridge",
]
