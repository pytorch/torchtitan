# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pluggable coding-agent harness for TorchTitan RL.

An external CLI agent (Claude Code first) runs unmodified inside a cloud sandbox
and is pointed at an on-box wire-format adapter that serves the trained policy and
captures every turn as on-policy training tokens. Three orthogonal axes, each its
own subpackage:

  - ``sandbox``: WHERE code runs -- provider-agnostic ``Sandbox`` contract +
    the Daytona backend + the Daytona fs-relay ``bridge``.
  - ``adapters``: HOW the model is served to the agent -- a token-capturing HTTP
    endpoint per wire format (``anthropic`` for Claude Code; add ``openai`` for
    Codex/OpenCode).
  - ``agents``: WHICH CLI agent + how to launch it (``claude_code``).

Adding a new CLI agent = a new ``agents`` runner (+ reuse/extend an ``adapters``
wire module); a new sandbox provider = a new ``sandbox`` backend. R2E (SWE) task
data + grading live in ``examples/swe_r2e``.
"""

from torchtitan.experiments.rl.harness.adapters import AnthropicAdapter, CapturedTurn
from torchtitan.experiments.rl.harness.agents import (
    apply_pre_commands,
    boot_agent_sandbox,
    git_diff,
    run_claude_code,
)
from torchtitan.experiments.rl.harness.sandbox import (
    DaytonaSandbox,
    make_sandbox,
    Sandbox,
)

__all__ = [
    "AnthropicAdapter",
    "CapturedTurn",
    "DaytonaSandbox",
    "Sandbox",
    "apply_pre_commands",
    "boot_agent_sandbox",
    "git_diff",
    "make_sandbox",
    "run_claude_code",
]
