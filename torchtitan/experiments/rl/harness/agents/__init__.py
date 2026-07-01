# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-CLI-agent runners: install the agent binary in a sandbox and launch it
pointed at a wire-format adapter. ``claude_code`` runs Claude Code against the
Anthropic adapter; add a sibling module (e.g. ``codex``) for another CLI agent."""

from torchtitan.experiments.rl.harness.agents.claude_code import (
    apply_pre_commands,
    boot_agent_sandbox,
    git_diff,
    run_claude_code,
)
from torchtitan.experiments.rl.harness.agents.host_loop import run_host_loop

__all__ = [
    "apply_pre_commands",
    "boot_agent_sandbox",
    "git_diff",
    "run_claude_code",
    "run_host_loop",
]
