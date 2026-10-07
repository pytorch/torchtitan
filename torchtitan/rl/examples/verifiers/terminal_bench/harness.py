# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure Verifiers' Terminus-2 program to send reasoning back to the policy."""

import logging
import sys

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.terminus_2.harness import (
    PROGRAM_SOURCE,
    Terminus2Harness,
    Terminus2HarnessConfig,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# Verifiers 0.3.1 runs a fixed program that constructs Harbor's Terminus-2
# agent, and its harness config exposes only the Harbor version. We insert
# this constructor argument after _PROGRAM_MARKER:
# - interleaved_thinking=True: Harbor's default (False) drops each turn's
#   reasoning from the history it sends back to the model. Verifiers then no
#   longer recognizes earlier turns in the next request, so each turn becomes
#   a separate training sample with the full context, not one per rollout.
# Verifiers is pinned, and the marker must occur exactly once, so an upgrade
# that changes the program fails loudly instead of dropping the option.
# TODO: drop this patch once Verifiers' Terminus-2 config accepts agent arguments.
_PROGRAM_MARKER = "        record_terminal_session=False,\n"


def terminus_program_source(config: Terminus2HarnessConfig) -> str:
    """Return the Verifiers Terminus-2 program with interleaved thinking enabled."""
    if PROGRAM_SOURCE.count(_PROGRAM_MARKER) != 1:
        raise RuntimeError("Verifiers Terminus-2 program constructor has changed")
    source = PROGRAM_SOURCE.replace(
        _PROGRAM_MARKER, _PROGRAM_MARKER + "        interleaved_thinking=True,\n"
    )
    return source.replace("{version}", config.version)


class TerminalBenchTerminusHarness(Terminus2Harness):
    """Verifiers' Terminus-2 harness, running the program above."""

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            terminus_program_source(self.config), self.config.resolved_env
        )

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("Terminus 2 does not support disabling tools")
        system_prompt, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("Terminus 2 requires a task prompt")
        tmux_dir = f"/tmp/vf-terminus-2-{trace.id}"
        environment = {**self.config.resolved_env, "TMUX_TMPDIR": tmux_dir}
        arguments = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            f"--task={prompt}",
        ]
        try:
            program = await runtime.prepare_uv_script(
                terminus_program_source(self.config), self.config.resolved_env
            )
            return await runtime.run_program([*program, *arguments], environment)
        finally:
            try:
                await runtime.run(
                    [
                        "sh",
                        "-c",
                        'tmux kill-server >/dev/null 2>&1 || true; rm -rf "$TMUX_TMPDIR"',
                    ],
                    {"TMUX_TMPDIR": tmux_dir},
                )
            except Exception:
                logger.warning(
                    "failed to clean up Terminus 2 tmux server", exc_info=True
                )


def register_harness_alias() -> str:
    """Make the local harness importable in spawned Verifiers worker processes."""
    module = sys.modules[__name__]
    alias = __name__.replace(".", "_").lower()
    existing = sys.modules.get(alias)
    if existing is not None and existing is not module:
        raise ValueError(f"harness alias {alias!r} is already registered")
    sys.modules[alias] = module
    return alias


__all__ = ["TerminalBenchTerminusHarness"]
