# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure Verifiers' Terminus-2 program for the XML terminal-agent policy."""

import logging
import sys

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.terminus_2.harness import (
    PROGRAM_SOURCE,
    Terminus2Harness,
    Terminus2HarnessConfig,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

NUM_AGENT_TURNS = 120
_PROGRAM_MARKER = "        record_terminal_session=False,\n"
_TERMINUS_OPTIONS = (
    '        parser_name="xml",\n'
    "        enable_summarize=False,\n"
    f"        max_turns={NUM_AGENT_TURNS},\n"
    "        suppress_max_turns_warning=True,\n"
    '        model_info={"max_input_tokens": 63488, "max_output_tokens": 16384},\n'
)


def terminus_program_source() -> str:
    """Keep the Verifiers program and change only the policy's required knobs."""
    if PROGRAM_SOURCE.count(_PROGRAM_MARKER) != 1:
        raise RuntimeError("Verifiers Terminus-2 program constructor has changed")
    return PROGRAM_SOURCE.replace(_PROGRAM_MARKER, _PROGRAM_MARKER + _TERMINUS_OPTIONS)


class TerminalBenchTerminusHarnessConfig(Terminus2HarnessConfig):
    """Expose the XML Terminus-2 harness as a Verifiers plugin."""


class TerminalBenchTerminusHarness(
    Terminus2Harness, Harness[TerminalBenchTerminusHarnessConfig]
):
    """Use Harbor's XML scaffold without copying its agent implementation."""

    def __init__(self, config: TerminalBenchTerminusHarnessConfig) -> None:
        super().__init__(config)

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            terminus_program_source().replace("{version}", self.config.version),
            self.config.resolved_env,
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
            source = terminus_program_source().replace("{version}", self.config.version)
            program = await runtime.prepare_uv_script(source, self.config.resolved_env)
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
