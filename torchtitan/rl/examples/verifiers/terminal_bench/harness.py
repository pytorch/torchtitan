# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure Verifiers' Terminus-2 program for the XML terminal-agent policy."""

import logging
import sys

from pydantic import Field
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

# Verifiers 0.3.1 runs a fixed program that constructs Harbor's Terminus-2
# agent, and its harness config exposes only the Harbor version. We insert
# these constructor arguments after _PROGRAM_MARKER:
# - parser_name="xml": Terminus-2's XML prompt and action format instead of the
#   JSON default. Only the XML parser can salvage an action from a reply cut off
#   at the per-turn token limit.
# - enable_summarize=False: summarization (context compaction) makes three extra
#   policy calls (summary, questions, answers) that would be trained under the
#   task reward. TitanRL splits compacted rollouts into separate samples, but
#   that path is untested.
# - max_turns: the agent turn limit from the recipe, also enforced by Verifiers.
# - model_info: Terminus-2 calls the model through LiteLLM, a client library
#   that looks up context limits by model name. It does not know the served
#   name, so the limits come from the recipe.
# Verifiers is pinned, and the marker must occur exactly once, so an upgrade
# that changes the program fails loudly instead of dropping these options.
# TODO: drop this patch once Verifiers' Terminus-2 config accepts agent arguments.
_PROGRAM_MARKER = "        record_terminal_session=False,\n"


class TerminalBenchTerminusHarnessConfig(Terminus2HarnessConfig):
    """Expose the XML Terminus-2 harness as a Verifiers plugin."""

    max_input_tokens: int = Field(gt=0)
    """Model context length, i.e. the generator's maximum sequence length."""

    max_output_tokens: int = Field(gt=0)
    """Per-turn generation cap, i.e. the generator's sampling ``max_tokens``."""

    max_turns: int = Field(gt=0)
    """Agent turn limit, the same value as Verifiers' ``AgentConfig.max_turns``."""


def terminus_program_source(config: TerminalBenchTerminusHarnessConfig) -> str:
    """Return the Verifiers Terminus-2 program with this policy's options inserted."""
    if PROGRAM_SOURCE.count(_PROGRAM_MARKER) != 1:
        raise RuntimeError("Verifiers Terminus-2 program constructor has changed")
    model_info = {
        "max_input_tokens": config.max_input_tokens,
        "max_output_tokens": config.max_output_tokens,
    }
    options = (
        '        parser_name="xml",\n'
        "        enable_summarize=False,\n"
        f"        max_turns={config.max_turns},\n"
        "        suppress_max_turns_warning=True,\n"
        f"        model_info={model_info!r},\n"
    )
    source = PROGRAM_SOURCE.replace(_PROGRAM_MARKER, _PROGRAM_MARKER + options)
    return source.replace("{version}", config.version)


class TerminalBenchTerminusHarness(
    Terminus2Harness, Harness[TerminalBenchTerminusHarnessConfig]
):
    """Use Harbor's XML scaffold without copying its agent implementation."""

    def __init__(self, config: TerminalBenchTerminusHarnessConfig) -> None:
        super().__init__(config)

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
