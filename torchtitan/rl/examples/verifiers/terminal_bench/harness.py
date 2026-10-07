# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure Verifiers' Terminus-2 program with Harbor agent options."""

import logging
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
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
# agent, and its harness config exposes only the Harbor version. We insert the
# constructor arguments from TerminalBenchTerminusHarnessConfig after
# _PROGRAM_MARKER. Verifiers is pinned, and the marker must occur exactly once,
# so an upgrade that changes the program fails loudly instead of dropping them.
# TODO: drop this patch once Verifiers' Terminus-2 config accepts agent arguments.
_PROGRAM_MARKER = "        record_terminal_session=False,\n"


class TerminusModelInfo(BaseModel):
    """Model limits that Terminus-2 registers with LiteLLM under the served name."""

    model_config = ConfigDict(extra="forbid")

    max_input_tokens: int = Field(gt=0)
    """Model context length, i.e. the generator's maximum sequence length."""

    max_output_tokens: int = Field(gt=0)
    """Per-turn generation cap, i.e. the generator's sampling ``max_tokens``."""


class TerminalBenchTerminusHarnessConfig(Terminus2HarnessConfig):
    """Terminus-2 agent arguments that Verifiers' harness config does not expose.

    Each field is passed to Harbor's ``Terminus2`` under the same name. Defaults
    match Harbor's, except ``interleaved_thinking``.
    """

    parser_name: Literal["json", "xml"] = "json"
    """Terminus-2's prompt and action format.

    Only the XML parser can salvage an action from a reply cut off at the
    per-turn token limit.
    """

    enable_summarize: bool = True
    """Summarize the history once fewer than 8,000 tokens of the context remain.

    Each summary makes three extra policy calls (summary, questions, answers),
    which are trained under the task reward.
    """

    max_turns: int | None = Field(default=None, gt=0)
    """Terminus-2's own turn limit, at which it stops and the task is graded.

    ``None`` leaves only Verifiers' ``AgentConfig.max_turns``, which refuses
    the next model call instead.
    """

    model_info: TerminusModelInfo | None = None
    """Context and output limits of the served model.

    LiteLLM looks up limits by model name and does not know the served name.
    With ``None`` it reports a 1,000,000-token context, so summarization never
    triggers before the generator's cap.
    """

    interleaved_thinking: bool = True
    """Keep each turn's reasoning in the history sent back to the model.

    Harbor's default (False) drops it. Verifiers then no longer recognizes
    earlier turns in the next request, so each turn becomes a separate training
    sample with the full context, not one per rollout.
    """


def terminus_program_source(config: TerminalBenchTerminusHarnessConfig) -> str:
    """Return the Verifiers Terminus-2 program with ``config``'s options inserted."""
    if PROGRAM_SOURCE.count(_PROGRAM_MARKER) != 1:
        raise RuntimeError("Verifiers Terminus-2 program constructor has changed")
    options = {
        "parser_name": config.parser_name,
        "enable_summarize": config.enable_summarize,
        "max_turns": config.max_turns,
        "suppress_max_turns_warning": True,
        "model_info": config.model_info and config.model_info.model_dump(),
        "interleaved_thinking": config.interleaved_thinking,
    }
    arguments = "".join(
        f"        {name}={value!r},\n" for name, value in options.items()
    )
    source = PROGRAM_SOURCE.replace(_PROGRAM_MARKER, _PROGRAM_MARKER + arguments)
    return source.replace("{version}", config.version)


class TerminalBenchTerminusHarness(
    Terminus2Harness, Harness[TerminalBenchTerminusHarnessConfig]
):
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


__all__ = [
    "TerminalBenchTerminusHarness",
    "TerminalBenchTerminusHarnessConfig",
    "TerminusModelInfo",
]
