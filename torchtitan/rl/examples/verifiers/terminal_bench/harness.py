# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers harness that runs Harbor's Terminus-2 agent in the rollout's sandbox.

Copied from Verifiers 0.3.1's Terminus-2 harness (MIT License), whose config
exposes only the Harbor version:
https://github.com/PrimeIntellect-ai/verifiers/blob/v0.3.1/verifiers/v1/harnesses/terminus_2/harness.py
This copy runs ``terminus_harness.py`` and adds the ``interleaved_thinking`` and
``enable_summarize`` options of
https://github.com/PrimeIntellect-ai/verifiers/pull/2458, with the same names
and defaults.

TODO: once a Verifiers release includes
https://github.com/PrimeIntellect-ai/verifiers/pull/2458, delete this file and
``terminus_harness.py`` and use Verifiers' ``Terminus2HarnessConfig`` in the
recipe.
"""

import logging
import sys
from pathlib import Path

from pydantic import Field
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "terminus_harness.py").read_text()
logger = logging.getLogger(__name__)


class TerminalBenchTerminusHarnessConfig(HarnessConfig):
    version: str = Field(default="0.21.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Harbor release to install, pinned for reproducibility."""

    interleaved_thinking: bool = True
    """Keep each turn's reasoning in the history sent back to the model.

    Without it, Verifiers no longer recognizes earlier turns in the next
    request, so each turn becomes a separate training sample with the full
    context.
    """

    enable_summarize: bool = False
    """Let Terminus-2 summarize its history when the context fills up.

    Verifiers ends the rollout at the context limit, so only Terminus-2's
    proactive summarization can run, and that needs the model's context limit,
    which this harness does not pass.
    """


class TerminalBenchTerminusHarness(Harness[TerminalBenchTerminusHarnessConfig]):
    """Verifiers' Terminus-2 harness, running ``terminus_harness.py``."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    def _program_source(self) -> str:
        return PROGRAM_SOURCE.replace("{version}", self.config.version)

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            self._program_source(), self.config.resolved_env
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
        env = {
            **self.config.resolved_env,
            "TMUX_TMPDIR": tmux_dir,
        }
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt or ''}",
            f"--task={prompt}",
        ]
        if self.config.enable_summarize:
            args.append("--enable-summarize")
        if not self.config.interleaved_thinking:
            args.append("--no-interleaved-thinking")
        try:
            program = await runtime.prepare_uv_script(
                self._program_source(), self.config.resolved_env
            )
            return await runtime.run_program([*program, *args], env)
        finally:
            # Harbor normally destroys its whole sandbox; this adapter borrows the
            # Verifiers runtime, so clean up Terminus's detached tmux server ourselves.
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
                # Runtime teardown is the final backstop; preserve the rollout's
                # result or original failure when this best-effort cleanup cannot run.
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
]
