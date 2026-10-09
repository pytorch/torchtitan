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
and defaults. To grade as ``harbor run`` does, it also stops tmux only after
scoring, passes the prompts in a file instead of argv, and nests the agent's shell
(in ``terminus_harness.py``).

TODO: once a Verifiers release includes
https://github.com/PrimeIntellect-ai/verifiers/pull/2458 and the Harbor-parity
changes listed here, delete this file and ``terminus_harness.py`` and use
Verifiers' ``Terminus2HarnessConfig`` in the recipe.
"""

import asyncio
import json
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
        # A file, not argv or env: the agent's `pkill -f` matches argv, and some sandbox
        # runtimes copy the env into a wrapper's argv. Outside tmux_dir, so the program
        # still creates that dir with mode 0700.
        prompts_path = f"{tmux_dir}.prompts.json"
        await runtime.write(
            prompts_path,
            json.dumps({"system_prompt": system_prompt or "", "task": prompt}).encode(),
        )
        env = {
            **self.config.resolved_env,
            "TMUX_TMPDIR": tmux_dir,
        }
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--prompts={prompts_path}",
        ]
        if self.config.enable_summarize:
            args.append("--enable-summarize")
        if not self.config.interleaved_thinking:
            args.append("--no-interleaved-thinking")
        program = await runtime.prepare_uv_script(
            self._program_source(), self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], env)

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        """Stop Terminus-2's tmux server. Verifiers calls this after scoring, so
        ``tests/test.sh`` still sees the agent's jobs (``python3 server.py &``)
        running, as under ``harbor run``."""
        # Harbor normally destroys its whole sandbox; this adapter borrows the
        # Verifiers runtime, so clean up Terminus-2's detached tmux server ourselves.
        # Bounded: nothing above cleanup times out, and a stopped tmux server never exits.
        await asyncio.wait_for(
            runtime.run(
                [
                    "sh",
                    "-c",
                    "tmux kill-server >/dev/null 2>&1 || true; "
                    'rm -rf "$TMUX_TMPDIR" "$TMUX_TMPDIR.prompts.json"',
                ],
                {"TMUX_TMPDIR": f"/tmp/vf-terminus-2-{trace.id}"},
            ),
            timeout=30,
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
