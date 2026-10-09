# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor=={version}"]
# ///

"""Run Harbor's Terminus-2 agent in the rollout's sandbox.

Copied from Verifiers 0.3.1's Terminus-2 program (MIT License):
https://github.com/PrimeIntellect-ai/verifiers/blob/v0.3.1/verifiers/v1/harnesses/terminus_2/program.py
It adds the ``--enable-summarize`` and ``--no-interleaved-thinking`` flags of
https://github.com/PrimeIntellect-ai/verifiers/pull/2458, reads the prompts from
the ``--prompts`` file the harness writes, and nests the agent's shell so that an
``exit`` behaves as it does under ``harbor run``.

``TerminalBenchTerminusHarness`` in ``harness.py`` fills in the Harbor version
above and runs this file as a uv script, so it is never imported by TorchTitan.

TODO: delete this file together with ``harness.py``, as its TODO says.
"""

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class LocalEnvironment:
    default_user = None
    session_id = "verifiers"

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--enable-summarize", action="store_true")
    parser.add_argument("--no-interleaved-thinking", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    prompts = json.loads(Path(args.prompts).read_text())
    model, system_prompt, task = args.model, prompts["system_prompt"], prompts["task"]
    logs_dir = Path(os.environ["TMUX_TMPDIR"])
    logs_dir.mkdir(mode=0o700, exist_ok=True)
    EnvironmentPaths.agent_dir = PurePosixPath(logs_dir)

    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=model,
        api_base=args.base_url,
        llm_kwargs={"custom_llm_provider": "openai", "api_key": args.api_key},
        record_terminal_session=False,
        enable_summarize=args.enable_summarize,
        interleaved_thinking=not args.no_interleaved_thinking,
    )
    if system_prompt:
        call = agent._llm.call

        async def call_with_system_prompt(*args, message_history, **kwargs):
            return await call(
                *args,
                message_history=[
                    {"role": "system", "content": system_prompt},
                    *message_history,
                ],
                **kwargs,
            )

        agent._llm.call = call_with_system_prompt
    environment = LocalEnvironment()
    await agent.setup(environment)
    # Harbor's default recording (`asciinema rec`) nests the agent's shell, so an
    # `exit` ends only the recording. Recording is off here: nest a shell the same way.
    await agent._session.send_keys(keys=["bash", "Enter"], min_timeout_sec=1.0)
    await agent._session.send_keys(keys=["clear", "Enter"])
    await agent.run(task, environment, AgentContext())


if __name__ == "__main__":
    asyncio.run(main())
