# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from renderers import ParsedResponse
from torchtitan.rl.rollout import RolloutStatus

from torchtitan.rl.rollout.message_env import MessageEnvInitOutput, MessageEnvStepOutput
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.rollout.types import Completion

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


class _RecordingRenderer:
    """Records the kwargs of each renderer call the env makes."""

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        self.calls["render_ids"] = {"tools": tools}
        return [1, 2, 3]

    def parse_response(self, token_ids, *, tools=None):
        self.calls["parse_response"] = {"token_ids": token_ids, "tools": tools}
        return ParsedResponse(content="answer", reasoning_content=None, tool_calls=[])

    def bridge_to_next_turn(
        self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None
    ):
        self.calls["bridge_to_next_turn"] = {"tools": tools}
        return None

    def get_stop_token_ids(self):
        return []


class _ToolMessageEnv:
    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": "find bob"}],
            tools=_TOOLS,
        )

    async def step(self, completion_message) -> MessageEnvStepOutput:
        return MessageEnvStepOutput(
            env_messages=[{"role": "tool", "name": "search", "content": "bob: found"}]
        )

    async def close(self) -> None:
        pass


def test_env_passes_tools_to_render_parse_and_bridge() -> None:
    # Tool schemas are part of the chat template, and XML-style tool parsers need them
    # to type the arguments; every renderer call must see the same list.
    renderer = _RecordingRenderer()
    env = TokenEnv.Config().build(message_env=_ToolMessageEnv(), renderer=renderer)

    async def run():
        await env.init()
        return await env.step(
            Completion(
                min_policy_version=0,
                max_policy_version=0,
                request_id="r0",
                token_ids=[7, 8],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
            )
        )

    env_output = asyncio.run(run())
    assert env_output.status == RolloutStatus.ONGOING
    assert renderer.calls["render_ids"]["tools"] == _TOOLS
    assert renderer.calls["parse_response"] == {"token_ids": [7, 8], "tools": _TOOLS}
    assert renderer.calls["bridge_to_next_turn"]["tools"] == _TOOLS


class _ScriptedRenderer(_RecordingRenderer):
    """Renders the given prompts in order and never bridges."""

    def __init__(self, prompts: list[list[int]]) -> None:
        super().__init__()
        self.prompts = prompts

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return self.prompts.pop(0)


def test_env_returns_each_next_prompt_as_a_delta() -> None:
    renderer = _ScriptedRenderer(prompts=[[1, 2, 3], [1, 2, 3, 4, 7], [1, 2, 3, 4, 9]])
    env = TokenEnv.Config().build(message_env=_ToolMessageEnv(), renderer=renderer)

    def completion(token_ids: list[int]) -> Completion:
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id="r0",
            token_ids=token_ids,
            token_logprobs=[-0.1] * len(token_ids),
            finish_reason="stop",
        )

    async def run():
        return [
            await env.init(),
            await env.step(completion([4])),
            await env.step(completion([5])),
        ]

    outputs = asyncio.run(run())
    assert [
        (output.next_prompt_prefix_len, output.next_prompt_delta_token_ids.tolist())
        for output in outputs
    ] == [
        (0, [1, 2, 3]),  # first prompt: nothing to reuse
        (4, [7]),  # extends [1, 2, 3] + completion [4]
        (
            4,
            [9],
        ),  # edited history: shares [1, 2, 3, 4] with [1, 2, 3, 4, 7] + completion [5]
    ]
