# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import numpy as np
import pytest
from renderers import MultiModalData, ParsedResponse, PlaceholderRange, RenderedTokens
from torchtitan.components.renderer import hash_image_file
from torchtitan.rl.rollout import PromptImage, RolloutStatus, token_env

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

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        self.calls["render"] = {"tools": tools}
        return RenderedTokens(token_ids=[1, 2, 3])

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
    assert renderer.calls["render"]["tools"] == _TOOLS
    assert renderer.calls["parse_response"] == {"token_ids": [7, 8], "tools": _TOOLS}
    assert renderer.calls["bridge_to_next_turn"]["tools"] == _TOOLS


class _ScriptedRenderer(_RecordingRenderer):
    """Renders the given prompts in order and never bridges."""

    def __init__(self, prompts: list[list[int]]) -> None:
        super().__init__()
        self.prompts = prompts

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        return RenderedTokens(token_ids=self.prompts.pop(0))


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
    # A text renderer's prompts carry no images.
    assert all(output.next_prompt.images == () for output in outputs)


def _image_data(*images: tuple[str, tuple[int, int, int], int]) -> MultiModalData:
    """A renderer's `multi_modal_data` for ``(key, grid_thw, offset)`` images with merge size 2."""
    return MultiModalData(
        mm_hashes={"image": [key for key, _, _ in images]},
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=offset, length=int(np.prod(grid)) // 4)
                for _, grid, offset in images
            ]
        },
        mm_items={
            "image": [
                {"pixel_values": None, "image_grid_thw": np.array([grid])}
                for _, grid, _ in images
            ]
        },
    )


class _ImageRenderer(_RecordingRenderer):
    """A multimodal renderer: full renders and bridges place images."""

    def __init__(self) -> None:
        super().__init__()
        self.bridge_previous: list = []

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        # 1 text token, then a (1, 2, 4)-patch image -> 2 placeholders, then 1 text token.
        return RenderedTokens(
            token_ids=[1, 9, 9, 2],
            multi_modal_data=_image_data(("img-a", (1, 2, 4), 1)),
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
        previous_multi_modal_data=None,
    ):
        self.bridge_previous.append(previous_multi_modal_data)
        prefix = list(previous_prompt_ids) + list(previous_completion_ids)
        return RenderedTokens(
            token_ids=prefix + [3, 9, 9, 9, 9],
            multi_modal_data=_image_data(
                ("img-a", (1, 2, 4), 1), ("img-b", (1, 4, 4), len(prefix) + 1)
            ),
        )


class _ImageMessageEnv:
    """The user shows a.png; the tool's reply shows b.png."""

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": "a.png"},
                        {"type": "text", "text": "find bob"},
                    ],
                }
            ],
            tools=_TOOLS,
        )

    async def step(self, completion_message) -> MessageEnvStepOutput:
        return MessageEnvStepOutput(
            env_messages=[
                {
                    "role": "tool",
                    "name": "search",
                    "content": [{"type": "image", "path": "b.png"}],
                }
            ]
        )

    async def close(self) -> None:
        pass


def test_env_carries_rendered_images_across_turns(monkeypatch) -> None:
    files = {("a.png", "img-a"): b"file a", ("b.png", "img-b"): b"file b"}
    monkeypatch.setattr(
        token_env,
        "_read_image_file",
        lambda path, image_hash: files[(path, image_hash)],
    )
    renderer = _ImageRenderer()
    env = TokenEnv.Config().build(message_env=_ImageMessageEnv(), renderer=renderer)

    async def run():
        first = await env.init()
        second = await env.step(
            Completion(
                min_policy_version=0,
                max_policy_version=0,
                request_id="r0",
                token_ids=[7, 8],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
            )
        )
        return first, second

    first, second = asyncio.run(run())
    image_a = PromptImage(
        hash_value="img-a", source_path="a.png", grid_thw=(1, 2, 4), data=b"file a"
    )
    image_b = PromptImage(
        hash_value="img-b", source_path="b.png", grid_thw=(1, 4, 4), data=b"file b"
    )
    assert first.next_prompt.images == (image_a,)
    assert first.next_prompt.image_placeholders == (
        PlaceholderRange(offset=1, length=2),
    )
    # The bridge saw the first prompt's images and extended them.
    assert renderer.bridge_previous[0].mm_hashes == {"image": ["img-a"]}
    assert second.next_prompt.images == (image_a, image_b)
    assert second.next_prompt.image_placeholders == (
        PlaceholderRange(offset=1, length=2),
        PlaceholderRange(offset=7, length=4),
    )
    # The turn stores only the image after the shared prefix.
    assert second.next_prompt_delta_images == (image_b,)
    assert second.next_prompt_delta_image_placeholders == (
        PlaceholderRange(offset=7, length=4),
    )
    # Equality skips the bytes (hash_value is their hash): check them too.
    assert [image.data for image in second.next_prompt.images] == [b"file a", b"file b"]


def _init_prompt(multi_modal_data: MultiModalData | None, messages: list) -> None:
    """Run `TokenEnv.init` on ``messages`` with a renderer that places ``multi_modal_data``."""

    class _Renderer(_RecordingRenderer):
        def render(self, messages, *, tools=None, add_generation_prompt=False):
            return RenderedTokens(
                token_ids=[1, 9, 9, 2], multi_modal_data=multi_modal_data
            )

    class _MessageEnv(_ToolMessageEnv):
        async def init(self) -> MessageEnvInitOutput:
            return MessageEnvInitOutput(init_prompt_messages=messages)

    env = TokenEnv.Config().build(message_env=_MessageEnv(), renderer=_Renderer())
    asyncio.run(env.init())


def _image_message(*paths: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "image", "path": path} for path in paths],
    }


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_env_rejects_other_modalities(modality: str) -> None:
    data = MultiModalData(mm_hashes={modality: ["v"]})
    with pytest.raises(NotImplementedError, match="images only"):
        _init_prompt(data, [{"role": "user", "content": "hi"}])


def test_env_rejects_pixels_from_a_pixel_renderer() -> None:
    data = _image_data(("pil-hash", (1, 2, 4), 1))
    data.mm_items["image"][0]["pixel_values"] = np.zeros((8, 1536), dtype=np.float32)
    with pytest.raises(ValueError, match="image-reference renderer"):
        _init_prompt(data, [_image_message("a.png")])


def test_env_rejects_a_message_image_count_mismatch() -> None:
    data = _image_data(("img-a", (1, 2, 4), 1))
    with pytest.raises(ValueError, match="1 images but its messages have 2"):
        _init_prompt(data, [_image_message("a.png", "b.png")])


def test_env_rejects_image_parts_rendered_without_images() -> None:
    # E.g. Muse Spark renders an image part as one <|patch|> token with no image data.
    with pytest.raises(ValueError, match="1 image parts but the renderer placed no"):
        _init_prompt(None, [_image_message("a.png")])


def test_read_image_file_rejects_a_changed_file(tmp_path) -> None:
    path, changed = tmp_path / "a.png", tmp_path / "b.png"
    path.write_bytes(b"file a")
    changed.write_bytes(b"file b")
    image_hash = hash_image_file(b"file a")

    assert token_env._read_image_file(str(path), image_hash) == b"file a"
    with pytest.raises(ValueError, match="changed after it was rendered"):
        token_env._read_image_file(str(changed), image_hash)
