# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass, field

import numpy as np
from renderers import Message, MultiModalData, RenderedTokens, Renderer, ToolSpec

from torchtitan.components.renderer import image_content_key
from torchtitan.config import Configurable
from torchtitan.rl.rollout.message_env import MessageEnv
from torchtitan.rl.rollout.types import (
    Completion,
    Prompt,
    PromptImage,
    RolloutStatus,
    split_prompt,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=64)
def read_image_file(path: str, key: str) -> bytes:
    """Return the bytes of the image file at ``path``, which must hash to ``key``.

    The image-reference renderer hashes the file into the key (`image_content_key`) but
    keeps only the key and the grid (the renderers library carries nothing else from
    `_process_image`), so `TokenEnv` reads the file again for the prompt. Cached: the samples of a GRPO group
    render together in one process, so they share one bytes object, and a message that
    holds several of them pickles the bytes once.
    """
    with open(path, "rb") as f:
        data = f.read()
    if image_content_key(data) != key:
        raise ValueError(
            f"{path!r} changed after it was rendered: its content key is "
            f"{image_content_key(data)}, the prompt has {key}."
        )
    return data


@dataclass(kw_only=True, slots=True)
class TokenEnvOutput:
    """What `TokenEnv` produced this turn. Output only — it does not carry the prompt that was fed
    in or generator metadata (logprobs, policy_version). The rollout loop folds it together
    with those inputs into a durable `RolloutTurn`."""

    next_prompt: Prompt | None
    """Tokens and images for the next `generate` call. `None` once no further prompt is
    rendered (COMPLETED, or a completion-level error)."""

    next_prompt_prefix_len: int = 0
    """Length of the longest prefix shared with the previous prompt + completion; 0 on the first prompt.
    Set it with `split_prompt` on every later prompt, from the tokens in `next_prompt`."""

    next_prompt_messages: list[Message] | None = None  # [num_prompt_messages] or None
    """Message form of `next_prompt`; `None` whenever that is `None`."""

    status: RolloutStatus
    """`ONGOING` while the rollout runs; a terminal `RolloutStatus` otherwise."""

    completion_message: Message | None = None
    """This turn's completion decoded from the generator's tokens into a message by the renderer
    (this is the TokenEnv's parse, not the raw generator output). `None` on init and on parse failure."""

    env_messages: list[Message] = field(default_factory=list)  # [num_env_messages]
    """The env's reply on this step (tool / user). Empty on init and on terminals
    where the env was not stepped."""

    env_rewards: dict[str, float] = field(default_factory=dict)
    """Per-step reward signals from the env; empty when the env was not stepped. This
    can be used by the rubric to compute the final rollout reward."""

    @property
    def next_prompt_delta_token_ids(self) -> np.ndarray:
        """The next prompt after its first `next_prompt_prefix_len` tokens."""
        token_ids = self.next_prompt.token_ids if self.next_prompt is not None else []
        delta = token_ids[self.next_prompt_prefix_len :]
        return np.asarray(delta, dtype=np.int32)


class TokenEnv(Configurable):
    """Token-space env used to wrap a `MessageEnv` and drive it in tokens.

    In a rollout, the input and output of a generator are in tokens (Tokens-In-Tokens-Out).
    However, the MessageEnv is in messages (Messages-In-Messages-Out).

    Some process is necessary to:
    a. Decode the generator's completion tokens into messages;
    b. Call the MessageEnv;
    c. Encode the env response back into tokens for the next turn;

    This wrapper fills this role, using a renderer to convert between the generator and MessageEnv.

    Beyond encoding/decoding, several checks are necessary, e.g. prompt too long, too many
    turns, parse errors, timeouts. This wrapper also takes care of that, so we can keep the
    rollout loop clean and simple.

    If users have extra or different logic, they can wrap their MessageEnv with another class instead.

    Args:
        config: `TokenEnv.Config`.
        message_env: the user's `MessageEnv` subclass instance.
        renderer: a `renderers.Renderer` that converts messages <-> token ids.

    Example:

        env = TokenEnv.Config().build(message_env=MyMessageEnv(...), renderer=renderer)
        env_output = await env.init()
        while not env_output.status.is_terminal():
            completion = await generate_fn(env_output.next_prompt, ...)
            env_output = await env.step(completion)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Limits enforced by the wrapper"""

        max_rollout_tokens: int | None = None
        """Max prompt tokens before the next turn; multi-turn envs use it to stop before the
        prompt outgrows context (`None` disables)."""

        # TODO: its unclear if timeout should be on this layer or handled by the messageEnv
        step_timeout_s: float | None = 1800.0
        """Wall-clock timeout for one `MessageEnv.step` call."""

        max_num_turns: int | None = None
        """Cap on assistant turns. If the env would continue past this many turns,
        the rollout ends as `TRUNCATED_MAX_TURNS`; `None` disables the check."""

    def __init__(
        self,
        config: Config,
        *,
        message_env: MessageEnv,
        renderer: Renderer,
    ) -> None:
        self._message_env = message_env
        self._renderer = renderer
        self._config = config
        self._tools: list[ToolSpec] | None = None
        self._messages: list[Message] = []
        self._last_prompt_token_ids: list[int] = []
        # The renderer's record of the last prompt's images, which its bridge extends;
        # None for text prompts.
        self._last_multi_modal_data: MultiModalData | None = None
        self._num_turns = 0

    async def init(self) -> TokenEnvOutput:
        """Render the initial conversation into the first generator prompt."""
        env_init_output = await self._message_env.init()

        # Copy our running conversation (messages) so we avoid mutating previous states
        self._messages = list(env_init_output.init_prompt_messages)

        # Render messages into tokens. `render`, not `render_ids`: its `multi_modal_data`
        # places the prompt's images (None for text).
        self._tools = env_init_output.tools if env_init_output.tools else None
        rendered = await asyncio.to_thread(
            self._renderer.render,
            messages=self._messages,
            tools=self._tools,
            add_generation_prompt=True,
        )

        # Carry the prompt either way: ONGOING if it fits, else
        # TRUNCATED_PROMPT_TOO_LONG so the over-budget prompt stays debuggable.
        prompt_too_long = self._is_prompt_too_long(prompt_len=len(rendered.token_ids))
        if not prompt_too_long:
            self._last_prompt_token_ids = rendered.token_ids
            self._last_multi_modal_data = rendered.multi_modal_data
        return TokenEnvOutput(
            next_prompt=self._to_prompt(rendered),
            next_prompt_messages=self._messages,
            status=(
                RolloutStatus.TRUNCATED_PROMPT_TOO_LONG
                if prompt_too_long
                else RolloutStatus.ONGOING
            ),
        )

    async def step(self, completion: Completion) -> TokenEnvOutput:
        """Advance the env by one sampled completion from the generator.

        Args:
            completion: Generator output for the current prompt.

        Returns:
            `TokenEnvOutput` for the next generator call, or a terminal turn
            when the rollout completes, truncates, or errors.
        """
        self._num_turns += 1

        # Parse first, so a truncated / aborted response still carries its message
        try:
            parsed = await asyncio.to_thread(
                self._renderer.parse_response,
                token_ids=completion.token_ids,
                tools=self._tools,
            )
        except Exception:
            logger.exception(
                "parse_response failed (finish_reason=%s, %d tokens); -> ERROR_PARSE",
                completion.finish_reason,
                len(completion.token_ids),
            )
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.ERROR_PARSE,
            )

        completion_message: Message = {
            "role": "assistant",
            "content": parsed.content,
        }
        if parsed.reasoning_content:
            completion_message["reasoning_content"] = parsed.reasoning_content
        if parsed.tool_calls:
            completion_message["tool_calls"] = parsed.tool_calls

        # Truncated / aborted: the response is final and partial. Keep it for
        # partial-reward grading and debugging; don't step the env on it.
        # TODO: check if we should step the env on an incomplete message
        if completion.finish_reason == "length":
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.TRUNCATED_LENGTH,
                completion_message=completion_message,
            )
        if completion.finish_reason == "abort":
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.ERROR_ABORT,
                completion_message=completion_message,
            )

        # Apply the user's env step under a timeout
        timeout = self._config.step_timeout_s
        try:
            if timeout is None:
                step_output = await self._message_env.step(completion_message)
            else:
                step_output = await asyncio.wait_for(
                    self._message_env.step(completion_message), timeout=timeout
                )
        except TimeoutError:
            logger.warning("step timed out after %ss; -> ERROR_TIMEOUT", timeout)
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.ERROR_TIMEOUT,
                completion_message=completion_message,
            )

        # TODO(history-edit): We hard-code the logic to only append new messages.
        # This may not satisfy all uses cases, such as compacting the history.
        # Update this when such cases arise.

        # Create a new list to avoid mutating previous states
        self._messages = (
            self._messages + [completion_message] + step_output.env_messages
        )

        if step_output.done:
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.COMPLETED,
                completion_message=completion_message,
                env_messages=step_output.env_messages,
                env_rewards=step_output.env_rewards,
            )

        # Out of turn budget: the env would continue, but we stop here.
        if (
            self._config.max_num_turns is not None
            and self._num_turns >= self._config.max_num_turns
        ):
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.TRUNCATED_MAX_TURNS,
                completion_message=completion_message,
                env_messages=step_output.env_messages,
                env_rewards=step_output.env_rewards,
            )

        # Prepare the next prompt; full re-render if the renderer can't bridge.
        # `tools` is passed because tool schemas are part of the chat template, so
        # the bridged tokens must match what a full re-render (also tools-aware) produces.
        bridge_kwargs = {}
        if self._last_multi_modal_data is not None:
            # The bridge must know the earlier images to number and place new ones. Only
            # multimodal renderers' bridges take them, and only they render images.
            bridge_kwargs["previous_multi_modal_data"] = self._last_multi_modal_data
        rendered = await asyncio.to_thread(
            self._renderer.bridge_to_next_turn,
            previous_prompt_ids=self._last_prompt_token_ids,
            previous_completion_ids=completion.token_ids,
            new_messages=step_output.env_messages,
            tools=self._tools,
            **bridge_kwargs,
        )
        if rendered is None:
            rendered = await asyncio.to_thread(
                self._renderer.render,
                messages=self._messages,
                tools=self._tools,
                add_generation_prompt=True,
            )
        next_prompt_token_ids = rendered.token_ids

        # Terminal if the next prompt is over budget
        if self._is_prompt_too_long(prompt_len=len(next_prompt_token_ids)):
            return TokenEnvOutput(
                next_prompt=None,
                next_prompt_messages=None,
                status=RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
                completion_message=completion_message,
                env_messages=step_output.env_messages,
                env_rewards=step_output.env_rewards,
            )

        # RolloutTurn stores the next prompt as a delta on this prompt + completion
        next_prompt_prefix_len, _ = split_prompt(
            prompt_token_ids=next_prompt_token_ids,
            previous_token_ids=self._last_prompt_token_ids + completion.token_ids,
        )
        self._last_prompt_token_ids = next_prompt_token_ids
        self._last_multi_modal_data = rendered.multi_modal_data
        return TokenEnvOutput(
            next_prompt=self._to_prompt(rendered),
            next_prompt_prefix_len=next_prompt_prefix_len,
            next_prompt_messages=self._messages,
            status=RolloutStatus.ONGOING,
            completion_message=completion_message,
            env_messages=step_output.env_messages,
            env_rewards=step_output.env_rewards,
        )

    async def close(self) -> None:
        await self._message_env.close()

    def _is_prompt_too_long(self, *, prompt_len: int) -> bool:
        cap = self._config.max_rollout_tokens
        return cap is not None and prompt_len >= cap

    def _to_prompt(self, rendered: RenderedTokens) -> Prompt:
        """The generator's input for the rendered conversation: its tokens and images.

        Images need an image-reference renderer (e.g. ``Qwen35ImageRefRenderer``): each
        image's mm_hash is its content key and its ``pixel_values`` is None. Raises
        otherwise, and on non-image modalities. Renderers emit images in message order,
        so the n-th image part of the conversation names the n-th image's file, read
        here (`read_image_file`).
        """
        multi_modal_data = rendered.multi_modal_data
        if multi_modal_data is None:
            return Prompt(token_ids=rendered.token_ids)
        modalities = {
            modality
            for items in (
                multi_modal_data.mm_hashes,
                multi_modal_data.mm_placeholders,
                multi_modal_data.mm_items,
            )
            for modality, values in items.items()
            if values
        }
        if modalities - {"image"}:
            # TODO: video prompts need video grids in PromptImage, the generator and the batcher.
            raise NotImplementedError(
                f"RL prompts support images only, got modalities {sorted(modalities)}."
            )
        keys = multi_modal_data.mm_hashes.get("image", [])
        paths = [
            part["path"]
            for message in self._messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image"
        ]
        if len(paths) != len(keys):
            raise ValueError(
                f"The rendered prompt has {len(keys)} images but its messages have "
                f"{len(paths)} image parts."
            )
        images = []
        image_placeholders = []
        for key, path, placeholder, item in zip(
            keys,
            paths,
            multi_modal_data.mm_placeholders.get("image", []),
            multi_modal_data.mm_items.get("image", []),
            strict=True,
        ):
            if item.get("pixel_values") is not None:
                raise ValueError(
                    "RL image prompts need an image-reference renderer (e.g. "
                    "Qwen35ImageRefRenderer), which sets each image's mm_hash to its "
                    f"content key and leaves pixel_values None; got pixel_values for "
                    f"image {key!r}."
                )
            t, h, w = item["image_grid_thw"][0]
            images.append(
                PromptImage(
                    key=key,
                    source_path=path,
                    grid_thw=(int(t), int(h), int(w)),
                    data=read_image_file(path, key),
                )
            )
            image_placeholders.append(placeholder)
        return Prompt(
            token_ids=rendered.token_ids,
            images=tuple(images),
            image_placeholders=tuple(image_placeholders),
        )
