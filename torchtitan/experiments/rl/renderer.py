# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import create_renderer, Renderer

from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for message <-> token conversion.

    Wraps upstream `PrimeIntellect-ai/renderers`. `build` returns a live
    `Renderer` constructed from a HF `PreTrainedTokenizer` loaded from
    `model_path`.

    Args:
        name: Renderer name passed to `renderers.create_renderer`
            (e.g. `"qwen3"`, `"auto"`).
        tool_parser: Tool-call parser name used by the default renderer.
        reasoning_parser: Reasoning parser name used by the default renderer.
        preserve_all_thinking: Forward historical assistant reasoning back
            into future prompts.
        preserve_thinking_between_tool_calls: Keep assistant reasoning during
            active tool-call loops.

    Example:

        renderer = RendererConfig(name="qwen3").build(
            model_path="./Qwen3-0.6B",
        )
        prompt_ids = renderer.render_ids(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
        stop_token_ids = renderer.get_stop_token_ids()
    """

    name: str = "auto"
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    preserve_all_thinking: bool = False
    preserve_thinking_between_tool_calls: bool = False

    def build(self, *, model_path: str) -> Renderer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return create_renderer(
            tokenizer,
            self.name,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            preserve_all_thinking=self.preserve_all_thinking,
            preserve_thinking_between_tool_calls=self.preserve_thinking_between_tool_calls,
        )
