# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter
from renderers import create_renderer, Renderer, RendererConfig as _RendererConfig

from torchtitan.config import Configurable

# `renderers` exposes a pydantic discriminated union (on `name`); let it route
# `name` -> the matching config class
_RENDERER_CONFIG = TypeAdapter(_RendererConfig)


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for message <-> token conversion.

    Wraps `PrimeIntellect-ai/renderers`. `build` loads a HF tokenizer from
    `model_path` and constructs the `renderers` config matching `name`.

    Args:
        name: Renderer name (e.g. `"qwen3"`, `"auto"`).
        tool_parser: Tool-call parser name (renderer-specific).
        reasoning_parser: Reasoning parser name (renderer-specific).
        enable_thinking: Let the model emit reasoning.
        preserve_all_thinking: Keep historical reasoning in future prompts.
        preserve_thinking_between_tool_calls: Keep reasoning during tool loops.

    Example:

        renderer = RendererConfig(name="qwen3").build(model_path="./Qwen3-0.6B")
        prompt_ids = renderer.render_ids(
            [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
    """

    name: str = "auto"
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    enable_thinking: bool = True
    preserve_all_thinking: bool = False
    preserve_thinking_between_tool_calls: bool = False

    def build(self, *, model_path: str) -> Renderer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Resolve `name` to its config class via the library's discriminator,
        # then pass only the knobs that class supports.
        config_cls = type(_RENDERER_CONFIG.validate_python({"name": self.name}))
        config_args: dict[str, bool | str] = {
            k: v
            for k, v in {
                "enable_thinking": self.enable_thinking,
                "preserve_all_thinking": self.preserve_all_thinking,
                "preserve_thinking_between_tool_calls": self.preserve_thinking_between_tool_calls,
                "tool_parser": self.tool_parser,
                "reasoning_parser": self.reasoning_parser,
            }.items()
            if k in config_cls.model_fields and v is not None
        }
        return create_renderer(tokenizer, config_cls(**config_args))
