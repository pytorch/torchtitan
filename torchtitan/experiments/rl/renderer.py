# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, fields

from renderers import config_from_name, create_renderer, Renderer

from torchtitan.config import Configurable

# Map a TorchTitan model name to its `renderers` renderer. Models not listed fall
# back to "auto" (renderers resolves from the tokenizer)
# https://github.com/PrimeIntellect-ai/renderers/blob/942449c37ab6e9fab26d59b40336514c8baa6b13/renderers/configs.py#L404
_RENDERER_BY_MODEL = {
    "qwen3": "qwen3",
    "qwen3_vl": "qwen3-vl",
    "gpt_oss": "gpt-oss",
    "deepseek_v3": "deepseek-v3",
    "default": "default",  # llama3, llama4
    "auto": "auto",  # ignores knobs, resolves from tokenizer,
}


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for chat message <-> token conversion.

    Wraps `PrimeIntellect-ai/renderers`. `build` loads a tokenizer from
    `tokenizer_path`, maps the model `name` to a renderer, and forwards any
    supported knobs.

    Args:
        name: TorchTitan model name (e.g. `"qwen3"`, `"llama3"`). Mapped to a
            `renderers` renderer via `_RENDERER_BY_MODEL`; unmapped models use
            `"auto"` (renderers resolves from the tokenizer).
        tool_parser: Tool-call parser name, when the renderer supports it.
        reasoning_parser: Reasoning parser name, when the renderer supports it.
        enable_thinking: Let the model emit reasoning, when supported.
        preserve_all_thinking: Keep historical reasoning in future prompts.
        preserve_thinking_between_tool_calls: Keep reasoning during tool loops.

    Example:

        renderer = RendererConfig(name="qwen3").build(tokenizer_path="./Qwen3-0.6B")
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

    def build(self, *, tokenizer_path: str) -> Renderer:
        # TODO(renderers#70): use TorchTitan's tokenizer once `renderers` supports
        # bring-your-own-tokenizer (PR adds a Tokenizer protocol; drops transformers).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        renderer_name = _RENDERER_BY_MODEL.get(self.name)

        # `config_from_name` returns the typed renderers config, or `None` for
        # "auto" (which makes `create_renderer` resolve from the tokenizer).
        renderer_config = config_from_name(renderer_name)
        if renderer_config is None:
            return create_renderer(tokenizer, None)

        # Forward our knobs (every field except `name`) that this renderer config
        # supports. Configs are frozen pydantic models, so update via model_copy.
        overrides = {}
        for field in fields(self):
            if (
                field.name != "name"
                and field.name in type(renderer_config).model_fields
            ):
                overrides[field.name] = getattr(self, field.name)

        if overrides:
            renderer_config = renderer_config.model_copy(update=overrides)

        return create_renderer(tokenizer, renderer_config)
