# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

from renderers import config_from_name, create_renderer, Renderer

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)

# Map a TorchTitan model name to its `renderers` renderer. Models not listed fall
# back to "auto" (renderers resolves from the tokenizer)
# https://github.com/PrimeIntellect-ai/renderers/blob/942449c37ab6e9fab26d59b40336514c8baa6b13/renderers/configs.py#L404
_RENDERER_BY_MODEL = {
    "qwen3": "qwen3",
    "qwen3_vl": "qwen3-vl",
    # Qwen3.5 evolved from qwen3_vl; resolve its chat template from the tokenizer
    # (no dedicated renderers entry yet). TODO: map to a qwen3_5 renderer if added.
    "qwen3_5": "auto",
    "gpt_oss": "gpt-oss",
    "deepseek_v3": "deepseek-v3",
    "default": "default",  # llama3
    "auto": "auto",  # ignores knobs, resolves from tokenizer,
}


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for chat message <-> token conversion.

    Wraps `PrimeIntellect-ai/renderers`. `build` loads a tokenizer from
    `tokenizer_path`, maps the model `name` to a renderer, and forwards any
    supported knobs.

    Args:
        name: TorchTitan model name (e.g. `"qwen3"`, `"llama3"`), mapped to a
            `renderers` renderer via `_RENDERER_BY_MODEL`. `None` (the default)
            resolves the renderer from the tokenizer.
        tool_parser: Tool-call parser name, when the renderer supports it.
        reasoning_parser: Reasoning parser name, when the renderer supports it.
        enable_thinking: Let the model emit reasoning, when supported.
        preserve_all_thinking: Keep historical reasoning in future prompts.
        preserve_thinking_between_tool_calls: Keep reasoning during tool loops.

    Every field defaults to `None`; a non-`None` value overrides that knob on the
    chosen renderer's config, otherwise the renderer keeps its own default.

    Example:

        renderer = RendererConfig(name="qwen3").build(tokenizer_path="./Qwen3-0.6B")
        prompt_ids = renderer.render_ids(
            [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
    """

    name: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    enable_thinking: bool | None = None
    preserve_all_thinking: bool | None = None
    preserve_thinking_between_tool_calls: bool | None = None

    def build(self, *, tokenizer_path: str) -> Renderer:
        # TODO(renderers#70): use TorchTitan's tokenizer once `renderers` supports
        # bring-your-own-tokenizer (PR adds a Tokenizer protocol; drops transformers).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # `name=None` (or "auto") -> let `create_renderer` resolve from the tokenizer.
        renderer_name = _RENDERER_BY_MODEL.get(self.name, self.name)
        renderer_config = config_from_name(renderer_name) if renderer_name else None
        if renderer_config is None:
            return create_renderer(tokenizer, None)

        # Rebuild the typed config and pass parameters
        # that are not None and are supported
        config_type = type(renderer_config)
        args = {
            field.name: getattr(self, field.name)  # {key: value}
            for field in fields(self)
            if field.name != "name"  # Get all self.fields, except name
            and getattr(self, field.name) is not None  # Only consider provided fields
            and field.name in config_type.model_fields  # Config supports this field
        }
        logger.info(
            f"Using renderer {renderer_name}, of type {config_type}, with args {args}"
        )
        return create_renderer(tokenizer, config_type(**args))
