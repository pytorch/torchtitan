# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Literal

from renderers import AutoRendererConfig, config_from_name, create_renderer, Renderer

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)

# Map a TorchTitan model name to its `renderers` renderer. Models not listed fall
# back to "auto" (renderers resolves from the tokenizer)
# https://github.com/PrimeIntellect-ai/renderers/blob/942449c37ab6e9fab26d59b40336514c8baa6b13/renderers/configs.py#L404
_RENDERER_BY_MODEL = {
    "qwen3": "qwen3",
    "qwen3_vl": "qwen3-vl",
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
        preserve_all_thinking: Removed upstream; see `thinking_retention`.
        preserve_thinking_between_tool_calls: Removed upstream; see
            `thinking_retention`.
        thinking_retention: Historical-reasoning policy, forwarded to `renderers`.
            `"all"` keeps reasoning across the whole history, `"tool_cycle"` keeps
            it within a tool loop, and `None` defers to the chat template. It is
            carried through auto-resolution; `DefaultRenderer` is the one renderer
            that cannot honour an explicit policy.

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
    thinking_retention: Literal["tool_cycle", "all"] | None = None

    def build(self, *, tokenizer_path: str) -> Renderer:
        # TODO(renderers#70): use TorchTitan's tokenizer once `renderers` supports
        # bring-your-own-tokenizer (PR adds a Tokenizer protocol; drops transformers).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # `name=None` (or "auto") -> let `create_renderer` resolve from the tokenizer.
        renderer_name = _RENDERER_BY_MODEL.get(self.name, self.name)
        renderer_config = config_from_name(renderer_name) if renderer_name else None

        # `renderers` replaced these two bools with the single `thinking_retention`
        # enum (PrimeIntellect-ai/renderers#88), so neither is a field on a renderer
        # config any more and the name-match forwarding below drops them without a
        # word -- and drops them before the library's own validator, which raises
        # and names the replacement, can see them. Naming a renderer does not bring
        # them back, so report the replacement rather than the path.
        removed = sorted(
            name
            for name in (
                "preserve_all_thinking",
                "preserve_thinking_between_tool_calls",
            )
            # Only a `True` asked for something the renderer can no longer be told;
            # an explicit `False` meant "defer to the chat template", which is what
            # dropping it already does.
            if getattr(self, name)
            and name not in getattr(type(renderer_config), "model_fields", {})
        )
        if removed:
            raise ValueError(
                f"{removed} were replaced by `thinking_retention` in `renderers`. "
                'Set thinking_retention="all" to keep reasoning across the whole '
                'history, or "tool_cycle" to keep it within a tool loop. '
                "DefaultRenderer cannot honour an explicit policy; the "
                "model-specific renderers can."
            )

        if renderer_config is None:
            # `name=None` / "auto" resolves the renderer from the tokenizer.
            # AutoRendererConfig carries `thinking_retention` into the resolved
            # config and deliberately nothing else, so forward that one knob and
            # report the rest instead of dropping them.
            unused = sorted(
                field.name
                for field in fields(self)
                if field.name not in ("name", "thinking_retention")
                and getattr(self, field.name) is not None
            )
            if unused:
                raise ValueError(
                    f"RendererConfig set {unused} with name={self.name!r}. "
                    "Auto-resolution forwards only `thinking_retention`; name the "
                    "renderer to use the rest."
                )
            if self.thinking_retention is None:
                return create_renderer(tokenizer, None)
            return create_renderer(
                tokenizer, AutoRendererConfig(thinking_retention=self.thinking_retention)
            )

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
