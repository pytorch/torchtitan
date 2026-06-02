# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from renderers import create_renderer, Renderer, RendererConfig as _RendererConfig

from torchtitan.config import Configurable

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec

# `renderers` exposes a pydantic discriminated union (on `name`); let it route
# `name` -> the matching config class
_RENDERER_CONFIG = TypeAdapter(_RendererConfig)

# Map a TorchTitan model family (`ModelSpec.name`) to a `renderers` name, used to
# resolve `name="auto"` without relying on the HF tokenizer's `name_or_path`
# (a local path for our checkpoints, which misses the renderers auto map).
_TORCHTITAN_RENDERER_BY_MODEL: dict[str, str] = {
    "qwen3": "qwen3",
}


def _resolve_renderer_name(name: str, model_spec: "ModelSpec | None") -> str:
    """Resolve `name="auto"` via the TorchTitan model; explicit names pass through."""
    if name != "auto":
        return name
    if model_spec is not None:
        mapped = _TORCHTITAN_RENDERER_BY_MODEL.get(model_spec.name)
        if mapped is not None:
            return mapped
    # Last resort: let `renderers` resolve from the tokenizer's model id.
    return "auto"


def _build_renderer_config(name: str, cfg: "RendererConfig"):
    """Build the `renderers` config for `name`, passing only the knobs it supports."""
    if name == "auto":
        return None
    config_cls = type(_RENDERER_CONFIG.validate_python({"name": name}))
    supported_args: dict[str, bool | str] = {
        field: getattr(cfg, field)
        for field in (
            "enable_thinking",
            "preserve_all_thinking",
            "preserve_thinking_between_tool_calls",
            "tool_parser",
            "reasoning_parser",
        )
        if field in config_cls.model_fields and getattr(cfg, field) is not None
    }
    return config_cls(**supported_args)


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for message <-> token conversion.

    Wraps `PrimeIntellect-ai/renderers`. `build` loads a tokenizer from
    `tokenizer_path` and constructs the `renderers` config for `name`.

    Args:
        name: Renderer name. `"auto"` resolves from the TorchTitan model
            (`ModelSpec.name`); or name it explicitly: `"qwen3"`, `"gpt-oss"`,
            `"deepseek-v3"`, ... (see renderers `RENDERER_REGISTRY`).
        tool_parser: Tool-call parser name (renderer-specific).
        reasoning_parser: Reasoning parser name (renderer-specific).
        enable_thinking: Let the model emit reasoning.
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

    def build(
        self, *, tokenizer_path: str, model_spec: "ModelSpec | None" = None
    ) -> Renderer:
        # TODO(renderers#70): use TorchTitan's tokenizer once `renderers` supports
        # bring-your-own-tokenizer (PR adds a Tokenizer protocol; drops transformers).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        renderer_name = _resolve_renderer_name(self.name, model_spec)
        return create_renderer(tokenizer, _build_renderer_config(renderer_name, self))
