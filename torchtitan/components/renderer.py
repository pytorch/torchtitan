# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from renderers import create_renderer, Message, Renderer
from renderers.configs import RendererConfig as PrimeRendererConfig

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable


def hash_image_file(data: bytes) -> str:
    """The hash of an image file's bytes, which a rendered prompt carries for each image
    (its ``mm_hashes`` entry), so a file gets the same hash from any path."""
    return hashlib.sha256(data).hexdigest()[:32]


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Base config of a renderer; `build` returns a `renderers.Renderer` on TorchTitan's tokenizer.

    `RenderersConfigAdapter` adapts library configs, including in-tree renderers
    selected with `CustomRendererConfig`.
    """

    # pyrefly: ignore[bad-override]
    def build(
        self,
        *,
        tokenizer: HuggingFaceTokenizer,
        vision_processor: VisionProcessor | None = None,
    ) -> Renderer:
        """Build the renderer on ``tokenizer``.

        Args:
            tokenizer: The model's tokenizer.
            vision_processor: The model's image preprocessing, for a renderer that reads
                image parts from files (e.g. `Qwen35ImageRefRenderer`). None: the renderer
                reads no image files.
        """
        raise NotImplementedError


@dataclass(kw_only=True, slots=True)
class RenderersConfigAdapter(RendererConfig):
    """TorchTitan config adapter for a `renderers` library config.

    Example:

        from renderers import Qwen3RendererConfig

        from torchtitan.components.renderer import from_renderers
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        renderer = from_renderers(
            Qwen3RendererConfig(enable_thinking=False)
        ).build(tokenizer=HuggingFaceTokenizer(tokenizer_path="./Qwen3-0.6B"))
        prompt_ids = renderer.render_ids(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )

    In-tree renderers use `CustomRendererConfig(import_path=..., **options)`.
    Workers import the renderer and build it with the checkpoint tokenizer. It must
    provide a `BaseRendererConfig` `config_class` and implement the `Renderer`
    protocol, including `parse_response(..., prompt_ids=...)`. Pass the custom
    config to `from_renderers` to use it in TorchTitan and Verifiers alike.
    """

    renderers_config: PrimeRendererConfig
    """The library's typed config for the model, e.g. `Qwen3RendererConfig(enable_thinking=False)`.
    Renderers and their options:
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.12/docs/renderer-config.md"""

    def to_dict(self) -> dict[str, Any]:
        return {"renderers_config": self.renderers_config.model_dump(mode="json")}

    def get_renderers_config(self) -> PrimeRendererConfig:
        """Return the `renderers` config that Verifiers workers rebuild the renderer from."""
        return self.renderers_config

    def build(
        self,
        *,
        tokenizer: HuggingFaceTokenizer,
        vision_processor: VisionProcessor | None = None,
    ) -> Renderer:
        if self.renderers_config.name == "auto":
            raise ValueError(
                f"AutoRendererConfig resolves by exact match of tokenizer.name_or_path ({tokenizer.tokenizer_path!r}) "
                "against renderers' MODEL_RENDERER_MAP, else falls back to DefaultRenderer (unsupported here). "
                "Pick the model's renderer, e.g. Qwen3RendererConfig(...), or name an in-tree renderer with "
                "CustomRendererConfig(import_path=...)."
            )
        if self.renderers_config.name == "default":
            raise ValueError(
                "DefaultRenderer needs Hugging Face apply_chat_template; TorchTitan passes "
                "bos_token and eos_token only, not the rest of the special-token map "
                "(pad_token, unk_token, ...), and would silently produce different tokens. "
                "Pick the model's renderer, e.g. Qwen3RendererConfig(...)."
            )
        renderers_config = self.renderers_config
        if vision_processor is not None:
            if renderers_config.name != "custom":
                raise ValueError(
                    f"The {renderers_config.name!r} renderer comes from the renderers "
                    "library, so it cannot use TorchTitan's VisionProcessor: it either "
                    "reads no images, or preprocesses them inside render() with the "
                    "Hugging Face image processor. Use an in-tree renderer that takes the "
                    "VisionProcessor, e.g. CustomRendererConfig(import_path="
                    '"torchtitan.models.qwen3_5.renderer.Qwen35ImageRefRenderer").'
                )
            if (renderers_config.model_extra or {}).get("vision_processor") is not None:
                raise ValueError(
                    "The renderer config sets vision_processor too; pass it only to build "
                    "(TitanRL passes Controller.Config.vision_processor)."
                )
            # The renderer's own config validates the field, and rejects it if the renderer
            # takes no image processor.
            renderers_config = renderers_config.model_copy(
                update={"vision_processor": vision_processor.config}
            )
        return create_renderer(
            tokenizer=RendererTokenizerWrapper(tokenizer), config=renderers_config
        )


def from_renderers(config: PrimeRendererConfig) -> RendererConfig:
    """Adapt a `renderers` config to TorchTitan's renderer config interface."""
    return RenderersConfigAdapter(renderers_config=config)


TEXT_CONTENT_TYPES = frozenset({"text", "input_text", "output_text", "thinking"})


def require_content_types(
    messages: list[Message], allowed_types: frozenset[str]
) -> None:
    """Raise if a message has a content part whose type is not in ``allowed_types``.

    Renderers may render a media part as a placeholder token with no data behind
    it, or skip an unknown part. A part with no type key counts as text.
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type", "text")
            if kind not in allowed_types:
                raise ValueError(f"Unsupported {kind!r} content part.")


class RendererTokenizerWrapper:
    """Adapt TorchTitan's loaded tokenizer to `renderers.OffsetTokenizer`.

    Protocol and bring-your-own-tokenizer guide:
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.12/renderers/base.py#L680-L711
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.12/README.md#install

    `renderers` needs Hugging Face-style special-token attributes, raw encoding
    without automatic BOS/EOS, token-to-id lookup, and character offsets. The
    offsets identify tokens from message content (`is_content`). This adapter
    exposes that interface from TorchTitan's underlying `tokenizers.Tokenizer`;
    it does not load a second tokenizer.

    Example:

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.components.renderer import RendererTokenizerWrapper

        tokenizer = RendererTokenizerWrapper(
            HuggingFaceTokenizer(tokenizer_path="./Qwen3-0.6B")
        )
        tokenizer.encode("hi")  # [6023]
        tokenizer(
            "hi",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )  # ids + character offsets
        tokenizer.convert_tokens_to_ids("<|im_end|>")  # 151645
    """

    def __init__(self, tokenizer: HuggingFaceTokenizer):
        # The `tokenizers.Tokenizer` inside; it has the offsets and token -> id lookup.
        self._tokenizer_backend = tokenizer.tokenizer
        self.name_or_path = tokenizer.tokenizer_path
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.bos_token_id = tokenizer.bos_id
        self.eos_token_id = tokenizer.eos_id
        # `tokenizers` returns None for unknown tokens; it has no unk id.
        self.unk_token_id = None

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> list[int]:
        return self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        return self._tokenizer_backend.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(
        self, tokens: str | list[str]
    ) -> int | None | list[int | None]:
        if isinstance(tokens, str):
            return self._tokenizer_backend.token_to_id(tokens)
        return [self._tokenizer_backend.token_to_id(token) for token in tokens]

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict:
        encoding = self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        )
        output: dict[str, Any] = {"input_ids": encoding.ids}
        if return_offsets_mapping:
            output["offset_mapping"] = encoding.offsets
        return output
