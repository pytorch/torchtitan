# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import tyro
from renderers import create_renderer, Renderer
from renderers.configs import BaseRendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Base config of a renderer; `build` returns a `renderers.Renderer` on TorchTitan's tokenizer.

    Subclasses: `RenderersLibraryConfig` for a renderer from the `renderers` library, and
    in-tree renderers such as `MuseGlimmerRendererConfig`.
    """

    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        raise NotImplementedError


@dataclass(kw_only=True, slots=True)
class RenderersLibraryConfig(RendererConfig):
    """Builds one of the `renderers` library's renderers on TorchTitan's tokenizer.

    Example:

        from renderers import Qwen3RendererConfig

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.rl.renderer import RenderersLibraryConfig

        renderer = RenderersLibraryConfig(
            renderers_config=Qwen3RendererConfig(enable_thinking=False)
        ).build(tokenizer=HuggingFaceTokenizer(tokenizer_path="./Qwen3-0.6B"))
        prompt_ids = renderer.render_ids(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
    """

    renderers_config: Annotated[BaseRendererConfig, tyro.conf.Suppress]
    """The library's typed config for the model, e.g. `Qwen3RendererConfig(enable_thinking=False)`.
    Renderers and their options:
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/docs/renderer-config.md"""

    def to_dict(self) -> dict[str, Any]:
        return {"renderers_config": self.renderers_config.model_dump(mode="json")}

    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        if self.renderers_config.name == "auto":
            raise ValueError(
                f"AutoRendererConfig resolves by exact match of tokenizer.name_or_path ({tokenizer.tokenizer_path!r}) "
                "against renderers' MODEL_RENDERER_MAP, else falls back to DefaultRenderer (unsupported here). "
                "Pick the model's renderer, e.g. Qwen3RendererConfig(...)."
            )
        if self.renderers_config.name == "default":
            raise ValueError(
                "DefaultRenderer needs Hugging Face apply_chat_template; TorchTitan's template rendering lacks "
                "its special-token variables (bos_token, ...) and would silently produce different tokens. "
                "Pick the model's renderer, e.g. Qwen3RendererConfig(...)."
            )
        return create_renderer(
            tokenizer=RendererTokenizerWrapper(tokenizer), config=self.renderers_config
        )


class RendererTokenizerWrapper:
    """Adapt TorchTitan's loaded tokenizer to `renderers.OffsetTokenizer`.

    Protocol and bring-your-own-tokenizer guide:
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/renderers/base.py#L668-L699
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/README.md#install

    `renderers` needs Hugging Face-style special-token attributes, raw encoding
    without automatic BOS/EOS, token-to-id lookup, and character offsets. The
    offsets identify tokens from message content (`is_content`). This adapter
    exposes that interface from TorchTitan's underlying `tokenizers.Tokenizer`;
    it does not load a second tokenizer.

    Example:

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.experiments.rl.renderer import RendererTokenizerWrapper

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
        output = {"input_ids": encoding.ids}
        if return_offsets_mapping:
            output["offset_mapping"] = encoding.offsets
        return output
