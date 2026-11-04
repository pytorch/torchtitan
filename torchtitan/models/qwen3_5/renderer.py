# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 renderer for RL image prompts: placeholder counts and content keys, no pixels.

Why not the library's ``Qwen35Renderer``: its ``_process_image`` runs the Hugging Face
image processor (decode, resize, normalize, patchify) and puts float32 ``pixel_values``
in ``multi_modal_data``. In RL that

- costs ~7.5 ms per Geometry3K prompt, once per sample of every group;
- carries megabytes of pixels per prompt between the rollout worker, the controller
  and the generator;
- is a second preprocessing implementation next to the generator's and the trainer's,
  which can silently disagree with them.

Rendering needs only each image's placeholder count (N in ``<|vision_start|>`` +
N x ``<|image_pad|>`` + ``<|vision_end|>``, from the patch grid) and a stable content
key. So this renderer overrides only ``_process_image``: `lookup_image` reads the file,
hashes it into the key (the image's ``mm_hashes`` entry) and computes the grid from the
header with the model's `ImageSpec`, without decoding; ``pixel_values`` stays ``None``,
and the generator and the trainer decode the image file with the same spec. Rendering takes
~1.1 ms per Geometry3K prompt, and the token ids equal the library renderer's (checked on
Geometry3K's 2,101 train images) when the spec uses the HF processor's pixel budget (the
``ImageSpec`` defaults).

Image parts must name a file, e.g. ``{"type": "image", "path": ...}``. Select it
with ``CustomRendererConfig(import_path=..., image_spec=ImageSpec())``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import model_validator
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from torchtitan.components.data.multimodal.utils.image import (
    ImageSpec,
    lookup_image,
    num_image_tokens,
)


class Qwen35ImageRefRendererConfig(Qwen35RendererConfig):
    """Qwen3.5 renderer config plus the image spec it computes grids with."""

    # pyrefly: ignore [bad-override]
    name: Literal["qwen3.5_image_ref"] = "qwen3.5_image_ref"
    # image_cache_max is inherited and unused: it bounds the HF-processor cache
    # of the `_process_image` this renderer replaces.
    _internal_fields = frozenset({"image_cache_max", "image_spec"})

    image_spec: ImageSpec
    """How the model turns an image into vision tokens (resize, patch and merge sizes),
    which gives each image's grid and so its number of placeholder tokens. In RL it must
    equal the generator's and the trainer's (`VLLMEngine.Config.image_spec`,
    `Batcher.Config.image_spec`), which decode the image files with it."""

    @model_validator(mode="after")
    def _reject_image_cache_max(self) -> "Qwen35ImageRefRendererConfig":
        if "image_cache_max" in self.model_fields_set:
            raise ValueError(
                "image_cache_max bounds the HF-processor cache, which "
                f"{type(self).__name__} never uses; remove it."
            )
        return self


class Qwen35ImageRefRenderer(Qwen35Renderer):
    """``Qwen35Renderer`` that resolves image parts with `lookup_image`.

    Example:

        renderer = Qwen35ImageRefRenderer(
            tokenizer,
            Qwen35ImageRefRendererConfig(image_spec=ImageSpec()),
        )
        rendered = renderer.render(
            [{"role": "user", "content": [
                {"type": "image", "path": "./assets/datasets/geometry3k/images/train/00000_0.png"},
                {"type": "text", "text": "Find x."},
            ]}],
            add_generation_prompt=True,
        )
        rendered.multi_modal_data.mm_hashes["image"]  # [content key]
    """

    config_class = Qwen35ImageRefRendererConfig

    def __init__(self, tokenizer: Any, config: Qwen35ImageRefRendererConfig) -> None:
        super().__init__(tokenizer, config)
        self._image_spec = config.image_spec

    # pyrefly: ignore [bad-override]
    def _process_image(self, part: dict[str, Any]):
        """Return ``(None, {"pixel_values": None, "image_grid_thw"}, num_image_tokens, key)``.

        The grid and placeholder count come from the file header; no pixels are decoded.
        """
        path = part.get("path")
        if path is None:
            raise ValueError(
                f"{type(self).__name__} needs image parts with a 'path' to an image "
                f"file, got keys {sorted(part)}."
            )
        image = lookup_image(path, self._image_spec)
        out = {"pixel_values": None, "image_grid_thw": np.array([image.grid_thw])}
        return None, out, num_image_tokens(image.grid_thw, self._image_spec), image.key
