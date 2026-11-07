# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 renderer for RL image prompts: placeholder counts and image hashes, no pixels.

Why not the library's ``Qwen35Renderer``: its ``_process_image`` runs the Hugging Face
image processor (decode, resize, normalize, patchify) and puts float32 ``pixel_values``
in ``multi_modal_data``. In RL that

- costs ~9 ms per Geometry3K prompt, once per sample of every group;
- carries megabytes of pixels per prompt between the rollout worker, the controller
  and the generator;
- is a second preprocessing implementation next to the generator's and the trainer's,
  which can silently disagree with them.

Rendering needs only each image's placeholder count (N in ``<|vision_start|>`` +
N x ``<|image_pad|>`` + ``<|vision_end|>``, from the patch grid) and a stable hash. So
this renderer overrides only ``_process_image``: it reads the file, hashes it (the
image's ``mm_hashes`` entry, `hash_image_file`) and computes the grid from the size in
the header with ``ImageProcessor.grid_thw``, without decoding; ``pixel_values`` stays
``None``, and the generator and the trainer preprocess the image file with the same
image processor. Rendering takes ~1.2 ms per Geometry3K prompt, and the
token ids equal the library renderer's (checked on Geometry3K's 2,101 train images).

Image parts must name a file, e.g. ``{"type": "image", "path": ...}``. Select it with
``CustomRendererConfig(import_path="torchtitan.models.qwen3_5.renderer.Qwen35ImageRefRenderer")``
and ``Controller.Config.image_processor=ImageProcessor.Config(**QWEN3_5_IMAGE_PROCESSOR_KWARGS)``.
"""

from __future__ import annotations

import io
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import model_validator
from renderers.configs import Qwen35RendererConfig
from renderers.qwen35 import Qwen35Renderer

from torchtitan.components.data.multimodal.image_processor import ImageProcessor
from torchtitan.components.renderer import hash_image_file


class Qwen35ImageRefRendererConfig(Qwen35RendererConfig):
    """Qwen3.5 renderer config plus the image processor it computes grids with."""

    # pyrefly: ignore [bad-override]
    name: Literal["qwen3.5_image_ref"] = "qwen3.5_image_ref"
    # image_cache_max is inherited and unused: it bounds the HF-processor cache
    # of the `_process_image` this renderer replaces.
    _internal_fields = frozenset({"image_cache_max", "image_processor"})

    image_processor: ImageProcessor.Config | None = None
    """The model's image preprocessing, which gives each image's grid and so its number
    of placeholder tokens. Required. `RenderersConfigAdapter.build(image_processor=...)`
    sets it; TitanRL passes `Controller.Config.image_processor`, so the renderer, the
    generator and the trainer share one. Set it here only to build the renderer
    directly."""

    @model_validator(mode="after")
    def _reject_image_cache_max(self) -> "Qwen35ImageRefRendererConfig":
        if "image_cache_max" in self.model_fields_set:
            raise ValueError(
                "image_cache_max bounds the HF-processor cache, which "
                f"{type(self).__name__} never uses; remove it."
            )
        return self


class Qwen35ImageRefRenderer(Qwen35Renderer):
    """``Qwen35Renderer`` that resolves image parts from the file header, without
    decoding pixels.

    Example:

        renderer = Qwen35ImageRefRenderer(
            tokenizer,
            Qwen35ImageRefRendererConfig(
                image_processor=ImageProcessor.Config(**QWEN3_5_IMAGE_PROCESSOR_KWARGS)
            ),
        )
        rendered = renderer.render(
            [{"role": "user", "content": [
                {"type": "image", "path": "./assets/datasets/geometry3k/images/train/00000_0.png"},
                {"type": "text", "text": "Find x."},
            ]}],
            add_generation_prompt=True,
        )
        rendered.multi_modal_data.mm_hashes["image"]  # [hash_image_file(file bytes)]
    """

    config_class = Qwen35ImageRefRendererConfig

    def __init__(self, tokenizer: Any, config: Qwen35ImageRefRendererConfig) -> None:
        super().__init__(tokenizer, config)
        if config.image_processor is None:
            raise ValueError(
                f"{type(self).__name__} needs an image processor; set "
                "Controller.Config.image_processor, e.g. "
                "ImageProcessor.Config(**QWEN3_5_IMAGE_PROCESSOR_KWARGS)."
            )
        self._image_processor: ImageProcessor = config.image_processor.build()

    # pyrefly: ignore [bad-override]
    def _process_image(self, part: dict[str, Any]):
        """Return ``(None, {"pixel_values": None, "image_grid_thw"}, num_placeholder_tokens,
        hash)``.

        The grid and placeholder count come from the size in the file header; no pixels
        are decoded. E.g. a 640x480 JPEG with Qwen3.5's 16-pixel patches resizes to
        640x480, a ``(1, 30, 40)`` patch grid and 300 placeholder tokens (2x2 merge).
        """
        path = part.get("path")
        if path is None:
            raise ValueError(
                f"{type(self).__name__} needs image parts with a 'path' to an image "
                f"file, got keys {sorted(part)}."
            )
        with open(path, "rb") as f:
            data = f.read()
        # The header gives the stored size; decoding applies no EXIF rotation either.
        width, height = Image.open(io.BytesIO(data)).size
        grid_thw = self._image_processor.grid_thw(height, width)
        out = {"pixel_values": None, "image_grid_thw": np.array([grid_thw])}
        num_tokens = self._image_processor.num_placeholder_tokens(grid_thw)
        return None, out, num_tokens, hash_image_file(data)
