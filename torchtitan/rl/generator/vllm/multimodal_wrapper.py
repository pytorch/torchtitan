# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image prompts in the vLLM generator: the TorchTitan vision encoder behind vLLM's
multimodal interface.

Requests carry a `Prompt`: token ids with each image's placeholder run, and the
`PromptImage`s (hash, patch grid and the image file's bytes). TitanRL preprocesses the
images, not vLLM: ``VLLMEngine`` turns each image file into pixel values with the
`VisionProcessor` (``Controller.Config.vision_processor``) on the CPU, as the SFT dataloader
does, and submits vLLM's multimodal input directly (`image_prompt_to_engine_input`), so
vLLM's Hugging Face processor never runs.

An image's embedding depends on the vision weights, which train, and vLLM's encoder cache
keeps it after its requests finish (only ``reset_encoder_cache`` clears it, not a prefix
cache reset). So the image's mm hash is its file's hash salted like the prefix cache with
the request's pinned policy version (``hash@version``): vLLM's encoder and prefix caches
reuse an image's embedding only among requests pinned to the same version, e.g. the samples
of a GRPO group, and never across versions. With ``reset_kv_cache_on_model_state_dict_load``
there is no salt; the engine resets both caches on every weight load instead:

    PromptImage(hash_value, ..., data)              rollout worker -> generator: the file's
                                                    bytes (Geometry3K median 15 KB)
      -> VisionProcessor.resize_and_normalize,       each TP rank, once per file while it is
         then patchify (CPU): float32               in the engine's LRU cache (Geometry3K:
         pixel_values (num_patches, D)              6.5 ms median, 2.4 MB)
      -> vLLM mm_input(prompt ids, {"pixel_values", "image_grid_thw"},
                       hashes=hash[@version], placeholders)
      -> embed_multimodal: the vision encoder, once per mm hash while cached
                           (vLLM copies pixel_values to the GPU)
      -> embed_input_ids: token embeddings with image embeddings at the placeholders
      -> forward(inputs_embeds, positions: 3D M-RoPE for Qwen3.5, 1D for Muse Spark)

vLLM requires a registered processor for a multimodal model (`register_multimodal_processor`);
it serves only vLLM's startup: the vision processor's largest image (`_largest_patch_grid`)
sizes the encoder budget, and a dummy image of that size profiles the encoder's memory.

Supported models: a ``vision_encoder`` behind ``get_vision_embeds`` and merged patches as
placeholder tokens; image tokens take 3D positions with ``MRoPE.Config`` attention
(Qwen3.5) and 1D positions otherwise (Muse Spark). Before the engine is built, the
controller checks that the model is a `MultimodalModel` with a vision encoder.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import renderers
import torch
from vllm.inputs import mm_input, MultiModalInput
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)

from torchtitan.components.data.multimodal.mrope import get_mrope_positions

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper
from torchtitan.rl.rollout.types import Prompt, PromptImage

# The keyword arguments vLLM batches per image and passes to `embed_multimodal`.
_IMAGE_FIELDS = {
    # float32 (num_patches, patch_dim) from `VisionProcessor.patchify`; stacked to
    # (num_images, num_patches, patch_dim) when the images have the same patch count,
    # else a list.
    "pixel_values": MultiModalFieldConfig.batched("image"),
    # (3,) raw patch grid, stacked to (num_images, 3).
    "image_grid_thw": MultiModalFieldConfig.batched("image"),
}


def image_prompt_to_engine_input(
    prompt: Prompt,
    pixel_values: Sequence[torch.Tensor],
    *,
    cache_salt: str | None = None,
) -> MultiModalInput:
    """Turn a prompt with images, and its images' pixel values, into vLLM's engine input.

    Called by `VLLMEngine._prompt_to_engine_input` (``vllm/engine.py``) for each image
    request the engine admits, and by `_ImageMultiModalProcessor.apply` for vLLM's startup
    profiling.

    Args:
        prompt: Token ids with each image's placeholder run, and the images.
        pixel_values: Each image's ``(num_patches, patch_dim)`` pixel values
            (`VisionProcessor.resize_and_normalize`, then `patchify`), in ``prompt.images``
            order.
        cache_salt: vLLM prefix-cache salt. TorchTitan sets it, not vLLM: the engine
            passes the request's pinned policy version, as for text prompts, the oldest
            version its KV may come from. Here it also salts the image hashes
            (``hash@salt``), because an image's encoder output depends on the vision
            weights, which train. None (the engine resets its caches on every weight
            load): the hashes are the bare file hashes. vLLM's ``mm_input`` only packs
            these fields into the engine input.
    """
    kwargs = MultiModalKwargsItems.from_hf_inputs(
        {
            "pixel_values": list(pixel_values),
            "image_grid_thw": torch.tensor([image.grid_thw for image in prompt.images]),
        },
        _IMAGE_FIELDS,
    )
    return mm_input(
        prompt_token_ids=prompt.token_ids,
        mm_kwargs=kwargs,
        mm_hashes={
            "image": [
                (
                    image.hash_value
                    if cache_salt is None
                    else f"{image.hash_value}@{cache_salt}"
                )
                for image in prompt.images
            ]
        },
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=placeholder.offset, length=placeholder.length)
                for placeholder in prompt.image_placeholders
            ]
        },
        cache_salt=cache_salt,
    )


# A side longer than any resize budget allows, so the resize scales it down to the budget.
_SIDE_ABOVE_EVERY_RESIZE_BUDGET = 1 << 20


def _largest_patch_grid(
    processor: VisionProcessor, max_tokens: int
) -> tuple[int, int, int]:
    """Patch grid of the image with the most placeholder tokens ``processor`` produces,
    capped at ``max_tokens`` tokens.

    Called once at generator startup: vLLM sizes its vision encoder budget and profiles
    the encoder's memory with a dummy image of this grid. The encoder's cost depends on
    the token count, not the shape. E.g. with Qwen3.5's processor (16-pixel patches,
    2x2 merge, max_pixels = 4096 * 4096):

        _largest_patch_grid(processor, max_tokens=100_000)  # (1, 256, 256): 4096 / 16 per side
        _largest_patch_grid(processor, max_tokens=4_096)    # (1, 128, 128): 64 * 64 tokens

    The largest image is not always square: with max_pixels = 1,003,520 and the same
    patches, the largest square has 31 x 31 = 961 tokens of 32 x 32 pixels, but a
    28 x 35 grid fits exactly (980 tokens, (1, 56, 70) in patches). So, as vLLM's
    Qwen2-VL ``get_image_size_with_most_features`` does, this tries token counts from
    the top down, each as its most square grid, and takes the first grid the resize
    keeps unchanged, i.e. the first one an image can be resized to.

    Exact for a resize that keeps every image within an area cap: a cap on the
    merged-patch count, or `PixelBudgetResize` unless ``min_pixels`` is so close to
    ``max_pixels`` that rounding up an image it enlarges to ``min_pixels`` crosses
    ``max_pixels``. A resize that pads after applying its budget
    (`NaViTPatchGridResize`) can give an elongated image more, by its padding.
    """
    # TODO: an exact bound for padding resizes, once a NaViT model serves images.
    merge = processor.config.spatial_merge_size
    patch_size = processor.config.patch_size
    _, grid_h, grid_w = processor.grid_thw(
        _SIDE_ABOVE_EVERY_RESIZE_BUDGET, _SIDE_ABOVE_EVERY_RESIZE_BUDGET
    )
    rows = cols = side = min(grid_h, grid_w) // merge
    # The huge image resized to the largest square, so under an area cap no image
    # reaches (side + 1)**2 tokens. Whether the resize keeps a size depends on its area
    # and aspect ratio, so the most square grid of a count is kept if any is.
    for num_tokens in range((side + 1) ** 2 - 1, side**2, -1):
        candidate_rows = next(
            divisor
            for divisor in range(math.isqrt(num_tokens), 0, -1)
            if num_tokens % divisor == 0
        )
        candidate_cols = num_tokens // candidate_rows
        grid = (1, candidate_rows * merge, candidate_cols * merge)
        try:
            # The grid of an image exactly this size: equal if the resize keeps it.
            kept = (
                processor.grid_thw(grid[1] * patch_size, grid[2] * patch_size) == grid
            )
        except ValueError:  # the resize rejects this aspect ratio
            continue
        if kept:
            rows, cols = candidate_rows, candidate_cols
            break
    if rows * cols > max_tokens:
        rows = cols = math.isqrt(max_tokens)
        if rows == 0:
            raise ValueError(
                f"No image fits {max_tokens} tokens with patch size "
                f"{processor.config.patch_size} and spatial merge size {merge}."
            )
    return (1, rows * merge, cols * merge)


class _ImageProcessingInfo(BaseProcessingInfo):
    """Sizes vLLM's encoder budget and memory profiling: the vision processor's largest
    image that fits ``max_model_len``."""

    def __init__(
        self, ctx: InputProcessingContext, *, vision_processor: VisionProcessor
    ) -> None:
        super().__init__(ctx)
        self.vision_processor = vision_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def largest_grid(self, seq_len: int) -> tuple[int, int, int]:
        return _largest_patch_grid(self.vision_processor, seq_len)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {
            "image": self.vision_processor.num_placeholder_tokens(
                self.largest_grid(seq_len)
            )
        }


class _ImageDummyInputsBuilder(BaseDummyInputsBuilder[_ImageProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        _, h, w = self.info.largest_grid(seq_len)
        patch_size = self.info.vision_processor.config.patch_size
        return {
            "image": self._get_dummy_images(
                width=w * patch_size,
                height=h * patch_size,
                num_images=mm_counts.get("image", 0),
            )
        }


class _ImageMultiModalProcessor(BaseMultiModalProcessor[_ImageProcessingInfo]):
    """Bypasses vLLM's image preprocessing: TitanRL preprocesses images itself.

    vLLM normally runs the model's Hugging Face processor on every prompt (decode, resize,
    normalize, patchify, expand placeholders). Here ``VLLMEngine`` turns each image file
    into pixel values with the `VisionProcessor` and submits them to vLLM directly
    (`image_prompt_to_engine_input`). So this processor builds only the dummy inputs vLLM
    profiles with at startup.
    """

    # Prompts arrive as token ids and images preprocessed, so no tokenizer is needed
    # (the engine runs with skip_tokenizer_init).
    requires_tokenizer: ClassVar[bool] = False

    def _get_mm_fields_config(
        self, hf_inputs: Any, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _IMAGE_FIELDS

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        """One zero image of the vision processor's largest size per dummy image item."""
        vision_processor = self.info.vision_processor
        num_images = inputs.mm_data_items.get_count("image", strict=False)
        grid = self.info.largest_grid(self.info.ctx.model_config.max_model_len)
        length = vision_processor.num_placeholder_tokens(grid)
        patch_size = vision_processor.config.patch_size
        pixel_values, _ = vision_processor.patchify(
            torch.zeros(1, grid[1] * patch_size, grid[2] * patch_size, 3)
        )
        images = tuple(
            PromptImage(
                hash_value=f"profile-{index}",
                source_path=f"profile-{index}",
                grid_thw=grid,
                data=b"",  # unused: the pixel values are `pixel_values`
            )
            for index in range(num_images)
        )
        prompt = Prompt(
            token_ids=[0] * (num_images * length),
            images=images,
            image_placeholders=tuple(
                renderers.PlaceholderRange(offset=index * length, length=length)
                for index in range(num_images)
            ),
        )
        return image_prompt_to_engine_input(prompt, [pixel_values] * num_images)


class VLLMMultimodalModelWrapper(VLLMModelWrapper, SupportsMultiModal, SupportsMRoPE):
    """`VLLMModelWrapper` that keeps the vision encoder and serves image prompts.

    Implements vLLM's multimodal (`embed_multimodal`, `embed_input_ids`) and M-RoPE
    (`get_mrope_input_positions`) interfaces for models with a ``vision_encoder`` reached
    through the model's ``get_vision_embeds`` and merged patches as placeholder tokens:
    Qwen3.5, whose image tokens take 3D M-RoPE positions, and Muse Spark, whose take 1D
    positions like text (vLLM then never calls `get_mrope_input_positions`). vLLM feeds
    multimodal models embeddings, also for text-only steps.

    """

    def __init__(self, *, vision_processor: VisionProcessor, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vision_processor = vision_processor
        # FSDP makes the first module that runs forward its root. This wrapper calls
        # the embeddings, the vision encoder and the layers directly, never the root
        # model, so make the root the whole model before any of them runs. Otherwise
        # the grouped [tok_embeddings, norm, lm_head] state raises ("requires a single
        # root module") when embed_input_ids runs first.
        # TODO: remove once pytorch/pytorch#204267 lands; FSDP2 then handles this case.
        self.model._get_fsdp_state()._lazy_init()

    def embed_multimodal(
        self,
        *,
        pixel_values: torch.Tensor | list[torch.Tensor],
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """vLLM required API. Encode a batch of images, one embedding block per image.

        Args:
            pixel_values: Each image's ``(num_patches, patch_dim)`` pixel values on the GPU
                (see `_IMAGE_FIELDS`): a list, or stacked when the images have the same
                patch count.
            image_grid_thw: ``(num_images, 3)`` raw patch grids.
        """
        # The encoder takes the images' patches packed into one (total_num_patches, D).
        packed_pixel_values = torch.cat(list(pixel_values))
        with self.parallelism_context.activate_spmd():
            vision_embeds, num_tokens = self.model.get_vision_embeds(
                packed_pixel_values, grid_thw=image_grid_thw
            )
        return vision_embeds.split(num_tokens.tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Sequence[torch.Tensor] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vLLM required API. Token embeddings with image embeddings written over the
        placeholder tokens that ``is_multimodal`` marks."""
        inputs_embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        # vLLM orders the embeddings as the placeholder tokens appear in the batch, the
        # order masked_scatter_ fills them in. Unlike assigning through a boolean mask,
        # whose nonzero syncs the host when the mask is on the GPU, masked_scatter_
        # never syncs (vLLM's V1 runner passes a pinned CPU mask).
        inputs_embeds.masked_scatter_(
            is_multimodal.to(inputs_embeds.device, non_blocking=True).unsqueeze(-1),
            torch.cat(list(multimodal_embeddings)).to(inputs_embeds.dtype),
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """vLLM required API. The decoder on precomputed embeddings.

        Args:
            input_ids: Unused; vLLM passes embeddings to multimodal models.
            positions: ``(3, num_tokens)`` M-RoPE positions (Qwen3.5), or ``(num_tokens,)``
                for a model whose image tokens take 1D positions (Muse Spark).
            inputs_embeds: ``(num_tokens, dim)`` from `embed_input_ids`.
        """
        assert inputs_embeds is not None and positions is not None
        if positions.ndim == 2:
            # TorchTitan's MRoPE takes (num_tokens, 3).
            positions = positions.transpose(0, 1)
        with self.parallelism_context.activate_spmd():
            h = self.model.forward_embeddings(
                inputs_embeds,
                positions,
                **self._prepare_step_kwargs(inputs_embeds.shape[0]),
            )
        return h

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """vLLM required API: vLLM calls it once per request of an M-RoPE model for the
        prompt's ``(3, num_tokens)`` positions and the offset ``position - token index`` of
        every token after the prompt.

        `get_mrope_positions` computes them from each image's placeholder offset and grid,
        as the batcher does for the trainer, so the generator's and the trainer's positions
        match.
        """
        vision_items = []
        for feature in mm_features:
            # `image_prompt_to_engine_input` submits only images, and the processor cache
            # is off, so vLLM keeps each feature's data.
            assert (
                feature.modality == "image" and feature.data is not None
            ), feature.modality
            t, h, w = feature.data["image_grid_thw"].data.tolist()
            merge = self._vision_processor.config.spatial_merge_size
            vision_items.append(
                (feature.mm_position.offset, (t, h // merge, w // merge))
            )
        positions = get_mrope_positions(len(input_tokens), vision_items)
        delta = int(positions.max()) + 1 - len(input_tokens)
        return positions.transpose(0, 1), delta


def register_multimodal_processor(
    model_cls: type, vision_processor: VisionProcessor
) -> None:
    """Register `_ImageMultiModalProcessor` with vLLM for ``model_cls``, instead of the
    Hugging Face processor vLLM would run.

    Required: vLLM's engine build raises "has no registered multimodal processor" for a
    model class that implements ``SupportsMultiModal`` without one. vLLM uses this one
    only at startup: ``vision_processor``'s largest image sizes the encoder budget and the
    encoder cache, and a dummy image of that size profiles the encoder's memory. Prompts
    bypass it (`image_prompt_to_engine_input`).
    """
    MULTIMODAL_REGISTRY.register_processor(
        _ImageMultiModalProcessor,
        info=functools.partial(_ImageProcessingInfo, vision_processor=vision_processor),
        dummy_inputs=_ImageDummyInputsBuilder,
    )(model_cls)
