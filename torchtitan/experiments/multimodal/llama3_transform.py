# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Optional, Tuple

from torchtitan.datasets.tokenizer.tiktoken import IMAGE_TOKEN_ID, TikTokenizer

from clip import CLIPPreprocess
from vision_attention_mask import VisionCrossAttentionMask

# NOTE Inspired from torchtune.models.llama3_2_vision._transform.py
class Llama3VisionFormatter:
    """
    This class combines the transforms for the different modalities of Llama 3.2 Vision. It
    performs the following transforms:
    - Tokenizing the text field using :class:`torchtitan.datasets.tokenizer.titoken.TikTokenizer`
    - Preprocessing the images for the CLIP encoder using :class:`torchtitan.experiments.multimodal.clip.ClipPreprocess`
    - Generating the Vision Cross Attention mask for the Fused layers
        using :class:`torchtitan.datasets.multimodal.utils.VisionCrossAttentionMask`

    Args:
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode_multimodal` method.
        tile_size (int): Size of the tiles to divide the image into.
        patch_size (int): Size of the patches used in the CLIP vision tranformer model. This is
            used to calculate the number of image embeddings per image.
        max_num_tiles (int): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
            Default 4.
        image_mean (Optional[Tuple[float, float, float]]): Mean values of each channel, used for normalization.
        image_std (Optional[Tuple[float, float, float]]): Standard deviations for each channel, used for normalization.

    Examples:
        >>> model_transform = Llama3VisionFormatter(tokenizer, tile_size=224, patch_size=14)
        >>> transformed_data = model_transform({"messages": user_message, "images": [img1, img2]})
        >>> print(transformed_data["tokens"])
        [1, 31587, 29644, 102, 2]
        >>> print(transformed_data["images"][0].shape)
        torch.Size([4, 3, 224, 224])
    """

    def __init__(
        self,
        tokenizer: TikTokenizer,
        tile_size: int,
        patch_size: int,
        max_num_tiles: int = 4,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
    ):
        self.tokenizer = tokenizer

        self.transform_image = CLIPPreprocess(
            image_mean=image_mean,
            image_std=image_std,
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )
        self.xattn_mask = VisionCrossAttentionMask(
            tile_size=tile_size,
            patch_size=patch_size,
            image_token_id=IMAGE_TOKEN_ID, 
            max_num_tiles=max_num_tiles,
        )

        self.image_seq_len = max_num_tiles * (self.xattn_mask.patches_per_tile + 1)
        # TODO(tj.solergibert) self.pad_id = self.tokenizer.pad_id

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Apply image decoding, transformations and tokenization to messages in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field.

        Returns:
            Mapping[str, Any]: The transformed sample with the following fields:
                - tokens: List[int] of tokenized messages
                - encoder_input: Dict[str, Any] of transformed images
                - encoder_mask: List[bool] of masks for the transformed images
        """
        encoder_input = {"images": [], "aspect_ratio": []}
        for image in sample["images"]:
            out = self.transform_image({"image": image})
            encoder_input["images"].append(out["image"])
            encoder_input["aspect_ratio"].append(out["aspect_ratio"])

        sample["encoder_input"] = encoder_input
        sample = self.tokenizer.encode_multimodal(sample)
        # TODO(tj.solergibert) What should we do (Include y/n & Mask y/n) with both bos & eos
        # TODO(tj.solergibert) allowed_special to this fancy set OR set it to "all"?
        sample = self.xattn_mask(sample)
        return sample
