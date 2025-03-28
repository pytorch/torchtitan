# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

import torch

import torchvision

from torchtitan.datasets.tokenizer.tiktoken import IMAGE_TOKEN_ID, TikTokenizer
from torchtitan.tools.logging import logger
from torchvision.transforms.v2 import functional as F

from utils import (
    find_supported_resolutions,
    get_canvas_best_fit,
    resize_with_pad,
    tile_crop,
)

class CLIPTransform:
    """
    This class accepts images of any size and dynamically resizes, pads, normalizes and tiles it
    based on the image aspect ratio and the number of image tiles we allow.

    The algorithm will NOT distort the image to fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    The user can choose if they want to allow upscaling by using the flag ``resize_to_max_canvas``.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image tiles, with side 224px, then:

    If ``resize_to_max_canvas=False``, then:
    best_resolution = (448, 896) -> smallest canvas, up to 16 tiles, that doesn't require downscaling
    image is NOT resized
    image is padded (300, 800) -> 448,896
    Image is tiled 2x4, for a final output shape of (8, 3, 224, 224)

    If ``resize_to_max_canvas=True``, then:
    best_resolution = (448, 1344) # canvas that allows maximum upscaling, with minimum padding, up to 16 tiles
    image is resized without distortion (300,800) -> (448, 1194) #448 is the limiting side for the resize
    image is padded (448, 1194) -> (448, 1344)
    Image is tiled 2x6, for a final output shape of (10, 3, 224, 224)

    Args:
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed. Default None.
        image_std (Optional[List[float]]): Standard deviation values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed. Default None.
        possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
            where each tuple represents a possible canvas to fit the image into when calling ``get_canvas_best_fit``.
            If None, this will be calculated using max_num_tiles and tile_size. Default None.
        tile_size (int): Size of the tiles to divide the image into. Default 224.
        max_num_tiles (Optional[int]): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
            Default 4.
        dtype (torch.dtype): Data type of the output image. Default torch.bfloat16.
        resample (str): Resampling method used when resizing images. Supports any enum of
            ``torchvision.transforms.InterpolationMode``, e.g. "nearest", "nearest_exact", "bilinear", "bicubic".
            Default 'bilinear'.
        resize_to_max_canvas (bool): "If True, the image will be upscaled without distortion to fit the largest possible
            resolution from possible_resolutions.
            If False, it will pick the resolution that minimizes downscaling, including no downscaling at all.
            In this case, the image will only be upscaled if it's size < tile_size. Default False.

    Examples:
        >>> image_transform = CLIPImageTransform(
        ...    image_mean=None,
        ...    image_std=None,
        ...    tile_size=224,
        ...    possible_resolutions=None,
        ...    max_num_tiles=4,
        ...    resample="bilinear",
        ...    resize_to_max_canvas=True,
        ...)
        >>> # create random image
        >>> image = (np.random.rand(100,200,3) * 255).astype(np.uint8)
        >>> image = PIL.Image.fromarray(image)
        >>> output = image_transform(image)
        >>> output['image'].shape # [num_tiles, num_channels, tile_size, tile_size]
        torch.Size([2, 3, 224, 224])
        >>> output['ar'] # image best fits the canvas 224x448
        torch.tensor([1,2])
    """

    def __init__(
        self,
        *,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        possible_resolutions: Optional[List[Tuple[int, int]]] = None,
        tile_size: int = 224,
        max_num_tiles: Optional[int] = 4,
        dtype: torch.dtype = torch.bfloat16,
        resample: str = "bilinear",
        resize_to_max_canvas: bool = False,
    ) -> None:

        # get_canvas_best_fit
        assert (
            possible_resolutions is not None or max_num_tiles is not None
        ), f"Either possible_resolutions or max_num_tiles must be given. Got {possible_resolutions} and {max_num_tiles}"

        # If possible_resolutions are not given, then calculate possible ones based on max_num_tiles
        if not possible_resolutions and max_num_tiles:
            possible_resolutions = find_supported_resolutions(
                max_num_tiles=max_num_tiles, tile_size=tile_size
            )
        else:
            possible_resolutions = possible_resolutions

        self.possible_resolutions = torch.tensor(possible_resolutions).reshape(-1, 2)
        logger.debug(
            f"Found possible_resolutions: {self.possible_resolutions}. Will fit the images into the canvas with best fit."
        )

        self.resize_to_max_canvas = resize_to_max_canvas

        # normalize
        assert (image_mean is None) == (
            image_std is None
        ), f"Need to provide both or none of image_mean and image_std. Got {image_mean=} and {image_std=}"
        self.mean = image_mean
        self.std = image_std

        # resize_with_pad
        self.max_size = None if resize_to_max_canvas else tile_size
        self.dtype = dtype
        self.resample = torchvision.transforms.InterpolationMode[resample.upper()]

        # tile_crop
        self.tile_size = tile_size

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Apply image decoding and transformations to the "image" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with an "image" field containing
                a List[Message] to tokenize

        Returns:
            Mapping[str, Any]: The sample with an updated "image" filed and added
                "aspect_ratio" field.
        """
        image = sample["image"]
        assert isinstance(image, torch.Tensor), "Input image must be a torch.Tensor."

        image = F.to_image(image)
        image = F.grayscale_to_rgb_image(image)
        image = F.to_dtype(image, dtype=self.dtype, scale=True)

        # Find the best canvas to fit the image without distortion
        best_resolution = get_canvas_best_fit(
            image=image,
            possible_resolutions=self.possible_resolutions,
            resize_to_max_canvas=self.resize_to_max_canvas,
        )

        # resize without distortion + pad to fit best_resolution
        image = resize_with_pad(
            image=image,
            target_size=best_resolution,
            resample=self.resample,
            max_size=self.max_size,
        )

        # Normalize
        if self.mean:
            image = F.normalize(image, mean=self.mean, std=self.std)

        # Divide the image into equally sized tiles
        image = tile_crop(image=image, tile_size=self.tile_size)

        aspect_ratio = torch.tensor(best_resolution).reshape(-1) // self.tile_size

        sample.update(
            {
                "image": image,
                "aspect_ratio": aspect_ratio,
            }
        )

        return sample


class VisionFormatter:
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
        max_num_tiles: int = 4,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
    ):
        self.tokenizer = tokenizer

        self.transform_image = CLIPTransform(
            image_mean=image_mean,
            image_std=image_std,
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )
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
        return sample
