# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from tokenizer.tiktoken import IGNORE_INDEX

from torch.nn.utils.rnn import pad_sequence


def padded_collate(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    {"input_ids": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"input_ids": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["input_ids"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    input_ids = pad_sequence(
        [x["input_ids"] for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [x["labels"] for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    return {"input_ids": input_ids, "labels": labels}


# NOTE Inspired from torchtune.data._collate.py
@dataclass
class MultiModalCollator:
    padding_idx: int = 128004
    ignore_idx: int = IGNORE_INDEX
    pad_max_tiles: Optional[int] = None
    pad_max_images: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad a batch of text sequences, tiled image tensors, aspect ratios,
        and cross attention masks. This can be used for both training and inference.

        ``batch`` is expected to be a list of sample dicts containing the following::
            - "input_ids": List[int] of length text_seq_len, varies across samples
            - "labels": List[int] of length text_seq_len, varies across samples
            - "encoder_input": Dict[str, List[torch.Tensor]]
                - "images": List[torch.Tensor], each with shape (n_tiles, c, h, w)
                - "aspect_ratio": List[torch.Tensor], each with shape (2, ) to indicate h_ratio, w_ratio

        Shape notation:
            - c = channel dim
            - h = height dim
            - w = weight dim

        Note:
            For each element in the batch, ``len(images) == len(aspect_ratio)``.

        This collater does the following:
            (1) Pad text sequence and encoder mask to the longest sequence length in the batch
            (2) Pad image tensors in the tile dimension with zeros to the largest number
                of tiles in the batch
            (3) Add empty images of zeros to samples up to max number of images in the batch
            (4) Pad aspect ratios with (1,1) for all added padding images

        Args:
            batch (List[Dict[str, Any]]): A list of sample dicts containing input_ids,
                labels, images, and aspect_ratio.
            padding_idx (int): Padding index for input token ids. Defaults to 0.
            ignore_idx (int): Padding index for labels. Defaults to -100.
            pad_max_tiles (Optional[int]): Maximum number of tiles to pad to. If None, will pad to the largest number of tiles
                in the batch. Defaults to None.
            pad_max_images (Optional[int]): Maximum number of images to pad to. If None, will pad to the largest number of images
                in the batch. Defaults to None.

        Returns:
            Dict[str, Tensor]: Collated tokens, labels, images, aspect_ratio tensors.
                - tokens: Tensor of shape (bsz, max_seq_len)
                - labels: Tensor of shape (bsz, max_seq_len)
                - images: Tensor of shape (bsz, max_num_images, max_num_tiles, c, h, w)
                - aspect_ratio: Tensor of shape (bsz, max_num_images, 2)

        Example:
            >>> image_id = 1
            >>> tokens_per_tile = 5
            >>> c, h, w = 1, 1, 1
            >>> batch = [
            ...     {
            ...         "input_ids": [1, 2, 1, 3], "labels": [4, 5, 6, 7],
            ...         "encoder_input": {
            ...             # One image with two tiles, one image with three tiles
            ...             "images": [torch.ones(2, c, h, w), torch.ones(3, c, h, w)],
            ...             "aspect_ratio": [torch.tensor([1, 2]), torch.tensor([1, 3])],
            ...         },
            ...     },
            ...     {
            ...         "input_ids": [1, 4], "labels": [8, 9],
            ...         "encoder_input": {
            ...             # One image with four tiles
            ...             "images": [torch.ones(4, c, h, w)],
            ...             "aspect_ratio": [torch.tensor([2, 2])],
            ...         },
            ...     },
            ... ]
            ... collator = MultiModalCollator(pad_max_tiles=4)
            >>> model_inputs = collator(batch=batch)
            >>> print(model_inputs["input_ids"])
            tensor([[1, 2, 1, 3],
                    [1, 4, 0, 0]])
            >>> print(model_inputs["labels"])
            tensor([[4, 5, 6, 7],
                    [8, 9, -100, -100]])
            >>> print(model_inputs["encoder_input"]["images"].shape)  # (bsz, max_num_images, max_num_tiles, c, h, w)
            torch.Size([2, 2, 4, 1, 1, 1])
            >>> print(model_inputs["encoder_input"]["aspect_ratio"].shape)  # (bsz, max_num_images, 2)
            torch.Size([2, 2, 2])
            >>> print(model_inputs["encoder_input"]["images"][0, 0, ...])  # Image with two tiles got padded to four
            tensor([[[[1.]]], [[[1.]]], [[[0.]]], [[[0.]]]])
            >>> print(model_inputs["encoder_input"]["images"][0, 1, ...])  # Image with three tiles got padded to four
            tensor([[[[1.]]], [[[1.]]], [[[1.]]], [[[0.]]]])
            >>> print(model_inputs["encoder_input"]["images"][1, 0, ...])  # Image with four tiles did not get padded
            tensor([[[[1.]]], [[[1.]]], [[[1.]]], [[[1.]]]])
            >>> print(model_inputs["encoder_input"]["images"][1, 1, ...])  # Extra padding image was added to second sample
            tensor([[[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]]])
        """
        # Text tokens can be handled independently by existing collaters
        text_only = [
            {"input_ids": sample["input_ids"], "labels": sample["labels"]}
            for sample in batch
        ]
        collated_text = padded_collate(text_only, self.padding_idx, self.ignore_idx)

        if self.pad_max_tiles is None:
            # Get max number of tiles in batch
            max_num_tiles = max(sample["images_tiles"].shape[0] for sample in batch)
        else:
            max_num_tiles = self.pad_max_tiles

        # Pad images and aspect ratios to max number of tiles
        batch_images = []
        batch_aspect_ratios = []

        for sample in batch:
            sample_images = []
            for image in sample["encoder_input"]["images"]:
                # Single image in each sample has shape (n_tiles, c, h, w)
                n_tiles = image.shape[0]
                # Single mask in each sample corresponds to a single image and has shape (text_seq_len, image_seq_len)
                # where image_seq_len = n_tiles * tokens_per_tile
                padding_tiles = max_num_tiles - n_tiles

                # Image should now have shape (max_num_tiles, c, h, w)
                padded_image = F.pad(
                    image, (0, 0, 0, 0, 0, 0, 0, padding_tiles), value=0
                )

                sample_images.append(padded_image)
            # Stack multiple images and masks per sample in num_images dimension
            batch_images.append(torch.stack(sample_images))
            batch_aspect_ratios.append(
                torch.stack(sample["encoder_input"]["aspect_ratio"])
            )
        # Finally, pad images, masks, aspect ratios to max number of images in batch
        # (bsz, max_num_images, max_num_tiles, c, h, w)
        collated_images = pad_sequence(batch_images, batch_first=True, padding_value=0)
        # (bsz, max_num_images, 2)
        collated_aspect_ratios = pad_sequence(
            batch_aspect_ratios, batch_first=True, padding_value=1
        )

        batch_dict = {
            "input_ids": collated_text["input_ids"],
            "labels": collated_text["labels"],
            "encoder_input": {
                "images": collated_images,
                "aspect_ratio": collated_aspect_ratios,
            },
        }

        return batch_dict
