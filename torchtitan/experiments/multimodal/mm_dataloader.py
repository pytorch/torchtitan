# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader


class ParallelAwareDataloaderWithCollator(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Callable
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(dataset, batch_size, collate_fn=collate_fn) # TODO(tj.solergibert) Delete collate_fn=?
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning( # NOTE(tj.solergibert) Missing import?
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))

def padded_collate_sft(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = -100,  # NOTE(tj.solergibert) Hardcoded!
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
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
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
    return {"tokens": input_ids.long(), "labels": labels.long()}


# NOTE Inspired from torchtune.data._collate.py
@dataclass
class MultiModalCollator:
    padding_idx: int = 0
    ignore_idx: int = -100  # NOTE(tj.solergibert) Hardcoded!
    pad_max_tiles: Optional[int] = None
    pad_max_images: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad a batch of text sequences, tiled image tensors, aspect ratios,
        and cross attention masks. This can be used for both training and inference.

        ``batch`` is expected to be a list of sample dicts containing the following::
            - "tokens": List[int] of length text_seq_len, varies across samples
            - "labels": List[int] of length text_seq_len, varies across samples
            - "encoder_input": Dict[str, List[torch.Tensor]]
                - "images": List[torch.Tensor], each with shape (n_tiles, c, h, w)
                - "aspect_ratio": List[torch.Tensor], each with shape (2, ) to indicate h_ratio, w_ratio
            - "encoder_mask": List[Tensor], each with shape (text_seq_len, image_seq_len)

        Shape notation:
            - c = channel dim
            - h = height dim
            - w = weight dim

        Note:
            For each element in the batch, ``len(images) == len(encoder_mask) == len(aspect_ratio)``.

        This collater does the following:
            (1) Pad text sequence and encoder mask to the longest sequence length in the batch
            (2) Pad image tensors in the tile dimension with zeros to the largest number
                of tiles in the batch
            (3) Add empty images of zeros to samples up to max number of images in the batch
            (4) Pad aspect ratios with (1,1) for all added padding images

        Args:
            batch (List[Dict[str, Any]]): A list of sample dicts containing tokens,
                labels, images, encoder_mask, and aspect_ratio.
            padding_idx (int): Padding index for input token ids. Defaults to 0.
            ignore_idx (int): Padding index for labels. Defaults to -100.
            pad_max_tiles (Optional[int]): Maximum number of tiles to pad to. If None, will pad to the largest number of tiles
                in the batch. Defaults to None.
            pad_max_images (Optional[int]): Maximum number of images to pad to. If None, will pad to the largest number of images
                in the batch. Defaults to None.

        Returns:
            Dict[str, Tensor]: Collated tokens, labels, images, encoder_mask, aspect_ratio tensors.
                - tokens: Tensor of shape (bsz, max_seq_len)
                - labels: Tensor of shape (bsz, max_seq_len)
                - images: Tensor of shape (bsz, max_num_images, max_num_tiles, c, h, w)
                - encoder_mask: Tensor of shape (bsz, max_seq_len, tokens_per_tile * max_num_tiles * max_num_images)
                - aspect_ratio: Tensor of shape (bsz, max_num_images, 2)

        Raises:
            ValueError: if pad_max_tiles is set to a value less than the largest number of tiles in an image.

        Example:
            >>> image_id = 1
            >>> tokens_per_tile = 5
            >>> c, h, w = 1, 1, 1
            >>> batch = [
            ...     {
            ...         "tokens": [1, 2, 1, 3], "labels": [4, 5, 6, 7],
            ...         "encoder_input": {
            ...             # One image with two tiles, one image with three tiles
            ...             "images": [torch.ones(2, c, h, w), torch.ones(3, c, h, w)],
            ...             "aspect_ratio": [torch.tensor([1, 2]), torch.tensor([1, 3])],
            ...         },
            ...         # Mask is shape (text_seq_len, tokens_per_tile * n_tiles)
            ...         "encoder_mask": [torch.ones(4, 5 * 2), torch.ones(4, 5 * 3)],
            ...     },
            ...     {
            ...         "tokens": [1, 4], "labels": [8, 9],
            ...         "encoder_input": {
            ...             # One image with four tiles
            ...             "images": [torch.ones(4, c, h, w)],
            ...             "aspect_ratio": [torch.tensor([2, 2])],
            ...         },
            ...         # Mask is shape (text_seq_len, tokens_per_tile * n_tiles)
            ...         "encoder_mask": [torch.ones(2, 5 * 4)],
            ...     },
            ... ]
            >>> model_inputs = padded_collate_tiled_images_and_mask(batch=batch)
            >>> print(model_inputs["tokens"])
            tensor([[1, 2, 1, 3],
                    [1, 4, 0, 0]])
            >>> print(model_inputs["labels"])
            tensor([[4, 5, 6, 7],
                    [8, 9, -100, -100]])
            >>> print(model_inputs["encoder_input"]["images"].shape)  # (bsz, max_num_images, max_num_tiles, c, h, w)
            torch.Size([2, 2, 4, 1, 1, 1])
            >>> print(model_inputs["encoder_mask"].shape)
            >>> # (bsz, max_text_seq_len, tokens_per_tile * max_num_tiles * max_num_images)
            torch.Size([2, 4, 40])
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
            {"tokens": sample["tokens"], "labels": sample["labels"]} for sample in batch
        ]
        collated_text = padded_collate_sft(text_only, self.padding_idx, self.ignore_idx)

        max_seq_len = collated_text["tokens"].shape[-1]
        bsz = len(batch)

        # TODO: Figure out how to make this more efficient or vectorized. Setting
        # max_num_tiles beforehand will save one nested for loop but may incur more
        # memory and compute costs in attention if max_num_tiles > batch_max_num_tiles

        if self.pad_max_tiles is None:
            # Get max number of tiles in batch
            max_num_tiles = max(
                image.shape[0]
                for sample in batch
                for image in sample["encoder_input"]["images"]
            )
            if self.pad_max_tiles < max_num_tiles:
                raise ValueError(
                    f"More tiles in image {max_num_tiles}, than pad_max_tiles {self.pad_max_tiles}"
                )
        max_num_tiles = self.pad_max_tiles

        # Second loop: pad images and masks to max number of tiles, max text seq len in batch
        batch_images = []
        batch_masks = []
        batch_aspect_ratios = []
        token_len = []  # DEBUG(tj.solergibert)
        image_len = []  # DEBUG(tj.solergibert)
        tile_len = []  # DEBUG(tj.solergibert)
        for sample in batch:
            sample_images = []
            sample_masks = []
            token_len.append(len(sample["tokens"]))  # DEBUG(tj.solergibert)
            image_len.append(
                len(sample["encoder_input"]["images"])
            )  # DEBUG(tj.solergibert)
            tmp_tile_len = []  # DEBUG(tj.solergibert)
            for image, mask in zip(
                sample["encoder_input"]["images"], sample["encoder_mask"]
            ):
                # Single image in each sample has shape (n_tiles, c, h, w)
                n_tiles = image.shape[0]
                tmp_tile_len.append(n_tiles)  # DEBUG(tj.solergibert)
                # Single mask in each sample corresponds to a single image and has shape (text_seq_len, image_seq_len)
                # where image_seq_len = n_tiles * tokens_per_tile
                text_seq_len, image_seq_len = mask.shape
                tokens_per_tile = image_seq_len // n_tiles
                padding_tiles = max_num_tiles - n_tiles

                # Image should now have shape (max_num_tiles, c, h, w)
                padded_image = F.pad(
                    image, (0, 0, 0, 0, 0, 0, 0, padding_tiles), value=0
                )
                # Mask should now have shape (max_seq_len, max_image_seq_len), where
                # max_image_seq_len = max_num_tiles * tokens_per_tile
                padded_mask = F.pad(
                    mask,
                    (
                        0,
                        padding_tiles * tokens_per_tile,
                        0,
                        max_seq_len - text_seq_len,
                    ),
                    value=0,
                )

                sample_images.append(padded_image)
                sample_masks.append(padded_mask)
            tile_len.append(tmp_tile_len)  # DEBUG(tj.solergibert)
            # Stack multiple images and masks per sample in num_images dimension
            batch_images.append(torch.stack(sample_images))
            batch_masks.append(torch.stack(sample_masks))
            batch_aspect_ratios.append(
                torch.stack(sample["encoder_input"]["aspect_ratio"])
            )
        # Finally, pad images, masks, aspect ratios to max number of images in batch
        # (bsz, max_num_images, max_num_tiles, c, h, w)
        collated_images = pad_sequence(batch_images, batch_first=True, padding_value=0)
        # (bsz, max_num_images, max_seq_len, max_image_seq_len)
        collated_masks = pad_sequence(batch_masks, batch_first=True, padding_value=0)
        # (bsz, max_num_images, 2)
        collated_aspect_ratios = pad_sequence(
            batch_aspect_ratios, batch_first=True, padding_value=1
        )

        # Concatenate masks for multiple images across image_seq_len dimension
        concat_masks = collated_masks.view(bsz, max_seq_len, -1)
        if self.pad_max_images is not None:
            _, _, img_seq = concat_masks.shape
            concat_masks = F.pad(
                concat_masks, (0, self.pad_max_images * image_seq_len - img_seq)
            )

        batch_dict = {
            "tokens": collated_text["tokens"],
            "labels": collated_text["labels"],
            "encoder_input": {
                "images": collated_images,
                "aspect_ratio": collated_aspect_ratios,
            },
            "encoder_mask": concat_masks,
            "token_len": torch.tensor(token_len),  # DEBUG(tj.solergibert)
            "image_len": torch.tensor(image_len),  # DEBUG(tj.solergibert)
            "tile_len": tile_len,  # DEBUG(tj.solergibert)
        }

        return batch_dict
