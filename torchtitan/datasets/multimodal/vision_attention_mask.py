# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional

import torch

# NOTE Inspired from torchtune.modules.transforms._transforms.py
class VisionCrossAttentionMask:
    """
    Computes the cross-attention mask for text + image inputs. Text tokens that
    participate in cross-attention with an image token will show True in the mask
    and follow the interleaved structure laid out in Fig. 7 of the Flamingo paper
    (https://arxiv.org/pdf/2204.14198):

        (1) Text tokens immediately following the image token up until the next image token
        (2) Consecutive image tokens attend to subsequent text tokens

    ::

             ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
        img1 │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │   │ │   │ │   │ │   │ │   │
             └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
             ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
        img2 │   │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │   │ │   │ │   │ │   │ │   │
             └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
             ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
        img3 │   │ │   │ │   │ │   │ │   │ │   │ │ ■ │ │ ■ │ │ ■ │ │ ■ │ │ ■ │
             └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
            <img1> <img2>These  are   two  dogs. <img3> This   is    a    cat.



    Resultant mask is constructed per image and is of shape (text_seq_len, image_seq_len),
    where True indicates that the token outputted from the image encoder attends
    to the token in the text sequence in cross-attention. A list of these masks
    are returned with length equal to number of images in the sample.

    Args:
        tile_size (int): The size of the image tiles from the image transform
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        image_token_id (int): Token ID of the image special token.
        max_num_tiles (Optional[int]): Maximum number of tiles in an image, used to
            pad mask during inference. Defaults to None
    """

    def __init__(
        self,
        tile_size: int,
        patch_size: int,
        image_token_id: int,
        max_num_tiles: Optional[int] = None,
    ):
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.image_token_id = image_token_id
        self.max_num_tiles = max_num_tiles

    def _get_image_attention_intervals(self, tokens: List[int]) -> List[List[int]]:
        """
        Returns a list of lists of the form [start, end) where start is the index
        of the current image token and end is the index of the next image token, exclusive.

        Args:
            tokens (List[int]): List of token IDs in the text sequence

        Returns:
            List[List[int]]: List of lists of the form [start, end) indicating
                range of positions in text sequence that should attend to the image

        Example:
            >>> text = "<img1><img2>These are two dogs. <img3>This is a cat."
            >>> image_token_id = 1
            >>> tokens = [1, 1, 9673, 527, 1403, 12875, 13, 1, 1115, 374, 264, 8415]
            >>> transform = VisionCrossAttentionMask(tile_size=400, patch_size=40, image_token_id=1)
            >>> intervals = transform._get_image_attention_intervals(tokens)
            >>> print(intervals)
            [[0, 7], [1, 7], [7, 12]]
        """
        end = len(tokens)
        vision_token_locations = [
            i for i, token in enumerate(tokens) if token == self.image_token_id
        ]
        # Return empty list if there are no images
        if len(vision_token_locations) == 0:
            return []
        # If there is only one image, it will attend to subsequent text until end
        if len(vision_token_locations) == 1:
            return [[vision_token_locations[0], end]]

        # Construct intervals from previous image token to next image token
        vision_masks = [
            [tok_idx_prev, tok_idx_next]
            # Offset by one to get consecutive indices
            for tok_idx_prev, tok_idx_next in zip(
                vision_token_locations[:-1], vision_token_locations[1:]
            )
        ]
        # Last image will attend to subsequent text until end
        vision_masks.append([vision_token_locations[-1], end])

        # If there are consecutive vision tokens, they should all attend to the
        # same subsequent text
        last_mask_end = vision_masks[-1][1]
        for vision_mask in vision_masks[::-1]:
            if vision_mask[0] == vision_mask[1] - 1:
                vision_mask[1] = last_mask_end
            last_mask_end = vision_mask[1]
        return vision_masks

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Generates the vision cross-attention mask for the given sample based on
        the image token locations interleaved in the text sequence.

        Args:
            sample (Mapping[str, Any]): Sample dict containing the following keys:
                - tokens (List[int]): List of token IDs in the text sequence. Number of
                    image token IDs in the sequence must match the number of images.
                - images (List[torch.Tensor]): List of image Tensors post-tiling of shape
                    (n_tiles, c, h, w) each.

        Returns:
            Mapping[str, Any]: sample with a new key encoder_mask, with a mask per image with shape
                (text_seq_len, image_seq_len) where text_seq_len == len(tokens) and
                image_seq_len == max_tiles * (patches_per_tile + 1). These masks get padded and concatenated
                in the batch collator.

        Raises:
            RuntimeError: if the number of images in the batch does not match the number of image tokens in the batch.
        """
        tokens, images = sample["tokens"], sample["encoder_input"]["images"]
        # One sample can have multiple images - verify the number of image tokens
        # is the same
        n_img = len(images)
        intervals = self._get_image_attention_intervals(tokens)
        if len(intervals) != n_img:
            raise RuntimeError(
                f"The number of image tokens ({len(intervals)}) does not match the number of images ({n_img})."
            )

        # Create mask for each individual image based on its number of tokens,
        # which can vary based on number of tiles since they are not yet tile padded.
        # The masks are padded and concatenated together in the batch collator
        text_seq_len = len(tokens)
        max_image_size = None
        masks = []
        for image_num, interval in enumerate(intervals):
            # Identify what part of text sequence should be attended
            start, end = interval
            # Compute this image's number of tokens based on num tiles, patches per tile
            n_tiles = images[image_num].shape[0]
            image_seq_len = n_tiles * (self.patches_per_tile + 1)  # +1 for CLS token
            # Mask will be block of 1s at the corresponding interval in the text.
            # It is not a causal block because all the image tokens correspond
            # to a single image, so text tokens attend to all the image's tokens.
            # The mask is text_seq_len x mask_image_size if defined, otherwise
            # it uses current text/image sequence lengths.
            mask = torch.zeros(
                text_seq_len, max_image_size or image_seq_len, dtype=torch.bool
            )
            mask[start:end, :image_seq_len] = True
            masks.append(mask)

        sample.update({"encoder_mask": masks})
        return sample
