# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Any, Tuple

import torch
import torch.nn as nn


class TilePositionEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    For details, please check the documentation of :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        emb_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(
        self,
        max_num_tiles: int,
        emb_dim: int,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.emb_dim = emb_dim
        self.embedding = nn.Parameter(
            torch.randn(max_num_tiles, max_num_tiles, 1, emb_dim) / math.sqrt(emb_dim)
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor):
        """
        args:
            x (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, num_tiles, num_tokens, emb_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_num_imgs, num_tiles, num_tokens, emb_dim = x.shape

        for batch_idx, (num_tiles_h, num_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            num_non_padded_tiles = int(num_tiles_h * num_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. num_tiles_h, num_tiles_w.
            pos_embed = self.embedding[:num_tiles_h, :num_tiles_w, :, :]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(num_non_padded_tiles, 1, self.emb_dim)
            x[batch_idx, :num_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()

        return x


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Args:
        emb_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, emb_dim: int, tile_size: int, patch_size: int) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        scale = emb_dim**-0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((patch_grid_size**2 + 1, emb_dim))  # +1 for CLS token
        )

    def forward(self, x: torch.Tensor, *args: Tuple[Any]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., num_tokens, emb_dim)
            *args (Tuple[Any]): Optional args.

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """

    Token positional embedding for tiled images. There are two positional embeddings in this module:

    * local_token_positional_embedding: same for every tile, different for every token. Equivalent \
        to :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        emb_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(
        self, max_num_tiles: int, emb_dim: int, tile_size: int, patch_size: int
    ) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        self.num_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token
        scale = emb_dim**-0.5

        # different for every token, same for every tile
        self.local_token_positional_embedding = nn.Parameter(
            scale * torch.randn((patch_grid_size**2 + 1, emb_dim))  # +1 for CLS token
        )

        # different for every token, different for every tile
        self.global_token_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.num_tokens_per_tile,
                emb_dim,
            )
        )

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, num_tiles, num_tokens, emb_dim).
            aspect_ratio (torch.Tensor): torch.Tensor with shape (bsz * num_imgs, 2),
                where aspect_ratio[k] represents the aspect ratio of the k^th image
                of the batch before tile-cropping,  e.g. aspect_ratio[k] = (2,1).
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_num_imgs, num_tiles, num_tokens, emb_dim = x.shape

        # apply local position embedding (same for every tile)
        x = x + (self.local_token_positional_embedding * (1 - self.gate.tanh()))

        # apply global positional embedding (different for every tile)
        x = x.view(bsz_and_num_imgs, num_tiles, num_tokens, emb_dim)
        for batch_idx, (num_tiles_h, num_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            num_non_padded_tiles = int(num_tiles_h * num_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. num_tiles_h, num_tiles_w.
            pos_embed = self.global_token_positional_embedding[
                :num_tiles_h, :num_tiles_w, :, :
            ]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(
                num_non_padded_tiles, self.num_tokens_per_tile, emb_dim
            )
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :num_non_padded_tiles, :, :] += pos_embed

        return x
