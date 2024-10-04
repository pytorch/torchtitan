# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import List, Optional

import torch
import torch.nn as nn
from torchtitan.models.llama_multimodal.attention import Attention
from torchtitan.models.llama_multimodal.feed_forward import FeedForward
from torchtitan.models.llama_multimodal.norms import Fp32LayerNorm
from torchtitan.models.llama_multimodal.position_embeddings import (
    TiledTokenPositionalEmbedding,
    TilePositionEmbedding,
    TokenPositionalEmbedding,
)
from torchtitan.models.llama_multimodal.tanh_gate import TanhGate


class Conv2dModule(torch.nn.Module):
    """Conv2D Module.
    This is a equivalent to Conv2D in PyTorch, but has support better flexibility
    in parallelism. The output of this module is slightly different from PyTorch's
    Conv2D. While the latter has output to be (*, out_channels, h_out, w_out),
    this module just has output to be (*, h_out * w_out, out_channels).

    More details can be found in the PyTorch documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    We might later column-wise parallelize over unfolded. And we assume that
    the shape of the kernel is always square.

    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._unfold = torch.nn.Unfold(
            kernel_size=(kernel_size, kernel_size), stride=stride
        )
        self._linear = torch.nn.Linear(
            in_channels * kernel_size * kernel_size,
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (bsz, in_channels, width, height)
        # Output: (bsz, in_channels * kernel_size * kernel_size, num_tokens)
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        # Output: (bsz, num_tokens, out_channels), when stride = kernel_size,
        # num_tokens = grid ** 2 and out_channels is emd_dim.
        return self._linear(x)


class VitTransformerBlock(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
        attn_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.attn = Attention(model_args)
        self.ln_attn = Fp32LayerNorm(model_args.encoder_embed_dim, eps=1e-5)
        self.mlp = FeedForward(
            dim=model_args.encoder_embed_dim,
            hidden_dim=4 * model_args.encoder_embed_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
            activation=model_args.activation,
            enable_w3=False,
        )
        self.ln_mlp = Fp32LayerNorm(model_args.encoder_embed_dim, eps=1e-5)
        self.attn_scale = attn_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, emd_dim = x.shape
        x = x.view(bsz * seq_len, emd_dim)
        x = x + self.attn_scale(self.attn(x=self.ln_attn(x), freqs_cis=None))
        x = x + self.mlp_scale(self.mlp(self.ln_mlp(x)))
        return x.view(bsz, seq_len, emd_dim)


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile of an image in the beginning of each token.

    Args:
        emb_dim (int): The dimensionality of the input patch embedding.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        scale = emb_dim**-0.5
        self.weight = nn.Parameter(scale * torch.randn(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # add 1 CLS token to every tile
        bsz_and_num_imgs, num_tiles, _, emb_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_num_imgs, num_tiles, 1, emb_dim)
        return torch.cat([cls_emb, x], dim=2)


class Vit(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, outputting of hidden layers.

    (credit for the documentation below: `vision_transformer.py

    <https://github.com/pytorch/torchtune/blob/b4fea322189f16629264ee44826f2ac080e922ec/torchtune/modules/vision_transformer.py>`_).

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into **patches** by a convolution.
    These patches are flattened and subsequently treated as **tokens** by the transformer.

    To further enhance the performance of ViT and avoid downscaling images, we support tile-cropped images,
    which are images divided into **tiles** during the preprocessing stage. For example, instead of
    downscaling an 800x400 image to fit 400x400, we may crop it into two 400x400 tiles,
    if the ``tile_size=400``.

    Each of these tiles is further broken down into patches by a convolution operation. For example, if
    your ``patch_size=40``, then each (400, 400) tile will become a grid of 10x10 patches, and your whole image will have
    num_tiles * n_tokens -> num_tiles * (10x10 patches + 1 CLS token) -> num_tiles * 101.

    Before the transformer layers, a CLS token is added to each tile as the first token.
    In transformers, a token called CLS is a special token that is added to the beginning of each sequence.
    This token can be used to represent the whole input, instead of using a pooling operation, for example.

    To help the model "see" the whole image, we use positional embeddings. If your image
    was tile-cropped, then you need to use tile positional embeddings:

    - token_pos_embedding (tiled): :class:`torchtitan.models.llama_vision.position_embeddings.TiledTokenPositionalEmbedding`
    - pre_tile_pos_embed: :class:`torchtitan.models.llama_vision.position_embeddings.TilePositionalEmbedding`
    - post_tile_pos_embed: :class:`torchtitan.models.llama_vision.position_embeddings.TilePositionalEmbedding`

    Otherwise, pre and post tile_pos_embed should be None and all you need is a simple
    token positional embedding:

    - token_pos_embedding (not tiled): :class:`torchtitan.models.llama_vision.position_embeddings.TokenPositionalEmbedding`

    All images will be considered as a stack of tiles, even if your image was not tile-cropped. In such cases,
    your image would be composed of a single tile.

    In summary:

    1) An image is broken down into tiles during preprocessing.
    2) In the ViT, the tiles will be broken down into patches.
    3) The patches will be flattened and transformed. We call them tokens, because that's how the transformer sees them.

    Image: shape (8x8)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
        | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |

    Tiles: shape (4,4,4) # (num_tiles, tile_size, tile_size)

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  5 |  6 |  7 |  8 |
        |  9 | 10 | 11 | 12 |    | 13 | 14 | 15 | 16 |
        | 17 | 18 | 19 | 20 |    | 21 | 22 | 23 | 24 |
        | 25 | 26 | 27 | 28 |    | 29 | 30 | 31 | 32 |

        | 33 | 34 | 35 | 36 |    | 37 | 38 | 39 | 40 |
        | 41 | 42 | 43 | 44 |    | 45 | 46 | 47 | 48 |
        | 49 | 50 | 51 | 52 |    | 53 | 54 | 55 | 56 |
        | 57 | 58 | 59 | 60 |    | 61 | 62 | 63 | 64 |

    Patches: shape (4,4,2,2) # (num_tiles, num_patches_per_tile, patch_size, patch_size)

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    token: shape (4, 4, 4) # (num_tiles, num_patches_per_tile, emb_dim)

    .. code-block:: text

        |  1 |  2 |  9 |  10 |    |  3 |  4 |  11 |  12 |    |  17 |  18 |  25 |  26 |    | 19 | 20 |  27 |  28 |
        | ... continuation of data ...
        | ... continuation of data ...
        | 37 | 38 | 45 |  46 |    | 39 |  40 | 47 |  48 |    | 53 | 54 |  61 |  62 |    | 55 | 56 |  63 |  64 |

    For the positional embeddings:

    Same for every tile, different for every token.

    - :class:`torchtitan.models.llama_vision.position_embeddings.TokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

        |  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
        |  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
        | 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
        | 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

    Different for every tile, different for every token.

    - :class:`torchtitan.models.llama_vision.position_embeddings.TiledTokenPositionalEmbedding`

    .. code-block:: text

        |  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
        |  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

        | 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
        | 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

        | 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
        | 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

        | 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
        | 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

    different for every tile, same for every token within a tile.

    - :class:`torchtitan.models.llama_vision.position_embeddings.TilePositionalEmbedding`

    .. code-block:: text

        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
        |  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |

        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
        |  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |

    Args:
        model_args (ModelArgs): The model args.

    Raises:
        ValueError: If `patch_size` is not greater than 0.
        ValueError: If `len(return_intermediate)` is greater than `num_layers`.
    """

    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()
        if model_args.patch_size <= 0:
            raise ValueError(f"kernel size of conv {model_args.patch_size} must be > 0")
        if return_intermediate and (
            len(model_args.return_intermediate) > model_args.n_layers
        ):
            raise ValueError(
                f"len(return_intermediate) must be <= num_layers. Got {return_intermediate=} and {num_layers=}"
            )

        self.return_intermediate = model_args.return_intermediate

        self.conv = Conv2dModule(
            in_channels=model_args.in_channels,
            out_channels=model_args.encoder_embed_dim,
            kernel_size=model_args.patch_size,
            stride=model_args.patch_size,
            bias=False,
        )

        self.ln_post = Fp32LayerNorm(model_args.encoder_embed_dim)
        self.ln_pre = Fp32LayerNorm(model_args.encoder_embed_dim)
        self.transformer_layers = nn.ModuleList(
            [VitTransformerBlock(model_args) for _ in range(model_args.n_layers)]
        )

        self.class_embedding = CLSEmbedding(model_args.encoder_embed_dim)
        # pre and post tile position embedding
        if model_args.max_num_tiles > 1:
            self.pre_tile_pos_embed = TilePositionEmbedding(
                num_tiles=model_args.max_num_tiles,
                emb_dim=model_args.encoder_embed_dim,
            )
            self.post_tile_pos_embed = TilePositionEmbedding(
                num_tiles=model_args.max_num_tiles,
                emb_dim=model_args.model_args.encoder_embed_dim,
            )
            self.token_pos_embedding = TokenPositionalEmbedding(
                emb_dim=model_args.encoder_embed_dim,
                tile_size=model_args.tile_size,
                patch_size=model_args.patch_size,
            )
        else:
            self.pre_tile_pos_embed = None
            self.post_tile_pos_embed = None
            self.token_pos_embedding = TiledTokenPositionalEmbedding(
                max_num_tiles=model_args.max_num_tiles,
                emb_dim=model_args.encoder_embed_dim,
                tile_size=model_args.tile_size,
                patch_size=model_args.patch_size,
            )

    def forward(
        self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension num_imgs to the input. This is useful when a single
        sample constains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, num_imgs=2, num_tiles, num_channels, tile_size_w, tile_size_h).

        Notice that to batch it, you will have to pad num_imgs to max_num_imgs and max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, num_imgs, num_tiles, num_channels, tile_size_w, tile_size_h).
            aspect_ratio (Optional[torch.Tensor]): torch.Tensor with shape (bsz, n_imgs, 2). If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple: (x, hidden_states),
                where x is a torch.tensor of shape (bsz, num_imgs, num_tiles, num_tokens, emb_dim) and
                hidden_states has shape is a list of len(out_indices) torch.tensor with shape
                (bsz, num_imgs, num_tiles, num_tokens, emb_dim).

        Raises:
            ValueError: If aspect_ratio is None, but num_tiles > 1 in the batch.
        """

        bsz, num_imgs, num_tiles, num_channels, width, height = images.shape

        if aspect_ratio is None:
            aspect_ratio = torch.ones((bsz * num_imgs, 2), dtype=torch.int).to(
                device=images.device
            )
            if num_tiles > 1:
                raise ValueError(
                    f"aspect_ratio was not provided, but found num_tiles > 1 "
                    f"for {images.shape=}. Please provide aspect_ratio."
                )

        aspect_ratio = aspect_ratio.reshape(bsz * num_imgs, 2)

        # patch embedding
        images = images.view(bsz * num_imgs * num_tiles, num_channels, width, height)
        # The op is not behaving completely same as conv2d it contains a permute inside.
        x = self.conv(images)  # shape = [*, emb_dim, grid ** 2]
        _, num_tokens, emb_dim = x.shape  # num_tokens = grid ** 2
        x = x.reshape(bsz * num_imgs, num_tiles, num_tokens, emb_dim)

        # tile embeddings
        if self.pre_tile_pos_embed:
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # apply cls token
        x = self.class_embedding(x)
        num_tokens += 1

        # apply position embeddings
        x = self.token_pos_embedding(x, aspect_ratio)

        x = self.ln_pre(x)
        x = x.view(bsz * num_imgs, -1, emb_dim)

        int_x = []  # intermediate outputs
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if layer_idx in self.return_intermediate:
                h = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
                int_x.append(h)
            x = transformer_layer(x)

        x = self.ln_post(x)
        x = x.view(bsz * num_imgs, num_tiles, num_tokens + num_pad, emb_dim)

        if self.post_tile_pos_embed:
            x = self.post_tile_pos_embed(x, aspect_ratio)

        x = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
        return x, int_x


class LearnableProjection(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.

    Args:
        model_args (ModelArgs): configs for the model.
        num_hidden_inputs (int): Number of expected hidden state inputs
    """

    def __init__(
        self,
        model_args: ModelArgs,
        num_hidden_inputs: int,
    ) -> None:
        super().__init__()
        self.transformer_layers = nn.ModuleList(
            [
                VitTransformerBlock(
                    model_args, sa_scale=TanhGate(), mlp_scale=TanhGate()
                )
                for _ in range(model_args.n_layers)
            ]
        )
        self.output = nn.Linear(
            model_args.encoder_embed_dim * (num_hidden_inputs + 1), model_args.dim
        )
        self.num_hidden = len(model_args.return_intermediates or [])

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [bsz, num_imgs, num_tiles, num_tokens, encoder_emb_dim]
            hidden_states (Optional[List[torch.Tensor]]): list of hidden states
                from the encoder. Each hidden state has the same shape as x.

        Returns:
            Tensor: output tensor of a sequence of embedings [bsz x seq x decoder_emb_dim]
                where sequence length is num_imgs * num_tiles * num_tokens

        """
        bsz, imgs, tiles, embeds, dim = x.shape
        bsz, num_imgs, num_tiles, num_tokens, emb_dim = x.shape

        # apply transformer layers
        x = x.view(bsz * num_imgs, num_tiles * num_tokens, emb_dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)

        # interleave hidden states and cat with x
        if self.num_hidden > 0:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, num_imgs, num_tiles, num_tokens, -1)
            x = torch.cat([x, hidden_states], dim=-1)

        # [bsz x seq x decoder_emb_dim]
        return self.output(x).reshape(bsz, num_imgs * num_tiles * num_tokens, -1)


class VisionEncoder(nn.Module):
    """Vision encoder model for Llama 3.2 Vision. This combines a pretrained
    vision encoder with a learnable projection. We define two different components
    so that we can specify the freeze of the vit part during training easily.

    Args:
        model_args (ModelArgs): configs for the vision encoder.
    """

    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.vit = Vit(model_args)
        self.proj = LearnableProjection(model_args)

    def forward(
        self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor):
                Image tensor with shape [bsz x num_imgs x num_tiles x num_channels x width x height].
            aspect_ratio (Optional[torch.Tensor]): Tensor with shape [bsz x num_imgs x 2]. If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.
        Returns:
            Tensor: output tensor of a sequence of embedings [bsz x seq_len x decoder_emb_dim]
                where sequence length is num_imgs*num_tiles+num_embeds
        """
        return self.projection(self.vit(images, aspect_ratio))
