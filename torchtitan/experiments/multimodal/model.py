# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 3 is licensed under the LLAMA 3 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    # encoder part
    encoder_embed_dim: int = 4096
    encoder_num_layers: int = 32
    num_layers_projection: int = 32
    encoder_num_heads: int = 32
    encoder_num_kv_heads: Optional[int] = None
    patch_size: int = 1
    tile_size: int = 128
    max_num_tiles: int = 8
    activation: nn.Module = nn.GELU()
    # in_channels (int): The number of image input channels.
    in_channels: int = 3
    # return_intermediates (Optional[List[int]]): The indices of hidden layers to return.
    # If provided, it will return the intermediate results of the transformer layers
    # before they go through a next layer. For example, ``return_intermediates=[0,3]``
    # will return the tokens before they go through the first and fourth layers.
    return_intermediates: Optional[List[int]] = None
    is_causal: bool = True

    # decoder part
    decoder_embed_dim: int = 4096  # This is for linear projection to convert the output of encoder to decoder
    fusion_interval: int = 1  # This is the interval of layers that are used for fusion
    num_special_tokens: int = 2  # This is the number of special tokens in the tokenizer
    decoder_num_layers: int = 16
    decoder_num_heads: int = 32
    decoder_num_kv_heads: Optional[int] = None

    # common part
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around :class:`~torch.nn.LayerNorm` to support mixed-precision training.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: The normalized output tensor having the same shape as ``x``.
        """
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, num_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=num_rep)"""
    bsz, seq_len, num_kv_heads, head_dim = x.shape
    if num_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bsz, seq_len, num_kv_heads, num_rep, head_dim)
        .reshape(bsz, seq_len, num_kv_heads * num_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        num_kv_heads (int): Number of key and value heads.
        num_heads (int): Number of query heads.
        num_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.num_heads = model_args.encoder_num_heads
        self.num_kv_heads = (
            model_args.encoder_num_heads
            if model_args.encoder_num_kv_heads is None
            else model_args.encoder_num_kv_heads
        )
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.encoder_embed_dim // model_args.encoder_num_heads

        self.wq = nn.Linear(
            model_args.encoder_embed_dim,
            model_args.encoder_num_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            model_args.encoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.encoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.encoder_num_heads * self.head_dim,
            model_args.encoder_embed_dim,
            bias=False,
        )
        self.is_causal = model_args.is_causal

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `num_heads` (or `num_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if (
            freqs_cis is not None
        ):  # Only used in the self attention layers for text decoder
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if num_kv_heads < num_heads
        keys = repeat_kv(xk, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=self.is_causal)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.
        activation: (nn.Module): Activation function to use. Defaults to nn.silu.

    Attributes:
        w1 (Linear): Linear transformation for the first layer, which projects input from input dim to
            hidden dim, and multiplies by the projection from w3 for activation and second layer.
        w2 (Linear): Linear transformation for the second layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.activation = activation
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)


class TanhGate(nn.Module):
    """Implements a basic learnable gate to scale layer outputs"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to gate

        Returns:
            torch.Tensor: The output tensor after gating. Has the same shape as ``x``.
        """
        return x * self.scale.tanh()


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    For details, please check the documentation of :class:`ViT`.

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
        to :class:`TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`ViT`.

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


class Conv2dModule(torch.nn.Module):
    """Conv2D Module.
    This is like Conv2D in PyTorch except:

    - PyTorch Conv2D outputs shape (*, out_channels, h_out, w_out), while this module
      outputs (*, h_out * w_out, out_channels).
    - We implement the conv as an unfold -> permute -> linear, where we can column-wise
      shard the linear.

    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel. This module also assumes a square kernel.
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
        # x = x.view(bsz * seq_len, emd_dim)
        x = x + self.attn_scale(self.attn(x=self.ln_attn(x), freqs_cis=None))
        x = x + self.mlp_scale(self.mlp(self.ln_mlp(x)))
        # return x.view(bsz, seq_len, emd_dim)
        return x


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

    - token_pos_embedding (tiled): :class:`TiledTokenPositionalEmbedding`
    - pre_tile_pos_embed: :class:`TilePositionalEmbedding`
    - post_tile_pos_embed: :class:`TilePositionalEmbedding`

    Otherwise, pre and post tile_pos_embed should be None and all you need is a simple
    token positional embedding:

    - token_pos_embedding (not tiled): :class:`TokenPositionalEmbedding`

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

    - :class:`TokenPositionalEmbedding`

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

    - :class:`TiledTokenPositionalEmbedding`

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

    - :class:`TilePositionalEmbedding`

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
        ValueError: If `len(return_intermediates)` is greater than `num_layers`.
    """

    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()
        if model_args.patch_size <= 0:
            raise ValueError(f"kernel size of conv {model_args.patch_size} must be > 0")
        if model_args.return_intermediates and (
            len(model_args.return_intermediates) > model_args.encoder_num_layers
        ):
            raise ValueError(
                "len(return_intermediates) must be <= num_layers."
                f" Got {model_args.return_intermediate=} and {model_args.encoder_num_layers=}"
            )

        # For test validation purposes
        patch_grid_size = model_args.tile_size // model_args.patch_size
        self.patches_per_tile = patch_grid_size**2

        self.return_intermediates = model_args.return_intermediates

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
            [
                VitTransformerBlock(model_args)
                for _ in range(model_args.encoder_num_layers)
            ]
        )

        self.class_embedding = CLSEmbedding(model_args.encoder_embed_dim)
        # pre and post tile position embedding
        if model_args.max_num_tiles > 1:
            self.pre_tile_pos_embed = TilePositionalEmbedding(
                max_num_tiles=model_args.max_num_tiles,
                emb_dim=model_args.encoder_embed_dim,
            )
            self.post_tile_pos_embed = TilePositionalEmbedding(
                max_num_tiles=model_args.max_num_tiles,
                emb_dim=model_args.encoder_embed_dim,
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
            if layer_idx in self.return_intermediates:
                h = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
                int_x.append(h)
            x = transformer_layer(x)

        x = self.ln_post(x)
        x = x.view(bsz * num_imgs, num_tiles, num_tokens, emb_dim)

        if self.post_tile_pos_embed:
            x = self.post_tile_pos_embed(x, aspect_ratio)

        x = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)
        return x, int_x


class Projection(nn.Module):
    """Projection transformer to adapt the output of a
    encoder (CLIP) to the decoder model.
    """

    def __init__(
        self,
        model_args: ModelArgs,
    ) -> None:
        super().__init__()
        self.transformer_layers = nn.ModuleList(
            [
                VitTransformerBlock(
                    model_args, attn_scale=TanhGate(), mlp_scale=TanhGate()
                )
                for _ in range(model_args.num_layers_projection)
            ]
        )

        self.num_hidden = len(model_args.return_intermediates or [])
        self.output = nn.Linear(
            model_args.encoder_embed_dim * (self.num_hidden + 1),
            model_args.decoder_embed_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, num_imgs, num_tiles, num_tokens, emb_dim = x.shape

        # apply transformer layers
        x = x.view(bsz * num_imgs, num_tiles * num_tokens, emb_dim)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.view(bsz, num_imgs, num_tiles, num_tokens, emb_dim)

        # interleave hidden states and cat with x
        if self.num_hidden > 0:
            assert hidden_states is not None
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, num_imgs, num_tiles, num_tokens, -1)
            x = torch.cat([x, hidden_states], dim=-1)

        # [bsz x seq x decoder_emb_dim]
        return self.output(x).reshape(bsz, num_imgs * num_tiles * num_tokens, -1)


class VisionEncoder(nn.Module):
    """Vision encoder model for Llama 3.2 Vision. This combines a vision
    encoder with a projection. We define two different components.

    Args:
        model_args (ModelArgs): configs for the vision encoder.
    """

    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        self.vit = Vit(model_args)
        self.proj = Projection(model_args)

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
        return self.proj(*self.vit(images, aspect_ratio))


class FeedForwardForDecoder(nn.Module):
    """
    FeedForward module for the decoder. It's different from the one in the encoder.
    This is the component which is orignally used in llama3.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    """
    Multi-head self attention module with rotary position.
    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.num_heads = model_args.decoder_num_heads
        self.num_kv_heads = (
            model_args.decoder_num_heads
            if model_args.decoder_num_kv_heads is None
            else model_args.decoder_num_kv_heads
        )
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.decoder_embed_dim // model_args.decoder_num_heads

        self.wq = nn.Linear(
            model_args.decoder_embed_dim,
            model_args.decoder_num_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.decoder_num_heads * self.head_dim,
            model_args.decoder_embed_dim,
            bias=False,
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `num_heads` (or `num_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if num_kv_heads < num_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class CrossAttention(nn.Module):
    """
    Multi-head cross attention module.
    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.num_heads = model_args.decoder_num_heads
        self.num_kv_heads = (
            model_args.decoder_num_heads
            if model_args.decoder_num_kv_heads is None
            else model_args.decoder_num_kv_heads
        )
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.decoder_embed_dim // model_args.decoder_num_heads

        self.wq = nn.Linear(
            model_args.decoder_embed_dim,
            model_args.decoder_num_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.decoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.decoder_num_heads * self.head_dim,
            model_args.decoder_embed_dim,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-05)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-05)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        encoder_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bs, seqlen_x, _ = x.shape
        seqlen_y = encoder_input.shape[1]
        xq, xk, xv = self.wq(x), self.wk(encoder_input), self.wv(encoder_input)

        # Use -1 instead of `num_heads` (or `num_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen_x, -1, self.head_dim)
        xk = xk.view(bs, seqlen_y, -1, self.head_dim)
        xv = xv.view(bs, seqlen_y, -1, self.head_dim)

        # repeat k/v heads if num_kv_heads < num_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen_y, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen_y, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen_x, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen_y, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen_y, head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, is_causal=False
        )
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen_x, n_local_heads, head_dim)
        output = output.view(bs, seqlen_x, -1)
        return self.wo(output)


class DecoderTransformerSelfAttnBlock(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()
        self.attn = SelfAttention(model_args)
        self.ln_attn = nn.RMSNorm(model_args.decoder_embed_dim, eps=1e-5)
        self.mlp = FeedForwardForDecoder(
            dim=model_args.decoder_embed_dim,
            hidden_dim=4 * model_args.decoder_embed_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.ln_mlp = nn.RMSNorm(model_args.decoder_embed_dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        **kwargs: Dict,
    ):
        bsz, seq_len, emd_dim = x.shape
        x = x + self.attn(self.ln_attn(x), freqs_cis)
        x = x + self.mlp(self.ln_mlp(x))
        return x


class DecoderTransformerCrossAttnBlock(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()
        self.attn = CrossAttention(model_args)
        self.ln_attn = nn.RMSNorm(model_args.decoder_embed_dim)
        self.mlp = FeedForward(
            dim=model_args.decoder_embed_dim,
            hidden_dim=4 * model_args.decoder_embed_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.ln_mlp = nn.RMSNorm(model_args.decoder_embed_dim)
        self.attn_scale = TanhGate()
        self.mlp_scale = TanhGate()

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.
        The False value means a token can't attend to an embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| F    F    F
        |red| T    F    T
        |car| F    T    T

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask is used to mask the outputs of attention and
        mlp resulting in the token being skipped.

        The above example would result in a skip mask of: [[True], [False], [False]]
        which specifies which tokens to fully mask out.

        """
        # no skip_mask if no masking
        if mask is None:
            return None
        # negate mask and convert to boolean mask
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        # True where all elements in a row are True
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ) -> torch.Tensor:
        # Skip cross attention when no secondary input as it's primary purpose
        # is to attend between x and encoder_input.
        if encoder_input is None:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)

        attn_out = self.attn(
            self.ln_attn(x),
            encoder_input,
            mask=encoder_mask,
        )
        if skip_mask is not None:
            attn_out.masked_fill_(skip_mask, 0)

        h = self.attn_scale(attn_out) + x
        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.ln_mlp(h))
        if skip_mask is not None:
            mlp_out.masked_fill_(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)

        return out


class FusionLayer(nn.Module):
    """
    Deep Fusion model architectures combine pretrained encoder models with pretrained
    language models by infusing the encoder outputs into the middle layers of the LLM.
    This allows the language model to interpret the enocder outputs as text and
    "understand" any modality for which you can train an decoder. To enable the language model
    to adapt to the encoder outputs, the FusionLayer fuses a new learnable layer to an existing
    decoder (language model) layer. This additional layer can take the encoder embeddings and
    learn to combine them with the token embeddings from the decoder.
    """

    def __init__(
        self, layer: nn.Module, fusion_layer: nn.Module, fusion_first: bool = True
    ):
        super().__init__()
        self.layer = layer
        self.fusion_layer = fusion_layer

    def forward(self, x: torch.Tensor, **kwargs: Dict) -> torch.Tensor:
        x = self.fusion_layer(x, **kwargs)
        x = self.layer(x, **kwargs)
        return x


class FusionEmbedding(nn.Module):
    """
    Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a language model,
    there may be some additional tokens needed to support the fused language model. For
    example, adding a vision encoder might necessitate additional tokens like ``<|image|>``
    to indicate an images position in text and require learning an embedding for this token.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.
    """

    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self.dim = embed_dim
        self.num_embeddings = vocab_size + fusion_vocab_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings

        mask = input < vocab_size
        # num_tokens = (input < vocab_size).sum()
        tokens = torch.masked_select(input, mask)
        # num_fusion_tokens = (input >= vocab_size).sum()
        fusion_tokens = torch.masked_select(input, ~mask) - vocab_size

        # [batch_size x num_tokens x embed_dim]
        embeds = self.embedding(tokens)
        # [batch_size x num_fusion_tokens x embed_dim]
        fusion_embeds = self.fusion_embedding(fusion_tokens)

        # [batch_size x seq_length x embed_dim]
        out = torch.empty(
            bsz,
            seq_len,
            self.dim,
            device=self.embedding.weight.device,
            dtype=self.embedding.weight.dtype,
        )
        mask = mask.unsqueeze(-1).expand(bsz, seq_len, self.dim)
        out.masked_scatter_(mask, embeds)
        out.masked_scatter_(~mask, fusion_embeds)
        return out


class MultimodalDecoder(nn.Module):
    """Decoder multimodal model for Llama 3.2.

    Args:
        model_args (ModelArgs): configs for the vision encoder.
    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(model_args), persistent=True
        )

        self.layers = []
        for idx in range(1, model_args.decoder_num_layers + 1):
            # define a llama3-like decoder layer, we don't train this part.
            decoder_layer = DecoderTransformerSelfAttnBlock(model_args)
            # cross attention layers, mixing text and vision,
            # placed every `fusion_interval` layers
            if idx % model_args.fusion_interval == 0:
                cross_attn_layer = DecoderTransformerCrossAttnBlock(model_args)
                fusion_layer = FusionLayer(
                    layer=decoder_layer, fusion_layer=cross_attn_layer
                )
                self.layers.append(fusion_layer)
            else:
                self.layers.append(decoder_layer)

        self.tok_embeddings = FusionEmbedding(
            model_args.vocab_size,
            model_args.num_special_tokens,
            model_args.decoder_embed_dim,
        )
        self.norm = nn.RMSNorm(model_args.decoder_embed_dim, eps=1e-05)
        self.output = nn.Linear(
            model_args.decoder_embed_dim, model_args.vocab_size, bias=False
        )

    def _precompute_freqs_cis(self, model_args) -> torch.Tensor:
        return precompute_freqs_cis(
            model_args.decoder_embed_dim // model_args.decoder_num_heads,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            model_args.max_seq_len * 2,
            model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(
                h,
                freqs_cis=self.freqs_cis,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
            )

        # shape: [b, s, d]
        h = self.norm(h)
        output = self.output(h).float()

        return output
