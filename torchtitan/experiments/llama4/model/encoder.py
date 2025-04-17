# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import EarlyFusionModelArgs as ModelArgs


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around :class:`~torch.nn.LayerNorm` to support mixed-precision training.
    """

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


def precompute_freqs_cis(
    dim: int,
    patch_size: int,
    tile_size: int,
    theta: float = 10000.0,
    append_cls_token=True,
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Embedding dimension.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the full input image. In this case, the function will consider your image as a single tile
        append_cls_token (bool): Set to True if CLS token embedding is at the end of the sequence in the vision transformer,
            False if is in the beginning of the sequence. RoPE is zeroed out for the CLS token. Default is True.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freq_dim = dim // 2
    freqs = 1.0 / (
        theta ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim)
    )
    grid_size = tile_size // patch_size
    end = grid_size**2
    t = torch.arange(end, dtype=freqs.dtype, device=freqs.device)

    # Add a placeholder index for CLS token - will not be used in RoPE
    t_cls = -1 * torch.ones(1, dtype=t.dtype, device=t.device)
    t = torch.cat([t, t_cls] if append_cls_token else [t_cls, t])

    # Encode x and y positions of each patch in the tile
    t_x = t % grid_size
    t_y = t // grid_size

    # Outer product of freqs and t; output tensor has shape [end + 1, dim // 4]
    freqs_x = torch.outer(t_x + 1, freqs).float()
    freqs_y = torch.outer(t_y + 1, freqs).float()

    # Shape: [end + 1, dim // 2]
    freqs = torch.cat([freqs_x, freqs_y], dim=-1)
    # Zero out CLS token position frequencies
    freqs = freqs.masked_fill(t.unsqueeze(-1) < 0, 0)

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # constant fixed shape for broadcasting since tile size is fixed
    return freqs_cis.view(1, 1, end + 1, 1, dim // 2)


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
    seq_len = freqs_cis.shape[2]  # (bsz, num_tiles, **seq_len**, n_h, h_d)
    bsz, _, n_h, h_d = xq.shape
    xq_ = torch.view_as_complex(xq.float().reshape(bsz, -1, seq_len, n_h, h_d // 2, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(bsz, -1, seq_len, n_h, h_d // 2, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(bsz, -1, n_h, h_d)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(bsz, -1, n_h, h_d)
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
        self.num_heads = model_args.encoder_n_heads
        self.num_kv_heads = (
            model_args.encoder_n_heads
            if model_args.encoder_n_kv_heads is None
            else model_args.encoder_n_kv_heads
        )
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = model_args.encoder_embed_dim // model_args.encoder_n_heads

        self.wq = nn.Linear(
            model_args.encoder_embed_dim,
            model_args.encoder_n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            model_args.encoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.encoder_embed_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.encoder_n_heads * self.head_dim,
            model_args.encoder_embed_dim,
            bias=False,
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (Optional[torch.Tensor]): Precomputed frequency tensor.

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

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if num_kv_heads < num_heads
        keys = repeat_kv(xk, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        # we use casual mask for training
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=False)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): mlp embedding dimension.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.
        activation: (nn.Module): uninitialized activation module to use. Defaults to nn.silu.

    Attributes:
        w1 (Linear): Linear transformation for the first layer, which projects input from input dim to
            hidden dim, and multiplies by the projection from w3 for activation and second layer.
        w2 (Linear): Linear transformation for the second layer.
    """

    def __init__(
        self,
        dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        hidden_dim = dim
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.activation = activation()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)


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
        self.scale = emb_dim**-0.5
        self.positional_embedding = nn.Parameter(
            self.scale
            * torch.randn((patch_grid_size**2 + 1, emb_dim))  # +1 for CLS token
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., num_tokens, emb_dim)

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding

    def reset_parameters(self):
        nn.init.normal_(self.positional_embedding, std=self.scale)


class VitTransformerBlock(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = Fp32LayerNorm(model_args.encoder_embed_dim, eps=1e-5)
        self.feed_forward = FeedForward(
            dim=model_args.encoder_embed_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.encoder_ffn_dim_multiplier,
            activation=model_args.encoder_activation,
        )
        self.ffn_norm = Fp32LayerNorm(model_args.encoder_embed_dim, eps=1e-5)
        self.weight_init_std = 0.02 / (2 * model_args.encoder_n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile of an image in the beginning of each token.

    Args:
        emb_dim (int): The dimensionality of the input patch embedding.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.scale = emb_dim**-0.5
        self.weight = nn.Parameter(self.scale * torch.randn(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add 1 CLS token to every tile
        bsz_and_num_imgs, num_tiles, _, emb_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_num_imgs, num_tiles, 1, emb_dim)
        return torch.cat([cls_emb, x], dim=2)

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=self.scale)


class ViT(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, outputting of hidden layers.

    (credit for the documentation below: `vision_transformer.py

    <https://github.com/pytorch/torchtune/blob/b4fea322189f16629264ee44826f2ac080e922ec/torchtune/modules/vision_transformer.py>`_).

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into **patches** by a convolution.
    These patches are flattened and subsequently treated as **tokens** by the transformer.

    Each image (or tile which is a fix resolution section of an image) is broken down into patches by a convolution operation.
    For example, if your ``patch_size=40``, then each (400, 400) tile will become a grid of 10x10 patches, and your whole image
    will have num_tiles * n_tokens -> num_tiles * (10x10 patches + 1 CLS token) -> num_tiles * 101.

    Before the transformer layers, a CLS token is added to each tile as the first token.
    In transformers, a token called CLS is a special token that is added to the beginning of each sequence.
    This token can be used to represent the whole input, instead of using a pooling operation, for example.

    This model uses two positional embeddings. Token positional embeddings and 2D Rope Embeddings.

    Args:
        model_args (ModelArgs): The model args.

    Raises:
        ValueError: If `patch_size` is not greater than 0.
    """

    def __init__(
        self,
        model_args: ModelArgs,
    ):
        super().__init__()
        if model_args.patch_size <= 0:
            raise ValueError(f"kernel size of conv {model_args.patch_size} must be > 0")

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # For test validation purposes
        patch_grid_size = model_args.tile_size // model_args.patch_size
        self.patches_per_tile = patch_grid_size**2

        self.conv = nn.Conv2d(
            in_channels=model_args.in_channels,
            out_channels=model_args.encoder_embed_dim,
            kernel_size=(model_args.patch_size, model_args.patch_size),
            stride=(model_args.patch_size, model_args.patch_size),
            bias=False,
        )

        self.ln_post = Fp32LayerNorm(model_args.encoder_embed_dim)
        self.ln_pre = Fp32LayerNorm(model_args.encoder_embed_dim)
        self.transformer_layers = nn.ModuleList(
            [
                VitTransformerBlock(model_args)
                for _ in range(model_args.encoder_n_layers)
            ]
        )

        self.class_embedding = CLSEmbedding(model_args.encoder_embed_dim)

        self.token_pos_embedding = TokenPositionalEmbedding(
            emb_dim=model_args.encoder_embed_dim,
            tile_size=model_args.tile_size,
            patch_size=model_args.patch_size,
        )
        self.model_args = model_args

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        self.conv.reset_parameters()
        self.class_embedding.reset_parameters()
        self.token_pos_embedding.reset_parameters()
        for layer in self.transformer_layers:
            layer.init_weights()
        for norm in (self.ln_pre, self.ln_post):
            norm.reset_parameters()

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.encoder_embed_dim // self.model_args.encoder_n_heads,
            self.model_args.patch_size,
            self.model_args.tile_size,
            self.model_args.rope_theta,
            True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Processes images and returns the embeddings.

        Multiple images per sample: all images are broken into fixed resolution tiles (in prepocessing)
        and then stacked in the batch dimension. This model therefore doesn't have to be aware of
        the number of images from a data sample or the number of tiles in an image. They're all just
        a batch of images to be processed.

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. So your input should have
        shape (num_tiles_img_1+num_tiles_img_2+num_tiles_img_3, num_channels, tile_size_w, tile_size_h).

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, num_channels, tile_size_w, tile_size_h).

        Returns:
            torch.Tensor: A tensor of shape (bsz, num_tokens, emb_dim) and
        """
        # The op is not behaving completely same as conv2d it contains a permute inside.
        x = self.conv(images)
        x = x.reshape(images.shape[0], -1, self.patches_per_tile)
        x = x.permute(0, 2, 1)
        bsz, num_tokens, emb_dim = x.shape  # num_tokens = grid ** 2

        # apply cls token
        x = self.class_embedding(x)
        num_tokens += 1

        # apply position embeddings
        x = self.token_pos_embedding(x)

        x = self.ln_pre(x)

        for layer in self.transformer_layers:
            x = layer(x, self.freqs_cis)

        x = self.ln_post(x)
        return x


class Llama4VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, nn.Sequential(CLIP(), Llama4VisionProjectionHead()).

    Note: this module assumes the CLS token embedding is added at the end
    of the sequence.

    Args:
        model_args (ModelArgs): The model args.
    """

    def __init__(self, model_args: ModelArgs) -> None:
        super().__init__()
        # Account for the pixel shuffle scaling factor ** 2
        input_dim = int(
            model_args.encoder_embed_dim // model_args.pixel_shuffle_scaling_factor**2
        )
        proj_dim = model_args.encoder_projection_embed_dim
        self.output = nn.Sequential(
            nn.Linear(input_dim, proj_dim, bias=False),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.GELU(),
            nn.Linear(proj_dim, model_args.dim, bias=False),
        )
        self.pixel_shuffle_scaling_factor = model_args.pixel_shuffle_scaling_factor

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(
            n,
            w,
            int(h * self.pixel_shuffle_scaling_factor),
            int(c / self.pixel_shuffle_scaling_factor),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * self.pixel_shuffle_scaling_factor),
            int(w * self.pixel_shuffle_scaling_factor),
            int(
                c
                / (
                    self.pixel_shuffle_scaling_factor
                    * self.pixel_shuffle_scaling_factor
                )
            ),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def init_weights(self):
        for layer in self.output:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b, e, d]

        Returns:
            Tensor: output tensor of a sequence of embeddings [b, s, d * pixel_shuffle_factor ** 2]

        Notation used for tensor shapes:
            - b: batch size
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by t * (e - 1) // (pixel_shuffle_factor ** 2)
            - d: embed dim
        """
        # Remove cls token - assumes it is the last token in the sequence
        x = x[:, :-1, :]
        bsz, embeds, dim = x.shape

        # apply pixel shuffle
        h_patches = w_patches = int(embeds**0.5)
        x = x.reshape(bsz, h_patches, w_patches, -1)
        x = self._pixel_shuffle(x)
        _, new_h_patches, new_w_patches, new_dim = x.shape
        # shape: [bsz, embeds // factor ** 2, dim * factor ** 2)]
        x = x.reshape(bsz, new_h_patches * new_w_patches, new_dim)
        # apply output - shape [bsz, embeds // factor ** 2, output_dim]
        x = self.output(x)

        return x


class Llama4VisionEncoder(nn.Module):
    """Vision encoder model for Llama 4. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        model_args (ModelArgs): The model args.
    """

    def __init__(self, model_args) -> None:
        super().__init__()
        self.clip = ViT(model_args)
        self.projection = Llama4VisionProjectionHead(model_args)

    def init_weights(
        self,
        buffer_device: Optional[torch.device] = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        self.clip.init_weights(buffer_device)
        self.projection.init_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b x c x w x h]

        Returns:
            Tensor: output tensor of a sequence of embeddings ``[b x s x d]``
                where sequence length (``s``) is ``(num_imgs*num_tiles)+num_embeds``

         Notation used for tensor shapes:
            - b: batch size, equal to flatten(batch x images x tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*clip_embeds_per_tile
            - d: embed dim
        """
        x = self.clip(images)
        x = self.projection(x)
        return x
