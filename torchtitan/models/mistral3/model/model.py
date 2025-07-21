import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.norms import RMSNorm
from torchtitan.models.norms import build_norm
from time import sleep


from torchtitan.logging import logger
from torchtitan.train_spec import ModelProtocol
from torchtitan.models.attention import build_attention, init_attention_mask
from torch.distributed._tensor import Replicate, Shard, distribute_tensor

@dataclass
class ModelArgs:
    # vision encoder part
    vision_embed_dim: int = 1024
    vision_num_layers: int = 24
    vision_num_heads: int = 16
    vision_feature_layer: int = -1
    patch_size: int = 14
    image_size: int = 1540
    in_channels: int = 3
    # For merging patches
    spatial_merge_size: int = 2
    
    # projection part
    num_layers_projection: int = 8
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = False

    # decoder part
    decoder_embed_dim: int = 5120
    decoder_num_layers: int = 40
    decoder_num_heads: int = 32
    decoder_num_kv_heads: int = 8
    fusion_interval: int = 8  # Interval for fusion of vision features into text model
    image_token_index: int = 10  # Token ID representing an image in the text
    
    # common part
    vocab_size: int = 131072
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 1000000000.0
    max_seq_len: int = 131072
    activation: nn.Module = nn.GELU()
    depth_init: bool = True

    n_layers: int = 40
    n_heads: int = 32
    n_embd: int = 5120
    dim: int = 4096


class Mistral3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Mistral3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1)[None, :, :, :]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(image_grid, kernel_size=sub_grid_size, stride=sub_grid_size)
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size, -1
        )  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids

class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        hidden_size = config.vision_embed_dim
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias=False)
        

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)

        return image_features.unsqueeze(0)


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.norm = build_norm("rmsnorm", config.vision_embed_dim, config.norm_eps)
        self.patch_merger = Mistral3PatchMerger(config)
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_embed_dim * num_feature_layers,
            config.decoder_embed_dim,
            bias=config.multimodal_projector_bias,
        )
        self.act = config.activation
        self.linear_2 = nn.Linear(
            config.decoder_embed_dim, config.decoder_embed_dim, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        image_features = self.norm(image_features)

        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


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
    #logger.info(freqs_cis.shape)
    #logger.info(x.shape)
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
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
        position_ids (torch.Tensor, optional): Custom position IDs of shape [batch_size, seq_len].
                                              If provided, will use these to index into freqs_cis.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    if position_ids is not None:
        # Custom position_ids handling
        bs, seqlen = position_ids.shape
        
        # Create output tensors
        xq_out = torch.empty_like(xq)
        xk_out = torch.empty_like(xk)
        
        # Apply rotations batch by batch
        for i in range(bs):
            # Get frequencies for this batch element's positions
            batch_freqs = freqs_cis[position_ids[i]]  # [seqlen, head_dim//2]
            # Reshape for broadcasting
            batch_freqs = batch_freqs.view(1, seqlen, 1, -1)  # [1, seqlen, 1, head_dim//2]
            
            # Apply rotation to this batch element
            batch_xq = xq_[i:i+1]  # [1, seqlen, heads, head_dim//2]
            batch_xk = xk_[i:i+1]
            
            # Multiply by complex exponential
            batch_xq_out = batch_xq * batch_freqs
            batch_xk_out = batch_xk * batch_freqs
            
            # Convert back to real and store in output tensors
            xq_out[i:i+1] = torch.view_as_real(batch_xq_out).flatten(3)
            xk_out[i:i+1] = torch.view_as_real(batch_xk_out).flatten(3)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)
    else:
        # Standard case with sequential positions
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


class SelfAttention(nn.Module):
    """
    Multi-head self attention module with rotary position encoding.
    """

    def __init__(self, config: ModelArgs, is_vision=True):
        super().__init__()
        if is_vision:
            self.num_heads = config.vision_num_heads
            self.num_kv_heads = config.vision_num_heads
            self.head_dim = config.vision_embed_dim // config.vision_num_heads
            self.embed_dim = config.vision_embed_dim
            self.is_causal = False
        else:
            self.num_heads = config.decoder_num_heads
            self.num_kv_heads = (
                config.decoder_num_heads if config.decoder_num_kv_heads is None else config.decoder_num_kv_heads
            )
            self.head_dim = config.decoder_embed_dim // config.decoder_num_heads
            self.embed_dim = config.decoder_embed_dim
            self.is_causal = True
            
        self.num_rep = self.num_heads // self.num_kv_heads


        self.wq = nn.Linear(self.embed_dim, int(self.num_heads * self.head_dim * 0.8), bias=False)
        self.wk = nn.Linear(self.embed_dim, int(self.num_kv_heads * self.head_dim * 0.8), bias=False)
        self.wv = nn.Linear(self.embed_dim, int(self.num_kv_heads * self.head_dim * 0.8), bias=False)
        self.wo = nn.Linear(int(self.num_heads * self.head_dim * 0.8), self.embed_dim, bias=False)

        self.sdpa = build_attention(True, "causal")




    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor, optional): Precomputed frequency tensor.
            position_ids (torch.Tensor, optional): Custom position ids tensor of shape [batch, seq_len].

        Returns:
            torch.Tensor: Output tensor after attention.
        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `num_heads` (or `num_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, 128)
        xk = xk.view(bs, seqlen, -1, 128)
        xv = xv.view(bs, seqlen, -1, 128)

        if freqs_cis is not None:
            # Apply RoPE with position_ids if provided
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, position_ids=position_ids)

        # repeat k/v heads if num_kv_heads < num_heads
        keys = repeat_kv(xk, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.num_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
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
        w1 (Linear): Linear transformation for the first layer.
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
        hidden_dim = 32768
        self.activation = activation

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)


class FeedForwardForDecoder(nn.Module):
    """
    FeedForward module for the decoder. It's different from the one in the encoder.
    This is the component which is originally used in Mistral3/Llama3.
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
        hidden_dim = 32768

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


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


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        attn_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.attn = SelfAttention(config, is_vision=True)
        self.ln_attn = build_norm("rmsnorm", config.vision_embed_dim, config.norm_eps)
        self.mlp = FeedForward(
            dim=config.vision_embed_dim,
            hidden_dim=4 * config.vision_embed_dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            activation=config.activation,
        )
        self.ln_mlp = build_norm("rmsnorm", config.vision_embed_dim, config.norm_eps)
        self.attn_scale = attn_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(x=self.ln_attn(x), freqs_cis=None)
        x = x + self.mlp(self.ln_mlp(x))
        return x


class DecoderTransformerSelfAttnBlock(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
    ):
        super().__init__()
        self.attn = SelfAttention(config, is_vision=False)
        self.ln_attn = build_norm("rmsnorm", config.decoder_embed_dim, config.norm_eps)
        self.mlp = FeedForwardForDecoder(
            dim=config.decoder_embed_dim,
            hidden_dim=4 * config.decoder_embed_dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.ln_mlp = build_norm("rmsnorm", config.decoder_embed_dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs: Dict,
    ):
        # Handle custom position_ids if provided
        if position_ids is not None:
            # Custom handling for position_ids
            # We need to index into freqs_cis with the position_ids
            # First, we'll do a custom reshape_for_broadcast implementation that uses position_ids
            x_norm = self.ln_attn(x)
            # Get the appropriate freqs_cis based on position_ids
            x = x + self.attn(x_norm, freqs_cis, position_ids=position_ids)
        else:
            # Standard forwarding without custom position_ids
            x = x + self.attn(self.ln_attn(x), freqs_cis)
            
        x = x + self.mlp(self.ln_mlp(x))
        return x


class VisionEncoder(nn.Module):
    """Vision encoder model using Pixtral Vision for Mistral3. This integrates the Pixtral vision encoder
    with a projection to connect to the multimodal decoder.

    Args:
        config (ModelArgs): configs for the vision encoder.
    """

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        
        # Import Pixtral components here to avoid circular imports
        from .modeling_pixtral import PixtralVisionModel, PixtralVisionConfig
        
        # Create a PixtralVisionConfig based on the ModelArgs 
        pixtral_config = PixtralVisionConfig(
            hidden_size=config.vision_embed_dim,
            intermediate_size=4 * config.vision_embed_dim,  # Standard multiplier
            num_hidden_layers=config.vision_num_layers,
            num_attention_heads=config.vision_num_heads,
            num_channels=config.in_channels,
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_act="gelu",  # Standard activation
            attention_dropout=0.0,  # No dropout by default
            rope_theta=config.rope_theta,
            initializer_range=0.02  # Standard initialization
        )
        
        # Initialize the Pixtral vision model
        self.pixtral_vision = PixtralVisionModel(pixtral_config)
        
        # Add projection to connect to the decoder
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

    def forward(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor):
                Input image tensor with shape [batch_size, channels, height, width].
            image_sizes (torch.Tensor):
                Tensor with actual image sizes (height, width) for each image in the batch.
                
        Returns:
            Tensor: output tensor of embeddings [batch_size, seq_len, decoder_embed_dim]
        """
        # Pass through Pixtral vision model
        vision_outputs = self.pixtral_vision(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return vision_outputs

        
        # Get the last hidden state
        image_features = vision_outputs.last_hidden_state
        
        # Project to decoder dimension
        return self.multi_modal_projector(image_features, image_sizes)

class MultimodalDecoder(nn.Module):
    """Decoder multimodal model for Mistral3.

    Args:
        config (ModelArgs): configs for the model.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(config), persistent=True
        )

        self.layers = nn.ModuleDict()
        for idx in range(config.decoder_num_layers):
            # define a llama3-like decoder layer
            decoder_layer = DecoderTransformerSelfAttnBlock(config)
            # cross attention layers, mixing text and vision,
            # placed every `fusion_interval` layers
            """
            if idx % config.fusion_interval == 0:
                cross_attn_layer = DecoderTransformerCrossAttnBlock(config)
                fusion_layer = FusionLayer(
                    layer=decoder_layer, fusion_layer=cross_attn_layer
                )
                self.layers[str(idx)] = fusion_layer
            else:
            """
            self.layers[str(idx)] = decoder_layer

        self.tok_embeddings = nn.Embedding(131072, config.decoder_embed_dim)
        self.norm = build_norm(
            config.norm_type, dim=config.decoder_embed_dim, eps=config.norm_eps
        )
        self.output = nn.Linear(
            config.decoder_embed_dim, 131072, bias=False
        )

    def _precompute_freqs_cis(self, config) -> torch.Tensor:
        return precompute_freqs_cis(
            int(config.decoder_embed_dim // config.decoder_num_heads * 0.8),
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            config.max_seq_len,
            config.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. Shape ``[b x s x s_e]``. Default is None.
            position_ids (Optional[torch.Tensor]): Optional tensor of position ids with shape ``[b x s]``.
                If provided, will use these position ids to index into freqs_cis.
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        if inputs_embeds is None:
            h = self.tok_embeddings(tokens)
        else:
            h = inputs_embeds

        # Setup freqs_cis based on position_ids or sequence length
        if position_ids is not None:
            # Use custom position_ids to index into freqs_cis
            # We still need freqs_cis with the right device/dtype
            freqs_cis = self.freqs_cis
        else:
            # Default: use standard positions based on sequence length
            freqs_cis = self.freqs_cis

        for layer in self.layers.values():
            # shape: [b, s, d]
            h = layer(
                h,
                freqs_cis=freqs_cis,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                position_ids=position_ids,
            )

        # shape: [b, s, d]
        h = self.norm(h)
        output = self.output(h)

        return output


class Mistral3ForConditionalGeneration(nn.Module, ModelProtocol):
    """
    Mistral3 model which consists of a vision backbone and a language model.
    
    Args:
        config (ModelArgs): Configuration for the model.
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # Language model decoder
        self.language_model = MultimodalDecoder(config)
        
        # Special token for representing images in the text
        self.image_token_index = config.image_token_index

        self.vision_model_initialized = False


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

        buffer_device = buffer_device or self.language_model.freqs_cis.device
        with torch.device(buffer_device):
            self.language_model._precompute_freqs_cis(self.config)
    

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        image_sizes: torch.Tensor,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        #image_outputs = self.vision_tower(pixel_values.to(dtype=torch.float16), image_sizes=image_sizes, output_hidden_states=True)

        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        #with open("my_file.txt", "w") as f:
        #    #print(pixel_values, file=f)

        with torch.no_grad():
            image_outputs = self.vision_tower(pixel_values, image_sizes=image_sizes, output_hidden_states=False)
            # If we have one vision feature layer, return the corresponding hidden states,
            # otherwise, select the hidden states of each feature layer and concatenate them
            if isinstance(vision_feature_layer, int):
                selected_image_feature = image_outputs.last_hidden_state #[vision_feature_layer]
            else:
                hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
                selected_image_feature = torch.cat(hs_pool, dim=-1)

            image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
            return image_features


        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        tp_mesh: Optional[Any] = None,
        device: Optional[Any] = None,
        vision_tower: Optional[Any] = None,
    ):
        if not self.vision_model_initialized:
            from transformers import Mistral3ForConditionalGeneration as Mistral3Model
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch

            full = Mistral3Model.from_pretrained(
                "mistralai/Mistral-Small-3.1-24B-Instruct-2503", trust_remote_code=True, device_map="auto", torch_dtype=torch.float16
            ).eval()

            from safetensors.torch import load_file

            if device == None:
                device = torch.device("cuda")

            self.vision_tower = full.vision_tower.eval().to(device, dtype=torch.bfloat16)

            self.multi_modal_projector = full.multi_modal_projector.eval().to(device, dtype=torch.bfloat16)

            del full
            self.vision_model_initialized = True
        
        if pixel_values is not None and image_sizes is not None:

            inputs_embeds = self.language_model.tok_embeddings(input_ids)

            if type(inputs_embeds) is torch.Tensor:

                if image_features is None:
                    image_features = self.get_image_features(
                        pixel_values=pixel_values,
                        vision_feature_layer=-1,
                        image_sizes=image_sizes
                    ).to(device=device, dtype=inputs_embeds.dtype)

                image_token_index = self.config.image_token_index

                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds)
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype) 
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features) #.detach().contiguous()
            
            else:
                # dtensor
                inputs_embeds = inputs_embeds.redistribute(tp_mesh, [Replicate()]).to_local()   # all-gather

                image_features = self.get_image_features(
                        pixel_values=pixel_values,
                        vision_feature_layer=-1,
                        image_sizes=image_sizes,
                ).to(device=device, dtype=inputs_embeds.dtype)

                image_token_index = self.config.image_token_index

                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds)
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)


                leaf = inputs_embeds.masked_scatter(special_image_mask, image_features).detach().contiguous()

                rep_dt = distribute_tensor(leaf, tp_mesh, [Replicate()])         # OK now
                sharded_dt = rep_dt.redistribute(tp_mesh, [Shard(1)])            # reduce-scatter

                inputs_embeds = sharded_dt 

            logits = self.language_model(
                        tokens=input_ids,
                        encoder_input=inputs_embeds,
                        encoder_mask=None,
                        position_ids=position_ids
                    )
        else:
            if position_ids is not None:
                # for the case where we want to do sequence packing, we need to pass the nonstandard position_ids
                logits = self.language_model(
                            tokens=input_ids,
                            encoder_mask=None,
                            position_ids=position_ids
                        )
            else:
                logits = self.language_model(
                            tokens=input_ids,
                            encoder_mask=None,
                        )
    
        return logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text from a prompt with optional image input.
        
        Args:
            input_ids (torch.LongTensor): The input token IDs.
            pixel_values (torch.FloatTensor, optional): The input pixel values.
            image_features (torch.FloatTensor, optional): Pre-extracted image features.
            image_sizes (torch.Tensor, optional): The sizes of the images.
            position_ids (torch.Tensor, optional): Custom position ids of shape [batch, seq_len].
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Sampling temperature for generation.
            top_k (int, optional): If set, only use the top k tokens for sampling.
            
        Returns:
            torch.LongTensor: The generated token IDs.
        """
        # Ensure batch dimension (T,) --> (B, T)

        if pixel_values is not None and image_features is not None:
            raise ValueError("You can't provide both pixel_values and image_features at the same time")

        # Handle image sizes if not provided
        if pixel_values is not None and image_sizes is None:
            # If image_sizes not provided, use the dimensions of pixel_values
            image_sizes = torch.tensor([[p.shape[1], p.shape[2]] for p in pixel_values])
            
        # Process image features if provided through pixel_values
        vision_output = None
        if pixel_values is not None:
            # Process pixel values with the vision encoder to get image features
            vision_output = self.vision_tower(pixel_values, image_sizes)
        elif image_features is not None:
            vision_output = image_features
            
        # Start with the input_ids and incrementally generate tokens
        generated_ids = input_ids.clone()
        
        # Set up RNG for sampling
        rng = None
        if "seed" in kwargs and kwargs["seed"] is not None:
            rng = torch.Generator(input_ids.device).manual_seed(kwargs["seed"])
        
        # Initialize position_ids tracking if custom positions are provided
        current_position_ids = position_ids
        
        for i in range(max_length):
            # For generation we need to handle position_ids specially
            # If position_ids were provided for the input, we need to extend them for each new token
            if current_position_ids is not None:
                # For the next token, we'll need to extend position_ids with the next position
                if i > 0:  # Only need to extend after the first iteration
                    # Calculate the next position values based on the last position in each sequence
                    # This logic may need to be customized based on your specific position_ids encoding
                    last_positions = current_position_ids[:, -1].unsqueeze(-1)
                    # Add 1 to get the next position (this assumes sequential positions)
                    next_positions = last_positions + 1
                    # Append to current position_ids
                    current_position_ids = torch.cat([current_position_ids, next_positions], dim=1)
            
            # Generate next token
            outputs = self.forward(
                input_ids=generated_ids,
                pixel_values=None,  # Don't pass pixel_values again after first token
                image_features=vision_output,  # Pass pre-extracted features
                image_sizes=image_sizes,
                position_ids=current_position_ids
            )
            
            # Get the next token's logits (last token in the sequence)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature and top-k if specified
            if temperature != 1.0 or top_k is not None:
                from scripts.generate._generation import logits_to_probs, multinomial_sample_one
                
                # Convert logits to probabilities
                probs = logits_to_probs(next_token_logits, temperature, top_k)
                
                # Sample from the distribution
                next_token = multinomial_sample_one(probs, rng=rng)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Concatenate to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS token
            eos_token_id = kwargs.get("eos_token_id", -1)
            if eos_token_id >= 0 and (next_token == eos_token_id).any():
                break
                
        return generated_ids

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a TransformerModelArgs object.

        Args:
            model_args (TransformerModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
