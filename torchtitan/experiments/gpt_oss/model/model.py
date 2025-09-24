# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.experiments.simple_fsdp import model
from torchtitan.models.attention import build_attention
from torchtitan.protocols.train_spec import ModelProtocol

from .args import GptOssModelArgs
from .moe import GptOssMoE


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args: GptOssModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (GptOssModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor
    original_seq_len = args.original_seq_len

    # YaRN default m-scale (attention_factor). Matches HF when attention_factor is None.
    mscale = 0.1 * math.log(factor) + 1.0

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> Tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Basic RoPE frequency calculation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices
    t = torch.arange(seqlen)

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.full_like(freqs, fill_value=mscale), freqs)

    return freqs_cis


def apply_rotary_emb_inner(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """
    HF-style inputs (half-split last dim) -> interleave -> Torchtitan complex RoPE -> de-interleave.
    Shapes:
      q, k: [B, T, H, D] with D even (HF half-split: first D/2 real, last D/2 imag)
      freqs_cis: complex, last dim == D/2. Typically [T, D/2] or [1, T, D/2].
    Returns:
      q_out, k_out in HF half-split layout (same shape as q, k).
    """
    B, T, H, D = q.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"
    rot = D // 2
    assert freqs_cis.shape[-1] == rot, "freqs_cis last dim must be D/2"
    freqs_cis = freqs_cis[:T, :]

    # Memory layout comparison for head_dim=8:
    # HF Format:     [r0][r1][r2][r3][i0][i1][i2][i3]
    #             ↑-- reals --↑   ↑-- imags --↑

    # Interleaved:   [r0][i0][r1][i1][r2][i2][r3][i3]  
    #             ↑-pair-↑ ↑-pair-↑ ↑-pair-↑ ↑-pair-↑
    # --- inline: HF half-split -> interleaved (real0, imag0, real1, imag1, ...)
    # q_i, k_i: [B, T, H, D]
    q_i = torch.empty_like(q)
    k_i = torch.empty_like(k)
    q_i[..., 0::2] = q[..., :rot]
    q_i[..., 1::2] = q[..., rot:]
    k_i[..., 0::2] = k[..., :rot]
    k_i[..., 1::2] = k[..., rot:]

    # --- Torchtitan default complex apply (expects interleaved last dim)
    # freqs_cis will be reshaped inside to [1, T, 1, rot]
    # TODO(jianiw): I think we shoud go with sin/cos representation to simplify the conversion between paired real/imaginary <-> half-split real/imaginary 
    q_rot_i = apply_rotary_emb_inner(q_i, freqs_cis)  # uses TT's complex path
    k_rot_i = apply_rotary_emb_inner(k_i, freqs_cis)

    # --- inline: interleaved -> HF half-split
    # TODO(jianiw): convert it back
    q_out = torch.cat([q_rot_i[..., 0::2], q_rot_i[..., 1::2]], dim=-1)
    k_out = torch.cat([k_rot_i[..., 0::2], k_rot_i[..., 1::2]], dim=-1)
    return q_out, k_out

# Torch Attention backup implementation (for debugging and sampling) from HuggingFace
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# TODO(jianw): This is eager version from HuggingFace
def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # attention_mask can be [Tq, Tk] or [B, H, Tq, Tk]
        # Convert boolean "allowed" -> additive mask
        if attention_mask.dtype == torch.bool:
            m = attention_mask
            add_mask = torch.zeros_like(m, dtype=attn_weights.dtype)
            add_mask = add_mask.masked_fill(~m, -float("inf"))
        else:
            add_mask = attention_mask.to(attn_weights.dtype)

        # Truncate to current key length and add (broadcasts if needed)
        add_mask = add_mask[..., : key_states.shape[-2]]
        attn_weights = attn_weights + add_mask

    sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output

class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, model_args: GptOssModelArgs, use_sliding_attention: bool = False):
        super().__init__()

        self.sliding_window = model_args.sliding_window if use_sliding_attention else None
        self.head_dim = model_args.head_dim
        self.n_heads = model_args.num_attention_heads
        self.n_kv_heads = model_args.num_key_value_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            model_args.hidden_size, model_args.num_attention_heads * model_args.head_dim, bias=True
        )
        self.wk = nn.Linear(
            model_args.hidden_size, model_args.num_key_value_heads * model_args.head_dim, bias=True
        )
        self.wv = nn.Linear(
            model_args.hidden_size, model_args.num_key_value_heads * model_args.head_dim, bias=True
        )
        self.wo = nn.Linear(
            model_args.num_attention_heads * model_args.head_dim, model_args.hidden_size, bias=True
        )
        self.sinks = nn.Parameter(torch.empty(model_args.num_attention_heads))

        self.use_flex_attn = model_args.use_flex_attn

        if self.use_flex_attn:
            # Only apply sliding window to every other layer
            if use_sliding_attention:
                self.attn = build_attention(use_flex_attn=True, attn_mask_type="sliding_window", sliding_window=self.sliding_window)
            else:
                self.attn = build_attention(use_flex_attn=True, attn_mask_type=model_args.attn_mask_type)
        else:
            # NOTE: sampling with FlexAttn seems broken; use TorchAttn if needed
            self.attn = eager_attention_forward

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(k, self.n_rep)
        values = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2).contiguous()
        k = keys.transpose(1, 2).contiguous()
        v = values.transpose(1, 2).contiguous()

        if self.use_flex_attn:
            output = self.attn(
                q, k, v,
                scale=None,
                sink_weights=self.sinks.to_local() if isinstance(self.sinks, DTensor) else self.sinks,
            )
        else:
            # eager attention forward
            output = self.attn(
                q, k, v, self.sinks,
                attention_mask=self.sliding_window_causal(seqlen, x.device),
                scaling=self.head_dim**-0.5,
                dropout=0.0,
            )
        output = output.transpose(1, 2).contiguous()   # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.reshape(bsz, seqlen, -1).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        return output

    def init_weights(self, init_std: float):
        linear_list = [
            self.wq,
            self.wk,
            self.wv,
        ]

        nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    # TODO: statically init the mask using train.seq_len
    def sliding_window_causal(self, seqlen, device):
        i = torch.arange(seqlen, device=device)
        q_idx  = i[:, None]
        kv_idx = i[None, :]

        causal_mask = q_idx >= kv_idx
        if self.sliding_window is None:
            return causal_mask
        window_mask = q_idx - kv_idx <= self.sliding_window
        return causal_mask & window_mask


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: GptOssModelArgs):

        super().__init__()
        use_sliding_attention = layer_id % 2 == 0
        self.attention = Attention(model_args, use_sliding_attention=use_sliding_attention)
        self.attention_norm = nn.RMSNorm(model_args.hidden_size, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.hidden_size, eps=model_args.norm_eps)

        self.moe = GptOssMoE(model_args, dim=model_args.hidden_size, hidden_dim=model_args.moe_inter_dim)
        self.moe_enabled = True  # for composability with load balancing

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


class GptOssModel(nn.Module, ModelProtocol):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: GptOssModelArgs):
        super().__init__()
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.hidden_size)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=True
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.num_hidden_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args).to(torch.bfloat16)
            # convert_submodules_to_bf16(self.layers[str(layer_id)])

        self.norm = nn.RMSNorm(model_args.hidden_size, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.hidden_size,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()
        # convert_submodules_to_bf16(self)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.hidden_size**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h)
        output = self.output(h)
        return output
