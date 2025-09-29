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


def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, num_heads, head_dim]
    head_dim = xq.shape[-1]

    # reshape for broadcast
    rope_cache = reshape_for_broadcast(rope_cache, xq)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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


# TODO(jianw): This is eager version from HuggingFace. Remove it once FlexAttention is ready.
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
    key_values = key.transpose(2, 3)  # When TP is enabled, key should be shard()
    print(f"key_values : {key_values.placements} {key_values.shape}")
    print(f"query : {query.placements} {query.shape}")

    # [rank0]:key_values : (Shard(dim=1),) torch.Size([8, 64, 64, 2048])
    # [rank0]:query : (Shard(dim=1),) torch.Size([8, 64, 2048, 64])

    attn_weights = query @ key_values * scaling
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
        add_mask = add_mask[..., : key.shape[-2]]
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

    def __init__(
        self, model_args: GptOssModelArgs, use_sliding_attention: bool = False
    ):
        super().__init__()

        self.sliding_window = (
            model_args.sliding_window if use_sliding_attention else None
        )
        self.head_dim = model_args.head_dim
        self.n_heads = model_args.num_attention_heads
        self.n_kv_heads = model_args.num_key_value_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            model_args.hidden_size,
            model_args.num_attention_heads * model_args.head_dim,
            bias=True,
        )
        self.wk = nn.Linear(
            model_args.hidden_size,
            model_args.num_key_value_heads * model_args.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            model_args.hidden_size,
            model_args.num_key_value_heads * model_args.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(
            model_args.num_attention_heads * model_args.head_dim,
            model_args.hidden_size,
            bias=True,
        )
        self.sinks = nn.Parameter(torch.empty(model_args.num_attention_heads))

        self.use_flex_attn = model_args.use_flex_attn

        if self.use_flex_attn:
            # Only apply sliding window to every other layer
            if use_sliding_attention:
                self.attn = build_attention(
                    use_flex_attn=True,
                    attn_mask_type="sliding_window",
                    sliding_window=self.sliding_window,
                )
            else:
                self.attn = build_attention(
                    use_flex_attn=True, attn_mask_type=model_args.attn_mask_type
                )
        else:
            # NOTE: sampling with FlexAttn seems broken; use TorchAttn if needed
            self.attn = eager_attention_forward

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies for rope embedding.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb(q, k, rope_cache)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(k, self.n_rep)
        values = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2).contiguous()
        k = keys.transpose(1, 2).contiguous()
        v = values.transpose(1, 2).contiguous()

        if self.use_flex_attn:
            # FlexAttention
            output, lse = self.attn(
                q,
                k,
                v,
                scale=None,
                return_lse=False,
            )

            # Apply attention sink rescaling: rescale by σ(lse - w[h])
            # This is mathematically equivalent to concatenating learnable sink weights
            sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(
                -1
            )  # [B,H,S,1]
            output = output * sink_scale.to(output.dtype)

        else:
            # eager attention forward
            output = self.attn(
                q,
                k,
                v,
                self.sinks,
                attention_mask=self.sliding_window_causal(seqlen, x.device),
                scaling=self.head_dim**-0.5,
                dropout=0.0,
            )
        output = output.transpose(1, 2).contiguous()  # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.reshape(
            bsz, seqlen, -1
        ).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
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
        q_idx = i[:, None]
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
        self.attention = Attention(
            model_args, use_sliding_attention=use_sliding_attention
        )
        self.attention_norm = nn.RMSNorm(
            model_args.hidden_size, eps=model_args.norm_eps
        )
        self.ffn_norm = nn.RMSNorm(model_args.hidden_size, eps=model_args.norm_eps)

        self.moe = GptOssMoE(
            model_args, dim=model_args.hidden_size, hidden_dim=model_args.moe_inter_dim
        )
        self.moe_enabled = True  # for composability with load balancing

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(self.attention_norm(x), rope_cache)
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
        self.model_args = model_args
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size, model_args.hidden_size
        )
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.num_hidden_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args).to(
                torch.bfloat16
            )

        self.norm = nn.RMSNorm(model_args.hidden_size, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.hidden_size,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
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

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
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
            h = layer(h, self.rope_cache)
        h = self.norm(h)
        output = self.output(h)
        return output
