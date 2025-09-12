# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

# Import building blocks from torchtitan's Llama3 implementation
from torchtitan.models.llama3.model.model import (
    repeat_kv,
)
from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger
from torchtitan.models.moe import MoE, FeedForward, MoEArgs

from .args import Qwen3TransformerModelArgs

class Qwen3Attention(nn.Module):
    """
    Attention module for Qwen3.
    This version includes RMSNorm for query and key states, a key architectural
    difference from Llama3.
    """

    def __init__(self, model_args: Qwen3TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim # model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=model_args.attention_bias)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=model_args.attention_bias)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=model_args.attention_bias)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=model_args.attention_bias)

        # Qwen3 specific: RMSNorm on Query and Key heads
        self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)

        self.scaling = self.head_dim**-0.5

        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, position_ids: torch.Tensor | None):
        """Forward pass using TT attention backends (RoPE cis like Llama)."""
        bs, seqlen, _ = x.shape
        # Project and reshape Q, K, V
        xq = self.wq(x).view(bs, seqlen, -1, self.head_dim)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        # Apply QK Norm (Qwen3 specific) before RoPE
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply Rotary Positional Embeddings (cis path like Llama)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, packing_mode=position_ids is not None)

        # Grouped Query Attention: repeat K, V heads
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        # Transpose for attention calculation
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # SDPA/Flex attention
        output = self.sdpa(xq, keys, values, position_ids=position_ids, scale=self.scaling)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float) -> None:
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()


class Qwen3TransformerBlock(nn.Module):
    """
    A single transformer block for Qwen3.
    It contains an attention block and either a dense MLP or a sparse MoE block.
    """

    def __init__(self, layer_id: int, model_args: Qwen3TransformerModelArgs):
        super().__init__()
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.attention = Qwen3Attention(model_args)

        # Conditionally instantiate MoE or dense MLP based on layer index
        moe_enabled = (
            model_args.num_experts > 0
            and (layer_id + 1) % model_args.decoder_sparse_step == 0
            and layer_id not in model_args.mlp_only_layers
        )
        self.moe_enabled = moe_enabled

        if moe_enabled:
            self.mlp = MoE(
                moe_args=MoEArgs(
                    num_experts=model_args.num_experts,
                    num_shared_experts=0,
                    score_func="softmax",
                    route_norm=True,
                    route_scale=model_args.route_scale,
                    top_k=model_args.num_experts_per_tok,
                    use_grouped_mm=model_args.use_grouped_mm,
                    load_balance_coeff=None,
                    score_before_experts=False,
                ),
                dim=model_args.dim,
                hidden_dim=model_args.moe_intermediate_size,
            )
        else:
            self.mlp = FeedForward(model_args.dim, model_args.intermediate_size)

        # Depth-scaled init standard deviation (match Llama pattern)
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, position_ids: torch.Tensor | None, router_logits: list[torch.Tensor]):
        """Forward pass for the transformer block."""
        # Attention sub-block with pre-normalization
        h = x + self.attention(self.attention_norm(x), freqs_cis=freqs_cis, position_ids=position_ids)

        # FFN/MoE sub-block with pre-normalization
        mlp_input = self.ffn_norm(h)
        if self.moe_enabled:
            mlp_output, new_router_logits = self.mlp(mlp_input)
        else:
            mlp_output = self.mlp(mlp_input)
            new_router_logits = None
        router_logits.append(new_router_logits)

        out = h + mlp_output
        # Return router_logits (will be None for dense layers)
        return out, router_logits

    def init_weights(self, buffer_device: torch.device) -> None:
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.mlp.init_weights(self.weight_init_std, buffer_device)
        else:
            self.mlp.init_weights(self.weight_init_std)


class Transformer(nn.Module, ModelProtocol):
    """
    The main Qwen3 Transformer model.
    """

    def __init__(self, model_args: Qwen3TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = Qwen3TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        if model_args.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies (cis) like Llama to work with TT infra
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Initialize weights similar to Llama/DeepSeek
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        logger.info(f"Precomputing {self.model_args.max_seq_len}")
        return precompute_freqs_cis(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        eos_id: int | None = None,
        input_batch: torch.Tensor | None = None,
    ):
        """
        Forward pass for the entire model.
        Returns final logits. Router logits from MoE layers are stored in `self._last_router_logits` for optional aux loss.
        """
        if self.model_args.use_flex_attn:
            init_attention_mask(input_batch if input_batch is not None else tokens, eos_id=eos_id)

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        if position_ids is not None:
            freqs_cis = self.freqs_cis[position_ids]
        else:
            freqs_cis = self.freqs_cis

        router_logits = []
        for layer in self.layers.values():
            # I am passing the router logits every time, because I imagine that this works better with pipeline parallel, where certain layers become no-ops.
            # I have not checked this extensively.
            h, router_logits = layer(h, freqs_cis=freqs_cis, position_ids=position_ids, router_logits=router_logits)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h

        # Just pass the router logits, if you need them for the auxiliary loss.
        return {
            "logits": output,
            "router_logits": router_logits,
        }



def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, packing_mode: bool = False) -> torch.Tensor:
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
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    if not packing_mode:
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        assert freqs_cis.shape == (x.shape[0], seqlen, x.shape[-1]), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}, seqlen {seqlen}"
        shape = [d if i <= 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    packing_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(torch.view_as_complex(freqs_cis.float()), xq_, packing_mode)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2) # JT: CHANGED FROM 3 TO -2 BECAUSE MORE RELIABLE
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return torch.view_as_real(freqs_cis)