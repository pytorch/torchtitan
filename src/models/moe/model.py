"""Custom MoE Transformer model for training from scratch."""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch import nn

from src.config.config import ModelConfig
from src.models.moe import FeedForward, build_moe


# --- RMSNorm ---


class RMSNorm(nn.Module):
    """RMSNorm that optionally dispatches to quack.rmsnorm for faster memory-bound ops."""

    def __init__(self, dim: int, eps: float = 1e-5, use_quack: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_quack = use_quack

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quack:
            from quack import rmsnorm

            orig_shape = x.shape
            y = rmsnorm(x.reshape(-1, orig_shape[-1]), self.weight, eps=self.eps)
            return y.reshape(orig_shape)
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


# --- RoPE ---


def precompute_freqs_cis(dim: int, end: int, theta: float = 500_000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# --- Attention ---


class Attention(nn.Module):
    """Standard GQA attention using FlashAttention 2.

    flash_attn_func handles GQA natively when q has more heads than k/v,
    and expects (batch, seqlen, nheads, headdim) layout.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq = self.wq(x).view(bs, seqlen, -1, self.head_dim)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # flash_attn expects (bs, seqlen, nheads, headdim) and supports GQA natively
        output = flash_attn_func(xq, xk, xv, causal=True)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


# --- Transformer Block ---


class TransformerBlock(nn.Module):
    """Transformer block with MoE FFN. Attribute layout matches apply_fsdp/apply_moe_ep expectations."""

    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        from src.models.moe import MoEArgs

        self.attention = Attention(cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, use_quack=cfg.use_quack)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, use_quack=cfg.use_quack)

        self.moe_enabled = layer_id >= cfg.n_dense_layers
        if self.moe_enabled:
            moe_args = MoEArgs(
                num_experts=cfg.num_experts,
                num_shared_experts=cfg.num_shared_experts,
                top_k=cfg.top_k,
                score_func="sigmoid",
                route_norm=False,
                route_scale=1.0,
                score_before_experts=True,
                use_grouped_mm=True,
                load_balance_coeff=1e-3,
                _debug_force_load_balance=getattr(cfg, "force_load_balance", False),
            )
            self.moe = build_moe(args=moe_args, dim=cfg.dim, hidden_dim=cfg.ffn_dim)
        else:
            self.feed_forward = FeedForward(cfg.dim, cfg.ffn_dim)

        if cfg.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * cfg.n_layers) ** 0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)  # type: ignore
        else:
            self.feed_forward.init_weights(self.weight_init_std)


# --- Full Model ---


class MoETransformer(nn.Module):
    """
    MoE Transformer model. Attribute layout matches apply_fsdp expectations:
    - model.tok_embeddings (nn.Embedding)
    - model.layers (nn.ModuleDict with string keys)
    - model.norm (nn.RMSNorm)
    - model.output (nn.Linear)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)  # type: ignore
            if hasattr(cfg, "max_seq_len")
            else precompute_freqs_cis(cfg.head_dim, 8192, cfg.rope_theta),
            persistent=False,
        )

        self.layers = nn.ModuleDict()
        for layer_id in range(cfg.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, cfg)

        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, use_quack=cfg.use_quack)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(
                self.cfg.head_dim,
                getattr(self.cfg, "max_seq_len", 8192),
                self.cfg.rope_theta,
            )
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)  # type: ignore
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.cfg.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
