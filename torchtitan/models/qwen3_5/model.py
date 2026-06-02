# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import AttentionMasksType, BaseAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from .sharding import set_qwen35_sharding_config
from .vision_encoder import Qwen35VisionEncoder


class _Conv1d(nn.Conv1d, Module):
    pass


try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm using rsqrt(sum(x²) + eps), not x/max(norm, eps) like F.normalize, to match FLA kernel."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_naive_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop — use FLA kernels for GPU efficiency.

    Args:
        q, k: (bs, seqlen, n_heads, key_head_dim)
        v: (bs, seqlen, n_heads, value_head_dim)
        g: (bs, seqlen, n_heads) — log-space decay, always negative
        beta: (bs, seqlen, n_heads) — update gate ∈ (0, 1)

    Returns:
        output: (bs, seqlen, n_heads, value_head_dim)
    """
    B, L, H, D_k = q.shape
    D_v = v.shape[-1]
    dtype = q.dtype

    # Upcast to float32 — recurrence accumulates over seqlen steps
    q = _l2norm(q.float(), dim=-1) * (D_k**-0.5)
    k = _l2norm(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()

    output = torch.zeros(B, L, H, D_v, dtype=torch.float32, device=q.device)
    state = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)

    for t in range(L):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        b_t = beta[:, t].unsqueeze(-1)

        state = state * g_t
        kv_mem = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv_mem) * b_t
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        output[:, t] = torch.einsum("bhkv,bhk->bhv", state, q_t)

    return output.to(dtype)


class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class RMSNormGated(Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``.

    Takes ``(x, gate)`` separately. Weight is ones-initialized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = (self.weight.float() * x).to(input_dtype)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)


class GatedDeltaKernel(Module):
    """Stateless dispatch to FLA kernel or pure-torch fallback.

    Provides a module boundary for the sharding code to wrap forward with
    DTensor→local conversion — same pattern as FlexAttention. Handles Q/K
    head expansion for grouped linear attention internally so that
    repeat_interleave runs on local tensors under TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "fla_chunked": parallel within chunks, fast for training (default)
        # "fla_fused_recurrent": token-by-token, lower memory for long sequences
        # "torch_naive": pure-Python reference, for numerical testing only
        backend: Literal[
            "fla_chunked", "fla_fused_recurrent", "torch_naive"
        ] = "fla_chunked"

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if q.shape[2] != v.shape[2]:
            assert v.shape[2] % q.shape[2] == 0
            repeat = v.shape[2] // q.shape[2]
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        if self.backend == "torch_naive":
            return _torch_naive_gated_delta(q, k, v, g, beta)

        if not _HAS_FLA:
            raise RuntimeError(
                f"Backend '{self.backend}' requires the `fla` package. "
                "Install: pip install flash-linear-attention"
            )

        if self.backend == "fla_chunked":
            result = _fla_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                use_qk_l2norm_in_kernel=True,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent', 'torch_naive'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0]


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, no attention masks, different head structure from standard
    attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        key_head_dim: int
        value_head_dim: int
        conv_kernel_size: int = 4

        # Sub-module configs
        in_proj_q: Linear.Config
        in_proj_k: Linear.Config
        in_proj_v: Linear.Config
        in_proj_z: Linear.Config
        in_proj_a: Linear.Config
        in_proj_b: Linear.Config
        kernel: GatedDeltaKernel.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        key_dim = config.in_proj_q.out_features
        value_dim = config.in_proj_v.out_features

        self.in_proj_q = config.in_proj_q.build()
        self.in_proj_k = config.in_proj_k.build()
        self.in_proj_v = config.in_proj_v.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_a = config.in_proj_a.build()
        self.in_proj_b = config.in_proj_b.build()

        self.conv_q = _Conv1d(
            in_channels=key_dim,
            out_channels=key_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=key_dim,
            padding=0,
        )
        self.conv_k = _Conv1d(
            in_channels=key_dim,
            out_channels=key_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=key_dim,
            padding=0,
        )
        self.conv_v = _Conv1d(
            in_channels=value_dim,
            out_channels=value_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=value_dim,
            padding=0,
        )

        n_value_heads = value_dim // config.value_head_dim
        self.A_log = nn.Parameter(torch.empty(n_value_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_value_heads))

        self.kernel = config.kernel.build()
        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()

    def _causal_conv(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        # pyrefly: ignore [bad-argument-type]
        x = F.pad(x.transpose(1, 2), (self.conv_kernel_size - 1, 0))
        if isinstance(x, DTensor):
            # TODO: Remove once DTensor Conv1d dispatch handles sharded groups.
            mesh, plc = x.device_mesh, x.placements
            w: torch.Tensor = conv.weight  # pyrefly: ignore [bad-assignment]
            if isinstance(w, DTensor):
                w = w.to_local()
            local_groups = w.size(0)
            # pyrefly: ignore [no-matching-overload]
            out = F.conv1d(
                x.to_local(),
                w,
                None,
                conv.stride,
                conv.padding,
                conv.dilation,
                local_groups,
            )
            x = DTensor.from_local(out, mesh, plc, run_check=False)
        else:
            x = conv(x)
        return F.silu(x).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Split projections (not fused QKV) so each is ColwiseParallel for TP.
        xq = self._causal_conv(self.in_proj_q(x), self.conv_q)
        xk = self._causal_conv(self.in_proj_k(x), self.conv_k)
        xv = self._causal_conv(self.in_proj_v(x), self.conv_v)
        xz = self.in_proj_z(x)
        xa = self.in_proj_a(x)
        xb = self.in_proj_b(x)

        xq = xq.view(bs, seqlen, -1, self.key_head_dim)
        xk = xk.view(bs, seqlen, -1, self.key_head_dim)
        xv = xv.view(bs, seqlen, -1, self.value_head_dim)

        g = -torch.exp(self.A_log.float()) * F.softplus(
            xa.float() + self.dt_bias
        )  # decay rate, always negative
        beta = torch.sigmoid(xb)  # update gate ∈ (0, 1)

        output = self.kernel(xq, xk, xv, g, beta)

        xz = xz.view(bs, seqlen, -1, self.value_head_dim)
        output = self.norm(output, xz)

        output = output.reshape(bs, seqlen, -1)
        return self.out_proj(output)


class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config
        mask_type: str = "causal"

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # wq is 2x wider: produces query + gate
        xq_gate = self.wq(x).view(bs, seqlen, -1, self.head_dim * 2)
        xq, gate = xq_gate.chunk(2, dim=-1)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        # QK norm (before RoPE)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_rot, xq_pass = xq[..., : self.rotary_dim], xq[..., self.rotary_dim :]
        xk_rot, xk_pass = xk[..., : self.rotary_dim], xk[..., self.rotary_dim :]
        xq_rot, xk_rot = apply_rotary_emb_cos_sin(xq_rot, xk_rot, rope_cache, positions)
        xq = torch.cat([xq_rot, xq_pass], dim=-1)
        xk = torch.cat([xk_rot, xk_pass], dim=-1)

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        output = output * torch.sigmoid(gate)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class Qwen35TransformerBlock(Module):
    """Hybrid transformer block for Qwen3.5.

    Each layer uses either full attention (Qwen35Attention) or linear
    attention (GatedDeltaNet), determined by which config is provided.
    Both types share the same FFN/MoE structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen35Attention.Config | None = None
        delta_net: GatedDeltaNet.Config | None = None
        feed_forward: Module.Config | None = None
        moe: Module.Config | None = None
        attention_norm: OffsetRMSNorm.Config
        ffn_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None

        if self.full_attn:
            self.attn = config.attention.build()  # pyrefly: ignore [missing-attribute]
        else:
            assert config.delta_net is not None
            self.attn = config.delta_net.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.attention_norm(x)
        if self.full_attn:
            h = self.attn(h, freqs_cis, attention_masks, positions)
        else:
            h = self.attn(h)
        x = x + h

        h = self.ffn_norm(x)
        if self.moe_enabled:
            x = x + self.moe(h)
        else:
            x = x + self.feed_forward(h)
        return x


class Qwen35Model(Decoder):
    """Qwen3.5: Multimodal model with hybrid attention.

    Combines a hybrid decoder (GatedDeltaNet linear attention + full
    attention with output gating and partial RoPE) with a Vision
    Transformer encoder for multimodal understanding.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MRoPE: 3D position IDs (temporal, height, width) for vision tokens
    - MoE variant: routed experts + shared expert with sigmoid gate

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ _compute_vision_positions             → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          ├─ _compute_mrope_freqs                      → 3D position IDs → interleaved cos/sin
          │
          └─ transformer layers (hybrid)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config

        # MRoPE section sizes for interleaved multi-dimensional RoPE
        # [temporal, height, width] - controls how position dimensions are interleaved
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                attn_cfg = next(
                    (l.attention for l in self.layers if l.attention is not None),
                    None,
                )
                if attn_cfg is not None and (
                    attn_cfg.n_heads % tp != 0 or attn_cfg.n_kv_heads % tp != 0
                ):
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({attn_cfg.n_heads}) and "
                        f"n_kv_heads ({attn_cfg.n_kv_heads})."
                    )
                dn_cfg = next(
                    (l.delta_net for l in self.layers if l.delta_net is not None),
                    None,
                )
                if dn_cfg is not None:
                    n_key_heads = dn_cfg.in_proj_q.out_features // dn_cfg.key_head_dim
                    n_value_heads = (
                        dn_cfg.in_proj_v.out_features // dn_cfg.value_head_dim
                    )
                    if n_key_heads % tp != 0 or n_value_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_key_heads ({n_key_heads}) and "
                            f"n_value_heads ({n_value_heads})."
                        )

            set_qwen35_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attn_cfg = next(
                (l.attention for l in self.layers if l.attention is not None),
                None,
            )
            # pyrefly: ignore [missing-attribute]
            n_heads = attn_cfg.n_heads
            # pyrefly: ignore [missing-attribute]
            head_dim = attn_cfg.head_dim
            num_full_attn = sum(1 for l in self.layers if l.attention is not None)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                n_heads,
                2 * head_dim,
                seq_len,
                num_full_attn=num_full_attn,
            )

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = config.vision_encoder.build()

        self.mrope_section = config.mrope_section
        self.spatial_merge_size = config.vision_encoder.spatial_merge_size

    def _compute_mrope_freqs(
        self,
        tokens: torch.Tensor,
        *,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build 3D position IDs and compute interleaved MRoPE cos/sin frequencies.

        Constructs (temporal, height, width) position IDs for each token, then
        looks up cos/sin from the 1D RoPE table and overwrites H/W-assigned dims
        with their own position lookups.

        Args:
            tokens: (batch, seq_len) token IDs
            grid_thw: (num_images, 3) grid dimensions for images
            grid_thw_videos: (num_videos, 3) grid dimensions for videos
            special_tokens: Special token definitions
            positions: (batch, seq_len) per-token position IDs for packed
                sequences. When provided, document boundaries are detected
                where positions reset (positions[t] < positions[t-1]), and
                pos_id_offset resets to 0 at each boundary

        Returns:
            (batch, seq_len, 1, head_dim * 2) pre-computed MRoPE cos/sin
        """
        # --- Build 3D position IDs ---

        # Expand each video [T, H, W] into T rows of [1, H, W] so that
        # each frame is treated like an image in the MRoPE code below
        # Temporal position comes from frame ordering in the sequence
        if grid_thw_videos is not None:
            grid_thw_videos = torch.repeat_interleave(
                grid_thw_videos, grid_thw_videos[:, 0], dim=0
            )
            grid_thw_videos[:, 0] = 1

        spatial_merge_size = self.spatial_merge_size
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        batch_size, seq_len = tokens.shape
        position_ids = torch.zeros(
            3,
            batch_size,
            seq_len,
            dtype=tokens.dtype,
            device=tokens.device,
        )

        # Precompute document boundaries and vision token positions across batch
        if positions is not None:
            resets = positions[:, 1:] < positions[:, :-1]  # (batch, seq_len-1)
        # Find the first token of each consecutive vision region (image or video)
        # E.g. for [text, img, img, img, text, vid, vid] → positions [1, 5]
        vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
        prev_vision = torch.cat(
            [torch.zeros_like(vision_mask[:, :1]), vision_mask[:, :-1]], dim=1
        )
        batch_vision_starts = vision_mask & ~prev_vision  # (batch, seq_len)
        # Cache vision grid indices by shape to avoid redundant construction
        grid_cache: dict[tuple[int, int, int], torch.Tensor] = {}

        image_index, video_index = 0, 0
        # Build MRoPE 3D position IDs per sample
        # With sample packing, each sample may contain multiple documents
        for sample_i in range(batch_size):
            llm_pos_ids_list: list[torch.Tensor] = []

            if positions is not None:
                # Detect document boundaries within one packed sample
                # pyrefly: ignore [unbound-name]
                reset_indices = torch.where(resets[sample_i])[0] + 1
                doc_starts = [0] + reset_indices.tolist()
                doc_ranges = [
                    (
                        doc_starts[d],
                        doc_starts[d + 1] if d + 1 < len(doc_starts) else seq_len,
                    )
                    for d in range(len(doc_starts))
                ]
            else:
                doc_ranges = [(0, seq_len)]

            sample_tokens = tokens[sample_i]
            sample_vision_starts = torch.where(batch_vision_starts[sample_i])[
                0
            ].tolist()
            vision_start_index = 0

            for doc_start, doc_end in doc_ranges:
                doc_pos_ids_list: list[torch.Tensor] = []

                # Advance pointer to collect vision region starts in this document
                doc_vision_starts: list[int] = []
                while (
                    vision_start_index < len(sample_vision_starts)
                    and sample_vision_starts[vision_start_index] < doc_end
                ):
                    doc_vision_starts.append(sample_vision_starts[vision_start_index])
                    vision_start_index += 1

                # Process [text tokens][vision tokens] pairs within this document
                pair_cursor = doc_start
                for vision_start in doc_vision_starts:
                    if sample_tokens[vision_start] == image_token_id:
                        # pyrefly: ignore [unsupported-operation]
                        t, h, w = grid_thw[image_index]
                        image_index += 1
                    else:
                        # pyrefly: ignore [unsupported-operation]
                        t, h, w = grid_thw_videos[video_index]
                        video_index += 1

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        int(t.item()),
                        int(h.item()) // spatial_merge_size,
                        int(w.item()) // spatial_merge_size,
                    )
                    text_len = vision_start - pair_cursor

                    # pos_id_offset may differ from pair_cursor due to compact
                    # spatial position IDs for vision regions
                    pos_id_offset = (
                        doc_pos_ids_list[-1].max() + 1
                        if len(doc_pos_ids_list) > 0
                        else 0
                    )
                    # [text tokens] — sequential positions, identical on all 3 axes
                    doc_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                    )
                    # [vision tokens] — 3D grid positions (T, H, W)
                    grid_key = (llm_grid_t, llm_grid_h, llm_grid_w)
                    if grid_key not in grid_cache:
                        hw = llm_grid_h * llm_grid_w
                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, hw)
                            .flatten()
                        )
                        h_index = (
                            torch.arange(llm_grid_h)
                            .view(1, -1, 1)
                            .expand(llm_grid_t, -1, llm_grid_w)
                            .flatten()
                        )
                        w_index = (
                            torch.arange(llm_grid_w)
                            .view(1, 1, -1)
                            .expand(llm_grid_t, llm_grid_h, -1)
                            .flatten()
                        )
                        grid_cache[grid_key] = torch.stack([t_index, h_index, w_index])
                    doc_pos_ids_list.append(
                        grid_cache[grid_key] + text_len + pos_id_offset
                    )
                    pair_cursor = vision_start + llm_grid_t * llm_grid_h * llm_grid_w

                # Trailing [text tokens] after the last [text tokens][vision tokens] pair
                if pair_cursor < doc_end:
                    pos_id_offset = (
                        doc_pos_ids_list[-1].max() + 1
                        if len(doc_pos_ids_list) > 0
                        else 0
                    )
                    text_len = doc_end - pair_cursor
                    doc_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                    )

                llm_pos_ids_list.extend(doc_pos_ids_list)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[:, sample_i, :] = llm_positions.to(position_ids.device)

        # --- Compute interleaved MRoPE cos/sin from position IDs ---
        # Convert to local — DTensor doesn't support fancy indexing with
        # plain-tensor indices (cos_cache[t_pos], sin_cache[:, col][dim_pos]).
        freqs_cis = self.freqs_cis
        if isinstance(freqs_cis, DTensor):
            freqs_cis = freqs_cis.to_local()
        head_dim = freqs_cis.shape[-1] // 2
        cos_cache = freqs_cis[:, :head_dim]
        sin_cache = freqs_cis[:, head_dim:]

        # Initialize with temporal positions, then overwrite H/W slices
        t_pos = position_ids[0].long()
        mrope_cos = cos_cache[t_pos]
        mrope_sin = sin_cache[t_pos]

        # Overwrite H and W slices with their own position lookups
        # Both halves of head_dim must be updated (head_dim = cat([freqs, freqs]))
        half = head_dim // 2
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            low = torch.arange(offset, length, 3, device=freqs_cis.device)
            col_indices = torch.cat([low, low + half])
            dim_pos = position_ids[dim].long()
            mrope_cos[..., col_indices] = cos_cache[:, col_indices][dim_pos]
            mrope_sin[..., col_indices] = sin_cache[:, col_indices][dim_pos]

        return torch.cat([mrope_cos, mrope_sin], dim=-1).unsqueeze(2)

    def _compute_vision_positions(
        self,
        tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
    ) -> list[tuple[int, int, int, int]]:
        """Compute (item_idx, sample_idx, vision_start, n_tokens) for each vision item.

        Finds where each contiguous run of vision placeholder tokens starts
        in the text sequence.

        Args:
            tokens: Token IDs (batch, seq_len)
            num_tokens_per_item: (num_items,) actual tokens per vision item
            vision_token_id: Placeholder token ID

        Returns:
            List of (item_idx, sample_idx, vision_start, n_tokens) tuples
        """
        vision_mask = tokens == vision_token_id
        flat_mask = vision_mask.view(-1)
        prev_mask = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat_mask.device), flat_mask[:-1]]
        )
        region_starts = torch.where(flat_mask & ~prev_mask)[0]
        seq_len = tokens.shape[1]

        positions = []
        for i in range(num_tokens_per_item.shape[0]):
            start = int(region_starts[i].item())
            n_tokens = int(num_tokens_per_item[i].item())
            positions.append((i, start // seq_len, start % seq_len, n_tokens))
        return positions

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vision encoder and return padded embeddings with token counts.

        Args:
            pixel_values: Padded patches (num_items, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            merged_embeds: (num_items, max_tokens, dim) padded vision embeddings
            num_tokens_per_item: (num_items,) actual token count per item
        """
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        merged_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return merged_embeds, num_tokens_per_item

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        merged_embeds: torch.Tensor,
        vision_positions: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Scatter vision embeddings into text embeddings at placeholder positions.

        Copies directly from the padded vision encoder output into the text
        sequence.

        Args:
            inputs_embeds: Text embeddings (batch, seq_len, dim)
            merged_embeds: Padded vision embeddings (num_items, max_tokens, dim)
            vision_positions: List of (item_idx, sample_idx, vision_start, n_tokens)

        Returns:
            Updated embeddings
        """
        for item_idx, sample_idx, vision_start, n_tokens in vision_positions:
            inputs_embeds[
                sample_idx, vision_start : vision_start + n_tokens, :
            ] = merged_embeds[item_idx, :n_tokens, :]
        return inputs_embeds

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
    ) -> torch.Tensor:
        """Embed tokens, run vision encoder, scatter vision into text.

        Args:
            tokens: Input token IDs (batch_size, seq_len)
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            (batch, seq_len, dim) embeddings with vision tokens scattered in
        """
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None:
            merged_embeds, num_tokens = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = self._compute_vision_positions(
                tokens, num_tokens, image_token_id
            )
            if image_positions:
                inputs_embeds = self._scatter_vision_embeds(
                    inputs_embeds,
                    merged_embeds=merged_embeds,
                    vision_positions=image_positions,
                )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            merged_embeds, num_tokens = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = self._compute_vision_positions(
                tokens, num_tokens, video_token_id
            )
            if video_positions:
                inputs_embeds = self._scatter_vision_embeds(
                    inputs_embeds,
                    merged_embeds=merged_embeds,
                    vision_positions=video_positions,
                )

        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ):
        if self.tok_embeddings is not None:
            x = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,  # pyrefly: ignore [bad-argument-type]
            )
        else:
            x = tokens

        if grid_thw is not None or grid_thw_videos is not None:
            freqs_cis = self._compute_mrope_freqs(
                tokens,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,  # pyrefly: ignore [bad-argument-type]
                positions=positions,
            )
        else:
            freqs_cis = self.freqs_cis
        for layer in self.layers.values():
            x = layer(x, freqs_cis, attention_masks, positions)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
