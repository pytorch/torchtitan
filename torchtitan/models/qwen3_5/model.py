# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from .rope import MRoPE
from .sharding import set_qwen35_sharding_config
from .vision_encoder import Qwen35VisionEncoder

GatedDeltaBackend = Literal["fla_chunked", "fla_fused_recurrent"]


@dataclass(frozen=True, slots=True)
class Qwen35AttentionMasks:
    """Attention metadata for the full-attention and DeltaNet layers."""

    full_attention: AttentionMasksType | None
    delta_net: VarlenMetadata | None


def _causal_conv1d_varlen(
    x_TD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
) -> torch.Tensor:
    """FLA depthwise causal conv with per-document resets (CUDA-only).

    A pure-torch per-document reference lives in
    ``tests/unit_tests/test_qwen3_5_deltanet.py``.
    """
    if cu_seqlens_cpu is None:
        raise ValueError(
            "Qwen3.5 FLA varlen conv requires a CPU cu_seqlens tensor. "
            "Build VarlenMetadata with include_host_offsets=True."
        )

    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d

    out_BTD, _ = _fla_causal_conv1d(
        x=x_TD.unsqueeze(0),
        weight=weight.squeeze(1),
        bias=None,
        activation="silu",
        backend="triton",
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return out_BTD.squeeze(0)


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
    """Stateless dispatch to the configured FLA gated delta kernel.

    Provides a module boundary for the sharding code to wrap forward with
    DTensor→local conversion — same pattern as FlexAttention. Handles Q/K
    head expansion for grouped linear attention internally so that
    repeat_interleave runs on local tensors under TP. A pure-torch reference
    implementation lives in ``tests/unit_tests/test_qwen3_5_deltanet.py``;
    it is far too slow for training use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "fla_chunked": parallel within chunks for training (default)
        # "fla_fused_recurrent": for inference only in rl, no backward
        backend: GatedDeltaBackend = "fla_chunked"

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend

    def forward(
        self,
        xq_TNK: torch.Tensor,
        xk_TNK: torch.Tensor,
        xv_TNV: torch.Tensor,
        g_TN: torch.Tensor,
        beta_TN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if xq_TNK.shape[1] != xv_TNV.shape[1]:
            assert xv_TNV.shape[1] % xq_TNK.shape[1] == 0
            repeat = xv_TNV.shape[1] // xq_TNK.shape[1]
            xq_TNK = xq_TNK.repeat_interleave(repeat, dim=1)
            xk_TNK = xk_TNK.repeat_interleave(repeat, dim=1)

        xq_BTNK = xq_TNK.unsqueeze(0)
        xk_BTNK = xk_TNK.unsqueeze(0)
        xv_BTNV = xv_TNV.unsqueeze(0)
        g_BTN = g_TN.unsqueeze(0)
        beta_BTN = beta_TN.unsqueeze(0)

        if self.backend == "fla_chunked":
            if cu_seqlens is not None and cu_seqlens_cpu is None:
                raise ValueError(
                    "Qwen3.5 FLA varlen DeltaNet requires a CPU cu_seqlens tensor."
                )
            result = _fla_chunk_gated_delta_rule(
                xq_BTNK,
                xk_BTNK,
                xv_BTNV,
                g_BTN,
                beta_BTN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                xq_BTNK,
                xk_BTNK,
                xv_BTNV,
                g_BTN,
                beta=beta_BTN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0].squeeze(0)


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. When varlen
    metadata (``VarlenMetadata``) is provided -- i.e. under the ``varlen``
    attention backend -- conv and recurrent state are reset at document
    boundaries. Under other backends (e.g. ``flex``, which passes a
    ``BlockMask``) no reset occurs and the packed sequence is processed as a
    single continuous stream.
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
        conv_q: Conv1d.Config
        conv_k: Conv1d.Config
        conv_v: Conv1d.Config
        kernel: GatedDeltaKernel.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        value_dim = config.in_proj_v.out_features

        self.in_proj_q = config.in_proj_q.build()
        self.in_proj_k = config.in_proj_k.build()
        self.in_proj_v = config.in_proj_v.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_a = config.in_proj_a.build()
        self.in_proj_b = config.in_proj_b.build()

        self.conv_q = config.conv_q.build()
        self.conv_k = config.conv_k.build()
        self.conv_v = config.conv_v.build()

        n_value_heads = value_dim // config.value_head_dim
        self.A_log = nn.Parameter(torch.empty(n_value_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_value_heads))

        self.kernel = config.kernel.build()
        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()

    @staticmethod
    def _local_map_conv(
        x: DTensor,
        conv: Conv1d,
        conv_fn: Callable[..., torch.Tensor],
        *extra_args: torch.Tensor,
    ) -> torch.Tensor:
        """Run a depthwise, channel-sharded conv on local shards via local_map.

        ``conv_fn`` receives the local (x, weight, *extra_args) tensors. Trailing
        ``extra_args`` (e.g. ``cu_seqlens``) are plain replicated tensors passed
        through with a ``None`` placement (unmapped). Input is channel-sharded
        and the weight is ``Shard(0)``; DTensor-ness and gradient placements are
        restored explicitly.
        """
        x_plc = x.placements
        w = conv.weight
        w_plc = w.placements  # pyrefly: ignore [missing-attribute]
        extra_plc = (None,) * len(extra_args)
        conv_dt = local_map(
            conv_fn,
            out_placements=(x_plc,),
            in_placements=(x_plc, w_plc, *extra_plc),
            in_grad_placements=(x_plc, w_plc, *extra_plc),
            device_mesh=x.device_mesh,
        )
        return conv_dt(x, w, *extra_args)  # pyrefly: ignore

    def _causal_conv(
        self,
        x_TD: torch.Tensor,
        conv: Conv1d,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            if isinstance(x_TD, DTensor):

                def _conv_varlen(
                    x_local_TD: torch.Tensor,
                    w_local: torch.Tensor,
                    cu_seqlens_local: torch.Tensor,
                ) -> torch.Tensor:
                    return _causal_conv1d_varlen(
                        x_local_TD,
                        w_local,
                        cu_seqlens_local,
                        cu_seqlens_cpu,
                    )

                return self._local_map_conv(x_TD, conv, _conv_varlen, cu_seqlens)
            return _causal_conv1d_varlen(
                x_TD,
                conv.weight,
                cu_seqlens,
                cu_seqlens_cpu,
            )

        x_BDT = F.pad(
            x_TD.transpose(0, 1).unsqueeze(0),
            [self.conv_kernel_size - 1, 0],
        )
        if isinstance(x_BDT, DTensor):
            # TODO: Remove once the DTensor Conv1d dispatch fix for sharded
            # groups lands in a released torch.
            def _conv(x_local_BDT: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
                # groups == local out-channels (depthwise, channel-sharded)
                return F.conv1d(
                    x_local_BDT,
                    w_local,
                    None,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    w_local.size(0),
                )

            x_BDT = self._local_map_conv(x_BDT, conv, _conv)
        else:
            x_BDT = conv(x_BDT)
        return F.silu(x_BDT).squeeze(0).transpose(0, 1)

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        cu_seqlens = None
        cu_seqlens_cpu = None
        if isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q
            cu_seqlens_host = attention_masks.cu_seq_q_host
            if cu_seqlens_host is None:
                raise ValueError(
                    "Qwen3.5 GatedDeltaNet varlen requires CPU cu_seqlens "
                    "metadata. Build VarlenMetadata with include_host_offsets=True."
                )
            # Keep host metadata as Python values in VarlenMetadata so
            # SelectiveAC does not treat integer CPU tensors as checkpoint
            # inputs; build the FLA API tensor at the DeltaNet boundary.
            cu_seqlens_cpu = torch.tensor(
                cu_seqlens_host,
                dtype=cu_seqlens.dtype,
                device="cpu",
            )

        # Shapes:
        #   xq_TNK, xk_TNK: (T, n_key_heads, key_head_dim)
        #   xv_TNV, xz_TNV: (T, n_value_heads, value_head_dim)
        #   xa_TN, xb_TN: (T, n_value_heads)
        xq_TNK = self._causal_conv(
            self.in_proj_q(x_TD),
            self.conv_q,
            cu_seqlens,
            cu_seqlens_cpu,
        ).view(num_tokens, -1, self.key_head_dim)
        xk_TNK = self._causal_conv(
            self.in_proj_k(x_TD),
            self.conv_k,
            cu_seqlens,
            cu_seqlens_cpu,
        ).view(num_tokens, -1, self.key_head_dim)
        xv_TNV = self._causal_conv(
            self.in_proj_v(x_TD),
            self.conv_v,
            cu_seqlens,
            cu_seqlens_cpu,
        ).view(num_tokens, -1, self.value_head_dim)
        xz_TNV = self.in_proj_z(x_TD).view(num_tokens, -1, self.value_head_dim)
        xa_TN = self.in_proj_a(x_TD)
        xb_TN = self.in_proj_b(x_TD)

        # Gating signals have shape (T, n_value_heads).
        g_TN = -torch.exp(self.A_log.float()) * F.softplus(xa_TN.float() + self.dt_bias)
        beta_TN = torch.sigmoid(xb_TN)

        out_TNV = self.kernel(
            xq_TNK,
            xk_TNK,
            xv_TNV,
            g_TN,
            beta_TN,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

        out_TNV = self.norm(out_TNV, xz_TNV)

        out_TD = out_TNV.reshape(num_tokens, -1)
        return self.out_proj(out_TD)


class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``BaseAttention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config

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

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]

        # wq is 2x wider: produces query + gate
        xq_gate_TN2H = self.wq(x_TD).view(num_tokens, -1, self.head_dim * 2)
        xq_TNH, gate_TNH = xq_gate_TN2H.chunk(2, dim=-1)
        xk_TNH = self.wk(x_TD).view(num_tokens, -1, self.head_dim)
        xv_TNH = self.wv(x_TD).view(num_tokens, -1, self.head_dim)

        # QK norm (before RoPE)
        xq_TNH = self.q_norm(xq_TNH)
        xk_TNH = self.k_norm(xk_TNH)

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_TNR, xq_TNP = (
            xq_TNH[..., : self.rotary_dim],
            xq_TNH[..., self.rotary_dim :],
        )
        xk_TNR, xk_TNP = (
            xk_TNH[..., : self.rotary_dim],
            xk_TNH[..., self.rotary_dim :],
        )
        xq_TNR, xk_TNR = self.rope(xq_TNR, xk_TNR, positions)
        xq_TNH = torch.cat([xq_TNR, xq_TNP], dim=-1)
        xk_TNH = torch.cat([xk_TNR, xk_TNP], dim=-1)

        out_TNH = self.inner_attention(
            xq_TNH,
            xk_TNH,
            xv_TNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        out_TNH = out_TNH * torch.sigmoid(gate_TNH)
        out_TD = out_TNH.view(num_tokens, -1)
        return self.wo(out_TD)


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
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        delta_net_metadata: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        h_TD = self.attention_norm(x_TD)
        if self.full_attn:
            h_TD = self.attn(h_TD, attention_masks, positions)
        else:
            h_TD = self.attn(h_TD, delta_net_metadata)
        x_TD = x_TD + h_TD

        h_TD = self.ffn_norm(x_TD)
        if self.moe_enabled:
            x_TD = x_TD + self.moe(h_TD)
        else:
            x_TD = x_TD + self.feed_forward(h_TD)
        return x_TD


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
    - MRoPE: 3D (temporal/height/width) position IDs for multimodal batches;
      text batches use the plain 1D positions
    - MoE variant: routed experts + shared expert with sigmoid gate

    MRoPE positions (``mrope_positions``, shape ``(num_tokens, 3)``) are built by
    the dataloader and forwarded to every pipeline stage, so RoPE stays consistent
    across stages even though the raw vision inputs (``pixel_values``/``grid_thw``)
    only reach the first stage. Text batches carry no ``mrope_positions`` and use
    the 1D ``positions`` instead.

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, mrope_positions, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ get_vision_positions              → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          └─ transformer layers (hybrid), each given (mrope_positions or positions)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    │    (the layer's MRoPE builds the cos/sin cache from positions)
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config

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
                dn_cfg = next(
                    (
                        layer_cfg.delta_net
                        for layer_cfg in self.layers
                        if layer_cfg.delta_net is not None
                    ),
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
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # The shared helper excludes the vision encoder from the per-token
            # FLOP term (ViT cost scales with patches, not seq_len), so this MFU
            # is decoder-only. TODO: add a per-batch vision FLOP term for VLMs.
            attn_cfg = self.first_attention
            # pyrefly: ignore [missing-attribute]
            n_heads = attn_cfg.n_heads
            # pyrefly: ignore [missing-attribute]
            head_dim = attn_cfg.head_dim
            return get_moe_model_nparams_and_flops(
                self,
                model,
                n_heads,
                2 * head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = config.vision_encoder.build()
        self.spatial_merge_size = config.vision_encoder.spatial_merge_size

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> Qwen35AttentionMasks | VarlenMetadata:
        attn_config = self.config.first_attention
        if attn_config is not None and isinstance(
            attn_config.inner_attention, VarlenAttention.Config
        ):
            # Host offsets are a GatedDeltaNet-only need: the FLA varlen
            # kernels take cu_seqlens as a CPU tensor to size their launches,
            # whereas quadratic attention (torch.nn.attention.varlen) consumes
            # the device tensor directly. They are stored as Python ints so
            # SelectiveAC checkpoint metadata stays tensor-free.
            return create_varlen_metadata_for_document(
                positions,
                include_host_offsets=True,
            )
        full_attention_masks = super().get_attention_masks(positions)
        # Multimodal padding uses position 0 for every padded token. A real
        # document start is position 0 followed by position 1; keep index 0 as
        # the first start. This avoids routing a single padded sample through
        # the varlen kernel while retaining boundaries between packed samples.
        followed_by_one = torch.cat(
            [
                positions[1:] == 1,
                torch.zeros(1, dtype=torch.bool, device=positions.device),
            ]
        )
        first_token = torch.arange(positions.shape[0], device=positions.device) == 0
        sequence_starts = ((positions == 0) & followed_by_one) | first_token
        sequence_positions = torch.where(sequence_starts, 0, 1)
        delta_net_metadata = create_varlen_metadata_for_document(
            sequence_positions,
            include_host_offsets=True,
        )
        if (
            delta_net_metadata.cu_seq_q_host is not None
            and len(delta_net_metadata.cu_seq_q_host) == 2
        ):
            delta_net_metadata = None
        return Qwen35AttentionMasks(full_attention_masks, delta_net_metadata)

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision encoder and return packed embeddings with token counts.

        Args:
            pixel_values: Packed patches ``(total_num_patches, patch_dim)``.
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            vision_embeds: Packed vision embeddings ``(total_tokens, dim)``.
            num_tokens_per_item: (num_items,) actual token count per item
        """
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return vision_embeds, num_tokens_per_item

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
            tokens: Input token IDs ``(num_tokens,)``.
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            ``(num_tokens, dim)`` embeddings with vision tokens scattered in.
        """
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None:
            vision_embeds, num_tokens = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = get_vision_positions(tokens, num_tokens, image_token_id)
            if image_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=image_positions,
                )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            vision_embeds, num_tokens = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = get_vision_positions(tokens, num_tokens, video_token_id)
            if video_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
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
        attention_masks: Qwen35AttentionMasks | AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
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

        if isinstance(attention_masks, Qwen35AttentionMasks):
            full_attention_masks = attention_masks.full_attention
            delta_net_metadata = attention_masks.delta_net
        else:
            full_attention_masks = attention_masks
            delta_net_metadata = (
                attention_masks if isinstance(attention_masks, VarlenMetadata) else None
            )

        # 3D MRoPE positions for multimodal batches, else 2D text positions.
        rope_positions = mrope_positions if mrope_positions is not None else positions
        assert rope_positions is not None
        for layer in self.layers.values():
            x = layer(
                x,
                full_attention_masks,
                rope_positions,
                delta_net_metadata,
            )

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
