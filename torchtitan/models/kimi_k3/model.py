# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

# Shape suffixes:
# B = batch, L = sequence length, D = model dimension, H = heads,
# K = key head dimension, V = value head dimension, E = experts,
# C = projection channels, F = expert hidden dimension, R = routed tokens,
# S = selected experts per token, T = flattened tokens,
# N = attention-residual entries.


class KimiShortConvolution(ShortConvolution, Module):
    """KDA short causal convolution backed by FLA's fused kernel.

    Matches the released Kimi K3 HuggingFace model, which builds FLA's
    ``ShortConvolution`` per q/k/v projection. The Triton kernel runs only on
    accelerator devices.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        kernel_size: int
        activation: str = "silu"

    def __init__(self, config: Config):
        super().__init__(
            hidden_size=config.hidden_size,
            kernel_size=config.kernel_size,
            activation=config.activation,
        )

    def forward(
        self,
        x_BLD: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        y_BLD, _ = causal_conv1d(
            x=x_BLD,
            weight=self.weight.squeeze(1),
            activation=self.activation,
            backend=self.backend,
        )
        return y_BLD, None

class KimiRMSNormGated(Module):
    """Per-head RMSNorm followed by a sigmoid output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(variance + self.eps)
        x_float = self.weight.float() * x_float
        return (x_float * torch.sigmoid(gate.float())).to(input_dtype)


def _situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Kimi's SiTU-GLU activation, evaluated in FP32."""
    input_dtype = gate.dtype
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(input_dtype)


class KimiFeedForward(FeedForward):
    """FeedForward with Kimi's SiTU activation"""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            _situ_glu(self.w1(x), self.w3(x), self.beta, self.linear_beta),
        )


class KimiMLAAttention(BaseAttention):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``: the RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied, so this has no rope
    config at all. Attention delegates to the configured inner backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        gate: Linear.Config
        wo: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.gate = config.gate.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions

        B, L, _ = x_BLD.shape
        q_BLHK = self.wq_b(self.q_norm(self.wq_a(x_BLD))).view(
            B, L, self.n_heads, self.q_head_dim
        )

        compressed_kv_BLC = self.wkv_a(x_BLD)
        kv_latent_BLC, k_rope_BLK = torch.split(
            compressed_kv_BLC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_BLHC = self.wkv_b(self.kv_norm(kv_latent_BLC)).view(
            B,
            L,
            self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_BLHK, v_BLHV = torch.split(
            kv_BLHC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_rope_BLHK = k_rope_BLK.view(B, L, 1, self.qk_rope_head_dim).expand(
            -1, -1, self.n_heads, -1
        )
        k_BLHK = torch.cat((k_nope_BLHK, k_rope_BLHK), dim=-1)

        out_BLHV = self.inner_attention(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            attention_masks=attention_masks,
            scale=self.scale,
        )
        out_BLD = out_BLHV.reshape(B, L, self.n_heads * self.v_head_dim)
        out_BLD = out_BLD * torch.sigmoid(self.gate(x_BLD))
        return self.wo(out_BLD)


class KimiKDAKernel(Module):
    """Stateless dispatch to FLA's chunked KDA kernel.

    The gate activation, the beta sigmoid, and the query/key L2 norm are all
    fused into the kernel rather than materialized here, so the decay never
    exists as a full ``(B, L, H, K)`` tensor. 
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float | None = -5.0

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound
        if self.lower_bound is not None and not (-5.0 <= self.lower_bound < 0.0):
            raise ValueError("KDA lower_bound must be in the safe range [-5, 0).")

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> torch.Tensor:
        # safe_gate selects the bounded gate activation
        # lower_bound * sigmoid(exp(A_log) * (gate + dt_bias)); without it the
        # kernel applies -exp(A_log) * softplus(gate + dt_bias).
        out_BLHV, _ = chunk_kda(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            gate_BLHK,
            beta_BLH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK.reshape(-1),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
        )
        return out_BLHV


class KimiDeltaAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        head_dim: int
        conv_kernel_size: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv: Conv1d.Config
        k_conv: Conv1d.Config
        v_conv: Conv1d.Config
        forget_a: Linear.Config
        forget_b: Linear.Config
        beta: Linear.Config
        output_gate: Linear.Config
        kernel: Module.Config
        output_norm: KimiRMSNormGated.Config
        output_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.conv_kernel_size = config.conv_kernel_size

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.q_conv = config.q_conv.build()
        self.k_conv = config.k_conv.build()
        self.v_conv = config.v_conv.build()
        self.forget_a = config.forget_a.build()
        self.forget_b = config.forget_b.build()
        self.beta = config.beta.build()
        self.output_gate = config.output_gate.build()
        self.kernel = config.kernel.build()
        self.output_norm = config.output_norm.build()
        self.output_proj = config.output_proj.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def _causal_conv(self, x_BLC: torch.Tensor, conv: Conv1d) -> torch.Tensor:
        x_BCL = F.pad(x_BLC.transpose(1, 2), (self.conv_kernel_size - 1, 0))
        return F.silu(conv(x_BCL)).transpose(1, 2)

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_masks is not None:
            raise NotImplementedError(
                "Kimi K3 reference KDA does not support packed-document masks."
            )

        B, L, _ = x_BLD.shape
        q_BLHK = self._causal_conv(self.q_proj(x_BLD), self.q_conv).view(
            B, L, self.num_heads, self.head_dim
        )
        k_BLHK = self._causal_conv(self.k_proj(x_BLD), self.k_conv).view(
            B, L, self.num_heads, self.head_dim
        )
        v_BLHV = self._causal_conv(self.v_proj(x_BLD), self.v_conv).view(
            B, L, self.num_heads, self.head_dim
        )
        forget_BLHK = self.forget_b(self.forget_a(x_BLD)).view(
            B, L, self.num_heads, self.head_dim
        )
        beta_BLH = self.beta(x_BLD).float()

        out_BLHV = self.kernel(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            forget_BLHK,
            beta_BLH,
            self.A_log,
            self.dt_bias,
        )
        output_gate_BLHV = self.output_gate(x_BLD).view(
            B, L, self.num_heads, self.head_dim
        )
        out_BLHV = self.output_norm(out_BLHV, output_gate_BLHV)
        return self.output_proj(out_BLHV.reshape(B, L, -1))


class KimiGroupedExperts(GroupedExperts):
    """``common/moe.py::GroupedExperts`` with Kimi's SiTU activation.

    Inherits its stacked-weight shape (``w1_EFD``/``w2_EDF``/``w3_EFD``) and
    parameter allocation; only ``forward`` differs, since the activation is
    baked into the ``_grouped_mm`` call sequence rather than being a
    swappable argument.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        gate_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=w1_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        up_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )

        h_RF = _situ_glu(gate_RF, up_RF, self.beta, self.linear_beta)

        return self._grouped_mm(
            A=h_RF,
            B_t=w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)


class KimiLatentMoE(MoE):
    """``common/moe.py::MoE`` with Kimi's latent routed-expert path.

    Routed tokens are projected down to the expert latent width, run through
    the experts, then normed and projected back up to the model dimension.
    Shared experts still see the full-width input.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        routed_down: Linear.Config
        routed_norm: RMSNorm.Config
        routed_up: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.routed_down = config.routed_down.build()
        self.routed_norm = config.routed_norm.build()
        self.routed_up = config.routed_up.build()

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        weights_BLS, expert_ids_BLS, scores_BLE = self.router(x_BLD, self.expert_bias_E)
        routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
            -1, expert_ids_BLS, True
        )
        num_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_tokens_per_expert_E)

        routed_BLD = self.routed_experts(
            self.routed_down(x_BLD),
            weights_BLS,
            expert_ids_BLS,
            num_tokens_per_expert_E,
        )
        out_BLD = self.routed_up(self.routed_norm(routed_BLD))
        if self.shared_experts is not None:
            out_BLD = out_BLD + self.shared_experts(x_BLD)
        return out_BLD


def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32.

    The norm and projection weights are folded into a single score vector, so
    this reads them directly instead of calling the modules. That is only valid
    while both are replicated; sharding them would need a DTensor-aware path.
    """
    assert norm.eps is not None

    values_TND = torch.cat((block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_TN = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_TN, values_float).squeeze(1)
    return output_TD.to(values_TND.dtype)


class KimiK3TransformerBlock(Module):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        attn_res_block_size: int
        attention: KimiMLAAttention.Config | None
        delta_attention: KimiDeltaAttention.Config | None
        feed_forward: KimiFeedForward.Config | None
        moe: KimiLatentMoE.Config | None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        attention_res_norm: RMSNorm.Config
        attention_res_proj: Linear.Config
        ffn_res_norm: RMSNorm.Config
        ffn_res_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if (config.attention is None) == (config.delta_attention is None):
            raise ValueError(
                "Exactly one of attention or delta_attention must be configured."
            )
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError("Exactly one of feed_forward or moe must be configured.")
        self.layer_id = config.layer_id
        self.attn_res_block_size = config.attn_res_block_size
        self.attention = (
            config.attention.build() if config.attention is not None else None
        )
        self.delta_attention = (
            config.delta_attention.build()
            if config.delta_attention is not None
            else None
        )
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.moe = config.moe.build() if config.moe is not None else None
        self.moe_enabled = self.moe is not None
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attention_res_norm = config.attention_res_norm.build()
        self.attention_res_proj = config.attention_res_proj.build()
        self.ffn_res_norm = config.ffn_res_norm.build()
        self.ffn_res_proj = config.ffn_res_proj.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the residual on every block output to preserve its FSDP gradient path.
        B, L, D = x_BLD.shape
        prefix_sum_BLD: torch.Tensor | None = x_BLD

        if block_residual_TND.shape[1] > 0:
            assert prefix_sum_BLD is not None
            x_BLD = _apply_attention_residual(
                prefix_sum_BLD.reshape(-1, D),
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            ).view(B, L, D)

        if self.layer_id % self.attn_res_block_size == 0:
            assert prefix_sum_BLD is not None
            block_residual_TND = torch.cat(
                (
                    block_residual_TND,
                    prefix_sum_BLD.reshape(-1, D).unsqueeze(1),
                ),
                dim=1,
            )
            prefix_sum_BLD = None

        h_BLD = self.attention_norm(x_BLD)
        if self.attention is not None:
            h_BLD = self.attention(h_BLD, attention_masks, positions)
        else:
            assert self.delta_attention is not None
            h_BLD = self.delta_attention(h_BLD, None, positions)
        prefix_sum_BLD = h_BLD if prefix_sum_BLD is None else prefix_sum_BLD + h_BLD

        assert prefix_sum_BLD is not None
        h_BLD = _apply_attention_residual(
            prefix_sum_BLD.reshape(-1, D),
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        ).view(B, L, D)
        h_BLD = self.ffn_norm(h_BLD)
        if self.moe is not None:
            h_BLD = self.moe(h_BLD)
        else:
            assert self.feed_forward is not None
            h_BLD = self.feed_forward(h_BLD)
        return prefix_sum_BLD + h_BLD, block_residual_TND


class KimiK3Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None
        spatial_merge_size: int = 2

        def update_from_config(self, *, config, **kwargs) -> None:
            # Unsupported parallelisms are rejected in parallelize_kimi_k3.
            dataloader = getattr(config, "dataloader", None)
            if getattr(dataloader, "packing_buffer_size", 0) > 0:
                raise NotImplementedError(
                    "Kimi K3 v1 does not support packed documents."
                )
            Decoder.Config.update_from_config(self, config=config, **kwargs)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention_config = self.first_attention
            if not isinstance(attention_config, KimiMLAAttention.Config):
                raise ValueError(
                    "Kimi K3 requires at least one MLA layer for FLOP accounting."
                )
            # KDA and the vision encoder have no dedicated term here, so their
            # parameters only contribute the dense 6*N estimate; reported MFU is
            # approximate.
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attention_config.n_heads,
                attention_config.qk_nope_head_dim
                + attention_config.qk_rope_head_dim
                + attention_config.v_head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.spatial_merge_size = config.spatial_merge_size
        if self.vision_encoder is not None:
            # The decoder sizes each image's placeholder run from
            # spatial_merge_size while the encoder merges patches with
            # merge_kernel_size. A mismatch surfaces much later as a
            # placeholder-run misalignment that blames the prompt.
            merge_kernel_size = self.vision_encoder.merge_kernel_size
            if merge_kernel_size != (
                config.spatial_merge_size,
                config.spatial_merge_size,
            ):
                raise ValueError(
                    f"spatial_merge_size {config.spatial_merge_size} does not "
                    f"match the vision encoder's merge_kernel_size "
                    f"{merge_kernel_size}; each image would occupy a different "
                    "number of text positions than the encoder produces."
                )

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        embeddings = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        if pixel_values is None:
            return embeddings
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        num_tokens_per_item = (grid_thw[:, 1] // self.spatial_merge_size) * (
            grid_thw[:, 2] // self.spatial_merge_size
        )
        vision_positions = get_vision_positions(
            tokens,
            num_tokens_per_item,
            special_tokens["image_id"],
        )
        return scatter_vision_embeds(
            embeddings,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        if self.tok_embeddings is not None:
            h_BLD = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                special_tokens=special_tokens,
            )
        else:
            h_BLD = tokens

        B, L, D = h_BLD.shape
        block_residual_TND = h_BLD.new_zeros(B * L, 0, D)
        for layer in self.layers.values():
            h_BLD, block_residual_TND = layer(
                h_BLD,
                block_residual_TND,
                attention_masks,
                positions,
            )

        h_BLD = _apply_attention_residual(
            h_BLD.reshape(-1, D),
            block_residual_TND,
            self.output_res_proj,
            self.output_res_norm,
        ).view(B, L, D)
        h_BLD = self.norm(h_BLD) if self.norm is not None else h_BLD
        if self._skip_lm_head:
            return h_BLD
        return self.lm_head(h_BLD) if self.lm_head is not None else h_BLD
