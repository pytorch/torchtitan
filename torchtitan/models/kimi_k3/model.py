# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 language model components.

Every operator outside the KDA recurrence is plain eager PyTorch, so the model
mirrors the released HuggingFace math and stays directly inspectable. KDA runs
on FLA's chunked Triton kernel, which is what makes the model trainable at
speed; the pure-PyTorch recurrence it is checked against lives in
``tests/unit_tests/test_kimi_k3.py`` and is far too slow for training.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda
from torch import nn

from torchtitan.distributed.fsdp import add_zero_valued_dependency
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module, ModuleList

# Shape suffixes:
# B = batch, L = sequence length, D = model dimension, H = heads,
# K = key head dimension, V = value head dimension, E = experts,
# T = flattened tokens, N = attention-residual entries.


class KimiRMSNorm(RMSNorm):
    """RMSNorm that applies its weight after casting back to the input dtype.

    ``nn.RMSNorm`` scales by the weight while still in the reduction dtype,
    which does not match the released Kimi implementation under BF16. Keeping
    the subclass also exposes ``kimi_eps`` to ``_apply_attention_residual``,
    which needs the epsilon to normalize residual entries by hand.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.kimi_eps = config.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(variance + self.kimi_eps)
        assert self.weight is not None
        return self.weight * x_float.to(input_dtype)


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


class SituAndMul(Module):
    """Kimi's SiTU activation applied to concatenated gate/up projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        input_dtype = gate_up.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(input_dtype)


class KimiFeedForward(Module):
    """Three-projection feed-forward network using SiTU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config
        activation: SituAndMul.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()
        self.activation = config.activation.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat((self.w1(x), self.w3(x)), dim=-1)
        return self.w2(self.activation(gate_up))


class KimiMLAAttention(Module):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``.  The RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: KimiRMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: KimiRMSNorm.Config
        wkv_b: Linear.Config
        gate: Linear.Config
        wo: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
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

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_masks is not None:
            raise NotImplementedError(
                "Kimi K3 reference MLA does not support packed-document masks."
            )

        B, L, _ = x_BLD.shape
        q_BLNH = self.wq_b(self.q_norm(self.wq_a(x_BLD))).view(
            B, L, self.num_heads, self.q_head_dim
        )

        compressed_kv = self.wkv_a(x_BLD)
        kv_latent, k_rope = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_BLNH = self.wkv_b(self.kv_norm(kv_latent)).view(
            B,
            L,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v_BLNH = torch.split(
            kv_BLNH,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_rope = k_rope.view(B, L, 1, self.qk_rope_head_dim).expand(
            -1, -1, self.num_heads, -1
        )
        k_BLNH = torch.cat((k_nope, k_rope), dim=-1)

        # HuggingFace eager attention keeps the matmul in the input dtype and
        # performs softmax in FP32.
        scores_BNLS = torch.einsum("blnh,bsnh->bnls", q_BLNH, k_BLNH)
        scores_BNLS = scores_BNLS * self.scale
        causal_mask = torch.ones(L, L, dtype=torch.bool, device=x_BLD.device).triu(
            diagonal=1
        )
        scores_BNLS = scores_BNLS.masked_fill(
            causal_mask.view(1, 1, L, L),
            torch.finfo(scores_BNLS.dtype).min,
        )
        probs_BNLS = torch.softmax(scores_BNLS, dim=-1, dtype=torch.float32).to(
            v_BLNH.dtype
        )
        out_BLNV = torch.einsum("bnls,bsnv->blnv", probs_BNLS, v_BLNH)

        out_BLD = out_BLNV.reshape(B, L, self.num_heads * self.v_head_dim)
        out_BLD = out_BLD * torch.sigmoid(self.gate(x_BLD))
        return self.wo(out_BLD)


class KimiKDAKernel(Module):
    """Stateless dispatch to FLA's chunked KDA kernel.

    The gate activation, the beta sigmoid, and the query/key L2 norm are all
    fused into the kernel rather than materialized here, so the decay never
    exists as a full ``(B, L, H, K)`` tensor. ``ReferenceKimiKDAKernel`` in
    ``tests/unit_tests/test_kimi_k3.py`` implements the same interface as an
    explicit recurrence and is the numerical baseline for this kernel; it also
    lets the CPU test suite exercise the surrounding model without Triton.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        lower_bound: float | None = -5.0

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
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
        if q_BLHK.shape != k_BLHK.shape:
            raise ValueError(
                f"KDA q/k shapes must match, got {q_BLHK.shape} and {k_BLHK.shape}."
            )
        if q_BLHK.shape[:3] != v_BLHV.shape[:3]:
            raise ValueError("Kimi KDA requires equal q/k/value head counts.")

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
    """Kimi Delta Attention with causal convolutions and reference recurrence."""

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
        # Typed as the base config so the KDA kernel stays swappable, matching
        # how VisionAttention types its inner_attention. Anything assigned here
        # must accept KimiKDAKernel.forward's arguments.
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


class KimiRoutedExperts(ModuleList):
    """List-backed experts implementing TorchTitan's FSDP expert protocol.

    Kimi K3 keeps one module per expert so the eager implementation and
    HuggingFace state-dict mapping stay directly inspectable.  The shared FSDP
    wrapper discovers routed expert parameters through ``inner_experts`` and
    ``num_experts``; exposing those properties here lets it shard this
    list-backed layout without changing parameter names or forward math.
    """

    @property
    def inner_experts(self) -> "KimiRoutedExperts":
        return self

    @property
    def num_experts(self) -> int:
        return len(self)


class KimiLatentMoE(Module):
    """Eager trainable implementation of Kimi K3 latent MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        router: TokenChoiceTopKRouter.Config
        routed_down: Linear.Config
        routed_experts: list[KimiFeedForward.Config]
        routed_norm: KimiRMSNorm.Config
        routed_up: Linear.Config
        shared_experts: KimiFeedForward.Config
        load_balance_coeff: float | None = 1e-3

    def __init__(self, config: Config):
        super().__init__()
        if len(config.routed_experts) != config.num_experts:
            raise ValueError(
                "The number of routed expert configs must equal num_experts."
            )
        self.num_experts = config.num_experts
        self.router = config.router.build()
        self.routed_down = config.routed_down.build()
        self.routed_experts = KimiRoutedExperts(
            [expert.build() for expert in config.routed_experts]
        )
        self.routed_norm = config.routed_norm.build()
        self.routed_up = config.routed_up.build()
        self.shared_experts = config.shared_experts.build()
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            if self.load_balance_coeff <= 0.0:
                raise ValueError("load_balance_coeff must be positive.")
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(config.num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        weights_BLK, expert_ids_BLK, _ = self.router(x_BLD, self.expert_bias_E)
        B, L, _ = x_BLD.shape
        routing_map_BLE = torch.zeros(
            B,
            L,
            self.num_experts,
            dtype=torch.bool,
            device=x_BLD.device,
        ).scatter(-1, expert_ids_BLK, True)
        with torch.no_grad():
            # In place so the load-balancing hook registered on the optimizer
            # keeps referring to this buffer across steps.
            self.tokens_per_expert_E.add_(routing_map_BLE.sum(dim=(0, 1)).float())

        latent_TD = self.routed_down(x_BLD).reshape(B * L, -1)
        expert_ids_TK = expert_ids_BLK.reshape(B * L, -1)
        weights_TK = weights_BLK.reshape(B * L, -1)
        routed_TD = torch.zeros_like(latent_TD, dtype=torch.float32)

        for expert_idx, expert in enumerate(self.routed_experts):
            token_and_slot = torch.nonzero(expert_ids_TK == expert_idx, as_tuple=False)
            # Keep empty experts in the autograd graph so every FSDP rank
            # produces zero gradients instead of rank-dependent None gradients.
            token_ids = token_and_slot[:, 0]
            route_slots = token_and_slot[:, 1]
            expert_output = expert(latent_TD.index_select(0, token_ids))
            route_weight = weights_TK[token_ids, route_slots].unsqueeze(-1)
            routed_TD = routed_TD.index_add(
                0, token_ids, expert_output.float() * route_weight
            )

        routed_BLD = routed_TD.to(latent_TD.dtype).view(B, L, -1)
        routed_BLD = self.routed_up(self.routed_norm(routed_BLD))
        return routed_BLD + self.shared_experts(x_BLD)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.tokens_per_expert_E.device
        self.tokens_per_expert_E = torch.zeros(
            self.num_experts, dtype=torch.float32, device=buffer_device
        )
        if self.load_balance_coeff is not None:
            self.expert_bias_E = torch.zeros(
                self.num_experts, dtype=torch.float32, device=buffer_device
            )


def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: KimiRMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32."""

    values_TND = torch.cat((block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.kimi_eps)
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
        attention_norm: KimiRMSNorm.Config
        ffn_norm: KimiRMSNorm.Config
        attention_res_norm: KimiRMSNorm.Config
        attention_res_proj: Linear.Config
        ffn_res_norm: KimiRMSNorm.Config
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
        # Blocks that do not extend the attention residual return it unchanged.
        # FSDP2 aliases module inputs to drive its backward hooks, so passing it
        # back out makes this FSDP unit return a view, which draws PyTorch's
        # warning about in-place ops dropping the pre-backward hook. Nothing
        # mutates it in place, and routing it back through the module boundary
        # is what keeps FSDP gradients bitwise equal to eager -- returning None
        # here instead reassociates the residual's gradient accumulation and
        # perturbs tok_embeddings.weight.grad by ~2e-3 relative.
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
            h_BLD = self.delta_attention(h_BLD, attention_masks, positions)
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
    """Reduced Kimi K3 multimodal model used for first-version validation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: KimiRMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None
        spatial_merge_size: int = 2

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            parallelism = config.parallelism
            unsupported = {
                "tensor parallel": parallelism.tensor_parallel_degree,
                "pipeline parallel": parallelism.pipeline_parallel_degree,
                "context parallel": parallelism.context_parallel_degree,
                "expert parallel": parallelism.expert_parallel_degree,
            }
            enabled = [name for name, degree in unsupported.items() if degree > 1]
            if enabled:
                raise NotImplementedError(
                    "Kimi K3 eager reference supports FSDP2 data parallelism only; "
                    f"disable {', '.join(enabled)}."
                )
            dataloader = getattr(config, "dataloader", None)
            if getattr(dataloader, "packing_buffer_size", 0) > 0:
                raise NotImplementedError(
                    "Kimi K3 v1 does not support packed documents."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention_config = self.first_attention
            if not isinstance(attention_config, KimiMLAAttention.Config):
                raise ValueError(
                    "Kimi K3 requires at least one MLA layer for FLOP accounting."
                )
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attention_config.num_heads,
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

    def _encode_placeholder_image(
        self,
        embeddings_BLD: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vision encoder on the smallest grid it can merge.

        A batch without images must still drive the vision encoder, because it
        is its own FSDP unit and the data-parallel ranks that do have images
        will issue its collectives. See ``add_zero_valued_dependency``.
        """
        assert self.vision_encoder is not None
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        patch_dim = self.vision_encoder.patch_embed.in_features
        pixel_values_NPK = torch.zeros(
            1,
            kernel_h * kernel_w,
            patch_dim,
            dtype=embeddings_BLD.dtype,
            device=embeddings_BLD.device,
        )
        grid_thw_N3 = torch.tensor(
            [[1, kernel_h, kernel_w]],
            dtype=torch.long,
            device=embeddings_BLD.device,
        )
        return self.vision_encoder(pixel_values_NPK, grid_thw=grid_thw_N3)

    def get_attention_masks(self, positions: torch.Tensor) -> AttentionMasksType | None:
        del positions
        return None

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
            if self.vision_encoder is None:
                return embeddings
            return add_zero_valued_dependency(
                embeddings,
                self._encode_placeholder_image(embeddings),
            )
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
        if not vision_positions:
            raise ValueError(
                "pixel_values were provided but no image placeholder tokens were found."
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
