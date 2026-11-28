# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inkling's text target: full-sequence attention and a recurrent middle stack.

B: batch, T: sequence, D: hidden, H: heads, K: head width, R: relative width,
O: relative offsets, U/W: query/key blocks, Q: residual queries,
S: residual source blocks, V: vocabulary.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

from torchtitan.experiments.inkling.experts import InklingMoE
from torchtitan.experiments.inkling.state_dict_adapter import InklingStateDictAdapter

from torchtitan.models.base.model import BaseModel, FlopsEstimator
from torchtitan.models.base.module import Module
from torchtitan.models.common.flops import quadratic_attention_flops
from torchtitan.models.common.linear import Linear


@dataclass(frozen=True, kw_only=True)
class InklingShape:
    hidden_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window: int
    full_layers: tuple[int, ...]
    num_experts: int
    experts_per_token: int
    intermediate_size: int
    padded_vocab_size: int
    vocab_size: int
    block_layers: int
    loop_start: int
    loop_end: int
    num_passes: int
    gate_rank: int
    nvfp4_layers: int
    conv_kernel: int = 4
    rms_eps: float = 1e-6
    qk_clip: float = math.sqrt(512)
    rel_clip: float = 8.0
    log_scaling_floor: int = 8192
    log_scaling_alpha: float = 0.1
    logit_softcap: float = 30.0

    @property
    def loop_layers(self) -> range:
        return range(self.loop_start, self.loop_end)


class _FrozenState(Module):
    """Published weights outside the adaptable projections, kept in checkpoint dtype."""

    def __init__(
        self, *, dtype: torch.dtype = torch.bfloat16, **shapes: tuple[int, ...]
    ):
        super().__init__()
        self._preserve_dtype_params = tuple(shapes)
        for name, shape in shapes.items():
            self.register_parameter(
                name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            )

    def reset_parameters(self) -> None:
        for value in self.parameters(recurse=False):
            nn.init.normal_(value, std=0.02)


class _Norm(_FrozenState):
    weight: torch.Tensor

    def __init__(self, width):
        super().__init__(weight=(width,))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def _rms(x, eps=1e-6, weight=None):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y if weight is None else weight * y).to(x.dtype)


def _block_residual(
    queries: torch.Tensor, blocks: list[torch.Tensor], partial_BTD: torch.Tensor
) -> torch.Tensor:
    """Each query attends to completed blocks and the partial block at its token.

    A single checkpoint query (D) returns BTD; a bank (QD) returns BTQD.
    """
    states_BTSD = _rms(torch.stack([*blocks, partial_BTD], dim=-2).float())
    width = partial_BTD.size(-1)
    scores_BTQS = queries.reshape(-1, width) @ states_BTSD.transpose(-1, -2)
    mixed_BTQD = (scores_BTQS * width**-0.5).softmax(-1) @ states_BTSD
    return mixed_BTQD.reshape(*partial_BTD.shape[:-1], *queries.shape)


def _short_conv(x_BTD, weight):
    kernel = weight.size(-1)
    sequence_BDT = F.pad(x_BTD.transpose(1, 2).float(), (kernel - 1, 0))
    conv_BDT = F.conv1d(sequence_BDT, weight.float(), groups=x_BTD.size(-1))
    return (
        (conv_BDT.transpose(1, 2) + x_BTD.float()) / math.sqrt(kernel + 1)
    ).bfloat16()


def _dynamic_conv(x_BTD, source_BTD, conv, heads):
    kernel = conv.static_w.size(0)
    batch, tokens, width = x_BTD.shape
    windows = F.pad(x_BTD.transpose(1, 2), (kernel - 1, 0)).unfold(-1, kernel, 1)
    windows = windows.transpose(1, 2).reshape(batch, tokens, heads, -1, kernel).flip(-1)
    taps = (
        F.linear(source_BTD, conv.weight_proj)
        .float()
        .reshape(batch, tokens, heads, 1, kernel)
    )
    taps = taps + conv.static_w.float().T.reshape(heads, -1, kernel)
    result_BTD = (windows.float() * taps).sum(-1).reshape(batch, tokens, width)
    return ((result_BTD + x_BTD.float()) / math.sqrt(2 * kernel + 1)).bfloat16()


def _clip(x, threshold):
    return x * (
        threshold / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-6)
    ).clamp(max=1)


_compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)


@lru_cache(maxsize=32)
@torch.inference_mode(False)  # Backward may reuse a mask first requested by inference.
def _inkling_block_mask(
    tokens: int, window: int | None, device: torch.device
) -> BlockMask:
    # The mask depends only on sequence length and window, not on the training row.
    # Classify blocks by their nearest and farthest token pairs, not a T x T mask.
    block_size = 128
    starts_U = torch.arange(0, tokens, block_size, device=device)
    ends_U = (starts_U + block_size - 1).clamp(max=tokens - 1)
    nearest_UW = starts_U[:, None] - ends_U[None, :]
    farthest_UW = ends_U[:, None] - starts_U[None, :]
    visible_UW = farthest_UW >= 0
    full_UW = nearest_UW >= 0
    if window is not None:
        visible_UW = visible_UW & (nearest_UW < window)
        full_UW = full_UW & (farthest_UW < window)

    def ordered(blocks_UW: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts_U = blocks_UW.sum(-1, dtype=torch.int32)
        indices_UW = blocks_UW.int().argsort(dim=-1, descending=True, stable=True)
        return counts_U[None, None], indices_UW.int()[None, None]

    def visible(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        distance = q_idx - kv_idx
        if window is None:
            return distance >= 0
        return (distance >= 0) & (distance < window)

    return BlockMask.from_kv_blocks(
        *ordered(visible_UW & ~full_UW),
        *ordered(full_UW),
        BLOCK_SIZE=block_size,
        mask_mod=visible,
        seq_lengths=(tokens, tokens),
    )


def _eager_inkling_attention(
    q_BHTK: torch.Tensor,
    k_BHTK: torch.Tensor,
    v_BHTK: torch.Tensor,
    bias_BHTO: torch.Tensor,
    sinks_BHT: torch.Tensor,
    *,
    window: int | None,
) -> torch.Tensor:
    """CPU reference; native FlexAttention backward requires an accelerator."""
    group = q_BHTK.size(1) // k_BHTK.size(1)
    keys_BHTK = k_BHTK.repeat_interleave(group, dim=1)
    values_BHTK = v_BHTK.repeat_interleave(group, dim=1)
    scores_BHTT = (q_BHTK @ keys_BHTK.transpose(-1, -2)) * q_BHTK.size(-1) ** -0.5
    pos_T = torch.arange(q_BHTK.size(-2), device=q_BHTK.device)
    distance_TT = pos_T[:, None] - pos_T[None, :]
    extent = bias_BHTO.size(-1)
    bias_BHTT = bias_BHTO.gather(
        -1, distance_TT.clamp(0, extent - 1).expand_as(scores_BHTT)
    )
    causal_TT = distance_TT >= 0
    in_extent_TT = causal_TT & (distance_TT < extent)
    scores_BHTT = scores_BHTT + bias_BHTT.masked_fill(~in_extent_TT, 0)
    visible_TT = causal_TT if window is None else causal_TT & (distance_TT < window)
    scores_BHTT = scores_BHTT.masked_fill(~visible_TT, -torch.inf)
    probs_BHTT = (
        torch.cat((scores_BHTT, sinks_BHT[..., None]), -1)
        .float()
        .softmax(-1)[..., :-1]
        .to(q_BHTK.dtype)
    )
    return probs_BHTT @ values_BHTK


def _inkling_attention(
    q_BTHK: torch.Tensor,
    k_BTHK: torch.Tensor,
    v_BTHK: torch.Tensor,
    rel_BTHR: torch.Tensor,
    rel_logits_RO: torch.Tensor,
    sinks_BHT: torch.Tensor,
    *,
    window: int | None,
) -> torch.Tensor:
    """Attend within each contiguous causal row, using BF16 projections."""
    tokens = q_BTHK.size(1)
    if window is not None and window >= tokens:
        window = None
    # No causal pair needs an offset >= tokens; avoid projecting those columns.
    bias_BHTO = (rel_BTHR @ rel_logits_RO[:, :tokens]).transpose(1, 2)
    q_BHTK, k_BHTK, v_BHTK = (x.transpose(1, 2) for x in (q_BTHK, k_BTHK, v_BTHK))
    if q_BTHK.device.type == "cpu":
        return _eager_inkling_attention(
            q_BHTK, k_BHTK, v_BHTK, bias_BHTO, sinks_BHT, window=window
        )
    if q_BTHK.device.type != "cuda":
        raise ValueError(
            "Inkling attention supports CPU reference and CUDA FlexAttention"
        )

    extent = bias_BHTO.size(-1)

    def score_mod(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        distance = q_idx - kv_idx
        bias = bias_BHTO[b, h, q_idx, distance.clamp(0, extent - 1)]
        return score + torch.where((distance >= 0) & (distance < extent), bias, 0)

    # Triton dot requires at least 16 lanes, including for the width-8 debug flavor.
    width = q_BHTK.size(-1)
    if width < 16:
        q_BHTK, k_BHTK, v_BHTK = (
            F.pad(x, (0, 16 - width)) for x in (q_BHTK, k_BHTK, v_BHTK)
        )
    # Triton backward supports gradients to tensors captured by score_mod.
    context_BHTK, aux = _compiled_flex_attention(
        q_BHTK,
        k_BHTK,
        v_BHTK,
        score_mod=score_mod,
        scale=width**-0.5,
        block_mask=_inkling_block_mask(tokens, window, q_BTHK.device),
        enable_gqa=True,
        return_aux=AuxRequest(lse=True),
        kernel_options={"BACKEND": "TRITON"},
    )
    # The learned sink has zero value. Restore the mass of the real keys after
    # adding the sink to the denominator: exp(lse) / (exp(lse) + exp(sink)).
    assert aux.lse is not None
    key_mass_BHT = (aux.lse - sinks_BHT.float()).sigmoid()
    return (context_BHTK[..., :width].float() * key_mass_BHT[..., None]).to(
        q_BTHK.dtype
    )


class InklingAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        wq_du: Linear.Config
        wk_dv: Linear.Config
        wv_dv: Linear.Config
        wo_ud: Linear.Config
        wr_du: Linear.Config
        gate_attn: Linear.Config
        attn_sink_proj: Linear.Config

    def __init__(self, config: Config, *, shape: InklingShape, full: bool):
        super().__init__()
        self.shape = shape
        self.full = full
        self.wq_du = config.wq_du.build()
        self.wk_dv = config.wk_dv.build()
        self.wv_dv = config.wv_dv.build()
        self.wo_ud = config.wo_ud.build()
        self.wr_du = config.wr_du.build()
        self.gate_attn = config.gate_attn.build()
        self.attn_sink_proj = config.attn_sink_proj.build()
        heads, kv, dim, kernel = (
            shape.num_q_heads,
            shape.num_kv_heads,
            shape.head_dim,
            shape.conv_kernel,
        )
        if kv < 1 or heads < 1 or heads % kv:
            raise ValueError(
                "Inkling query heads must be a positive multiple of KV heads"
            )
        extent = shape.rel_extent if full else shape.sliding_window
        if extent < 1:
            raise ValueError(
                "Inkling relative bias extent and sliding window must be positive"
            )
        self.q_l2_clip = _FrozenState(scale=(heads, dim))
        self.rel_l2_clip = _FrozenState(scale=(heads, shape.d_rel))
        self.rel_logits_proj = _FrozenState(proj=(shape.d_rel, extent))

        def kv_conv() -> _FrozenState:
            if full:
                return _FrozenState(
                    static_w=(kernel, kv * dim),
                    weight_proj=(kv * kernel, shape.hidden_size),
                )
            return _FrozenState(weight=(kv * dim, 1, kernel))

        self.k_sconv = kv_conv()
        self.v_sconv = kv_conv()

    def forward(self, q_in_BTD, k_in_BTD, v_in_BTD):
        s = self.shape
        device = q_in_BTD.device
        batch, tokens, _ = q_in_BTD.shape
        if tokens < 1:
            raise ValueError("Inkling attention requires at least one token")

        with torch.autocast(device.type, dtype=torch.bfloat16):
            q_BTHK = self.wq_du(q_in_BTD).reshape(
                batch, tokens, s.num_q_heads, s.head_dim
            )
            k_BTD, v_BTD = self.wk_dv(k_in_BTD), self.wv_dv(v_in_BTD)
            rel_BTHR = self.wr_du(q_in_BTD).reshape(
                batch, tokens, s.num_q_heads, s.d_rel
            )
            sinks_BHT = self.attn_sink_proj(q_in_BTD).transpose(1, 2)
            gate_BTD = self.gate_attn(q_in_BTD).float().sigmoid()
        q_BTHK = _clip(
            q_BTHK.float() * self.q_l2_clip.scale.float(), s.qk_clip
        ).bfloat16()
        rel_BTHR = _clip(
            rel_BTHR.float() * self.rel_l2_clip.scale.float(), s.rel_clip
        ).bfloat16()
        if self.full:
            pos_T = torch.arange(tokens, device=device)
            factor_T = (
                1
                + s.log_scaling_alpha
                * ((pos_T + 1).float() / s.log_scaling_floor).clamp(min=1).log()
            )
            q_BTHK = (q_BTHK.float() * factor_T[:, None, None]).bfloat16()
            rel_BTHR = (rel_BTHR.float() * factor_T[:, None, None]).bfloat16()
            k_BTD = _dynamic_conv(k_BTD, k_in_BTD, self.k_sconv, s.num_kv_heads)
            v_BTD = _dynamic_conv(v_BTD, v_in_BTD, self.v_sconv, s.num_kv_heads)
        else:
            k_BTD = _short_conv(k_BTD, self.k_sconv.weight)
            v_BTD = _short_conv(v_BTD, self.v_sconv.weight)
        k_BTHK = _clip(
            k_BTD.reshape(batch, tokens, s.num_kv_heads, s.head_dim).float(), s.qk_clip
        ).bfloat16()
        v_BTHK = _rms(v_BTD.reshape_as(k_BTHK), s.rms_eps)
        context_BHTK = _inkling_attention(
            q_BTHK,
            k_BTHK,
            v_BTHK,
            rel_BTHR,
            self.rel_logits_proj.get_parameter("proj"),
            sinks_BHT,
            window=None if self.full else s.sliding_window,
        )
        context_BTD = context_BHTK.transpose(1, 2).reshape(batch, tokens, -1)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return self.wo_ud((context_BTD.float() * gate_BTD).bfloat16())


class _Controller(_FrozenState):
    down_weight: torch.Tensor
    forget_weight: torch.Tensor
    gain_weight: torch.Tensor

    def forward(self, x_BTD, message_BTD, eps):
        x_BTD, message_BTD = x_BTD.float(), message_BTD.float()
        inputs = torch.cat([_rms(x_BTD, eps), _rms(message_BTD, eps)], -1)
        latent = F.linear(inputs.bfloat16(), self.down_weight).float()
        forget = F.linear(latent, self.forget_weight.float()).tanh()
        gain = F.linear(latent, self.gain_weight.float()).tanh()
        return (1 - forget) * x_BTD + gain * message_BTD


class InklingLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attn: InklingAttention.Config
        mlp: InklingMoE.Config

    def __init__(self, config, *, shape, index):
        super().__init__()
        self.shape, self.index = shape, index
        self.attn = config.attn.build(shape=shape, full=index in shape.full_layers)
        self.mlp = config.mlp.build()
        width = shape.hidden_size
        self.mlp_norm = _Norm(width)
        self.attn_sconv = _FrozenState(weight=(width, 1, shape.conv_kernel))
        self.mlp_sconv = _FrozenState(weight=(width, 1, shape.conv_kernel))
        if self.attn.full:
            self.q_input_norm, self.k_input_norm, self.v_input_norm = (
                _Norm(width) for _ in range(3)
            )
        else:
            self.attn_norm = _Norm(width)
        if index in shape.loop_layers:
            self.controllers = nn.ModuleDict(
                {
                    name: _Controller(
                        down_weight=(shape.gate_rank, 2 * width),
                        forget_weight=(width, shape.gate_rank),
                        gain_weight=(width, shape.gate_rank),
                    )
                    for name in ("attn", "k", "v", "mlp")
                }
            )

    def forward(self, h_BTD, blocks, messages_BTQD, attn_query, mlp_query):
        eps = self.shape.rms_eps
        if self.attn.full:
            streams = _block_residual(attn_query, blocks, h_BTD).unbind(-2)
            norms = (self.q_input_norm, self.k_input_norm, self.v_input_norm)
        else:
            stream = (
                h_BTD
                if attn_query is None
                else _block_residual(attn_query, blocks, h_BTD)
            )
            streams, norms = (stream,) * 3, (self.attn_norm,) * 3
        if messages_BTQD is not None:
            streams = [
                self.controllers[name](stream, messages_BTQD[..., i, :], eps)
                for i, (name, stream) in enumerate(zip(("attn", "k", "v"), streams))
            ]
        attention_BTD = self.attn(
            *[
                _rms(stream, eps, norm.weight).bfloat16()
                for stream, norm in zip(streams, norms)
            ]
        )
        residual_BTD = _short_conv(attention_BTD, self.attn_sconv.weight).float()
        if self.index % self.shape.block_layers == 0:
            blocks = [*blocks, h_BTD]
        else:
            residual_BTD = h_BTD.float() + residual_BTD
        mlp_in_BTD = _block_residual(mlp_query, blocks, residual_BTD)
        if messages_BTQD is not None:
            mlp_in_BTD = self.controllers["mlp"](
                mlp_in_BTD, messages_BTQD[..., 3, :], eps
            )
        mlp_BTD = self.mlp(_rms(mlp_in_BTD, eps, self.mlp_norm.weight).bfloat16())
        return (
            residual_BTD + _short_conv(mlp_BTD, self.mlp_sconv.weight).float()
        ).bfloat16(), blocks


class InklingModel(BaseModel):
    state_dict_adapter_cls = InklingStateDictAdapter
    _preserve_dtype_params = (
        "middle_cycle_cross_pass_queries",
        "middle_cycle_cross_pass_kv_queries",
    )
    middle_cycle_cross_pass_queries: nn.Parameter
    middle_cycle_cross_pass_kv_queries: nn.Parameter

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        shape: InklingShape
        layers: list[InklingLayer.Config]
        unembed: Linear.Config

        def get_parameter_counts(self, model):
            s = self.shape
            # Packed gate/up/down matrices represent 3*D*I scalars per expert.
            expert_count = 3 * s.hidden_size * s.intermediate_size * s.num_layers
            other = sum(
                t.numel()
                for name, t in model.state_dict().items()
                if ".experts." not in name
            )
            return (
                other + expert_count * s.num_experts,
                other + expert_count * s.experts_per_token,
            )

        def build_flops_estimator(
            self, model: Module, *, seq_len: int
        ) -> FlopsEstimator:
            s = self.shape
            model = cast("InklingModel", model)

            # Count forward and dX for each projection, plus dW for trainable
            # weights. This estimate does not prune dX on frozen-only inputs
            # or include block-residual mixing, elementwise work, or remat.
            def projection_flops(module: nn.Module) -> int:
                return sum(
                    (6 if weight.requires_grad else 4) * weight.numel()
                    for weight in module.parameters()
                    if weight.ndim > 1
                )

            # The output projection computes the padded vocabulary before slicing.
            flops_per_token = projection_flops(model.unembed)
            num_full_visits = num_local_visits = 0
            for module in model.layers:
                layer = cast(InklingLayer, module)
                attn = layer.attn
                # Messages and controllers start on the second core visit.
                num_controller_visits = (
                    s.num_passes - 1 if layer.index in s.loop_layers else 0
                )
                num_visits = 1 + num_controller_visits
                if attn.full:
                    num_full_visits += num_visits
                else:
                    num_local_visits += num_visits
                dense_flops = sum(
                    projection_flops(projection)
                    for projection in attn.children()
                    if isinstance(projection, Linear)
                )
                conv_flops = sum(
                    projection_flops(conv)
                    for conv in (
                        attn.k_sconv,
                        attn.v_sconv,
                        layer.attn_sconv,
                        layer.mlp_sconv,
                    )
                )
                moe = layer.mlp.config
                weights_per_expert = 3 * moe.hidden_size * moe.intermediate_size
                flops_per_token += num_visits * (
                    dense_flops
                    + conv_flops
                    + 4
                    * s.num_q_heads
                    * s.d_rel
                    * min(seq_len, s.rel_extent if attn.full else s.sliding_window)
                    + projection_flops(layer.mlp.gate)
                    + 4 * weights_per_expert * moe.experts_per_token
                )
                if num_controller_visits:
                    flops_per_token += num_controller_visits * projection_flops(
                        layer.controllers
                    )

            def estimate(batch: Mapping[str, torch.Tensor]) -> int:
                keys_T = batch["positions"][~batch["padding_mask"]] + 1
                return flops_per_token * keys_T.numel() + quadratic_attention_flops(
                    num_heads=s.num_q_heads,
                    qk_head_dim=s.head_dim,
                    v_head_dim=s.head_dim,
                    num_keys=num_full_visits * keys_T
                    + num_local_visits * keys_T.clamp(max=s.sliding_window),
                )

            return estimate

    def __init__(self, config):
        super().__init__()
        self.config = config
        s = config.shape
        width = s.hidden_size
        self.embed = _FrozenState(weight=(s.padded_vocab_size, width))
        self.embed_norm, self.norm = _Norm(width), _Norm(width)
        self.unembed = config.unembed.build()
        self.layers = nn.ModuleList(
            layer.build(shape=s, index=i) for i, layer in enumerate(config.layers)
        )
        self.attn_res = _FrozenState(dtype=torch.float32, output_query=(width,))
        self.attn_res.mlp_queries = _FrozenState(
            dtype=torch.float32, **{str(i): (width,) for i in range(s.num_layers)}
        )
        self.attn_res.attn_queries = _FrozenState(
            dtype=torch.float32,
            **{
                str(i - 1): (3, width) if i in s.full_layers else (width,)
                for i in range(1, s.num_layers)
            },
        )
        for name in self._preserve_dtype_params:
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(2 * len(s.loop_layers), width, dtype=torch.float32),
                    requires_grad=False,
                ),
            )

    def reset_parameters(self) -> None:
        for value in self.parameters(recurse=False):
            nn.init.normal_(value, std=0.02)

    def _apply_fsdp(self, *, parallelism_context, precision, parallelism):
        if parallelism_context.world_size != 1:
            raise ValueError("Inkling supports single-device training only")

    def preprocess_inputs(self, input_dict, **kwargs):
        tokens_BT, labels_BT = input_dict["input"], input_dict["labels"]
        positions = input_dict.get("positions")
        padding = input_dict.get("padding_mask")
        invalid = None
        if positions is not None:
            invalid = positions.reshape_as(tokens_BT) != torch.arange(
                tokens_BT.size(-1), device=tokens_BT.device
            )
        if padding is not None:
            padding = padding.reshape_as(tokens_BT)
            # A token following a pad would attend to it and read its conv state.
            interior_pad = (padding[..., :-1] & ~padding[..., 1:]).any()
            invalid = (
                interior_pad
                if invalid is None
                else (invalid & ~padding).any() | interior_pad
            )
        if invalid is not None and invalid.any():
            raise ValueError(
                "Inkling requires one contiguous document per training row"
            )
        return tokens_BT, labels_BT, {}, input_dict.get("loss_kwargs", {})

    def forward(self, tokens_BT):
        flat = tokens_BT.ndim == 1
        if flat:
            tokens_BT = tokens_BT[None]
        s = self.config.shape
        h_BTD = _rms(
            F.embedding(tokens_BT, self.embed.get_parameter("weight")),
            s.rms_eps,
            self.embed_norm.weight,
        )
        blocks = []
        q, mlp = self.middle_cycle_cross_pass_queries.unflatten(0, (-1, 2)).unbind(1)
        k, v = self.middle_cycle_cross_pass_kv_queries.unflatten(0, (-1, 2)).unbind(1)
        message_queries = torch.stack((q, k, v, mlp), 1)

        def visit(i, h_BTD, blocks, messages=None):
            query = getattr(self.attn_res.attn_queries, str(i - 1)) if i else None
            return self.layers[i](
                h_BTD,
                blocks,
                messages,
                query,
                getattr(self.attn_res.mlp_queries, str(i)),
            )

        for i in range(s.loop_start):
            h_BTD, blocks = visit(i, h_BTD, blocks)
        prelude_BTD, prelude_blocks = h_BTD, blocks
        for p in range(s.num_passes):
            # The first core block is the shared prelude output, not a pass-specific message.
            previous_BTD, previous_blocks = h_BTD, blocks[len(prelude_blocks) + 1 :]
            h_BTD, blocks = prelude_BTD, prelude_blocks
            for core_idx, i in enumerate(s.loop_layers):
                messages = (
                    _block_residual(
                        message_queries[core_idx], previous_blocks, previous_BTD
                    )
                    if p
                    else None
                )
                h_BTD, blocks = visit(i, h_BTD, blocks, messages)
        for i in range(s.loop_end, s.num_layers):
            h_BTD, blocks = visit(i, h_BTD, blocks)
        final_BTD = _block_residual(
            self.attn_res.get_parameter("output_query"), blocks, h_BTD
        )
        final_BTD = _rms(final_BTD, s.rms_eps, self.norm.weight).bfloat16()
        with torch.autocast(tokens_BT.device.type, dtype=torch.bfloat16):
            scores_BTV = self.unembed(final_BTD)[..., : s.vocab_size]
        logits_BTV = (scores_BTV.float() / s.logit_softcap).tanh() * s.logit_softcap
        return logits_BTV[0] if flat else logits_BTV
