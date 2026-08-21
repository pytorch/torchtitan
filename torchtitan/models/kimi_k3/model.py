# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchtitan-idiom port of MoonshotAI/Kimi-Linear.

Reference: ``reference/modeling_kimi.py`` (verbatim fork from HF
``moonshotai/Kimi-Linear-48B-A3B-Base``). We keep the HF code for
diffing but do NOT import it — the HF version assumes Transformers'
PreTrainedModel + Cache, which don't compose with torchtitan's
trainer, FSDP, PP, or cache adapter.

Architectural faithfulness (per Kimi Linear tech report §5):

* Every layer's attention is EITHER :class:`KimiDeltaAttention` (KDA,
  linear-attention variant via fla-core) OR :class:`KimiMLAAttention`
  (NoPE MLA, faithful to Kimi's spec — not the DSv3 MLA in
  ``torchtitan.models.deepseek_v3``). Alternation pattern is
  layer-index-driven by ``config.kda_layers`` / ``config.full_attn_layers``.
* Every layer's FFN is EITHER :class:`KimiMLP` (dense SwiGLU, used on
  the first ``first_k_dense_replace`` layers) OR :class:`KimiMoE`
  (sparse sigmoid-gated grouped-topk, composed from torchtitan's
  common :class:`TokenChoiceTopKRouter` + :class:`GroupedExperts`
  infrastructure to get a training-capable forward that the HF
  release lacks).
* Pre-norm + residual structure identical to Kimi's reference.

AttnRes weaving is implemented as a separate subclass in
``attn_res_model.py``, matching the ``AttnResLlama3Model`` pattern
this experiment grew out of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from functools import partial
from typing import Literal

import spmd_types as spmd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.sharding import (
    contract_for_mode,
    HEAD_DIM,
    SEQ_DIM,
    ULYSSES,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig


def _vocab_parallel_embedding() -> ShardingConfig:
    """Vocab-sharded embedding: weight S(0) on tp, output Partial on tp.

    Both halves are required and neither is optional. Embedding.forward takes its
    vocab-parallel branch whenever a tp group exists; that branch indexes the weight
    with ``input - rank * ceil(vocab / tp)``, so the rows it holds must BE that chunk
    (the S(0) half), and it zeroes ids outside its range, so the per-rank results are
    partial sums that something has to add up (the P half).

    Upstream declares tok_embeddings exactly this way. Declaring only the weight leaves
    every rank holding its own slice's contribution with nothing summing them.
    """
    embed_input = dense_activation_placement(tp=spmd.R)
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings={"input": embed_input},
        in_dst_shardings={"input": embed_input},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.R),
        local_map=LocalMapConfig(in_grad_placements=None),
    )


def _tp_shard(dim: int) -> ShardingConfig:
    """Weight sharded on ``dim`` of the tp axis; colwise is 0, rowwise is 1."""
    return ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.S(dim))}
    )


def _tp_replicate() -> ShardingConfig:
    """Weight replicated on the tp axis (the NoParallel case)."""
    return ShardingConfig(state_shardings={"weight": dense_param_placement(tp=spmd.R)})


try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError as err:  # pragma: no cover - import-time guard
    raise ImportError(
        "Kimi Linear KDA path requires fla-core. Run `pip install fla-core`."
    ) from err


def splice_vision_embeds(
    h: torch.Tensor,
    vision_embeds: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    """Write ``vision_embeds`` into ``h`` at the positions ``image_mask`` marks.

    ``h[image_mask] = vision_embeds.reshape(-1, D)`` is the obvious spelling and
    is wrong in three ways this handles:

    * Under PP shape inference the scheduler runs forward once on zero-filled
      tokens, so the mask is all False and advanced-index assignment raises a
      shape mismatch. ``masked_scatter`` copies as many elements as the mask
      asks for, which is none.
    * The reshape assumes every row holds exactly ``vision_embeds.size(1)``
      image tokens. True for single-image data, false as soon as a batch mixes
      text-only rows or multi-image rows, so the source is filtered to the
      leading ``n`` slots of each row first.
    * Destinations must equal sources or ``masked_scatter`` trips a CUDA
      device-side assert (``masked_scatter_size_check``) that surfaces
      asynchronously in whatever kernel runs next -- typically a linear or an
      FSDP all-gather, which makes an embed-scatter mismatch look like an
      attention or MoE bug. A row can hold more sentinels than there are embeds
      when a text token tokenizes to the sentinel id, so destinations are capped
      per row. Surplus positions keep their text embedding, which is correct:
      they are text tokens that collided with the sentinel, not image slots.
    """
    n_per_row = image_mask.sum(dim=1)
    n_vis_max = vision_embeds.size(1)
    arange = torch.arange(n_vis_max, device=image_mask.device)
    valid = arange.unsqueeze(0) < n_per_row.unsqueeze(1)
    source = vision_embeds[valid].to(h.dtype)
    # Computed unconditionally rather than behind ``if (n_per_row >
    # n_vis_max).any()``. That test is a device-to-host sync on the embed path,
    # once per microbatch, and the two branches agree anyway: with no row over
    # the limit, every True has pos_rank < n_per_row, so the mask is unchanged.
    pos_rank = image_mask.long().cumsum(dim=1) - 1  # 0-based rank within the row
    keep = torch.clamp(n_per_row, max=n_vis_max)
    scatter_mask = image_mask & (pos_rank < keep.unsqueeze(1))
    return h.masked_scatter(scatter_mask.unsqueeze(-1).expand_as(h), source)


@dataclass(kw_only=True, slots=True)
class KimiK3Config:
    """Torchtitan-flavored config for Kimi Linear.

    Mirrors ``reference/configuration_kimi.py:KimiK3Config`` but
    as a plain dataclass (no HF ``PretrainedConfig`` machinery). All
    fields kept identical to the HF config.json knobs for the 48B-A3B
    release; scaling-law variants (194M..528M) override the ones that
    change per size (hidden_size, num_hidden_layers, etc.).

    The 1-indexed ``kda_layers`` / ``full_attn_layers`` convention is
    preserved from the HF config.json (so literal copy-paste from HF
    works).

    This class carries the Kimi model hyperparameters only. The
    torchtitan ``BaseModel.Config`` shim — ``KimiK3Spec`` — lives
    in this module below and wraps one of these for ModelSpec
    registration.
    """

    # ---- vocabulary / embedding ----
    vocab_size: int = 163840
    hidden_size: int = 2304
    tie_word_embeddings: bool = False

    # ---- depth / width ----
    num_hidden_layers: int = 27
    intermediate_size: int = 9216  # dense MLP intermediate (layer 0 + shared experts)

    # ---- MLA (full-attn layers) ----
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # no GQA for Kimi 48B
    q_lora_rank: int | None = None  # None = no Q compression
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    # Gated MLA (K3 delta): sigmoid output gate, near-identity init so a
    # non-gated-MLA-pretrained checkpoint's function is ~preserved at
    # step 0 (graft-viable: a near-identity gate init keeps the
    # pretrained function intact). PROVISIONAL: exact gate form
    # reconciles at 7.27. Off by default (plain MLA = validated path).
    mla_gated: bool = False
    rope_theta: float = 10000.0
    # Declared context length. Nothing in the forward consumes it -- the model
    # is NoPE (MLA applies no positional encoding; KDA carries position in its
    # recurrence), which is exactly why K3 can state 1M without retuning a RoPE
    # base or applying YaRN (report sec 2.1.2). Kept so a flavor records the
    # official 1048576 and downstream tooling (dataloader, eval) can read it.
    max_position_embeddings: int = 4096

    # ---- KDA (linear-attn layers) ----
    # linear_attn_config structure preserved from HF config.json
    kda_num_heads: int = 32
    kda_head_dim: int = 128
    kda_short_conv_kernel_size: int = 4
    # 1-indexed layer lists
    kda_layers: list[int] = field(default_factory=list)
    full_attn_layers: list[int] = field(default_factory=list)

    # ---- MoE ----
    num_experts: int | None = 256
    num_experts_per_token: int = 8
    moe_intermediate_size: int = 1024
    moe_renormalize: bool = True
    moe_router_activation_func: Literal["sigmoid", "softmax"] = "sigmoid"
    num_shared_experts: int = 1
    routed_scaling_factor: float = 2.446
    first_k_dense_replace: int = 1
    # Multi-token prediction. Report Table 1 lists one MTP layer; the RELEASED
    # config.json ships num_nextn_predict_layers: 0, so the artifact trains
    # without it. Default 0 to match what can actually be loaded; set 1 to
    # build the architecture the report describes.
    num_nextn_predict_layers: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    # Wired by KimiK3Spec.update_from_config from config.parallelism
    # BEFORE build; consumed by KimiMoE to populate the upstream
    # module-internal MoE sharding configs (EP/TP). False = the
    # previously validated FSDP/PP plain path, untouched.
    moe_enable_ep: bool = False
    moe_enable_tp: bool = False
    topk_group: int = 1

    # ---- norm / act ----
    rms_norm_eps: float = 1e-5
    hidden_act: Literal["silu", "gelu", "situ"] = "silu"
    # SiTU (Sigmoid Tanh Unit), K3's activation. Official config.json ships
    # activation_situ_beta=4.0 and activation_situ_linear_beta=25.0; both are
    # only read when hidden_act == "situ".
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float | None = 25.0
    # Output-gate parameterization for the gated MLA / KDA paths.
    # "full_rank" is K3's (tech report Eq. 6/7): an input-dependent
    # channel-wise projection, sigmoid, applied to the attention output
    # before W_o. "per_head_graft" is this repo's near-identity variant:
    # one scalar per head with a +LARGE bias so sigmoid(.) ~= 1, which makes
    # a graft onto pretrained weights an exact no-op at step 0. Use
    # full_rank for K3 fidelity, per_head_graft for grafting experiments.
    attn_gate_param: Literal["full_rank", "per_head_graft"] = "full_rank"

    # ---- Stable LatentMoE (K3 tech report sec 2.3, Eq. 11) ----
    # Routed experts operate in a compact latent space of width
    # ``routed_expert_hidden_size`` (K3: 3584 against hidden 7168), entered and
    # left through two SHARED projections, with an RMSNorm on the aggregated
    # routed representation before the up-projection:
    #     u = sum_{i in Tk(x)} p_i * E_i^routed(W_down x)
    #     y = sum_j E_j^shared(x) + W_up RMSNorm(u)
    # Shared experts stay full width. None disables the latent path (the
    # conventional MoE this repo shipped before the official release).
    routed_expert_hidden_size: int | None = None
    latent_moe_use_norm: bool = True

    # ---- KDA parameterization (K3 tech report sec 2.1.1) ----
    # Eq. 5, lower-bounded decay. Kimi Linear used the unbounded
    # g = -exp(A) * Softplus(z); K3 bounds it from below with a scaled
    # sigmoid, g = g_min * Sigmoid(exp(A) z) in (g_min, 0), which keeps the
    # reciprocal chunk rescaling inside the bf16 range and lets every causal
    # tile use dense Tensor Core matmuls. Official value: -5.0. None keeps
    # the Kimi Linear form. fla-core implements both (ops/kda/gate.py).
    kda_gate_lower_bound: float | None = None
    # Which CP scheme the KDA layers use. "kcp" is report sec 5.1.2 and the
    # default: the sequence stays sharded end to end via a prefix scan over
    # state fragments plus a conv halo (see kcp.py). "ulysses" all-to-alls the
    # head axis instead, so every rank materializes the WHOLE sequence for its
    # head subset -- which means activation memory does not fall with cp, and a
    # 1M-token context is out of reach. It is kept as the validated A/B, not as
    # a production path, and it is not what K3 does.
    #
    # The MLA layers are Ulysses either way, and that is not an alternative to
    # this field: KCP decomposes the delta-rule recurrence and has nothing to say
    # about softmax attention. A CP run is KCP on the KDA layers AND Ulysses on
    # the MLA layers, together.
    kda_cp_mode: str = "kcp"
    # Eq. 6, output gate. Kimi Linear used a low-rank projection; K3 uses an
    # input-dependent FULL-RANK one: y = W_o[Sigmoid(W_g x) (.) RMSNorm(o~)].
    kda_use_full_rank_gate: bool = False

    # ---- init ----
    initializer_range: float = 0.02

    # Derived convenience
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def is_mla(self) -> bool:
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope
        )

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None and self.num_experts > 0

    def is_kda_layer(self, layer_idx: int) -> bool:
        """1-indexed match, preserving HF config.json convention."""
        return (layer_idx + 1) in self.kda_layers


# ----- RMSNorm ------------------------------------------------------------- #
# Use torch's ``nn.RMSNorm`` directly. Faithful to HF reference's
# ``KimiRMSNorm`` (same math: fp32 variance, cast back to input dtype).
# ``torchtitan.models.common.rmsnorm.RMSNorm`` is a Module-protocol
# wrapper around ``nn.RMSNorm``; we don't need the Config plumbing here
# since we're not going through the torchtitan Config.build() chain for
# the ported Kimi Linear backbone.


def _leave_for_checkpoint(tensor: torch.Tensor) -> torch.Tensor:
    """Init function for packed quantized bytes: deliberately does nothing.

    Present so the declarative init map covers every parameter name a packed
    module can carry. A missing name raises; this says "the checkpoint owns
    these bytes" out loud instead.
    """
    return tensor


# ----- SiTU activation ---------------------------------------------------- #


def situ_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """K3's Sigmoid Tanh Unit, gated form (reference: SituAndMul).

    ``situ(g) = beta * tanh(g / beta) * sigmoid(g)`` -- a soft-clipped SiLU:
    the tanh caps the magnitude at +/- beta while sigmoid keeps the SiLU-like
    gating shape near 0. When ``linear_beta`` is set the linear branch is
    clipped the same way before the product. Computed in fp32 and cast back,
    as the reference does, because the product of two saturating nonlinearities
    is sensitive to bf16 rounding near the caps.
    """
    g = gate.float()
    u = up.float()
    out = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return (out * u).to(gate.dtype)


# ----- Dense SwiGLU MLP --------------------------------------------------- #


class KimiMLP(FeedForward):
    """SwiGLU dense FFN. Used for layer 0 (pre-MoE dense replace) AND
    as the shared-experts module in MoE layers.

    Faithful to ``reference:KimiMLP`` (gate_proj, up_proj, down_proj), and reusing
    ``common.FeedForward`` for the plain SwiGLU case -- finding 7, which the maintainer
    raised as "should use our fused feed forward".

    The reuse does NOT require renaming the projections, which is what made this look like
    a checkpoint migration. ``FeedForward.forward`` is
    ``w2(silu(w1(x)) * w3(x))`` and only READS w1/w2/w3, so the real modules stay
    registered under the release's names and w1/w2/w3 are read-only properties over them.
    Properties are not in ``_modules``, so every state-dict key is unchanged and no DCP
    checkpoint moves. Same mechanism as ``UpstreamFSDPNames``.

    ``forward`` is inherited for ``silu`` and overridden otherwise: ``gelu`` swaps the
    activation, and ``situ`` (report sec 4.1) is gated over BOTH branches -- it clips the
    linear branch too -- so it is not expressible as an activation swap inside the shared
    forward at all.
    """

    # w1/w2/w3 name what FeedForward.forward reads; gate/up/down are what the checkpoint
    # calls them. Read-only on purpose: FeedForward never assigns to them.
    @property
    def w1(self) -> nn.Module:
        return self.gate_proj

    @property
    def w2(self) -> nn.Module:
        return self.down_proj

    @property
    def w3(self) -> nn.Module:
        return self.up_proj

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        """Config-driven construction, inheriting w1/w2/w3 from the parent.

        The FIELDS are core's w1/w2/w3 so that core's ``set_dense_ffn_sharding``
        and the rest of ``decoder_sharding`` apply to this config unchanged. The
        ATTRIBUTES the fields build into keep the release's names -- w1 becomes
        ``gate_proj``, w3 ``up_proj``, w2 ``down_proj`` -- so no checkpoint key
        moves. That works because a declaration rides on the ``Linear.Config``
        instance rather than on the attribute it lands in.
        """

        hidden_act: Literal["silu", "gelu", "situ"] = "silu"
        situ_beta: float = 4.0
        situ_linear_beta: float | None = 25.0

    @staticmethod
    def make_config(
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Literal["silu", "gelu", "situ"] = "silu",
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
    ) -> "KimiMLP.Config":
        """The dimensions-in form, until the flavor builder owns the tree.

        Callers still think in dimensions; this is the one place that turns them
        into the three ``Linear.Config``s, so hoisting the whole tree into a
        flavor builder later is a move rather than a rewrite.
        """

        def _lin(fan_in: int, fan_out: int, dim: int) -> Linear.Config:
            return Linear.Config(
                in_features=fan_in,
                out_features=fan_out,
                bias=False,
                sharding_config=_tp_shard(dim),
            )

        return KimiMLP.Config(
            w1=_lin(hidden_size, intermediate_size, 0),
            w3=_lin(hidden_size, intermediate_size, 0),
            w2=_lin(intermediate_size, hidden_size, 1),
            hidden_act=hidden_act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def __init__(self, config: "KimiMLP.Config") -> None:
        # Skip FeedForward.__init__, which would build w1/w2/w3 as attributes of
        # those names. This class owns the release's names, so only the forward is
        # inherited; the grandparent call keeps torchtitan's Module setup.
        super(FeedForward, self).__init__()
        self.gate_proj = config.w1.build()
        self.up_proj = config.w3.build()
        self.down_proj = config.w2.build()
        hidden_act = config.hidden_act
        self.hidden_act = hidden_act
        self._situ_beta = config.situ_beta
        self._situ_linear_beta = config.situ_linear_beta
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        elif hidden_act == "situ":
            # SiTU is gated over BOTH branches, so there is no elementwise
            # act_fn to apply to the gate alone; forward dispatches instead.
            self.act_fn = None
        else:
            raise ValueError(f"Unknown hidden_act: {hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "silu":
            # The shared implementation, verbatim.
            return super().forward(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.hidden_act == "situ":
            return self.down_proj(
                situ_and_mul(gate, up, self._situ_beta, self._situ_linear_beta)
            )
        return self.down_proj(self.act_fn(gate) * up)


# ----- MLA (NoPE variant) -------------------------------------------------- #


def _cp_all_to_all_headseq(
    x: torch.Tensor, cp_group, *, src_dim: int, dst_dim: int
) -> torch.Tensor:
    """Differentiable Ulysses all-to-all moving the CP shard between tensor dims.

    ``(1, 2)``: ``[B, T/cp, H, K]`` (seq-sharded) -> ``[B, T, H/cp, K]``.
    ``(2, 1)``: ``[B, T, H/cp, K]`` -> ``[B, T/cp, H, K]``.

    The dims come from the CP contract's placement pair rather than a flag, so a
    contract that names a pair with no implementation raises here instead of being
    quietly ignored.

    Numerics (round-trip and per-head chunk_kda parity) validated
    bit-exact against a single-rank reference; backward is the
    transposed all-to-all via torch.distributed.nn.functional.
    """
    import torch.distributed.nn.functional as dist_nn

    if (src_dim, dst_dim) not in ((SEQ_DIM, HEAD_DIM), (HEAD_DIM, SEQ_DIM)):
        raise ValueError(
            f"no Ulysses all-to-all for CP shard dims {src_dim} -> {dst_dim}; "
            f"implemented pairs are {SEQ_DIM} <-> {HEAD_DIM}"
        )
    cp = dist.get_world_size(cp_group)
    B, d1, d2, K = x.shape
    if (src_dim, dst_dim) == (SEQ_DIM, HEAD_DIM):
        t_loc, num_heads = d1, d2
        # [B, T/cp, H, K] -> [cp, B, T/cp, H/cp, K] (split heads by dest)
        x_split = (
            x.reshape(B, t_loc, cp, num_heads // cp, K)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        out = dist_nn.all_to_all_single(
            torch.empty_like(x_split), x_split, group=cp_group
        )
        # recv[s] holds src s's T/cp for THIS rank's head subset -> stack seq
        return (
            out.permute(1, 0, 2, 3, 4)
            .reshape(B, cp * t_loc, num_heads // cp, K)
            .contiguous()
        )
    t_full, h_loc = d1, d2
    t_loc = t_full // cp
    x_split = x.reshape(B, cp, t_loc, h_loc, K).permute(1, 0, 2, 3, 4).contiguous()
    out = dist_nn.all_to_all_single(torch.empty_like(x_split), x_split, group=cp_group)
    # out[s] = src s's head subset for THIS rank's seq shard; put T/cp
    # before the src(cp) axis so reshape stacks heads in ascending order.
    return out.permute(1, 2, 0, 3, 4).reshape(B, t_loc, cp * h_loc, K).contiguous()


class KimiMLAAttention(Module):
    """Multi-head Latent Attention, Kimi NoPE variant.

    Faithful port of ``reference:KimiMLAAttention``. Key differences
    vs. DSv3 MLA:

    * ``q_lora_rank`` — when None, Q is projected directly to
      ``num_heads x q_head_dim`` (the 48B-A3B path). When set (K3 ships
      1536) Q goes through the compression pair
      ``q_a_proj -> q_a_layernorm -> q_b_proj``, mirroring DSv3.
    * ``mla_use_nope=True`` — no RoPE applied; the "rot" split is
      vestigial naming. Position info carried by the KDA recurrence.
    * K is split into ``kv_lora_rank + qk_rope_head_dim`` halves from
      ``kv_a_proj_with_mqa``; the "rope" half is broadcast across
      heads (not per-head), matching Kimi's structural choice.

    No cache path — we only support training-time forward. HF's
    ``past_key_values`` / ``Cache`` machinery is not ported since
    torchtitan training doesn't invoke incremental decoding.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Config-driven Gated MLA.

        The Q path is either a single projection or the low-rank pair, so those
        child configs are optional and exactly one group is populated; the same
        is true of the output gate, which only exists when the layer is gated.
        """

        layer_idx: int
        hidden_size: int
        num_attention_heads: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        mla_use_nope: bool
        kv_a_proj_with_mqa: "Linear.Config"
        kv_a_layernorm: "RMSNorm.Config"
        kv_b_proj: "Linear.Config"
        o_proj: "Linear.Config"
        q_lora_rank: int | None = None
        mla_gated: bool = False
        attn_gate_param: str = "full_rank"
        q_proj: "Linear.Config | None" = None
        q_a_proj: "Linear.Config | None" = None
        q_a_layernorm: "RMSNorm.Config | None" = None
        q_b_proj: "Linear.Config | None" = None
        attn_gate_proj: "Linear.Config | None" = None
        inner_attention: "ScaledDotProductAttention.Config" = field(
            default_factory=ScaledDotProductAttention.Config
        )

    @staticmethod
    def make_config(config: KimiK3Config, layer_idx: int) -> "KimiMLAAttention.Config":
        """Turn the flat model config into this module's config tree.

        The one place that reads the flat config for MLA, so hoisting the tree
        into a flavor builder later is a move rather than a rewrite.
        """
        heads = config.num_attention_heads
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        def _lin(fan_in, fan_out, *, bias=False, sharding=None):
            return Linear.Config(
                in_features=fan_in,
                out_features=fan_out,
                bias=bias,
                sharding_config=sharding,
            )

        cfg = KimiMLAAttention.Config(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=heads,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            mla_use_nope=config.mla_use_nope,
            q_lora_rank=config.q_lora_rank,
            mla_gated=config.mla_gated,
            attn_gate_param=config.attn_gate_param,
            kv_a_proj_with_mqa=_lin(
                config.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                sharding=_tp_replicate(),
            ),
            kv_a_layernorm=RMSNorm.Config(
                normalized_shape=config.kv_lora_rank,
                eps=config.rms_norm_eps,
                sharding_config=_tp_replicate(),
            ),
            kv_b_proj=_lin(
                config.kv_lora_rank,
                heads * (config.qk_nope_head_dim + config.v_head_dim),
                sharding=_tp_shard(0),
            ),
            o_proj=_lin(
                heads * config.v_head_dim,
                config.hidden_size,
                sharding=_tp_shard(1),
            ),
        )
        if config.q_lora_rank is None:
            # 48B-A3B path: Q straight to H * q_head_dim.
            cfg.q_proj = _lin(config.hidden_size, heads * q_head_dim)
        else:
            # K3 path (official config: q_lora_rank=1536). Same shape as DSv3's
            # wq_a/wq_b pair: the compression stays replicated because its output
            # is the lora rank, not a head axis, and only the expansion shards.
            cfg.q_a_proj = _lin(
                config.hidden_size, config.q_lora_rank, sharding=_tp_replicate()
            )
            cfg.q_a_layernorm = RMSNorm.Config(
                normalized_shape=config.q_lora_rank,
                eps=config.rms_norm_eps,
                sharding_config=_tp_replicate(),
            )
            cfg.q_b_proj = _lin(
                config.q_lora_rank, heads * q_head_dim, sharding=_tp_shard(0)
            )
        if config.mla_gated:
            # Gated MLA, report Eq. 7. full_rank gates per (head, v_head_dim);
            # the graft variant gates per head with a bias so a large positive
            # init makes sigmoid(gate) ~= 1 and the layer starts near identity.
            if config.attn_gate_param == "full_rank":
                cfg.attn_gate_proj = _lin(
                    config.hidden_size,
                    heads * config.v_head_dim,
                    sharding=_tp_shard(0),
                )
            else:
                cfg.attn_gate_proj = _lin(
                    config.hidden_size, heads, bias=True, sharding=_tp_shard(0)
                )
        return cfg

    def __init__(self, config: "KimiMLAAttention.Config") -> None:
        super().__init__()
        self.layer_idx = config.layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.use_nope = config.mla_use_nope
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim**-0.5
        self.mla_gated = config.mla_gated
        self.attn_gate_param = config.attn_gate_param

        assert self.use_nope, (
            "Only mla_use_nope=True is currently supported (Kimi 48B-A3B "
            "config). RoPE-on-MLA is not ported."
        )

        # Exactly one Q group is populated by make_config.
        if config.q_proj is not None:
            self.q_proj = config.q_proj.build()
        else:
            assert config.q_a_proj is not None
            assert config.q_a_layernorm is not None
            assert config.q_b_proj is not None
            self.q_a_proj = config.q_a_proj.build()
            self.q_a_layernorm = config.q_a_layernorm.build()
            self.q_b_proj = config.q_b_proj.build()
        self.kv_a_proj_with_mqa = config.kv_a_proj_with_mqa.build()
        self.kv_a_layernorm = config.kv_a_layernorm.build()
        self.kv_b_proj = config.kv_b_proj.build()
        self.o_proj = config.o_proj.build()
        if config.attn_gate_proj is not None:
            self.attn_gate_proj = config.attn_gate_proj.build()

        # SDPA-only sub-module so the TP plan can wrap it with
        # use_local_output=True (DSv3 pattern). Has no parameters. torchtitan's
        # own SDPA module rather than a local copy: it brings the backend
        # priority list (cuDNN, then flash, then math) that a bare
        # F.scaled_dot_product_attention call leaves to the default dispatcher,
        # and it is the type the upstream CP dispatcher recognises.
        #
        # Kept as a submodule for the same reason DSv3 does: apply_tp wraps this
        # call with PrepareModuleInput(use_local_output=True), so q/k/v are plain
        # Tensors before SDPA's kernel dispatcher runs. Without that, the
        # mem-efficient cutlass path fails with "aten.bmm got mixed Tensor and
        # DTensor".
        self.inner_attention = config.inner_attention.build()

    def _attn_gate(self, x: torch.Tensor, width: int) -> torch.Tensor:
        """Sigmoid output gate, broadcastable onto ``[..., width]``.

                full_rank (K3): one value per output channel, so the projection
                already has the right width. per_head_graft: one value per head,
                expanded across that head's v_head_dim.

        Under TP ``x`` arrives here as a DTensor (measured), so DTensor's own
                autograd redistributes the gradient this branch contributes to the
                residual; there is nothing to reduce by hand. An earlier attempt to
                all-reduce it explicitly was a no-op for exactly that reason -- see
                TP_GRAD_FINDING_2026-07-29.
        """
        g = torch.sigmoid(self.attn_gate_proj(x))
        if self.attn_gate_param == "full_rank":
            return g
        return (
            g.unsqueeze(-1)
            .expand(*g.shape, width // g.shape[-1])
            .reshape(*g.shape[:-1], width)
        )

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Q projection, with or without the compression pair.

        Returns the flat ``[..., num_heads * q_head_dim]`` tensor; callers
        reshape. Kept as one method so the direct and CP forward paths cannot
        drift apart.
        """
        if self.q_lora_rank is None:
            return self.q_proj(x)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal mask; no KV cache.

        Args:
            x: ``[B, T, D]`` hidden states.
        Returns:
            ``[B, T, D]`` attention output.
        """
        # Context parallel: Ulysses path (seq-local projections,
        # all-to-all seq<->head, full-seq SDPA on this rank's head
        # subset). Handles both plain x and DTensor x (TP), so there is
        # no silent CP skip under TP anymore.
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            return self._forward_cp(x, cp_group)
        B, T, _ = x.shape

        # Q path: direct projection -> (B, T, H, q_head_dim) -> (B, H, T, q_head_dim)
        #
        # H is DERIVED from the projection, not read off self.num_heads, because the
        # two differ under TP: the projection is column-parallel, so each rank
        # produces num_heads/tp of them. Under partial_dtensor its output is a
        # DTensor whose view() sees the global shape and the distinction never
        # surfaced; under spmd_types the output is a local tensor and the global
        # count fails with "shape [1, 4096, 4, 192] is invalid for input of size
        # 1572864" -- exactly half. Deriving works either way and needs no branch
        # on the backend.
        q_proj_out = self._project_q(x)
        h_local = q_proj_out.shape[-1] // self.q_head_dim
        q = q_proj_out.view(B, T, h_local, self.q_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # KV path: (B, T, kv_lora + qk_rope)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Expand low-rank KV to full heads:
        #   kv_b_proj: (kv_lora_rank) -> (num_heads * (qk_nope_head_dim + v_head_dim))
        kv_expanded = self.kv_b_proj(self.kv_a_layernorm(k_pass))
        kv_expanded = kv_expanded.view(
            B, T, h_local, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_pass_expanded, v = torch.split(
            kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # k_rot is broadcast across heads: (B, T, qk_rope_head_dim) -> (B, H, T, qk_rope)
        k_rot = k_rot.view(B, 1, T, self.qk_rope_head_dim).expand(
            B, h_local, T, self.qk_rope_head_dim
        )

        # Concat nope + rot halves (NO RoPE application under mla_use_nope)
        q_full = torch.cat((q_pass, q_rot), dim=-1)
        k_full = torch.cat((k_pass_expanded, k_rot), dim=-1)

        # Standard scaled-dot-product attention with causal mask.
        # PyTorch's default SDPA backend selection picks the right
        # kernel here: for Kimi MLA's asymmetric head_dim (Q/K=192,
        # V=128), flash-attention rejects (requires Q/K/V same dim)
        # and cuDNN attention is runtime-disabled in PyTorch 2.11,
        # so the *mem-efficient cutlass kernel* (fmha_cutlassF_bf16,
        # flash-style fused) is selected by default.
        #
        # Routing through ``self.inner_attention`` (a parameterless
        # submodule) is the DSv3 pattern: it lets ``apply_tp_kimi_k3``
        # wrap this call with ``PrepareModuleInput(use_local_output=True)``
        # so q/k/v are converted from DTensor (sharded on the head axis)
        # to plain Tensors before SDPA's mem-efficient cutlass kernel
        # path sees them — avoiding "aten.bmm got mixed Tensor and
        # DTensor" inside SDPA's internal dispatcher.
        # (B, H, T, D) -> (B, T, H, D) because the shared module takes the
        # head-minor layout and transposes internally; the two cancel, so this
        # costs nothing, and the output comes back head-minor already.
        attn_out = self.inner_attention(
            q_full.transpose(1, 2),
            k_full.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.scaling,
        )  # (B, T, H, v_head_dim)

        attn_out = attn_out.reshape(B, T, -1)  # (B, T, H*Dv)
        # SDPA has no DTensor rule, so inner_attention hands back a plain local
        # tensor. Re-wrap it on the way out, the same shape as the fla kernels'
        # _to_local_if_dtensor round trip: the unwrap is a kernel-call detail and
        # must not leak into the residual stream, which is DTensor end to end.
        if isinstance(x, DTensor) and not isinstance(attn_out, DTensor):
            attn_out = DTensor.from_local(
                attn_out, x.device_mesh, (Shard(2),), run_check=False
            ).redistribute(placements=(Replicate(),))
        if self.mla_gated:
            attn_out = attn_out * self._attn_gate(x, attn_out.shape[-1])
        out = self.o_proj(attn_out)
        return out

    def _forward_cp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """Ulysses CP forward.

        Tensor-name legend (shape suffixes): B batch, L local seq (T/cp),
        T full seq, H local head count before CP split (num_heads/tp),
        G CP-local head count (H/cp), Q q_head_dim, N qk_nope_head_dim,
        V v_head_dim, R qk_rope_head_dim, W concatenated feature dim.

        Input x is ``[B, L, D]`` -- plain, or DTensor(Replicate on
        tp_mesh) under TP. Projections run through their (possibly
        TP-wrapped) modules at seq length L; the CP collectives operate
        on plain local tensors only, in the same gap where the TP plan
        already strips DTensor (inner_attention use_local_output). Under
        TP the head axis is already tp-sharded, so this rank computes
        num_heads/(tp*cp) heads over the full sequence. No rank ever
        materializes ``[B, T, D]`` hidden states: activation memory
        follows the Ulysses contract, unlike the previous all-gather-SP
        path which kept O(T x D) per rank at any cp degree.
        """
        import torch.distributed.nn.functional as dist_nn

        cp_size = dist.get_world_size(cp_group)
        B, t_loc, _ = x.shape

        q_BLE = self._project_q(x)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot_BLR = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_BLF = self.kv_b_proj(self.kv_a_layernorm(k_pass))

        # Leave DTensor land (no-ops when TP is off). All CP collectives
        # below run on plain local tensors on the cp sub-mesh group.
        q_BLE = _to_local_if_dtensor(q_BLE)
        kv_BLF = _to_local_if_dtensor(kv_BLF)
        # k_rot needs the PARTIAL grad placement, the other two do not. q and kv
        # come from Colwise projections and are Shard on tp, so the default
        # (gradient carries the forward placement) is right. k_rot comes from
        # kv_a_proj_with_mqa, which is NoParallel -> Replicate, and below it is
        # expanded onto THIS rank's head subset: each rank does different work
        # with the same replicated value, so the gradient of that value is the
        # SUM across tp ranks, i.e. Partial. With the default the sum never
        # happens and kv_a_proj_with_mqa's gradient ends up rank-dependent while
        # its placement still says Replicate.
        k_rot_BLR = _to_local_partial_grad(k_rot_BLR)

        kv_head_dim = self.qk_nope_head_dim + self.v_head_dim
        h_loc = q_BLE.shape[-1] // self.q_head_dim
        if h_loc % cp_size != 0:
            raise ValueError(
                f"MLA CP: local head count {h_loc} is not divisible by "
                f"cp={cp_size} (num_attention_heads must divide tp*cp)"
            )

        # One fused all-to-all for q and kv (concat on the feature axis).
        qkv_BLHW = torch.cat(
            [
                q_BLE.view(B, t_loc, h_loc, self.q_head_dim),
                kv_BLF.view(B, t_loc, h_loc, kv_head_dim),
            ],
            dim=-1,
        )
        src_dim, dst_dim = ULYSSES.in_dims()
        qkv_BTGW = _cp_all_to_all_headseq(
            qkv_BLHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
        )
        q_BTGQ, k_pass_BTGN, v_BTGV = torch.split(
            qkv_BTGW,
            [self.q_head_dim, self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        t_full = t_loc * cp_size
        h_cp = h_loc // cp_size

        # k_rot is broadcast across heads (headless): all-gather the seq
        # shards (differentiable -> reduce-scatter backward) and expand
        # onto this rank's head subset. Tiny tensor (R per token).
        k_rot_BTR = torch.cat(
            dist_nn.all_gather(k_rot_BLR.contiguous(), group=cp_group), dim=1
        )
        k_BTGQ = torch.cat(
            [
                k_pass_BTGN,
                k_rot_BTR.view(B, t_full, 1, self.qk_rope_head_dim).expand(
                    B, t_full, h_cp, self.qk_rope_head_dim
                ),
            ],
            dim=-1,
        )

        attn_BTGV = self.inner_attention(
            q_BTGQ,
            k_BTGQ,
            v_BTGV,
            scale=self.scaling,
        )
        out_src_dim, out_dst_dim = ULYSSES.out_dims()
        attn_BLHV = _cp_all_to_all_headseq(
            attn_BTGV.contiguous(), cp_group, src_dim=out_src_dim, dst_dim=out_dst_dim
        )
        attn_BLE = attn_BLHV.reshape(B, t_loc, h_loc * self.v_head_dim)
        if self.mla_gated:
            # Gate from the seq-local x; pointwise, so it applies after the
            # heads return seq-local. Under TP the gate projection is
            # head-sharded exactly like the attention output, so the local
            # widths line up.
            attn_BLE = attn_BLE * _to_local_if_dtensor(
                self._attn_gate(x, attn_BLE.shape[-1])
            )
        # Ulysses runs its all-to-alls on plain local tensors, so everything
        # above is plain by design. Re-wrap before o_proj: the residual stream is
        # a DTensor, and leaving this plain was measured to hand o_proj -- and
        # only o_proj -- a plain input on tp x cp cells, which is what failed the
        # three remaining LoRA cells.
        if isinstance(x, DTensor) and not isinstance(attn_BLE, DTensor):
            # Shard(-1): o_proj is Rowwise under TP, so its input is sharded on
            # the contracted axis. Replicate here gives "a and b must have same
            # reduction dim" -- the local width is num_heads/tp * v_head_dim.
            attn_BLE = DTensor.from_local(
                attn_BLE,
                x.device_mesh,
                (Shard(attn_BLE.dim() - 1),),
                run_check=False,
            )
        out = self.o_proj(attn_BLE)
        return out


# ----- KDA (Kimi Delta-rule Attention) ------------------------------------ #


def _to_local_if_dtensor(t):
    """Strip DTensor wrapping for fla-core triton kernels.

    fla-core's chunk_kda / fused_kda_gate / ShortConvolution are Triton
    kernels that don't dispatch through DTensor. Under TP, KDA's
    self_attn is NoParallel-wrapped (params become DTensor(Replicate)
    on tp_mesh) and incoming x is also DTensor at the parent's
    boundary. KDA forward stashes the DTensor mesh+placements, strips
    DTensor from x and from each weight at the kernel call site, runs
    the kernels on plain tensors (each rank computes redundantly under
    Replicate), and re-DTensors at the end so the parent NoParallel
    output hook composes correctly.

    isinstance(t, DTensor) is the safe check that dynamo's fake-tensor
    mode honors (``hasattr(t, "to_local")`` is unreliable: dynamo's
    type tracking can elide attribute lookups on DTensor parameters).
    """
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def _to_local_partial_grad(t):
    """``to_local`` for a value each rank then consumes DIFFERENTLY.

    ``to_local()`` defaults the incoming gradient's placement to the forward
    placement. For a Replicate value that is correct only when every rank does the
    SAME work with it -- which is exactly KDA's redundant kernels, and why
    ``_to_local_if_dtensor`` keeps the default.

    It is wrong when the ranks diverge. MLA's CP path expands the replicated
    ``k_rot`` onto this rank's head subset, so each rank's gradient is one partial
    contribution and the gradient of the replicated value is their sum: Partial,
    not Replicate. Keeping the default drops that all-reduce silently, because the
    placement still reads Replicate afterwards.

    Measured on ``kimi_k3_debugmodel_report_arch`` at tp2 x cp2: all four MLA
    layers' ``kv_a_proj_with_mqa`` gradients differed across the tp pair by 1-6%
    relative on every step, while tp2 alone was bit-identical -- the non-CP path
    never leaves DTensor, so DTensor reduces it there.
    """
    if not isinstance(t, DTensor):
        return t
    return t.to_local(
        grad_placements=[
            Partial() if isinstance(p, Replicate) else p for p in t.placements
        ]
    )


def _local_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply ``linear`` with both weight and (optional) bias unwrapped to local.

    Used by :class:`KimiDeltaAttention.forward` so each projection can
    operate in plain-Tensor land alongside the fla-core triton kernels,
    even when the parent NoParallel(self_attn) wrap makes ``linear.weight``
    a DTensor(Replicate) on tp_mesh.
    """
    weight = _to_local_if_dtensor(linear.weight)
    bias = _to_local_if_dtensor(linear.bias) if linear.bias is not None else None
    return F.linear(x, weight, bias)


class KimiDeltaAttention(Module):
    """Kimi Delta Attention — linear-attention variant using
    fla-core's gated delta rule kernel.

    Faithful port of ``reference:KimiDeltaAttention`` minus the
    HF ``Cache`` / ``cu_seqlens`` / padding-aware fast-path (training
    fixed-seqlen doesn't exercise those).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Config-driven KDA.

        The scalar fields deliberately carry the SAME names as the flat model
        config's, so the constructor body reads them unchanged. fla's
        ``ShortConvolution`` and ``FusedRMSNormGated`` are not Configurable, so
        they stay constructed from scalars here rather than from child configs --
        upstream avoids that by using core's Conv1d and its own gated norm, and we
        keep fla's fused kernels deliberately (they are on every KDA layer's
        critical path).
        """

        layer_idx: int
        hidden_size: int
        kda_short_conv_kernel_size: int
        kda_head_dim: int
        kda_num_heads: int
        kda_use_full_rank_gate: bool
        kda_gate_lower_bound: float
        kda_cp_mode: str
        rms_norm_eps: float
        q_proj: "Linear.Config"
        k_proj: "Linear.Config"
        v_proj: "Linear.Config"
        f_a_proj: "Linear.Config"
        f_b_proj: "Linear.Config"
        b_proj: "Linear.Config"
        o_proj: "Linear.Config"
        g_proj: "Linear.Config | None" = None
        g_a_proj: "Linear.Config | None" = None
        g_b_proj: "Linear.Config | None" = None

    @staticmethod
    def make_config(
        config: KimiK3Config, layer_idx: int
    ) -> "KimiDeltaAttention.Config":
        """The one place that reads the flat config for KDA."""
        projection_size = config.kda_head_dim * config.kda_num_heads

        def _lin(fan_in, fan_out, *, replicate=True):
            # Replicate throughout, matching what NoParallel gave these. Their
            # outputs feed the fla kernels, which KDA unwraps at the call site
            # (_to_local_if_dtensor), so the kernels see plain tensors either way.
            return Linear.Config(
                in_features=fan_in,
                out_features=fan_out,
                bias=False,
                sharding_config=_tp_replicate() if replicate else None,
            )

        cfg = KimiDeltaAttention.Config(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            kda_short_conv_kernel_size=config.kda_short_conv_kernel_size,
            kda_head_dim=config.kda_head_dim,
            kda_num_heads=config.kda_num_heads,
            kda_use_full_rank_gate=config.kda_use_full_rank_gate,
            kda_gate_lower_bound=config.kda_gate_lower_bound,
            kda_cp_mode=config.kda_cp_mode,
            rms_norm_eps=config.rms_norm_eps,
            q_proj=_lin(config.hidden_size, projection_size),
            k_proj=_lin(config.hidden_size, projection_size),
            v_proj=_lin(config.hidden_size, projection_size),
            f_a_proj=_lin(config.hidden_size, config.kda_head_dim),
            f_b_proj=_lin(config.kda_head_dim, projection_size),
            b_proj=_lin(config.hidden_size, config.kda_num_heads),
            o_proj=_lin(projection_size, config.hidden_size),
        )
        # K3 (report Eq. 6) makes the output gate full rank; Kimi Linear factored
        # it through head_dim. Both feed the same FusedRMSNormGated.
        if config.kda_use_full_rank_gate:
            cfg.g_proj = _lin(config.hidden_size, projection_size)
        else:
            cfg.g_a_proj = _lin(
                config.hidden_size, config.kda_head_dim, replicate=False
            )
            cfg.g_b_proj = _lin(config.kda_head_dim, projection_size, replicate=False)
        return cfg

    def __init__(self, config: "KimiDeltaAttention.Config") -> None:
        super().__init__()
        self.layer_idx = config.layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.kda_short_conv_kernel_size
        self.head_dim = config.kda_head_dim
        self.num_heads = config.kda_num_heads

        projection_size = self.head_dim * self.num_heads
        projection_k_size = projection_size  # k heads == v heads for Kimi

        # Replicate, matching what NoParallel gave them. Their outputs feed the
        # fla kernels, which KDA unwraps at the call site
        # (_to_local_if_dtensor), so the kernels still see plain tensors.
        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()

        # Short causal convolutions with silu activation on q/k/v
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        # A_log: per-head log-decay parameter, init uniform in log([1, 16])
        # fla-core 0.5.0 expects shape [H]; HF reference had [1, 1, H, 1]
        # but it's fed through fused_kda_gate which reshapes internally.
        # Drawn and log'd in fp32 for the init math, then stored at the default
        # dtype like every other parameter. Keeping the parameter itself fp32
        # (which is what dtype= on the empty() used to do) makes the module's
        # dtypes non-uniform under training.dtype=bfloat16, and FSDP2 rejects
        # that outright: "FSDP expects uniform original parameter dtype".
        # No-op when the default dtype is fp32.
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            ).to(torch.get_default_dtype())
        )

        # dt_bias: per-(head, head_dim) bias, shape [H * K]. Applied
        # inside fused_kda_gate as softplus(g + dt_bias). Kept zero-init
        # to reproduce HF reference's default init behavior.
        self.dt_bias = nn.Parameter(torch.zeros(projection_size))

        # Declared here rather than driven by ``plan["self_attn"] = NoParallel(...)``:
        # A_log and dt_bias are this module's OWN parameters, so only a module-level
        # declaration can reach them. tp-Replicate matches what NoParallel does, and
        # keeps every parameter on one mesh for clip_grad_norm_'s stack.
        #
        # ``param_init`` is not optional once this class is a Module:
        # ``_init_self_parameters`` RAISES for own parameters when neither a param_init
        # map nor ``reset_parameters`` exists, and both of these are initialized above --
        # so the map re-applies exactly that, rather than leaving a trap for the first
        # caller that reaches init_states from the root.
        self._sharding_config = ShardingConfig(
            state_shardings={
                "A_log": dense_param_placement(tp=spmd.R),
                "dt_bias": dense_param_placement(tp=spmd.R),
            }
        )
        self._param_init = {
            "A_log": lambda t: t.copy_(
                torch.log(
                    torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
                ).to(t.dtype)
            ),
            "dt_bias": lambda t: t.zero_(),
        }

        # Low-rank forget-gate and output-gate projections
        self.f_a_proj = config.f_a_proj.build()
        self.f_b_proj = config.f_b_proj.build()
        # Output gate. K3 (report Eq. 6) makes W_g full rank; Kimi Linear
        # factored it through head_dim. Both feed the same
        # FusedRMSNormGated(o, g) = Sigmoid(g) (.) RMSNorm(o~) below.
        self.use_full_rank_gate = config.kda_use_full_rank_gate
        if self.use_full_rank_gate:
            self.g_proj = config.g_proj.build()
        else:
            self.g_a_proj = config.g_a_proj.build()
            self.g_b_proj = config.g_b_proj.build()
        self.gate_lower_bound = config.kda_gate_lower_bound
        self.cp_mode = config.kda_cp_mode
        # Validate against the CP contracts so the accepted modes are declared
        # in one place rather than restated here.
        contract_for_mode(self.cp_mode)

        # Beta: per-head, per-token scalar (delta-rule learning rate)
        self.b_proj = config.b_proj.build()

        # Output RMSNorm with sigmoid-gated modulation from g, then o_proj
        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            activation="sigmoid",
        )
        # Replicate, unlike MLA's o_proj: KDA's core runs on plain tensors, so
        # this projection's input is not head-sharded and has nothing to reduce.
        self.o_proj = config.o_proj.build()

    def _output_gate_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid output-gate logits, flat ``[..., H * head_dim]``.

        Full rank is K3's (report Eq. 6); the low-rank pair is Kimi Linear's.
        The sigmoid itself lives in FusedRMSNormGated.
        """
        if self.use_full_rank_gate:
            return _local_linear(self.g_proj, x)
        return _local_linear(self.g_b_proj, _local_linear(self.g_a_proj, x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without KV cache, fixed seq_len.

        Args:
            x: ``[B, T, D]`` hidden states.
        Returns:
            ``[B, T, D]`` KDA output.
        """
        # Under TP, the parent KimiDecoderLayer's NoParallel(self_attn)
        # wraps this forward: x arrives as DTensor(Replicate) on tp_mesh,
        # and all child params (q/k/v projections, conv1d weights,
        # A_log, dt_bias, FusedRMSNormGated) are DTensors on the same
        # mesh. The standard nn.Linear ops (DTensor x × DTensor weight)
        # dispatch correctly through DTensor's op set; the fla-core
        # triton kernels (causal_conv1d in ShortConvolution,
        # fused_kda_gate, chunk_kda, FusedRMSNormGated) do not. We
        # stash the input's DTensor metadata, run the body in plain-
        # tensor land, and re-DTensor at the end so the parent
        # NoParallel hook's prepare_output sees a DTensor.
        in_mesh = None
        in_placements = None
        if isinstance(x, DTensor):
            in_mesh = x.device_mesh
            in_placements = x.placements
        x = _to_local_if_dtensor(x)
        # Context parallel: Ulysses path (seq-local projections,
        # all-to-all seq<->head, full-seq conv + scan on this rank's head
        # subset). chunk_kda is bit-exactly per-head independent
        # (kda_ulysses_cp_probe), so head-sharding the scan is exact.
        # MLA layers get the same treatment in KimiMLAAttention.
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            out = (
                self._forward_kcp(x, cp_group)
                if self.cp_mode == "kcp"
                else self._forward_cp(x, cp_group)
            )
            if in_mesh is not None and in_placements is not None:
                out = DTensor.from_local(
                    out,
                    in_mesh,
                    in_placements,
                    run_check=False,
                )
            return out
        _, T, _ = x.shape
        # mode selection matches reference: chunk for long, recurrent for short
        # training gate: chunk required (ref asserts this)
        mode = "fused_recurrent" if T <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 1) Q/K/V projection + short causal conv with silu.
        # _local_linear unwraps DTensor weight to local before F.linear.
        # ShortConvolution.forward is patched at TP-init time to handle
        # DTensor input/weight by to_local + re-DTensor; we feed plain
        # x here so the patch is a no-op when x is already plain.
        q, _ = self.q_conv1d(
            x=_local_linear(self.q_proj, x),
            cache=None,
            output_final_state=False,
        )
        k, _ = self.k_conv1d(
            x=_local_linear(self.k_proj, x),
            cache=None,
            output_final_state=False,
        )
        v, _ = self.v_conv1d(
            x=_local_linear(self.v_proj, x),
            cache=None,
            output_final_state=False,
        )

        # 2) Forget-gate g: (B,T,D) low-rank via f_a/f_b, reshape to
        #    (B, T, H, K) for fla-core 0.5.0's fused_kda_gate API:
        #      fused_kda_gate(g: [..., H, K], A_log: [H], dt_bias: [H*K])
        #      → [..., H, K] log-decay
        g_raw = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        g = fused_kda_gate(
            g_raw,
            _to_local_if_dtensor(self.A_log),
            dt_bias=_to_local_if_dtensor(self.dt_bias),
            lower_bound=self.gate_lower_bound,
        )

        # 3) Beta: per-head, per-token learning-rate (delta-rule)
        beta = _local_linear(self.b_proj, x).float().sigmoid()

        # 4) Reshape to (..., H, D) for KDA kernel
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # 6) Output gate (computed before the head-shard so the slice below
        # covers it too).
        g_out = self._output_gate_raw(x)
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)

        # 5) Run KDA op
        kda_fn = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o, _ = kda_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

        # FusedRMSNormGated.forward is patched at TP-init time too, so
        # it handles DTensor weight transparently. We pass plain o + g_out
        # here (both are plain after the to_local+linear chain).
        o = self.o_norm(o, g_out)  # o * sigmoid(g_out), normed

        # 7) Reshape back and project
        o = rearrange(o, "b t h d -> b t (h d)")
        out = _local_linear(self.o_proj, o)

        # Re-wrap the output as DTensor so the parent NoParallel hook
        # gets the type it expects. Replicate placement matches the
        # incoming x's placement (input_layernorm output).
        if in_mesh is not None and in_placements is not None:
            out = DTensor.from_local(
                out,
                in_mesh,
                in_placements,
                run_check=False,
            )
        return out

    def _forward_kcp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """KCP forward: the sequence stays sharded (report sec 5.1.2).

        Unlike the Ulysses path, no rank ever holds the full sequence. The two
        cross-rank dependencies are handled separately because they have
        different structure:

        * the short convolutions need only the previous rank's tail, since their
          support is finite -- one fixed-size halo, no scan (see kcp.py);
        * the delta-rule recurrence needs the true incoming state, which does
          NOT decompose by summation, so fla's cp_context does a prefix scan
          over (cumulative transition, zero-started state) fragments.

        Constraints this path inherits from fla: ``output_final_state`` is
        unsupported under cp_context, which is fine for training (the final
        state is only needed for decoding), and the sequence must divide evenly
        across the CP ranks.

        A batch axis is handled by looping, because fla's ``causal_conv1d_cp``
        asserts ``[1, T, D]``: its CP path is built around a single packed
        sequence. Flattening ``[B, L, D]`` into one packed sequence instead would
        be cheaper in launches but wrong -- ``build_cp_context`` derives each
        rank's slice by cutting the GLOBAL packed sequence into contiguous
        rank-ordered pieces, while what this rank actually holds is piece ``r`` of
        every sequence, so the two layouts only coincide at B = 1. The loop is
        also what the recurrence wants: sequences in a batch are independent, and
        the delta-rule state must not carry from one into the next.

        The cost is B prefix-scan all-gathers instead of one. Each is fixed size
        (state fragments, not activations) and independent of sequence length, and
        B is identical on every rank, so the collective counts match and cannot
        deadlock. K3's own regime is the cheap end of this: local batch 1 with a
        long sequence, the batch coming from DP.
        """
        B = x.shape[0]
        if B > 1:
            return torch.cat(
                [self._forward_kcp_one(x[b : b + 1], cp_group) for b in range(B)],
                dim=0,
            )
        return self._forward_kcp_one(x, cp_group)

    def _forward_kcp_one(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """One sequence's KCP forward. ``x`` is this rank's ``[1, L, D]`` shard."""
        from torchtitan.models.kimi_k3.kcp import build_kcp_context, conv_with_halo

        t_loc = x.shape[1]

        # One context serves both the conv halo and the recurrence; the conv
        # needs the kernel width, the recurrence ignores it.
        ctx = build_kcp_context(
            t_loc, cp_group, x.device, conv1d_kernel_size=self.q_conv1d.kernel_size[0]
        )

        # Projections are seq-local: nothing to exchange yet.
        q = conv_with_halo(self.q_conv1d, _local_linear(self.q_proj, x), ctx)
        k = conv_with_halo(self.k_conv1d, _local_linear(self.k_proj, x), ctx)
        v = conv_with_halo(self.v_conv1d, _local_linear(self.v_proj, x), ctx)

        g_raw = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        g = fused_kda_gate(
            g_raw,
            _to_local_if_dtensor(self.A_log),
            dt_bias=_to_local_if_dtensor(self.dt_bias),
            lower_bound=self.gate_lower_bound,
        )
        beta = _local_linear(self.b_proj, x).float().sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
        g_out = rearrange(
            self._output_gate_raw(x), "... (h d) -> ... h d", d=self.head_dim
        )

        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            # fla asserts this is unsupported under cp_context.
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=ctx.cu_seqlens,
            cp_context=ctx,
        )
        o = self.o_norm(o, g_out)
        o = rearrange(o, "b t h d -> b t (h d)")
        return _local_linear(self.o_proj, o)

    def _forward_cp(self, x: torch.Tensor, cp_group) -> torch.Tensor:
        """Ulysses CP forward for KDA.

        Tensor-name legend (shape suffixes): B batch, L local seq (T/cp),
        T full seq, H head count (KDA is never tp-sharded), G CP-local
        head count (H/cp), K head_dim, C flattened head-subset channels
        (G*K).

        Input x is the plain local ``[B, L, D]`` shard (caller already
        stripped DTensor). Projections run seq-local at L; one fused
        all-to-all moves (q, k, v, g_raw, g_out, beta) to full-seq
        head-subset layout; the causal short conv, fused_kda_gate, and
        chunk_kda then run on the full sequence for this rank's G heads
        (conv weights channel-sliced -- depthwise conv, exact; validated
        bit-exact vs ShortConvolution). No rank materializes the full
        sequence at hidden dim D.

        Gradient note: each rank's param-grad contribution covers its
        (seq shard x head subset) sector with zeros elsewhere; FSDP's
        dp_shard_cp mesh reduces over cp, reconstructing full grads --
        the same contract the previous all-gather-SP path relied on.
        """
        from fla.modules.conv.causal_conv1d import causal_conv1d

        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        B, t_loc, _ = x.shape
        num_heads, head_dim = self.num_heads, self.head_dim
        if num_heads % cp_size != 0:
            raise ValueError(
                f"KDA CP: num_heads {num_heads} is not divisible by " f"cp={cp_size}"
            )
        h_cp = num_heads // cp_size
        h0 = cp_rank * h_cp

        # 1) Seq-local projections at L (no cross-seq ops here).
        q_BLHK = _local_linear(self.q_proj, x).view(B, t_loc, num_heads, head_dim)
        k_BLHK = _local_linear(self.k_proj, x).view(B, t_loc, num_heads, head_dim)
        v_BLHK = _local_linear(self.v_proj, x).view(B, t_loc, num_heads, head_dim)
        g_raw_BLHK = _local_linear(self.f_b_proj, _local_linear(self.f_a_proj, x)).view(
            B, t_loc, num_heads, head_dim
        )
        g_out_BLHK = self._output_gate_raw(x).view(B, t_loc, num_heads, head_dim)
        beta_BLH1 = _local_linear(self.b_proj, x).unsqueeze(-1)

        # 2) One fused all-to-all: seq-shard -> full-seq head-subset.
        packed_BLHW = torch.cat(
            [q_BLHK, k_BLHK, v_BLHK, g_raw_BLHK, g_out_BLHK, beta_BLH1],
            dim=-1,
        )
        src_dim, dst_dim = ULYSSES.in_dims()
        packed_BTGW = _cp_all_to_all_headseq(
            packed_BLHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
        )
        q_BTGK, k_BTGK, v_BTGK, g_raw_BTGK, g_out_BTGK, beta_BTG1 = torch.split(
            packed_BTGW,
            [head_dim, head_dim, head_dim, head_dim, head_dim, 1],
            dim=-1,
        )
        t_full = t_loc * cp_size

        mode = "fused_recurrent" if t_full <= 64 else "chunk"
        if self.training:
            assert mode == "chunk", "KDA training requires chunk mode (T > 64)"

        # 3) Short causal conv on the full sequence, weights sliced to
        # this rank's head-subset channels (depthwise conv -> exact).
        def conv_subset(conv: ShortConvolution, x_BTGK: torch.Tensor):
            w_CW = _to_local_if_dtensor(conv.weight).squeeze(1)[
                h0 * head_dim : (h0 + h_cp) * head_dim
            ]
            b_C = (
                _to_local_if_dtensor(conv.bias)[h0 * head_dim : (h0 + h_cp) * head_dim]
                if conv.bias is not None
                else None
            )
            y_BTC, _ = causal_conv1d(
                x_BTGK.reshape(B, t_full, h_cp * head_dim),
                weight=w_CW,
                bias=b_C,
                activation=conv.activation,
                backend=conv.backend,
            )
            return y_BTC.view(B, t_full, h_cp, head_dim)

        q_BTGK = conv_subset(self.q_conv1d, q_BTGK)
        k_BTGK = conv_subset(self.k_conv1d, k_BTGK)
        v_BTGK = conv_subset(self.v_conv1d, v_BTGK)

        # 4) Forget gate + beta on the head subset (A_log/dt_bias sliced).
        g_BTGK = fused_kda_gate(
            g_raw_BTGK,
            _to_local_if_dtensor(self.A_log)[h0 : h0 + h_cp],
            dt_bias=_to_local_if_dtensor(self.dt_bias)
            .view(num_heads, head_dim)[h0 : h0 + h_cp]
            .reshape(-1),
            lower_bound=self.gate_lower_bound,
        )
        beta_BTG = beta_BTG1.squeeze(-1).float().sigmoid()

        # 5) KDA scan on this rank's heads over the full sequence.
        kda_fn = chunk_kda if mode == "chunk" else fused_recurrent_kda
        o_BTGK, _ = kda_fn(
            q=q_BTGK,
            k=k_BTGK,
            v=v_BTGK,
            g=g_BTGK,
            beta=beta_BTG,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )
        o_BTGK = self.o_norm(o_BTGK, g_out_BTGK)

        # 6) All-to-all back to seq-shard full-head layout, then o_proj.
        out_src_dim, out_dst_dim = ULYSSES.out_dims()
        o_BLHK = _cp_all_to_all_headseq(
            o_BTGK, cp_group, src_dim=out_src_dim, dst_dim=out_dst_dim
        )
        out = _local_linear(self.o_proj, o_BLHK.reshape(B, t_loc, num_heads * head_dim))
        return out


# ----- MoE (training-capable via torchtitan.models.common.moe) ------------ #


class KimiLatentMoEProjection(Module):
    """The latent entry/exit of Stable LatentMoE (report Eq. 11).

    ``down`` maps a token from full width ``d`` into the routed-expert latent
    ``l``; ``norm`` (RMSNorm, report sec 2.3.1 "Normalized LatentMoE") is
    applied to the AGGREGATED routed representation ``u`` -- after the weighted
    expert combine, not per expert -- and ``up`` maps back to ``d``.

    Kept as a separate module because both projections are shared across all
    routed experts: they are applied once per token, which is what makes the
    896-expert routing affordable (dispatch traffic is O(l), not O(d)).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Config-driven, with the norm optional the way the module is.

        No declaration on ``norm``: it sits on the MoE's OUTPUT side, where the
        value arrives plain (the MoE unwraps at its boundary), so a declared
        DTensor weight would meet a plain input inside _fused_rms_norm. It keeps
        its imperative NoParallel entry.
        """

        down: "Linear.Config"
        up: "Linear.Config"
        norm: "RMSNorm.Config | None" = None

    @staticmethod
    def make_config(
        hidden_size: int,
        latent_size: int,
        use_norm: bool = True,
        rms_norm_eps: float = 1e-5,
    ) -> "KimiLatentMoEProjection.Config":
        # Replicated, NOT the column/row pair a SwiGLU would use. down's output
        # goes straight into the MoE, whose in_src_shardings expects Replicate --
        # the SP-island boundary that makes EP x TP work. Declaring Shard(0) here
        # gives the MoE a Shard(dim=2) activation and it refuses:
        # "MoE.x_BLD: input DTensor has placements (Shard(dim=2),), but
        # in_src_shardings expects (Replicate(),)". up is replicated to match.
        return KimiLatentMoEProjection.Config(
            down=Linear.Config(
                in_features=hidden_size,
                out_features=latent_size,
                bias=False,
                sharding_config=_tp_replicate(),
            ),
            up=Linear.Config(
                in_features=latent_size,
                out_features=hidden_size,
                bias=False,
                sharding_config=_tp_replicate(),
            ),
            norm=(
                RMSNorm.Config(normalized_shape=latent_size, eps=rms_norm_eps)
                if use_norm
                else None
            ),
        )

    def __init__(self, config: "KimiLatentMoEProjection.Config") -> None:
        super().__init__()
        self.down = config.down.build()
        self.up = config.up.build()
        self.norm = config.norm.build() if config.norm is not None else None

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

    def from_latent(self, u: torch.Tensor) -> torch.Tensor:
        return self.up(self.norm(u) if self.norm is not None else u)


class KimiMoE(Module):
    """Kimi's sigmoid-gated grouped-topk MoE, implemented via
    torchtitan's training-capable MoE primitives.

    The HF reference's :class:`KimiSparseMoeBlock` raises
    NotImplementedError in training mode (line 667 of
    ``reference/modeling_kimi.py``) — it's inference-only. Since we
    only care about training here, we rebuild the MoE forward using
    torchtitan common building blocks:

    * :class:`TokenChoiceTopKRouter` — supports sigmoid scoring,
      grouped topk (``num_expert_groups`` / ``num_limited_groups``),
      ``route_norm`` (Kimi's ``moe_renormalize``), ``route_scale``
      (Kimi's ``routed_scaling_factor``), and ``expert_bias``
      (Kimi's ``e_score_correction_bias``).
    * :class:`GroupedExperts` — grouped-GEMM SwiGLU experts,
      training-capable, with a for-loop fallback for CPU.
    * Shared experts (``num_shared_experts``): a single
      :class:`KimiMLP` instance whose output is added to the routed
      output unconditionally.

    Load-balancing hook: ``expert_bias`` is registered as a buffer on
    the router and updated externally by torchtitan's
    ``register_moe_load_balancing_hook`` at optimizer-step time. This
    mirrors DSv3's auxiliary-loss-free routing protocol.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """The latent MoE's config tree.

        ``moe`` is core's MoE.Config, because this class composes core's MoE
        rather than re-implementing it -- that composition is what makes expert
        parallel 16 lines (``_moe.parallelize(parallel_dims)``) instead of a
        hand-written dispatcher.
        """

        moe: "MoE.Config"
        latent_size: int | None = None
        latent: "KimiLatentMoEProjection.Config | None" = None
        shared_experts: "KimiMLP.Config | None" = None

    @staticmethod
    def make_config(config: KimiK3Config) -> "KimiMoE.Config":
        """Translate Kimi's flat knobs into this module's config tree.

        The body was already assembling core's MoE.Config; it now returns
        that tree instead of building it in place, which is what lets a
        sharding.py reach these sub-configs before build().
        """
        from torchtitan.models.common.config_utils import make_token_dispatcher_config

        # Full reuse: torchtitan.models.common.moe.MoE already wires
        # router + TokenReorderer + GroupedExperts + shared_experts +
        # expert_bias buffer + auxiliary-loss-free load balancing. We
        # just translate Kimi's config knobs into MoE.Config.
        from torchtitan.models.common.feed_forward import FeedForward
        from torchtitan.models.common.linear import Linear
        from torchtitan.models.common.moe import (
            GroupedExperts,
            MoE,
            RoutedExperts,
            TokenChoiceTopKRouter,
        )

        assert config.num_experts is not None and config.num_experts > 0
        # Stable LatentMoE (report Eq. 11): routed experts live in a compact
        # latent of width l, entered/left through two SHARED projections with
        # an RMSNorm on the aggregate. The router still reads the FULL-WIDTH
        # token (sec 2.3.3: s_i = Sigmoid(W_r x_i)), which is why MoE.forward
        # takes a separate router_input.
        latent_cfg = None
        latent_size: int | None = config.routed_expert_hidden_size
        expert_dim = config.hidden_size if latent_size is None else latent_size
        if latent_size is not None:
            latent_cfg = KimiLatentMoEProjection.make_config(
                config.hidden_size,
                latent_size,
                use_norm=config.latent_moe_use_norm,
                rms_norm_eps=config.rms_norm_eps,
            )

        router_cfg = TokenChoiceTopKRouter.Config(
            num_experts=config.num_experts,
            gate=Linear.Config(
                in_features=config.hidden_size,
                out_features=config.num_experts,
                bias=False,
            ),
            num_expert_groups=(
                config.num_expert_group if config.num_expert_group > 1 else None
            ),
            num_limited_groups=(
                config.topk_group if config.num_expert_group > 1 else None
            ),
            top_k=config.num_experts_per_token,
            score_func=config.moe_router_activation_func,
            route_norm=config.moe_renormalize,
            route_scale=config.routed_scaling_factor,
        )
        # K3 sets hidden_act="situ" globally, so the routed experts use
        # SiTU-GLU (Eq. 12); core GroupedExperts is SwiGLU-only.
        if config.hidden_act == "situ":
            from torchtitan.models.kimi_k3.moe import KimiSiTUGroupedExperts

            experts_config_cls = KimiSiTUGroupedExperts.Config
            experts_act_kwargs = {
                "situ_beta": config.activation_situ_beta,
                "situ_linear_beta": config.activation_situ_linear_beta,
            }
        else:
            experts_config_cls = GroupedExperts.Config
            experts_act_kwargs = {}
        # Declarative per-parameter init, the mechanism upstream models use
        # (deepseek_v3/__init__.py::_depth_experts_init). Module._init_param RAISES on a
        # parameter name absent from the map, which is why the map is the mechanism: a
        # rename then fails loudly instead of leaving that parameter uninitialised.
        expert_init = {
            name: partial(nn.init.trunc_normal_, std=config.initializer_range)
            for name in ("w1_EFD", "w2_EDF", "w3_EFD")
        }
        # Packed MXFP4/NF4 expert bytes replace the float parameters and come
        # from the checkpoint, so init must leave them untouched. Named
        # explicitly rather than skipped by dtype, so the map still fails loudly
        # on a name nobody has thought about.
        expert_init.update(
            {
                f"{n}_{part}": _leave_for_checkpoint
                for n in ("w1_EFD", "w2_EDF", "w3_EFD")
                for part in ("qdata", "scale", "nf4")
            }
        )
        experts_cfg = experts_config_cls(
            dim=expert_dim,
            hidden_dim=config.moe_intermediate_size,
            num_experts=config.num_experts,
            param_init=expert_init,
            **experts_act_kwargs,
            # torch._grouped_mm fuses all expert GEMMs into one batched call.
            # For-loop path (use_grouped_mm=False) launches one GEMM per
            # expert per layer, which hurts tensor core utilization badly
            # on small per-expert batches (typical at LOCAL_BS<=8). Requires
            # PyTorch ≥ 2.5 with grouped_mm support; works on Hopper / Ada /
            # Blackwell; CPU path raises so MoE forward is GPU-only.
        )

        # Shared experts — Kimi's reference uses KimiMLP at
        # intermediate = moe_int * num_shared_experts. We swap to
        # torchtitan's FeedForward for consistency with MoE.Config;
        # the SwiGLU math is identical.
        shared_cfg = None
        if config.num_shared_experts > 0 and latent_size is None:
            if config.hidden_act == "situ":
                raise ValueError(
                    'hidden_act="situ" with shared experts requires the latent '
                    "MoE path (routed_expert_hidden_size set), because the "
                    "non-latent path builds shared experts from core "
                    "FeedForward, which is SwiGLU-only. K3 always sets both."
                )
            shared_dim = config.moe_intermediate_size * config.num_shared_experts
            shared_cfg = FeedForward.Config(
                w1=Linear.Config(
                    in_features=config.hidden_size,
                    out_features=shared_dim,
                    bias=False,
                ),
                w2=Linear.Config(
                    in_features=shared_dim,
                    out_features=config.hidden_size,
                    bias=False,
                ),
                w3=Linear.Config(
                    in_features=config.hidden_size,
                    out_features=shared_dim,
                    bias=False,
                ),
            )

        # TODO(kimi-parity): upstream removed score_before_experts; Kimi's
        # reference applies router scores BEFORE the experts. Verify the
        # fixed upstream ordering against the official 48B ckpt (the
        # SGLang-side A/B from PR15 is the harness) before training.
        moe_cfg = MoE.Config(
            num_experts=config.num_experts,
            routed_experts=RoutedExperts.Config(
                inner_experts=experts_cfg,
                token_dispatcher=make_token_dispatcher_config(
                    num_experts=config.num_experts,
                    top_k=config.num_experts_per_token,
                    comm_backend="standard",
                    hidden_dim=expert_dim,
                ),
            ),
            router=router_cfg,
            load_balance_coeff=1e-3,
            shared_experts=shared_cfg,
        )
        if config.moe_enable_ep or config.moe_enable_tp:
            # Upstream (post-merge) parallelizes MoE module-internally:
            # sharding configs are declared on the Config BEFORE build,
            # then _moe.parallelize(parallel_dims) distributes states and
            # wires the token dispatcher (see parallelize.py). Same
            # expert-param TP layout as deepseek_v3.
            import spmd_types as spmd

            from torchtitan.models.common.moe_sharding import set_moe_sharding_config

            set_moe_sharding_config(
                moe_cfg,
                enable_ep=config.moe_enable_ep,
                # EXPERIMENT (EP x TP): with EP on, every layout upstream declares from
                # the router through the routed+shared add is sequence-parallel over the
                # flattened (CP, TP) axes -- because the sparse mesh folds tp into efsdp
                # and tp becomes a token axis inside the MoE region. Keying the DESIRED
                # layouts on enable_sp alone then asks for S(1) -> P(sum), which DTensor
                # rejects. Declaring SP when both are on makes src and dst agree.
                enable_sp=config.moe_enable_ep and config.moe_enable_tp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )
            if config.moe_enable_ep and config.moe_enable_tp:
                # Make the MoE a self-contained SP island with a REPLICATED external
                # boundary. Upstream's config assumes SP already arrives, because in its
                # models TP implies SP for the whole decoder. Ours does not: the layer
                # hands the FFN a tp-Replicate activation, by design (plain-ish
                # boundaries are what let PP's P2P, AttnRes's stack and fla's kernels
                # work). in_src describes what ARRIVES and in_dst what the module WANTS,
                # so declaring Replicate in / SP inside / Replicate out lets DTensor
                # insert the scatter and the all-gather instead of asking for the
                # impossible S(1) -> P(sum).
                import dataclasses as _dc

                from torchtitan.models.common.decoder_sharding import (
                    dense_activation_placement,
                )

                replicated = dense_activation_placement(tp=spmd.R)
                # router_input_BLD as well as x_BLD. The latent path calls
                # ``self._moe(to_latent(x), router_input_BLD=x)`` -- report Eq. 11 has the
                # router read the PRE-latent activation -- and upstream's config knows
                # only about x_BLD, so that second entry point was arriving Replicate at
                # a router whose gate declares SP. Both have to be named or the
                # redistribution reaches one of them.
                wanted = moe_cfg.sharding_config.in_dst_shardings["x_BLD"]
                moe_cfg.sharding_config = _dc.replace(
                    moe_cfg.sharding_config,
                    in_src_shardings={
                        "x_BLD": replicated,
                        "router_input_BLD": replicated,
                    },
                    in_dst_shardings={
                        "x_BLD": wanted,
                        "router_input_BLD": wanted,
                    },
                    out_dst_shardings=replicated,
                )

        # Under the latent path the shared experts are ours, at full width.
        shared_experts_cfg = None
        if config.num_shared_experts > 0 and latent_size is not None:
            shared_dim = config.moe_intermediate_size * config.num_shared_experts
            shared_experts_cfg = KimiMLP.make_config(
                config.hidden_size,
                shared_dim,
                hidden_act=config.hidden_act,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )

        return KimiMoE.Config(
            latent_size=latent_size,
            latent=latent_cfg,
            moe=moe_cfg,
            shared_experts=shared_experts_cfg,
        )

    def __init__(self, config: "KimiMoE.Config") -> None:
        super().__init__()
        self.latent_size = config.latent_size
        self.latent = config.latent.build() if config.latent is not None else None
        self._moe = config.moe.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )

    @property
    def routed_experts(self):
        """The composed core MoE's routed experts.

        ``distributed.fsdp.apply_fsdp_to_decoder`` reaches the grouped-GEMM child
        as ``block.moe.routed_experts.inner_experts``, which is upstream's flat
        layout. We compose core's MoE instead of re-implementing it, so the path
        needs one forward. A property is not in ``_modules``, so no parameter FQN
        changes and the object it returns is the one core already sharded.
        """
        return self._moe.routed_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_size is None:
            out = self._moe(x)
        else:
            # y = sum_j E_j^shared(x) + W_up RMSNorm( sum_i p_i E_i(W_down x) )
            # Router reads x; experts consume W_down x.
            out = self._moe(self.latent.to_latent(x), router_input_BLD=x)
        if isinstance(out, DTensor):
            # Module-internal MoE parallelization (EP/TP) emits DTensor.
            # This model's boundary convention is plain tensors (PP P2P,
            # AttnRes stacking, fla kernels), so redistribute to Replicate
            # if needed and unwrap. Measured under TP with EP off the
            # placements are ALREADY Replicate here, so the redistribute
            # does not fire and this is a plain unwrap; the gradient
            # arriving from downstream is replicated and agrees across tp
            # ranks to 5e-4, so to_local's default Replicate grad
            # placement is correct.
            if any(not p.is_replicate() for p in out.placements):
                out = out.redistribute(placements=[Replicate()] * len(out.placements))
            out = out.to_local()
        if self.latent_size is not None:
            out = self.latent.from_latent(out)
            if self.shared_experts is not None:
                out = out + self.shared_experts(x)
        return out


# ----- Decoder layer ------------------------------------------------------- #


class UpstreamFSDPNames:
    """Read-only aliases so ``distributed.fsdp.apply_fsdp_to_decoder`` can drive our layout.

    That helper reads five names off a decoder and two off each block, and spells three of
    them differently from us: ``tok_embeddings`` for our ``embed_tokens``,
    ``enable_weight_tying`` for the config flag, and ``moe`` / ``moe_enabled`` for our
    ``ffn._moe`` / ``is_moe``.

    Aliases rather than renames, because the helper only ever READS them -- it makes no
    assignment to any model attribute. A property is not in ``_modules``, so
    ``named_parameters()``, ``state_dict()`` and every FQN are untouched, and
    ``fully_shard(model.tok_embeddings)`` wraps exactly the object ``model.embed_tokens``
    already refers to. Renaming the submodules instead would have invalidated every DCP
    checkpoint written so far.

    ``moe`` is no longer among them: the MoE layer's own attribute is now called that,
    matching upstream, and a class-level property would SHADOW the module -- normal
    attribute lookup finds the property, and ``nn.Module.__getattr__`` only runs when
    that fails. The helper reaches the experts through ``KimiMoE.routed_experts``
    instead, which forwards into the composed core MoE.
    """

    @property
    def moe_enabled(self) -> bool:
        return bool(getattr(self, "is_moe", False))


class KimiDecoderLayer(Module, UpstreamFSDPNames):
    """One transformer block: pre-norm + attention + residual +
    pre-norm + MoE/MLP + residual.

    Faithful to ``reference:KimiDecoderLayer``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """One hybrid block. Attention and FFN are each an XOR pair.

        The layer type is readable off the populated field rather than by asking
        the model config, which is what lets parallelize and FSDP collect the MLA
        blocks without a config lookup.
        """

        layer_idx: int
        hidden_size: int
        input_layernorm: "RMSNorm.Config"
        post_attention_layernorm: "RMSNorm.Config"
        attention: "KimiMLAAttention.Config | None" = None
        delta_attention: "KimiDeltaAttention.Config | None" = None
        moe: "KimiMoE.Config | None" = None
        feed_forward: "KimiMLP.Config | None" = None

    @staticmethod
    def make_config(config: KimiK3Config, layer_idx: int) -> "KimiDecoderLayer.Config":
        """The one place this class reads the flat config."""

        def _norm() -> "RMSNorm.Config":
            return RMSNorm.Config(
                normalized_shape=config.hidden_size,
                eps=config.rms_norm_eps,
                sharding_config=_tp_replicate(),
            )

        cfg = KimiDecoderLayer.Config(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            input_layernorm=_norm(),
            post_attention_layernorm=_norm(),
        )
        # Attention: KDA vs MLA by layer index.
        if config.is_kda_layer(layer_idx):
            cfg.delta_attention = KimiDeltaAttention.make_config(config, layer_idx)
        elif config.is_mla:
            cfg.attention = KimiMLAAttention.make_config(config, layer_idx)
        else:
            # Reachable: a config with none of the MLA dims set and mla_use_nope
            # False is constructible, it is just not a model this port implements.
            raise ValueError(
                f"Layer {layer_idx}: neither KDA nor MLA configured. Set the "
                "MLA head dims (or mla_use_nope) or list the layer in kda_layers."
            )
        # FFN: dense MLP for the first `first_k_dense_replace` layers, MoE
        # otherwise. Kimi's reference uses `layer_idx >= first_k_dense_replace`
        # AND `layer_idx % moe_layer_freq == 0`; we follow that.
        if (
            config.is_moe
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            cfg.moe = KimiMoE.make_config(config)
        else:
            cfg.feed_forward = KimiMLP.make_config(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        return cfg

    def __init__(self, config: "KimiDecoderLayer.Config") -> None:
        super().__init__()
        self.layer_idx = config.layer_idx
        self.hidden_size = config.hidden_size
        self.attention = (
            config.attention.build() if config.attention is not None else None
        )
        self.delta_attention = (
            config.delta_attention.build()
            if config.delta_attention is not None
            else None
        )
        self.is_linear_attn = self.delta_attention is not None
        self.moe = config.moe.build() if config.moe is not None else None
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.is_moe = self.moe is not None
        self.input_layernorm = config.input_layernorm.build()
        self.post_attention_layernorm = config.post_attention_layernorm.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.input_layernorm(x)
        if self.attention is not None:
            x = self.attention(x)
        else:
            assert self.delta_attention is not None
            x = self.delta_attention(x)
        x = residual + x

        # FFN block
        residual = x
        x = self.post_attention_layernorm(x)
        if self.moe is not None:
            x = self.moe(x)
        else:
            assert self.feed_forward is not None
            x = self.feed_forward(x)
        x = residual + x
        return x


# ----- Top-level model ----------------------------------------------------- #


class KimiK3Model(Module):
    """Kimi Linear stack: embed -> decoder layers -> final RMSNorm -> LM head.

    No KV cache, no generation path. Training / loss is expected to be
    wired by the torchtitan trainer (cross-entropy over logits).

    ``_return_only_new_blocks`` and ``layers_per_block`` attributes
    are defined here so the cross-stage cache adapter can toggle
    forward output shape once ``KimiK3AttnResModel`` subclass
    adds the AttnRes block machinery. In the base (non-AttnRes) class
    the flag is ignored — forward always returns full hidden_states.
    """

    # See the note at the _skip_lm_head check in forward.
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """The trunk's config tree.

        ``tok_embeddings``, ``norm`` and ``lm_head`` carry the names core's
        ``set_decoder_sharding_config`` writes to; the module attribute for the
        first stays ``embed_tokens``, the release's name, so no checkpoint key
        moves. ``layers`` is walked by our own sharding.py -- that helper
        deliberately does not walk it.

        ``kimi_config`` rides along because the flat config is still read after
        construction: ``register_topology(model.config)`` in parallelize and the
        pipeline adapter, plus init_weights' initializer_range.
        """

        kimi_config: KimiK3Config
        tok_embeddings: "Embedding.Config"
        layers: list["KimiDecoderLayer.Config"]
        norm: "RMSNorm.Config"
        lm_head: "Linear.Config"

    @staticmethod
    def make_config(config: KimiK3Config) -> "KimiK3Model.Config":
        """The one place the trunk reads the flat config."""
        return KimiK3Model.Config(
            kimi_config=config,
            tok_embeddings=Embedding.Config(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            ),
            layers=[
                KimiDecoderLayer.make_config(config, i)
                for i in range(config.num_hidden_layers)
            ],
            norm=RMSNorm.Config(
                normalized_shape=config.hidden_size,
                eps=config.rms_norm_eps,
                sharding_config=_tp_replicate(),
            ),
            lm_head=Linear.Config(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                bias=False,
                sharding_config=_tp_shard(0),
            ),
        )

    def __init__(self, config: "KimiK3Model.Config") -> None:
        super().__init__()
        self.config = config.kimi_config

        self.embed_tokens = config.tok_embeddings.build()
        # ModuleDict (not ModuleList) so pipeline_module_split preserves
        # layer-id string keys and the adapter's layer_to_stage discovery
        # works unchanged. Matches the attn_res/ experiment's pattern.
        self.layers = nn.ModuleDict(
            {str(i): c.build() for i, c in enumerate(config.layers)}
        )
        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()

        if self.config.tie_word_embeddings:
            # Not used on 48B-A3B (tie_word_embeddings=False) but kept for
            # smaller debug flavors that might tie.
            self.lm_head.weight = self.embed_tokens.weight

        # Hook for AttnRes subclass + PP adapter.
        self._return_only_new_blocks: bool = False

    @property
    def tok_embeddings(self):
        """What ``apply_fsdp_to_decoder`` calls our ``embed_tokens``.

        Returns None on a PP stage that had it stripped, which is what the helper
        expects and already tests for.
        """
        return self.embed_tokens

    @property
    def enable_weight_tying(self) -> bool:
        return bool(getattr(self.config, "tie_word_embeddings", False))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with PP-split awareness.

        Args:
            tokens: Either ``[B, T]`` int64 token ids (stage 0 / non-PP)
                OR ``[B, T, D]`` hidden state from upstream PP stage
                (middle / last). Dispatch is decided by presence of
                ``self.embed_tokens`` (pipeline_module_split strips it
                off non-first stages).
            inputs_embeds: Optional ``[B, T, D]`` pre-computed
                embeddings. When provided, ``embed_tokens`` is skipped
                entirely (`tokens` is ignored as long as it's a valid
                placeholder dispatch on the right device). Used by
                multimodal training where image-token positions are
                replaced with vision-projector outputs before the LM
                forward — keeps the call as a single FSDP-root forward.
            **kwargs: Ignored. Accepts ``attention_masks=None`` and
                ``positions=...`` that torchtitan's Trainer / Validator
                may inject for FlexAttention / CP paths — Kimi Linear
                uses plain SDPA + KDA Triton kernels and doesn't need
                them.

        Returns:
            * Non-last PP stage: ``[B, T, D]`` hidden state to forward
              to the next stage.
            * Last stage / non-PP: ``[B, T, vocab_size]`` logits.
        """
        if inputs_embeds is not None:
            h = inputs_embeds
        elif self.embed_tokens is not None:
            h = self.embed_tokens(tokens)
            # Multimodal scatter: replace embed positions for image tokens
            # with externally-supplied vision_embeds. Done INSIDE this
            # forward so FSDP sees a single root call (calling
            # embed_tokens externally would split the root).
            if vision_embeds is not None and image_mask is not None:
                h = splice_vision_embeds(h, vision_embeds, image_mask)
        else:
            h = tokens  # middle/last PP stage: tokens IS the hidden state
        for layer in self.layers.values():
            h = layer(h)
        if self.norm is not None:
            h = self.norm(h)
        # _skip_lm_head is an attribute rather than a forward kwarg because PP
        # backward calls .requires_grad on all stage inputs and a bool kwarg
        # fails that -- the same reason core's decoder does it this way. Set by
        # the trainer when ChunkedLossWrapper is in use, which then applies
        # lm_head per sequence chunk so the [B, L, V] logits are never
        # materialised whole. That tensor, not depth or attention, is what caps
        # sequence length: at V=163840 and L=8192 its fp32 upcast alone is
        # 5.37 GiB.
        if self._skip_lm_head:
            return h
        if self.lm_head is not None:
            return self.lm_head(h)
        return h  # middle PP stage: ship hidden state downstream

    def verify_module_protocol(self) -> None:
        """No-op: our internals are plain nn.Module (not the torchtitan
        ``Module`` protocol), since KimiK3Model ports the HF
        reference layer-by-layer rather than going through the Config
        chain. Trainer calls this post-build; overriding as no-op keeps
        the FSDP + loss + optimizer paths intact without requiring every
        sub-module to register as a ``Module.Config``-built instance.
        """
        return None

    def get_attention_masks(self, *args, **kwargs):
        """Return ``None`` — KDA + MLA both use plain SDPA / Triton paths
        and don't take an external ``attention_masks`` kwarg through
        ``forward``. torchtitan's Validator and Trainer call this to
        precompute attention masks for FlexAttention/VarlenAttention
        models; for our SDPA-style stack the right answer is no mask
        passthrough.

        Defined as method (not raise NotImplementedError) so the trainer
        and validator paths don't crash on AttributeError. Returning
        ``None`` causes ``extra_kwargs["attention_masks"] = None`` and
        our forward signature ``(tokens)`` simply ignores extra kwargs
        the trainer might try to pass.
        """
        return None

    def init_weights(self, init_range: float | None = None, **kwargs) -> None:
        """Initialize *all* parameters and buffers from scratch.

        This must be exhaustive because torchtitan's trainer flow is
        ``meta-build → parallelize_fn (FSDP wrap) → to_empty(device=cuda)
        → init_weights``. ``to_empty`` discards every value set inside
        ``__init__`` (including RMSNorm.weight=1 defaults, KDA's A_log,
        dt_bias, ShortConvolution kernels, MoE expert weights, and
        load-balance buffers). Anything we forget here stays at whatever
        garbage ``torch.empty`` left on the device — which silently
        zeroes RMSNorm scales and produces near-uniform logits with no
        learning signal.
        """
        std = init_range if init_range is not None else self.config.initializer_range

        # Pass 1: leaf modules with well-typed init contracts.
        for m in self.modules():
            cls_name = type(m).__name__
            if isinstance(m, nn.Linear):
                if "weight" not in m._parameters:
                    # Packed-MXFP4 LoRA base: quantize_base_mxfp4 dropped
                    # base.weight (split qdata/scale storage); the packed
                    # values come from the checkpoint, not init.
                    continue
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif cls_name in (
                "ShortConvolution",
                "FusedRMSNormGated",
                "KimiLoRALinear",
            ):
                # fla-core modules + the LoRA wrapper ship reset_parameters()
                # (LoRA: kaiming lora_a, zero lora_b -- the generic Linear
                # pass above only covers their nn.Linear children).
                m.reset_parameters()

        # Pass 2: per-layer raw Parameters that don't belong to any nn.Module
        # subclass we can dispatch on -- KDA's A_log and dt_bias, and the MLA
        # output gate's graft init below. Both attention kinds reach this loop:
        # keying it on one of the two attribute names skips the other's init
        # silently, which is how the gated-MLA near-identity test caught this.
        for layer in self.layers.values():
            attn = getattr(layer, "delta_attention", None) or getattr(
                layer, "attention", None
            )
            if attn is None:
                continue
            if hasattr(attn, "A_log"):
                # Match KimiDeltaAttention.__init__: log(uniform(1, 16))
                attn.A_log.data.uniform_(1.0, 16.0).log_()
            if hasattr(attn, "dt_bias"):
                nn.init.zeros_(attn.dt_bias)
            # Output gate init. The graft variant is near-identity: zero
            # the projection and set a large positive bias so
            # sigmoid(gate) ~= 1 at step 0 (gated_out ~= plain attn_out).
            # K3's full_rank gate has no bias and is initialized normally,
            # so it is left to the generic Linear init above.
            gate_proj = getattr(attn, "attn_gate_proj", None)
            if gate_proj is not None and gate_proj.bias is not None:
                nn.init.zeros_(gate_proj.weight)
                nn.init.constant_(gate_proj.bias, 6.0)  # sigmoid(6)=0.9975

        # Pass 3: torchtitan MoE -- GroupedExperts holds raw [E, ...]
        # parameter tensors (not nn.Linear), and MoE/router carry
        # auxiliary-loss-free load-balance buffers that must start at 0.
        #
        # TODO(kimi-k3): upstream models declare per-parameter init functions
        # instead (see deepseek_v3/__init__.py's _depth_experts_init, which maps
        # "w1_EFD"/"w2_EDF"/"w3_EFD" to trunc_normal_ with depth-scaled std).
        # That table makes a missing entry visible in one place, and it also
        # gives the depth scaling this pass lacks. Migrating this whole
        # hand-rolled init_weights to that mechanism is the right follow-up --
        # a hand-maintained exhaustive walk is exactly what broke here.
        #
        # isinstance, not a class-name string: K3's routed experts are
        # KimiSiTUGroupedExperts, and the QAT/packing paths install further
        # subclasses. And the parameters are enumerated from _parameters
        # rather than a hardcoded ("w1", "w2", "w3") tuple, because upstream
        # renamed them to shape-suffixed w1_EFD / w2_EDF / w3_EFD -- a stale
        # name list here leaves every routed expert at to_empty garbage, which
        # trains to a plausible loss on the dense/shared/latent path alone
        # while the routed experts contribute nothing. Both mistakes are
        # silent, so test_expert_init_is_not_silently_skipped guards them.
        from torchtitan.models.common.moe import MoE

        for m in self.modules():
            if isinstance(m, MoE):
                # The protocol recurses into the experts and the router and
                # dispatches through the param_init maps declared in KimiMoE,
                # raising on any parameter no map covers. It also zeroes the
                # load-balance buffers via _init_self_buffers.
                m.init_states(buffer_device=kwargs.get("buffer_device"))


# ----- ModelSpec shim: BaseModel.Config wrapper --------------------------- #

# Imports at module bottom to keep the KimiLinear* classes usable as plain
# nn.Modules without dragging the torchtitan.protocols.model chain in
# when used from the CPU tests.


@dataclass(kw_only=True, slots=True)
class KimiK3Spec:
    """``BaseModel.Config``-compatible shim that wraps a
    :class:`KimiK3Config` and an optional ``num_blocks`` (None =
    plain :class:`KimiK3Model`; integer N = :class:`KimiK3AttnResModel`
    with ``num_blocks=N``).

    Methods implemented for torchtitan integration:

    * :meth:`build` — returns the constructed model instance (either
      :class:`KimiK3Model` or :class:`KimiK3AttnResModel`).
    * :meth:`update_from_config` — no-op for Kimi Linear: MLA uses
      NoPE (``mla_use_nope=True``) so no RoPE max_seq_len to propagate,
      and KDA is seq-len-agnostic (short conv + recurrent state).
    * :meth:`get_nparams_and_flops` — trainer uses this for MFU
      reporting. Returns (n_params, forward+backward FLOPs per step).

    Deliberately NOT inheriting from ``BaseModel.Config`` at class
    definition to keep the module importable in CPU tests without
    pulling in the ``torchtitan.protocols`` dependency chain. The
    trainer only needs duck-typing on ``build`` /
    ``update_from_config`` / ``get_nparams_and_flops``.
    """

    kimi_config: KimiK3Config
    num_blocks: int | None = None
    # Block size for Block AttnRes, when the flavor derives its block count
    # from one. num_blocks alone cannot express K3's "full blocks plus a short
    # tail" partition (see KimiK3AttnResModel.__init__), so carry the size and
    # let the model use it verbatim. None keeps the equal-split reading.
    attn_res_block_size: int | None = None
    param_init: dict | None = None  # torchtitan BaseModel.Config contract
    # Graft gate: alpha-gated AttnRes reads (alpha=0 == exact identity
    # with the plain backbone at step 0). For grafting onto pretrained
    # weights; from-scratch flavors keep the paper's ungated read.
    attn_res_gated: bool = False
    # Gate for the PP cross-stage cache adapter (finding 32: was
    # TORCHTITAN_ATTNRES_CACHE). Opt-in, because it changes what crosses a stage
    # boundary. Resolved through knobs.register_topology.
    attn_res_cache: bool = False
    # LoRA (module-level; see lora.py). rank=None disables. When set,
    # target projections are wrapped (lora_b zero-init -> step-0
    # identity) and the base freezes EXCEPT the AttnRes graft params
    # (alpha-fullparam exception).
    lora_rank: int | None = None
    lora_alpha: float = 16.0
    lora_quantize_base: str | None = None  # 'nf4' => QLoRA
    # MXFP8 activations on a packed-MXFP4 LoRA base, so the adapter trains
    # against the numerics the deployed model runs. Independent of mxfp4_qat,
    # which is the BACKBONE's training precision.
    lora_quantize_act: bool = False
    # K3's post-training QAT (report sec 4.1.4): MXFP4 routed-expert
    # weights + MXFP8 expert activations, fake-quant with a bf16 master.
    # Scope comes from quant_scope.py, not a name list.
    mxfp4_qat: bool = False
    # Per-Head Muon (report sec 2.5). Tagging has to happen at BUILD time, not in
    # post_optimizer_build_fn: the optimizer is constructed from the parameters,
    # and Muon reads _muon_heads off each one, so a tag applied afterwards is
    # invisible to it.
    per_head_muon: bool = False

    # Registry-discovery passthroughs. veRL's torchtitan engine identifies a
    # flavor by reading cfg.dim / cfg.n_layers / cfg.vocab_size off
    # model_registry(flavor).model -- torchtitan's llama-convention names, which
    # our KimiK3Config spells hidden_size / num_hidden_layers. Without these
    # the shape match silently finds nothing and flavor resolution fails.
    @property
    def dim(self) -> int:
        return self.kimi_config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.kimi_config.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.kimi_config.vocab_size

    def build(self, **kwargs):
        # Local import to defer the attn_res_model dep chain.
        from torchtitan.models.kimi_k3.attn_res_model import KimiK3AttnResModel

        if self.num_blocks is None:
            model = KimiK3Model.make_config(self.kimi_config).build()
        else:
            model = KimiK3AttnResModel(
                self.kimi_config,
                num_blocks=self.num_blocks,
                layers_per_block=self.attn_res_block_size,
                gated=self.attn_res_gated,
            )
        return self.apply_build_time_features(model)

    def apply_build_time_features(self, model):
        """Attach LoRA, Per-Head Muon tags and MXFP4 QAT to a built model.

        Separate from ``build`` because the multimodal spec overrides ``build``
        to construct a vision-bearing model; without a shared entry point every
        one of these config fields is silently dropped on multimodal flavors.
        None of them can match a MoonViT module: the LoRA target names and the
        routed-expert QAT scope do not exist in the tower.
        """
        if self.lora_rank is not None:
            from torchtitan.models.kimi_k3.lora import apply_lora

            apply_lora(
                model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                quantize_base=self.lora_quantize_base,
                quantize_act=self.lora_quantize_act,
            )
        if self.per_head_muon:
            from torchtitan.models.kimi_k3.muon import tag_per_head_muon

            from torchtitan.tools.logging import logger

            tagged = tag_per_head_muon(model)
            logger.info("Per-Head Muon: tagged %d Q/K/V projections.", tagged)
        if self.mxfp4_qat:
            from torchtitan.models.kimi_k3.mxfp4_qat import apply_mxfp4_qat

            # Disjoint from LoRA: QAT attaches to GroupedExperts (3-D params),
            # LoRA wraps nn.Linear, and K3's scope contains no Linear at all.
            apply_mxfp4_qat(model)
        return model

    def update_from_config(self, *, config, **kwargs) -> None:
        """Wire parallelism knobs the model must know BEFORE build.

        Signature matches ``BaseModel.Config.update_from_config``
        (keyword ``config`` = the Trainer.Config).

        MoE EP/TP: upstream parallelizes MoE module-internally via
        sharding configs declared at config-build time; KimiMoE reads
        these flags when constructing its MoE.Config. Seq-len needs no
        propagation (NoPE-MLA + KDA are seq-len-agnostic).
        """
        parallelism = getattr(config, "parallelism", None)
        if parallelism is not None:
            self.kimi_config.moe_enable_ep = parallelism.expert_parallel_degree > 1
            self.kimi_config.moe_enable_tp = parallelism.tensor_parallel_degree > 1
        return None

    def get_nparams_and_flops(
        self,
        model: nn.Module,
        seq_len: int,
    ) -> tuple[int, int]:
        """(total_n_params, flops_per_TOKEN) for MFU reporting.

        Follows torchtitan's MoE convention in
        ``torchtitan.models.utils.get_moe_model_nparams_and_flops``
        (6x = fwd 2x + bwd 4x), extended for this architecture:

            flops_per_token = 6 * activated_non_embedding          (linear)
                            + 6 * n_mla * n_heads * head_dims * seq (MLA)
                            + 12 * n_kda * kda_heads * kda_dim^2    (KDA)
                            + 6 * (2*n_layers + 1) * (N+1) * hidden (AttnRes)

        * MLA: O(seq) per token (softmax attention counted per-token).
        * KDA: linear attention -- the per-head [kda_head_dim x
          kda_head_dim] recurrent state is written (delta-rule update)
          and read (output) once per token, seq-len INDEPENDENT; the 2
          state touches give the 12x (= 6 * 2) factor. Projections are
          already inside the 6*W linear term.
        * AttnRes (only when ``num_blocks`` is set): each sub-layer read
          mixes up to N block sources + the partial block per token
          (softmax over sources + weighted sum over hidden), twice per
          layer (attn + mlp reads) plus the final read.

        Activated params: dense + shared_expert + router + routed*top_k/num_experts.

        Embedding excluded from the linear term (FLOPs-free lookup).
        """
        nparams_total = 0
        nparams_embedding = 0
        nparams_dense = 0
        nparams_router = 0
        nparams_shared = 0
        nparams_routed = 0
        for name, p in model.named_parameters():
            nparams_total += p.numel()
            if "embed_tokens" in name or "lm_head" in name:
                # lm_head is tied to embeddings in Kimi scaling-law configs,
                # but not always — only exclude embed_tokens.
                if "embed_tokens" in name:
                    nparams_embedding += p.numel()
                # Treat both as dense for non-attention FLOPs; embedding
                # lookup is free, lm_head is a real projection.
                nparams_dense += p.numel()
            # These must match the real module names, which are ``_moe`` with a
            # leading underscore and ``routed_experts`` rather than ``experts``.
            # A bucket that matches nothing sends every MoE parameter into `dense`, which
            # counts all experts as activated. Keep the trailing dot: it stops the router
            # pattern from also claiming a dense FFN's ``gate_proj``.
            elif "._moe.shared_experts." in name:
                nparams_shared += p.numel()
            elif "._moe.router." in name:
                nparams_router += p.numel()
            elif "._moe.routed_experts." in name:
                nparams_routed += p.numel()
            else:
                nparams_dense += p.numel()

        cfg = self.kimi_config
        top_k = cfg.num_experts_per_token
        n_experts = cfg.num_experts or 1
        nparams_active_linear = (
            nparams_dense
            - nparams_embedding
            + nparams_shared
            + nparams_router
            + nparams_routed * top_k // n_experts
        )

        # MLA attention FLOPs: only full_attn_layers (softmax, O(seq)/token).
        n_mla_layers = len(cfg.full_attn_layers) if cfg.full_attn_layers else 0
        head_dims_attn = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim
        attn_flops_per_token = (
            6 * n_mla_layers * cfg.num_attention_heads * head_dims_attn * seq_len
        )

        # KDA linear-attention state ops: per token each head writes and
        # reads its [kda_head_dim x kda_head_dim] recurrent state once.
        n_kda_layers = (
            len(cfg.kda_layers)
            if cfg.kda_layers
            else cfg.num_hidden_layers - n_mla_layers
        )
        kda_flops_per_token = (
            12 * n_kda_layers * cfg.kda_num_heads * cfg.kda_head_dim**2
        )

        # AttnRes source mixing: 2 reads per layer + the final read, each
        # mixing up to (num_blocks + 1) sources over hidden_size.
        if self.num_blocks is not None:
            attn_res_flops_per_token = (
                6
                * (2 * cfg.num_hidden_layers + 1)
                * (self.num_blocks + 1)
                * cfg.hidden_size
            )
        else:
            attn_res_flops_per_token = 0

        flops_per_token = (
            6 * nparams_active_linear
            + attn_flops_per_token
            + kda_flops_per_token
            + attn_res_flops_per_token
        )
        return nparams_total, flops_per_token

    def to_dict(self) -> dict:
        """Serialize to a plain dict for logging / checkpoint metadata.

        Trainer calls this on the model_config to pretty-print the
        configuration before building. We flatten the wrapped
        :class:`KimiK3Config` dataclass into this dict so the log
        shows the actual Kimi hyperparameters (not just a reference).
        """
        import dataclasses

        out = dataclasses.asdict(self.kimi_config)
        out["__spec__"] = {
            "num_blocks": self.num_blocks,
            "model_class": (
                "KimiK3AttnResModel" if self.num_blocks is not None else "KimiK3Model"
            ),
        }
        return out

    @property
    def layers(self) -> list[None]:
        """Fake list of length ``num_hidden_layers`` for torchtitan
        pipeline_llm's ``num_layers = len(model_config.layers)`` check.

        Kimi Linear's per-layer config is not a standalone dataclass
        (KDA/MLA/MoE types vary per layer), so we don't expose a real
        list of per-layer Config objects. This property gives
        pipeline_llm the count it needs. Downstream consumers that
        iterate layers should use the built model's ``model.layers``
        (nn.ModuleList) directly.
        """
        return [None] * self.kimi_config.num_hidden_layers

    @property
    def num_hidden_layers(self) -> int:
        """Expose num_hidden_layers at the spec level so adapter code
        (pipeline_adapter._inject_kimi_k3_fqns) can get layer count
        without reaching into kimi_config.
        """
        return self.kimi_config.num_hidden_layers

    def traverse(self, config_cls, *, recurse: bool = False, _prefix: str = ""):
        """Config-tree leaf: yield nothing.

        The Kimi Linear model is built as plain modules from
        :class:`KimiK3Config`, not from a ``Configurable.Config``
        tree, so there are no nested component configs to expose.
        Implemented because the Trainer chain requires it on every
        model config (``has_quantization``, the override mechanism via
        ``ModelSpec.traverse``).
        """
        return iter(())


@dataclass(kw_only=True, slots=True)
class KimiK3Float8Spec(KimiK3Spec):
    """:class:`KimiK3Spec` whose ``build()`` swaps eligible
    ``nn.Linear`` modules to torchao ``Float8Linear``.

    The Kimi Linear model is constructed as plain modules, not from a
    ``Linear.Config`` tree, so ``Float8LinearConverter.convert``'s
    config traversal cannot apply here. Instead the swap happens
    module-level right after construction (on the meta device, before
    parallelize/init), mirroring the converter's ``module_filter_fn``
    semantics: all dims divisible by 16, filtered FQNs skipped.
    Additionally every Linear inside a :class:`KimiDeltaAttention` is
    skipped structurally. A name-based filter is now expressible too, since
    KDA lives under ``delta_attention`` and MLA under ``attention``, but the
    structural skip does not depend on the spelling staying that way.
    ``init_weights`` still covers swapped modules because torchao's
    ``Float8Linear`` subclasses ``nn.Linear``.
    """

    torchao_float8_config: object = None
    filter_fqns: list[str] = field(default_factory=list)

    def build(self, **kwargs):
        from torchao.float8 import convert_to_float8_training

        # Explicit base call: zero-arg super() breaks under
        # @dataclass(slots=True), which recreates the class object.
        model = KimiK3Spec.build(self, **kwargs)

        kda_linear_fqns = {
            f"{name}.{sub_name}"
            for name, m in model.named_modules()
            if isinstance(m, KimiDeltaAttention)
            for sub_name, sub in m.named_modules()
            if sub_name and isinstance(sub, nn.Linear)
        }

        def _filter(mod: nn.Module, fqn: str) -> bool:
            return (
                mod.in_features % 16 == 0
                and mod.out_features % 16 == 0
                and fqn not in kda_linear_fqns
                and not any(f in fqn for f in self.filter_fqns)
            )

        return convert_to_float8_training(
            model,
            config=self.torchao_float8_config,
            module_filter_fn=_filter,
        )

    def traverse(self, config_cls, *, recurse: bool = False, _prefix: str = ""):
        """Yield a single synthetic Float8Linear.Config marker.

        The Float8 swap here is module-level (``build()``), so there is
        no real config tree to report. Config-tree consumers -- today
        only ``has_quantization``, which gates the misleading-under-fp8
        MFU metric -- still need to see that quantization is active.
        The marker's dims are placeholders (16x16, the fp8 alignment
        unit); treat it strictly as a boolean signal, never as a real
        layer description.
        """
        from torchtitan.components.quantization.float8 import Float8Linear

        if (
            self.torchao_float8_config is not None
            and Float8Linear is not None
            and issubclass(Float8Linear.Config, config_cls)
        ):
            fqn = (
                f"{_prefix}.module_level_float8_swap"
                if _prefix
                else "module_level_float8_swap"
            )
            marker = Float8Linear.Config(
                in_features=16,
                out_features=16,
                _torchao_config=self.torchao_float8_config,
            )
            yield fqn, marker, None, None
        else:
            yield from ()
