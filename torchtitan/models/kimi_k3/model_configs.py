# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scaling-law config registry for Kimi Linear + AttnRes.

    Parametric :class:`KimiK3Config` constructors for the five sizes in the AttnRes
    report's Table 2 (194M to 528M activated) plus the 48B-A3B upscale target, which is
    kept for reference since it needs multi-node.

    See ``phase13_k3like_48b_posttrain/SCALING_LAW_CONFIGS.md``.
    """

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torchtitan.models.kimi_k3.model import KimiK3Config


# ----- Paper Table 2 canonical sizes -------------------------------------- #
# Columns copied verbatim from Kimi Linear AttnRes tech report Table 2.
# d_ff is the MoE per-expert intermediate size (moe_intermediate_size in our
# config). L_b is the number of Kimi decoder layers (= num_hidden_layers).


@dataclass(frozen=True)
class _SweepSize:
    """One row of the tech report's scaling-law sweep (Table 2)."""

    name: str
    activated_params: int  # M parameters (reported, non-embedding)
    tokens: float  # B tokens
    n_layers: int  # L_b in paper (= num_hidden_layers in our config)
    num_heads: int  # H in paper (= num_attention_heads + kda_num_heads)
    d_model: int  # d_model in paper
    d_ff: int  # d_ff in paper (= moe_intermediate_size in our config)
    lr: float  # peak learning rate
    batch_size: int  # global batch size (sequences)


SCALING_LAW_TABLE: tuple[_SweepSize, ...] = (
    _SweepSize("194m", 194, 38.7, 12, 12, 896, 400, 2.99e-3, 192),
    _SweepSize("241m", 241, 45.4, 13, 13, 960, 432, 2.80e-3, 256),
    _SweepSize("296m", 296, 62.1, 14, 14, 1024, 464, 2.50e-3, 320),
    _SweepSize("436m", 436, 87.9, 16, 16, 1168, 528, 2.20e-3, 384),
    _SweepSize("528m", 528, 119.0, 17, 17, 1264, 560, 2.02e-3, 432),
    # SGLang-friendly aligned-dim variant of the 436M row.
    # d=1024 (vs 1168) → head_dim=64 is multiple of 16; qk_rope=32, v=64,
    # kv_lora=512 all 8/16/32-aligned; flashinfer / cublas / triton
    # extend kernels accept this layout on SM 12.0 (RTX 5090). d_ff
    # bumped 528 → 768 to keep activated-param count ~447M, roughly
    # matching the original 436M row's compute budget.
    # Reuses 436M's lr / batch_size / token_count from the same row.
    _SweepSize("447m_aligned", 447, 87.9, 16, 16, 1024, 768, 2.20e-3, 384),
    # Full Kimi Linear 48B-A3B target. From paper §"Training recipe":
    # "27 Transformer blocks (54 layers)" with Block AttnRes N=9
    # (6 paper-layers per AttnRes-block = 3 transformer-blocks per
    # AttnRes-block). d_ff here is the MoE-per-expert intermediate
    # size (1024 in HF config); the dense FFN at layer 0 uses
    # intermediate_size=9216 (set in build_kimi_linear_48b_a3b_config
    # via the override path, not from this row).
    # NOTE: 48B requires multi-node; this row exists for carrier
    # construction + config-correctness checks, not single-node training.
    # tokens/lr/batch from paper §Training recipe (1T pretrain +
    # 400B mid-train; "global batch size of 8M tokens" → 8M/4096
    # context = 1953 seqs ≈ 2048).
    _SweepSize("48b", 3000, 1400.0, 27, 32, 2304, 1024, 1.0e-3, 2048),
    # Kimi K3, VERBATIM from the official config.json (2026-07-27) -- no
    # longer provisional. 93 layers, hidden 7168, 96 heads (head_dim 128),
    # moe_intermediate 3072. activated_params 104B per the model card; the
    # tokens/lr/batch entries remain OUR training-recipe choice, not the
    # paper's (the report does not publish the 2.8T optimizer schedule).
    # Reference: the released Kimi K3 config.json.
    _SweepSize("2p8t", 104000, 1400.0, 93, 96, 7168, 3072, 4.0e-4, 4096),
)

_BY_NAME: dict[str, _SweepSize] = {s.name: s for s in SCALING_LAW_TABLE}

# CI debug size -- NOT a paper row (kept out of SCALING_LAW_TABLE so the
# table stays verbatim Table 2). 4 layers = 3 KDA + 1 MLA at the default
# 3:1 ratio; d=256/H=4 -> head_dim 64, kv_lora 128; builds and runs a
# forward on CPU in seconds with the bundled 2016-token test tokenizer.
_BY_NAME["debugmodel"] = _SweepSize("debugmodel", 1, 0.01, 4, 4, 256, 128, 3e-4, 8)

# 8-head debug size for deep tp x cp meshes: H=4 binds at tp*cp=4, so
# tp2cp4 / tp4cp2 (8 ranks) need H=8. d=512 keeps head_dim 64.
_BY_NAME["debugmodel8h"] = _SweepSize("debugmodel8h", 4, 0.01, 4, 8, 512, 128, 3e-4, 8)

# K3-FAITHFUL downscale. Every structural choice is K3's, only the extents
# shrink, so it is the carrier for anything that must behave like K3 rather
# than merely run:
#   * head_dim 128 exactly (d=512, H=4) -- required by FlashKDA (K=V=128), so
#     this is the only debug row that can exercise the official inference
#     kernel at all;
#   * 21 layers with K3's own attn_res_block_size 12 -> 2 blocks with a
#     9-layer tail, which mirrors the SHAPE of K3's 93 = 7*12 + 9 (same block
#     size, same tail length) instead of just being small;
#   * KDA:MLA 3:1 with the final layer forced global (layers 4, 8, 12);
#   * latent ratio 0.5 (routed_expert_hidden_size = d/2), Ns = 2 shared.
_BY_NAME["k3mini"] = _SweepSize("k3mini", 70, 0.01, 21, 4, 512, 224, 3e-4, 8)

# The flavor functions renamed kimi_linear_k3mini_* -> kimi_k3_mini_*, so the
# parsed size is now "mini". Alias rather than rename the row: "k3mini" is
# still what older launch scripts and logbook entries name it.
_BY_NAME["mini"] = _BY_NAME["k3mini"]


# ----- 48B-A3B reference (upscale target, kept for docs) ------------------ #
# Faithful to the HF config.json at moonshotai/Kimi-Linear-48B-A3B-Base.
# Listed here so the full scale sweep is visible in one file; the 48B
# config needs multi-node to train.

_KIMI_48B_A3B_KDA_LAYERS = (
    1,
    2,
    3,
    5,
    6,
    7,
    9,
    10,
    11,
    13,
    14,
    15,
    17,
    18,
    19,
    21,
    22,
    23,
    25,
    26,
)
_KIMI_48B_A3B_FULL_ATTN_LAYERS = (4, 8, 12, 16, 20, 24, 27)


# ----- Sweep config builders ---------------------------------------------- #


def _alternating_kda_mla_layers(
    n_layers: int,
    kda_mla_ratio: int = 3,
    *,
    force_final_full_attn: bool = False,
) -> tuple[list[int], list[int]]:
    """Build 1-indexed kda_layers / full_attn_layers lists with given ratio.

    Default ratio 3:1 matches the paper + 48B-A3B (3 KDA, 1 MLA, repeat); MLA
    lands every ``kda_mla_ratio+1``-th layer (1-indexed).

    ``force_final_full_attn`` adds the last layer to the MLA set even when the
    period would not select it. K3 does this -- report sec 2.1: "An additional
    Gated MLA layer is placed at the end of the backbone, ensuring that the
    final layer always performs global attention" -- which is why its official
    full_attn_layers is [4, 8, ..., 88, 92, 93], with 92 AND 93 both global.
    """
    period = kda_mla_ratio + 1
    kda, mla = [], []
    for i in range(1, n_layers + 1):
        if i % period == 0:
            mla.append(i)
        else:
            kda.append(i)
    if force_final_full_attn and n_layers not in mla:
        mla.append(n_layers)
        kda.remove(n_layers)
    return kda, sorted(mla)


# Sizes that run against the repo's bundled test tokenizer rather than K3's own.
# Keeping this in ONE place matters: model_registry (which veRL and
# convert_to_hf.py resolve through) and the Trainer.Config flavors are separate
# code paths, and when they disagreed about k3mini's vocab the seed checkpoint
# came out with a 2016-row embedding that could not load into the 163840-row model
# the registry built -- which is what blocked the veRL actor.
BUNDLED_TOKENIZER_SIZES: frozenset[str] = frozenset(
    {"k3mini", "debugmodel", "debugmodel8h"}
)
BUNDLED_TOKENIZER_VOCAB = 2016
K3_VOCAB = 163840


def default_vocab_size(size: str) -> int:
    """Vocab a size uses when the caller does not say otherwise."""
    return BUNDLED_TOKENIZER_VOCAB if size in BUNDLED_TOKENIZER_SIZES else K3_VOCAB


def build_kimi_linear_config(
    size: str,
    *,
    num_experts: int | None = None,
    vocab_size: int | None = None,
    tie_word_embeddings: bool | None = None,
    kda_mla_ratio: int = 3,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-5,
    dense_intermediate_size: int | None = None,
    use_grouped_topk: bool | None = None,
) -> KimiK3Config:
    """Construct a :class:`KimiK3Config` for one scaling-law size.

    Args:
        size: One of ``{"194m","241m","296m","436m","528m","48b"}``.
        num_experts: Total MoE experts (token-choice top-k). Default 32
            for scaling-law sizes; 256 for the full 48B-A3B target.
        vocab_size: Token vocabulary. Default 163840 (Kimi tokenizer).
        tie_word_embeddings: Tie input/output embedding. Default True
            for scaling-law (smaller model, more param-efficient); False
            for 48B-A3B (matches HF config.json).
        kda_mla_ratio: KDA:MLA layer ratio. Default 3 matches paper + 48B.
        rope_theta: RoPE base (unused when ``mla_use_nope=True``, which is
            the Kimi default).
        rms_norm_eps: RMSNorm epsilon.
        dense_intermediate_size: Dense FFN intermediate size used by
            layer 0 only (when ``first_k_dense_replace=1``). Defaults to
            ``spec.d_ff`` (= MoE per-expert intermediate). 48B-A3B
            overrides: dense=9216 while moe-per-expert=1024.
        use_grouped_topk: MoE router grouped-topk gate. Default False
            (simplified); 48B-A3B uses True (matches HF config.json).
    """
    if vocab_size is None:
        vocab_size = default_vocab_size(size)
    if size not in _BY_NAME:
        raise ValueError(f"Unknown size '{size}'. Valid: {sorted(_BY_NAME.keys())}")
    spec = _BY_NAME[size]
    d = spec.d_model
    H = spec.num_heads

    # Size-specific defaults that differ between scaling-law sweep and
    # full 48B-A3B. Each is overridable from the kwargs above.
    if size == "2p8t":
        # official config.json
        num_experts_default = 896
        tie_default = False
        dense_d_ff_default = 33792  # intermediate_size (dense layer 0)
        use_grouped_topk_default = True
    elif size == "k3mini":
        num_experts_default = 8
        tie_default = False
        dense_d_ff_default = spec.d_ff * 4
        use_grouped_topk_default = True
    elif size == "48b":
        num_experts_default = 256
        tie_default = False
        dense_d_ff_default = 9216  # HF config.json:intermediate_size
        use_grouped_topk_default = True  # HF config.json
    else:
        num_experts_default = 32
        tie_default = True
        dense_d_ff_default = spec.d_ff
        use_grouped_topk_default = False
    if num_experts is None:
        num_experts = num_experts_default
    if tie_word_embeddings is None:
        tie_word_embeddings = tie_default
    if dense_intermediate_size is None:
        dense_intermediate_size = dense_d_ff_default
    if use_grouped_topk is None:
        use_grouped_topk = use_grouped_topk_default

    # Head dims — scaled to fit d_model/H, following 48B-A3B where
    # num_heads * head_dim = hidden_size (Kimi has no d_head < hidden/head_count).
    # For KDA: head_dim = d_model / num_heads (round to pow-2 via max(32, ...))
    # For MLA (NoPE): qk_nope + qk_rope + v_head split. Paper's 48B uses
    # qk_nope=128, qk_rope=64, v_head=128 at d=2304, num_heads=32, so each
    # head takes 128 (nope) + 64 (rope, broadcast) + 128 (v) units. We keep
    # qk_rope proportional to d/num_heads * 0.5 (half of nope).
    if size in ("48b", "2p8t", "k3mini"):
        # Verbatim from the official config.json -- Kimi-Linear-48B-A3B-Base
        # and Kimi-K3 happen to share all five of these.
        head_dim_mla_nope = 128
        head_dim_mla_rope = 64
        head_dim_mla_v = 128
        kda_head_dim = 128
        kv_lora_rank = 512
    elif size.endswith("_aligned"):
        # SGLang flashinfer / cuBLAS / triton extend kernels on
        # SM 12.0 (RTX 5090) require head_dim multiple of 8 (16 preferred
        # so qk_rope = head_dim/2 is also 8-aligned). Round head_dim down
        # to multiple of 16, kv_lora_rank to multiple of 64.
        head_dim_mla_nope = max(32, (d // H) & ~15)
        head_dim_mla_rope = max(16, head_dim_mla_nope // 2)
        head_dim_mla_v = head_dim_mla_nope
        kda_head_dim = head_dim_mla_nope
        kv_lora_rank = (d // 2) & ~63
    else:
        head_dim_mla_nope = max(32, d // H)
        head_dim_mla_rope = max(16, head_dim_mla_nope // 2)
        head_dim_mla_v = head_dim_mla_nope
        kda_head_dim = head_dim_mla_nope
        kv_lora_rank = d // 2  # scale with model; 48B uses 512 at d=2304 ≈ d/4.5

    if size == "48b":
        # HF config.json has 7 MLA layers (full_attn) at indices 4,8,12,16,
        # 20,24,27 (1-indexed) and 20 KDA layers everywhere else. The pattern
        # is "every 4th layer is MLA, plus the last layer 27". Hand-emit this
        # exact split instead of going through _alternating_kda_mla_layers
        # (which would miss layer 27 because 27 % 4 != 0).
        full_attn_layers = [4, 8, 12, 16, 20, 24, 27]
        kda_layers = [
            i for i in range(1, spec.n_layers + 1) if i not in full_attn_layers
        ]
    else:
        # K3 places an extra Gated MLA at the very end (report sec 2.1), so its
        # official full_attn_layers is [4, 8, ..., 88, 92, 93] -- 92 AND 93 both
        # global. Without force_final_full_attn we would put 93 on KDA.
        kda_layers, full_attn_layers = _alternating_kda_mla_layers(
            spec.n_layers,
            kda_mla_ratio=kda_mla_ratio,
            force_final_full_attn=is_k3_shaped(size),
        )

    # ---- K3 structural deltas (official config.json, 2026-07-27) ----
    # Every one of these is a real architectural choice, not a hyperparameter,
    # so they key off the size rather than being global defaults.
    is_k3 = is_k3_shaped(size)
    return KimiK3Config(
        # Vocabulary / embedding
        vocab_size=vocab_size,
        hidden_size=d,
        tie_word_embeddings=tie_word_embeddings,
        # Depth / width
        num_hidden_layers=spec.n_layers,
        intermediate_size=dense_intermediate_size,  # dense FFN (layer 0)
        # MLA
        num_attention_heads=H,
        num_key_value_heads=H,  # no GQA
        q_lora_rank=(1536 if size == "2p8t" else d // 4) if is_k3 else None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=head_dim_mla_nope,
        qk_rope_head_dim=head_dim_mla_rope,
        v_head_dim=head_dim_mla_v,
        mla_use_nope=True,
        rope_theta=rope_theta,
        # KDA
        kda_num_heads=H,
        kda_head_dim=kda_head_dim,
        kda_short_conv_kernel_size=4,
        kda_layers=list(kda_layers),
        full_attn_layers=list(full_attn_layers),
        # MoE
        num_experts=num_experts,
        num_experts_per_token=(
            16 if size == "2p8t" else (2 if size == "k3mini" else 8)
        ),
        moe_intermediate_size=spec.d_ff,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=2 if is_k3 else 1,
        routed_scaling_factor=1.0 if is_k3 else 2.446,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=1,
        topk_group=1,
        # Norm / init
        rms_norm_eps=rms_norm_eps,
        hidden_act="situ" if is_k3 else "silu",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        initializer_range=0.02,
        # Gated MLA (Eq. 7) and KDA's Eq. 5 / Eq. 6 parameterization.
        # kda_gate_lower_bound is not optional for K3 fidelity: FlashKDA, the
        # official inference kernel, refuses to run without safe_gate.
        mla_gated=is_k3,
        attn_gate_param="full_rank",
        kda_gate_lower_bound=-5.0 if is_k3 else None,
        kda_use_full_rank_gate=is_k3,
        # Stable LatentMoE (Eq. 11): routed experts in a 3584 latent.
        routed_expert_hidden_size=(
            (3584 if size == "2p8t" else d // 2) if is_k3 else None
        ),
        latent_moe_use_norm=True,
        # 1M context.
        max_position_embeddings=1048576 if size == "2p8t" else 4096,
    )


Variant = Literal["baseline", "block_attn_res", "full_attn_res"]


def is_k3_shaped(size: str) -> bool:
    """True for rows that reproduce K3's architecture rather than the sweep's.

    One predicate so the K3 deltas (SiTU, both full-rank gates, the lower-bounded
    decay, q-compression, LatentMoE, the extra final global-attention layer,
    block size 12) cannot drift apart across the builder.

    Keyed on the ROW, not the name. A name list silently excluded the "mini"
    alias, so every flavor built as ``kimi_k3_mini_*`` after the rename got the
    sweep architecture instead of the K3 one -- silu rather than SiTU, 32 experts
    rather than 8, and block size 3 rather than 12 -- while the trainer flavor of
    the same name built the K3 architecture. That is a checkpoint that loads into
    neither, from two spellings of one row.
    """
    row = _BY_NAME.get(size)
    return row is not None and row in (_BY_NAME["2p8t"], _BY_NAME["k3mini"])


def attn_res_block_size(size: str) -> int:
    """Layers per AttnRes block.

    One rule for every row: the size that lands the block count nearest the
    paper's "N ~= 8 recovers most of the benefit" (report sec 2.2), i.e.
    ``round(n_layers / 8)``.

    Worth noting that this reproduces K3's official value without being told:
    93 layers -> round(93/8) = 12 = ``attn_res_block_size`` in the shipped
    config, giving 8 blocks with a 9-layer tail. The N ~= 8 heuristic this repo
    has used since before the release derives the official partition exactly.
    """
    if is_k3_shaped(size):
        return 12  # K3's official attn_res_block_size, kept verbatim
    return max(1, round(_BY_NAME[size].n_layers / 8))


def resolve_num_blocks(size: str, variant: Variant) -> int | None:
    """Pick ``num_blocks`` for the given (size, variant) combo.

    Returns ``None`` for the baseline (no AttnRes). ``n_layers`` for
    Full AttnRes.

    For Block AttnRes the partition is driven by BLOCK SIZE, not by an equal
    split -- K3 uses ``attn_res_block_size = 12`` over 93 layers, giving 7 full
    blocks plus a 9-layer tail (report sec 2.2). So ``num_blocks =
    ceil(n_layers / block_size)`` and no divisibility is required. Block size
    defaults to 12 for K3-shaped depths and otherwise to whatever lands nearest
    the paper's "N ~= 8" shorthand, which for the shallow sweep rows means a
    small size rather than a contrived divisor.
    """
    if size not in _BY_NAME:
        raise ValueError(f"Unknown size '{size}'")
    n_layers = _BY_NAME[size].n_layers
    if variant == "baseline":
        return None
    if variant == "full_attn_res":
        return n_layers
    if variant == "block_attn_res":
        block_size = attn_res_block_size(size)
        return max(1, -(-n_layers // block_size))  # ceil
    raise ValueError(f"Unknown variant '{variant}'")


def build(
    size: str,
    variant: Variant,
) -> tuple[KimiK3Config, int | None]:
    """Top-level entrypoint: return ``(kimi_config, num_blocks)``.

    Pass to :class:`KimiK3Model` (baseline) or
    :class:`KimiK3AttnResModel` (AttnRes) depending on
    ``num_blocks is None``.
    """
    return (
        build_kimi_linear_config(size),
        resolve_num_blocks(size, variant),
    )


# ----- Convenience: which (size, variant) pairs exist -------------------- #


def flavor_names() -> list[str]:
    """All registered flavor names: ``kimi_linear_{size}_{variant}``."""
    out: list[str] = []
    for s in SCALING_LAW_TABLE:
        for v in ("baseline", "block_attn_res", "full_attn_res"):
            out.append(f"kimi_linear_{s.name}_{v}")
    return out


# ----- Trainer.Config factories ------------------------------------------ #
# One function per flavor, hand-rolled so the torchtitan ConfigManager
# can import them by name. Pattern matches attn_res/config_registry.py.
