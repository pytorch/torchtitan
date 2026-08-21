# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bidirectional HF <-> torchtitan key map for the released K3 checkpoint.

    The part that is not a string rewrite: one released key can map to a SLICE of one of
    our stacked tensors (``...experts.3.w1.weight`` -> ``...w1_EFD[3]``), so the reverse
    direction needs an expert index. ``g_proj`` also resolves by layer type, which is why
    the map takes ``kda_layers``.

    See ``phase13_k3like_48b_posttrain/HF_KEY_MAP.md``.
    """

from __future__ import annotations

import re

TEXT_PREFIX = "language_model.model."
LM_HEAD = "language_model.lm_head.weight"

# The same tensors as seen in a TEXT-ONLY checkpoint, with no multimodal
# wrapper. Read but never written: titan_to_official always emits the released
# (multimodal) spelling, so a round-trip stays canonical.
TEXT_ONLY_PREFIX = "model."
TEXT_ONLY_LM_HEAD = "lm_head.weight"

# Per-layer names that differ only by spelling.
_LAYER_RENAME = {
    "self_attention_res_proj": "attention_res_proj",
    "self_attention_res_norm": "attention_res_norm",
    "mlp_res_proj": "ffn_res_proj",
    "mlp_res_norm": "ffn_res_norm",
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
}

# Attention leaves that keep their name. Both attention types are covered; the
# official g_proj is ambiguous between them, so it is resolved by layer type.
_ATTN_SAME = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_a_proj",
    "q_a_layernorm",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_a_layernorm",
    "kv_b_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "q_conv1d",
    "k_conv1d",
    "v_conv1d",
    "o_norm",
    "A_log",
    "dt_bias",
)

# Routed experts: w1 gate, w3 up, w2 down (reference annotates these).
EXPERT_W_TO_SUFFIXED = {"w1": "w1_EFD", "w2": "w2_EDF", "w3": "w3_EFD"}

_MOE_BLOCK_RENAME = {
    "routed_expert_down_proj": "latent.down",
    "routed_expert_up_proj": "latent.up",
    "routed_expert_norm": "latent.norm",
}

VISION_PREFIX = "vision_tower."
PROJECTOR_PREFIX = "mm_projector."

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


class UnmappedKey(ValueError):
    """A checkpoint key with no destination. Never ignored silently."""


def kda_layers_zero_based(kimi_config) -> set[int]:
    """``kda_layers`` renumbered to match CHECKPOINT key indices.

    The release lists linear-attention layers 1-BASED in ``linear_attn_config`` and this
    folder's configs follow it, while checkpoint keys are ``layers.<0-based>``. Comparing
    the two directly misclassifies every layer whose 0-based index happens to appear in
    the 1-based set -- and the only visible symptom is a gate tensor landing on the wrong
    name, so it reads as a missing key rather than an off-by-one.

    Exists because :func:`_mla_layer` puts normalisation on the caller and no caller had
    a helper to do it with.
    """
    return {i - 1 for i in (getattr(kimi_config, "kda_layers", None) or ())}


def _mla_layer(layer_idx: int, kda_layers: set[int]) -> bool:
    """The release uses 1-BASED layer indices in linear_attn_config, while
    checkpoint keys are 0-based, so the caller's kda_layers must already be
    normalized to whichever base it uses. See is_kda_layer in the reference."""
    return layer_idx not in kda_layers


def official_to_titan(key: str, *, kda_layers: set[int]) -> tuple[str, str]:
    """Translate one released key. Returns ``(our_key, kind)``.

    ``kind`` is one of ``"param"``, ``"buffer"``, ``"expert_packed"``,
    ``"expert_scale"``, ``"vision"``. Raises :class:`UnmappedKey` otherwise --
    a checkpoint tensor we cannot place is a bug, not something to skip.
    """
    if key in (LM_HEAD, TEXT_ONLY_LM_HEAD):
        return "lm_head.weight", "param"
    if key.startswith(VISION_PREFIX) or key.startswith(PROJECTOR_PREFIX):
        # Our MoonViT holds the projector as a child, so mm_projector.* becomes
        # a child path and vision_tower.* loses its prefix.
        if key.startswith(PROJECTOR_PREFIX):
            return f"vision_tower.mm_projector.{key[len(PROJECTOR_PREFIX):]}", "vision"
        return f"vision_tower.{key[len(VISION_PREFIX):]}", "vision"
    # The RELEASE is the multimodal wrapper, so its text keys carry
    # ``language_model.model.``. A text-only checkpoint -- what a text flavor
    # exports, and what vLLM's KimiLinearForCausalLM consumes -- carries a bare
    # ``model.``. Accept both: refusing the bare form meant our own adapter could
    # not read a checkpoint our own exporter had just written, which is where
    # veRL's actor stopped.
    if key.startswith(TEXT_PREFIX):
        rest = key[len(TEXT_PREFIX) :]
    elif key.startswith(TEXT_ONLY_PREFIX):
        rest = key[len(TEXT_ONLY_PREFIX) :]
    else:
        raise UnmappedKey(key)
    if rest == "embed_tokens.weight":
        return "embed_tokens.weight", "param"
    if rest == "norm.weight":
        return "norm.weight", "param"
    if rest == "output_attn_res_proj.weight":
        return "output_res_proj.weight", "param"
    if rest == "output_attn_res_norm.weight":
        return "output_res_norm.weight", "param"

    m = _LAYER_RE.match(rest)
    if m is None:
        raise UnmappedKey(key)
    idx, tail = int(m.group(1)), m.group(2)
    head = tail.split(".", 1)[0]

    if head in _LAYER_RENAME:
        return f"layers.{idx}.{_LAYER_RENAME[head]}.weight", "param"

    if head == "self_attn":
        leaf = tail.split(".", 1)[1]
        name = leaf.rsplit(".", 1)[0] if leaf.endswith(".weight") else leaf
        # The release calls both attention types self_attn; we hold MLA under
        # attention and KDA under delta_attention, so the layer type picks the
        # attribute -- the same resolution g_proj already needed.
        mla = _mla_layer(idx, kda_layers)
        attn_attr = "attention" if mla else "delta_attention"
        if name == "g_proj":
            # KDA keeps g_proj; MLA's gate is attn_gate_proj on our side.
            ours = "attn_gate_proj" if mla else "g_proj"
            return f"layers.{idx}.{attn_attr}.{ours}.weight", "param"
        if name in _ATTN_SAME:
            suffix = ".weight" if leaf.endswith(".weight") else ""
            return f"layers.{idx}.{attn_attr}.{name}{suffix}", "param"
        raise UnmappedKey(key)

    if head == "mlp":
        # the single dense layer (first_k_dense_replace)
        leaf = tail.split(".", 1)[1]
        return f"layers.{idx}.feed_forward.{leaf}", "param"

    if head == "block_sparse_moe":
        leaf = tail.split(".", 1)[1]
        first = leaf.split(".", 1)[0]
        if first in _MOE_BLOCK_RENAME:
            return f"layers.{idx}.moe.{_MOE_BLOCK_RENAME[first]}.weight", "param"
        if first == "shared_experts":
            return f"layers.{idx}.moe.{leaf}", "param"
        if leaf == "gate.weight":
            return f"layers.{idx}.moe._moe.router.gate.weight", "param"
        if leaf == "gate.e_score_correction_bias":
            return f"layers.{idx}.moe._moe.expert_bias_E", "buffer"
        em = re.match(r"^experts\.(\d+)\.(w[123])\.(.+)$", leaf)
        if em:
            expert, w, suffix = int(em.group(1)), em.group(2), em.group(3)
            base = (
                f"layers.{idx}.moe._moe.routed_experts.inner_experts."
                f"{EXPERT_W_TO_SUFFIXED[w]}"
            )
            if suffix == "weight_packed":
                return f"{base}[{expert}]", "expert_packed"
            if suffix == "weight_scale":
                return f"{base}[{expert}]", "expert_scale"
            if suffix == "weight":
                return f"{base}[{expert}]", "param"
            raise UnmappedKey(key)
        raise UnmappedKey(key)

    raise UnmappedKey(key)


def titan_to_official(
    key: str,
    *,
    kda_layers: set[int],
    expert_idx: int | None = None,
    text_only: bool = False,
) -> str:
    """Inverse of :func:`official_to_titan` for a single tensor.

    Expert weights need ``expert_idx`` because one stacked ``w1_EFD`` on our
    side corresponds to ``num_experts`` separate official keys.

    ``text_only`` emits the bare ``model.`` prefix instead of the release's
    ``language_model.model.``. A text flavor has no vision tower, so the
    multimodal wrapper spelling names a module that does not exist -- and
    because the checkpoint loader builds its expected-key list from this
    function, emitting it made our own adapter unable to read a checkpoint our
    own exporter had written.
    """
    if text_only:
        result = titan_to_official(
            key, kda_layers=kda_layers, expert_idx=expert_idx, text_only=False
        )
        if result.startswith(TEXT_PREFIX):
            return TEXT_ONLY_PREFIX + result[len(TEXT_PREFIX) :]
        if result == LM_HEAD:
            return TEXT_ONLY_LM_HEAD
        return result

    inv_layer = {v: k for k, v in _LAYER_RENAME.items()}
    inv_moe = {v: k for k, v in _MOE_BLOCK_RENAME.items()}
    inv_expert = {v: k for k, v in EXPERT_W_TO_SUFFIXED.items()}

    if key == "lm_head.weight":
        return LM_HEAD
    if key.startswith("vision_tower.mm_projector."):
        return PROJECTOR_PREFIX + key[len("vision_tower.mm_projector.") :]
    if key.startswith("vision_tower."):
        return VISION_PREFIX + key[len("vision_tower.") :]
    if key in ("embed_tokens.weight", "norm.weight"):
        return TEXT_PREFIX + key
    if key == "output_res_proj.weight":
        return TEXT_PREFIX + "output_attn_res_proj.weight"
    if key == "output_res_norm.weight":
        return TEXT_PREFIX + "output_attn_res_norm.weight"

    m = _LAYER_RE.match(key)
    if m is None:
        raise UnmappedKey(key)
    idx, tail = int(m.group(1)), m.group(2)
    prefix = f"{TEXT_PREFIX}layers.{idx}."

    stem = tail.rsplit(".weight", 1)[0]
    if stem in inv_layer:
        return f"{prefix}{inv_layer[stem]}.weight"

    for attn_attr in ("attention.", "delta_attention."):
        if not tail.startswith(attn_attr):
            continue
        leaf = tail[len(attn_attr) :]
        name = leaf.rsplit(".", 1)[0] if leaf.endswith(".weight") else leaf
        official = "g_proj" if name in ("g_proj", "attn_gate_proj") else name
        suffix = ".weight" if leaf.endswith(".weight") else ""
        return f"{prefix}self_attn.{official}{suffix}"

    if tail.startswith("feed_forward."):
        # the dense layer: HF calls it mlp
        return f"{prefix}mlp.{tail[len('feed_forward.'):]}"

    if tail.startswith("moe."):
        leaf = tail[len("moe.") :]
        base = leaf.rsplit(".weight", 1)[0]
        if base in inv_moe:
            return f"{prefix}block_sparse_moe.{inv_moe[base]}.weight"
        if leaf.startswith("shared_experts."):
            return f"{prefix}block_sparse_moe.{leaf}"
        if leaf == "_moe.router.gate.weight":
            return f"{prefix}block_sparse_moe.gate.weight"
        if leaf == "_moe.expert_bias_E":
            return f"{prefix}block_sparse_moe.gate.e_score_correction_bias"
        em = re.match(r"^_moe\.routed_experts\.inner_experts\.(w\d_\w+)$", leaf)
        if em:
            if expert_idx is None:
                raise UnmappedKey(
                    f"{key} is a stacked expert tensor; expert_idx is required"
                )
            w = inv_expert[em.group(1)]
            return f"{prefix}block_sparse_moe.experts.{expert_idx}.{w}.weight"
        # dense FFN
        return f"{prefix}mlp.{leaf}"

    raise UnmappedKey(key)


# ---------------------------------------------------------------------------
# The config half of the same contract.
#
# Names are only half of what an inference engine keys on: it builds its modules
# from config.json first, and a config that disagrees with the weights fails at
# load with a name error that looks like a naming bug. That happened here -- a
# fixture carried the low-rank KDA gate (g_a_proj / g_b_proj) while its nested
# linear_attn_config claimed use_full_rank_gate, and the official loader, which
# reads the NESTED flag, went looking for g_proj.
#
# The schema is vLLM's KimiLinearConfig (vllm/transformers_utils/configs/
# kimi_linear.py): flat text fields, with the KDA settings in a nested
# linear_attn_config dict. Deriving both from one KimiK3Config is what makes
# them unable to disagree.
# ---------------------------------------------------------------------------

# Fields whose name and meaning are identical on both sides.
_PASSTHROUGH_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "initializer_range",
    "rms_norm_eps",
    "tie_word_embeddings",
    "max_position_embeddings",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "mla_use_nope",
    "num_experts",
    "num_experts_per_token",
    "num_shared_experts",
    "moe_intermediate_size",
    "moe_renormalize",
    "moe_router_activation_func",
    "routed_scaling_factor",
    "routed_expert_hidden_size",
    "first_k_dense_replace",
    "moe_layer_freq",
    "use_grouped_topk",
    "num_expert_group",
    "topk_group",
    "num_nextn_predict_layers",
    "latent_moe_use_norm",
    "activation_situ_beta",
    "activation_situ_linear_beta",
)


def titan_config_to_official(
    kimi_config,
    *,
    num_blocks: int | None = None,
    layers_per_block: int | None = None,
) -> dict:
    """Serialize a ``KimiK3Config`` to the official HF text config schema.

    ``num_blocks`` is the Block AttnRes block count; the released config states
    the block SIZE instead, so it is derived here rather than stored twice.
    Pass None for a backbone without AttnRes.

    Renames, each because the two sides genuinely spell it differently:

    * ``mla_gated`` -> ``mla_use_output_gate`` (Gated MLA, report Eq. 7)
    * ``kda_num_heads`` / ``kda_head_dim`` / ``kda_short_conv_kernel_size`` /
      ``kda_gate_lower_bound`` / ``kda_use_full_rank_gate`` -> the unprefixed
      keys inside ``linear_attn_config``
    * ``kda_layers`` / ``full_attn_layers`` appear in ``linear_attn_config``;
      both sides use 1-based indices there.

    Deliberately NOT emitted: ``kda_cp_mode``, ``moe_enable_ep``,
    ``moe_enable_tp``, ``attn_gate_param`` -- training-side knobs with no
    inference meaning. Emitting them would invite an engine to key on a field we
    do not intend as part of the contract.
    """
    cfg: dict = {"model_type": "kimi_linear"}
    for name in _PASSTHROUGH_CONFIG_FIELDS:
        if hasattr(kimi_config, name):
            cfg[name] = getattr(kimi_config, name)

    cfg["mla_use_output_gate"] = bool(getattr(kimi_config, "mla_gated", False))
    cfg["topk_method"] = "noaux_tc"
    cfg["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": getattr(kimi_config, "rope_theta", 10000.0),
    }
    if layers_per_block is not None or num_blocks is not None:
        # A partial final block is the released arrangement, not an edge case:
        # block size 12 over 93 layers is 7 full blocks plus a 9-layer tail
        # (report sec 2.2). Prefer the model's actual layers_per_block; the ceil
        # fallback reproduces what KimiK3AttnResModel derives from num_blocks
        # alone, which is exact for a config-supplied count but cannot recover a
        # size-derived one (see that constructor for why it is not invertible).
        cfg["attn_res_block_size"] = (
            layers_per_block
            if layers_per_block is not None
            else -(-kimi_config.num_hidden_layers // num_blocks)
        )

    cfg["linear_attn_config"] = {
        "num_heads": kimi_config.kda_num_heads,
        "head_dim": kimi_config.kda_head_dim,
        "short_conv_kernel_size": kimi_config.kda_short_conv_kernel_size,
        "kda_layers": list(kimi_config.kda_layers),
        "full_attn_layers": list(kimi_config.full_attn_layers),
        "gate_lower_bound": kimi_config.kda_gate_lower_bound,
        # The official loader reads the gate form from HERE, not from a
        # top-level key, and it decides whether the checkpoint must carry
        # g_proj or the low-rank g_a_proj/g_b_proj pair.
        "use_full_rank_gate": bool(kimi_config.kda_use_full_rank_gate),
    }
    return cfg


# The vision half of the config contract. The release prefixes the tower's own
# dims with ``vt_`` while ours are unprefixed, which is the only real divergence;
# everything else is same-named.
_VISION_RENAME = {
    "num_hidden_layers": "vt_num_hidden_layers",
    "hidden_size": "vt_hidden_size",
    "num_attention_heads": "vt_num_attention_heads",
    "intermediate_size": "vt_intermediate_size",
}

_VISION_PASSTHROUGH = (
    "patch_size",
    "init_pos_emb_height",
    "init_pos_emb_width",
    "qkv_hidden_size",
    "text_hidden_size",
    "merge_kernel_size",
)


def titan_vision_config_to_official(vision_config) -> dict:
    """Serialize a ``MoonViTConfig`` to the official vision-config schema."""
    cfg: dict = {"model_type": "kimi_k3_vision"}
    for ours, theirs in _VISION_RENAME.items():
        if hasattr(vision_config, ours):
            cfg[theirs] = getattr(vision_config, ours)
    for name in _VISION_PASSTHROUGH:
        if hasattr(vision_config, name):
            value = getattr(vision_config, name)
            cfg[name] = list(value) if isinstance(value, tuple) else value
    return cfg


def titan_config_to_official_multimodal(
    kimi_config,
    vision_config,
    *,
    num_blocks: int | None = None,
    layers_per_block: int | None = None,
    media_placeholder_token_id: int = 163605,
) -> dict:
    """The released config shape: text and vision nested, not flattened.

    ``KimiK3Config`` exposes ``hidden_size`` and ``vocab_size`` as read-only
    properties delegating to ``text_config``, so a flat text field at the top
    level does not merely go unread -- it raises "property has no setter" when
    transformers tries to assign it. The nesting is required, not cosmetic.
    """
    return {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": titan_config_to_official(
            kimi_config, num_blocks=num_blocks, layers_per_block=layers_per_block
        ),
        "vision_config": titan_vision_config_to_official(vision_config),
        "media_placeholder_token_id": media_placeholder_token_id,
    }
