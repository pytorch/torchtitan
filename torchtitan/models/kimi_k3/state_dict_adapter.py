# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF <-> torchtitan state-dict adapter for the Kimi Linear (+AttnRes) LM.

    Wired as ``ModelSpec.state_dict_adapter``, so offline conversion, the Trainer's
    ``initial_load_in_hf`` path and veRL's engine all go through it.

    See ``phase13_k3like_48b_posttrain/STATE_DICT_KEYSPACE.md``.
    """

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.tools.logging import logger


_W_TO_HF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_HF_TO_W = {v: k for k, v in _W_TO_HF.items()}

# Post-merge GroupedExperts params carry shape suffixes (Noam convention):
# w1/w3 are [E, F, D], w2 is [E, D, F].
_EXPERT_W_SUFFIXED = {"w1": "w1_EFD", "w2": "w2_EDF", "w3": "w3_EFD"}
_EXPERT_SUFFIXED_TO_W = {v: k for k, v in _EXPERT_W_SUFFIXED.items()}

# Sidecar/packed key suffixes that signal a quantized HF checkpoint.
_QUANT_KEY_MARKERS = (
    ".weight_scale",
    ".weight_scale_inv",
    ".scales",
    ".weight_packed",
    ".qweight",
    ".weight_blocks",
    ".qzeros",
)

_DIRECT_MAP_FROM_HF = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
    "model.output_res_proj.weight": "output_res_proj.weight",
    "model.output_res_norm.weight": "output_res_norm.weight",
    "model.output_res_alpha": "output_res_alpha",
}

# Attention leaves whose released name differs from ours, so they must reach
# hf_key_map rather than being passed through. Kept as a set so a second
# divergence has one place to be added.
_ATTN_LEAVES_RENAMED_BY_HF_KEY_MAP = frozenset({"attn_gate_proj"})

_PASSTHROUGH_LAYER_TAGS = (
    "attention_res_alpha",
    "ffn_res_alpha",
    "attention_res_proj.weight",
    "attention_res_norm.weight",
    "ffn_res_proj.weight",
    "ffn_res_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


_MM_TEXT_PREFIX = "language_model."
"""Wrapper child prefix a multimodal model's TEXT tensors carry, both in tt naming
and in to_hf's export. Named rather than inlined because from_hf has to strip it on
the way in and re-attach it on every destination."""


class KimiLinearStateDictAdapter(MoEStateDictAdapter):
    """StateDictAdapter for KimiK3Model / KimiK3AttnResModel."""

    def __init__(self, model_config, hf_assets_path: str | None):
        # model_config is a KimiK3Spec (duck-typed shim); the base
        # class only reads the safetensors index from hf_assets_path.
        super().__init__(model_config, hf_assets_path)
        self.kimi_config = model_config.kimi_config
        # LoRA renames every wrapped projection's weight (q_proj.weight ->
        # q_proj.base.weight). to_hf already strips that on export; loading
        # needs the inverse, or a plain base checkpoint cannot be loaded into a
        # LoRA model at all -- which is the 48B graft path: take official
        # weights, attach adapters, train. Without it the load dies on
        # "Missing key: ...base.weight".
        self._lora_rank = getattr(model_config, "lora_rank", None)
        self._lora_targets: tuple[str, ...] = ()
        if self._lora_rank is not None:
            from torchtitan.models.kimi_k3.lora import DEFAULT_LORA_TARGETS

            self._lora_targets = DEFAULT_LORA_TARGETS

    def _add_lora_base(self, tt_key: str) -> str:
        """Insert ``.base`` for LoRA-wrapped projections, if LoRA is enabled.

        Matches the same leaf/qualified-suffix rule apply_lora uses, so the two
        cannot disagree about which modules are wrapped.
        """
        if not self._lora_targets or not tt_key.endswith((".weight", ".bias")):
            return tt_key
        stem, _, suffix = tt_key.rpartition(".")
        leaf = stem.rpartition(".")[2]
        matched = leaf in self._lora_targets or any(
            "." in t and stem.endswith(f".{t}") for t in self._lora_targets
        )
        return f"{stem}.base.{suffix}" if matched else tt_key

    # ----- quantization guard -------------------------------------- #

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            # torch's own reader rather than a local unpack path. Its MXFP4
            # handling is format-compatible with what packed_mxfp4.py implements
            # on every axis that can be checked without a released artifact: the
            # same 16-entry E2M1 value table, the same 32-value group, and it
            # dispatches on the `_blocks` / `_scales` suffixes that K3's
            # `.weight_blocks` / `.scales` keys carry.
            #
            # What is NOT checked, for want of a packed K3 checkpoint on this
            # box, is the blocks tensor's dimension order -- upstream expects
            # [a, b, groups, bytes]. So this path is exercised by an explicit
            # from_quantized=True and is not the default. It replaces a blanket
            # refusal whose stated reason (waiting on the report to fix the
            # packing) is stale: the packing is known and implemented.
            #
            # block_size is left at its default because MXFP4 does not use it --
            # it is the fp8 blockwise scale tile, and the group size for MXFP4
            # comes from the blocks tensor itself.
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            return QuantizedHuggingFaceStorageReader(path)
        return HuggingFaceStorageReader(path)

    @staticmethod
    def _check_not_packed(hf_state_dict: dict[str, Any]) -> None:
        packed = [
            k
            for k in hf_state_dict
            if k.endswith(_QUANT_KEY_MARKERS)
            or (
                isinstance(hf_state_dict[k], torch.Tensor)
                and hf_state_dict[k].dtype
                in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)
            )
        ]
        if packed:
            raise NotImplementedError(
                "HF checkpoint contains quantized/packed tensors "
                f"(e.g. {packed[:4]}); the MXFP4/packed unpack path is not "
                "implemented yet. Refusing to silently treat packed weights "
                "as ordinary values."
            )

    # ----- tt -> HF -------------------------------------------------- #

    def _is_text_only(self, state_dict=None) -> bool:
        """Decide the prefix from the STATE DICT, falling back to the config.

        No vision tower means the release's multimodal wrapper prefix names a
        module this model does not have, so the bare ``model.`` spelling is
        right. Reading that off the config alone is not reliable: depending on
        how the spec is threaded, ``model_config`` here can be the inner text
        config even for a multimodal model, and then a multimodal export gets
        written with text-only keys and cannot be read back.

        The state dict cannot be wrong about it -- a multimodal model has
        ``vision_tower.*`` parameters and a text one does not.
        """
        if state_dict is not None:
            return not any(k.startswith("vision_tower.") for k in state_dict)
        return getattr(self.model_config, "vision_config", None) is None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert tt state dict to HF naming; split stacked experts."""
        hf_state_dict: dict[str, Any] = {}
        num_experts = self.kimi_config.num_experts
        text_only = self._is_text_only(state_dict)
        for key, value in state_dict.items():
            # LoRA wrapping renames base weights (q_proj.weight ->
            # q_proj.base.weight); the HF destination is the original
            # name, and the value stays a view of the same storage so
            # the online read path fills the real param in place.
            key = key.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
            if (
                "attention_res" in key
                or "ffn_res" in key
                or "output_res" in key
                or "lora_a" in key
                or "lora_b" in key
            ):
                # Graft/LoRA extras have no HF-format destination: the HF
                # key space is the ORIGINAL Kimi architecture (so official
                # checkpoints load into graft flavors without phantom read
                # keys). Trained graft/adapter params ship as the
                # fork-native trainable_state_dict payload instead.
                continue
            if ".moe._moe.routed_experts.inner_experts." in key:
                # layers.{i}.moe._moe.routed_experts.inner_experts.w1_EFD
                # -> per-expert HF linears
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                w_suffixed = key.rsplit(".", 1)[-1]
                w_tag = _EXPERT_SUFFIXED_TO_W[w_suffixed]
                # Official Kimi-Linear-48B export style: routed experts are
                # block_sparse_moe.experts.{e}.w{1,2,3}.weight (w-naming),
                # while shared experts use gate/up/down_proj naming.
                # The wrapper prefix has to be honoured here too. _tt_key_to_hf
                # applies it to every other key, but the expert path builds its
                # own name, so a multimodal model emitted experts as "model.*"
                # while everything else was "language_model.model.*" -- and the
                # load then failed on exactly the expert keys, nothing else.
                expert_prefix = "" if text_only else "language_model."
                hf_abstract_key = (
                    expert_prefix
                    + "model.layers.{}.block_sparse_moe.experts.{}."
                    + w_tag
                    + ".weight"
                )
                if isinstance(value, DTensor):
                    # Online (sharded) path: record placement metadata so
                    # from_hf can rebuild the DTensor, emit local experts.
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    hf_state_dict.update(
                        self._get_local_experts_weights(
                            hf_abstract_key, abstract_key, layer_num, value
                        )
                    )
                else:
                    split_values = self._split_experts_weights(value, num_experts)
                    for e in range(num_experts):
                        hf_state_dict[
                            hf_abstract_key.format(layer_num, e)
                        ] = split_values[e].squeeze()
                continue

            if key.endswith("self_attn.A_log"):
                # File-side KDA A_log is [1, 1, H, 1]; the model holds [H].
                # The online HF reader validates placeholder shapes against
                # the saved file, so the view must happen on this side too
                # (from_hf flattens back).
                value = value.reshape(1, 1, -1, 1)
            hf_state_dict[self._tt_key_to_hf(key, text_only)] = value

        return hf_state_dict

    @staticmethod
    def _tt_key_to_hf(key: str, text_only: bool = False) -> str:
        """Single-tensor tt -> HF key mapping (experts handled separately)."""
        direct = {v: k for k, v in _DIRECT_MAP_FROM_HF.items()}
        if key in direct:
            return direct[key]
        if key.startswith(("vision_tower.", "language_model.")):
            # A multimodal model's keys carry a wrapper child prefix, and the
            # vision subtree has no "layers." at all, so both were rejected here
            # before the hf_key_map delegation below could see them. hf_key_map
            # owns the vision naming (vision_tower.mm_projector.* becomes the
            # release's mm_projector.*), so hand them straight over.
            from torchtitan.models.kimi_k3.hf_key_map import titan_to_official

            return titan_to_official(
                key.removeprefix("language_model."),
                kda_layers=set(),
                text_only=text_only,
            )
        if not key.startswith("layers."):
            raise ValueError(f"Unmapped tt key: {key!r}")
        rest = key[len("layers.") :]
        idx_s, _, sub = rest.partition(".")
        prefix = f"model.layers.{idx_s}"

        # The attention leaves match the Kimi-Linear-48B naming this adapter was
        # written for, so only the module name is translated below. K3 breaks the
        # leaf match for exactly one: our attn_gate_proj is the release's g_proj.
        # Emitting our own name made the checkpoint load fail on a key nothing
        # writes, so the renamed leaves fall through to hf_key_map instead of
        # being caught here.
        attn_attr = next(
            (a for a in ("attention.", "delta_attention.") if sub.startswith(a)),
            None,
        )
        if sub in _PASSTHROUGH_LAYER_TAGS:
            return f"{prefix}.{sub}"
        if (
            attn_attr is not None
            and sub.split(".")[1] not in _ATTN_LEAVES_RENAMED_BY_HF_KEY_MAP
        ):
            # The 48B naming spells both attention types self_attn, so our two
            # attributes collapse onto the single HF one.
            return f"{prefix}.self_attn.{sub[len(attn_attr):]}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if sub == f"feed_forward.{proj}.weight":
                return f"{prefix}.mlp.{proj}.weight"
        if sub == "moe._moe.router.gate.weight":
            return f"{prefix}.block_sparse_moe.gate.weight"
        if sub == "moe._moe.expert_bias_E":
            return f"{prefix}.block_sparse_moe.gate.e_score_correction_bias"
        if sub.startswith("moe._moe.shared_experts."):
            tail = sub[len("moe._moe.shared_experts.") :]
            w_tag, _, suff = tail.partition(".")
            return f"{prefix}.block_sparse_moe.shared_experts.{_W_TO_HF[w_tag]}.{suff}"
        # K3's layout (latent MoE projections, the released AttnRes and gate
        # names) is owned by hf_key_map, which is tested for full coverage
        # against the released checkpoint index. This adapter predates it and
        # targets the Kimi-Linear-48B naming, so K3-only keys arrive here;
        # delegate rather than keep the same table in two places that can drift.
        from torchtitan.models.kimi_k3.hf_key_map import titan_to_official, UnmappedKey

        try:
            return titan_to_official(key, kda_layers=set(), text_only=text_only)
        except UnmappedKey:
            pass
        raise ValueError(f"Unmapped tt key: {key!r}")

    # ----- HF -> tt -------------------------------------------------- #

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to tt naming; stack per-expert weights."""
        self._check_not_packed(hf_state_dict)

        from torchtitan.models.kimi_k3.hf_key_map import (
            kda_layers_zero_based,
            official_to_titan,
            UnmappedKey,
        )

        state_dict: dict[str, Any] = {}
        num_experts = self.kimi_config.num_experts
        kda_zero_based = kda_layers_zero_based(self.kimi_config)
        # {layer: {titan_abstract_key: {expert_id: tensor}}}
        expert_weights_by_layer: dict[str, dict[str, dict[int, Any]]] = {}

        # Iterate over a key snapshot and pop each entry as it is
        # consumed: on the online (sharded initial-load) path
        # hf_state_dict holds every loaded per-expert slice, and keeping
        # those references alive while the stacked copies are built
        # doubles the peak -- enough to OOM the 48B load on 32 GiB
        # cards. Consuming the input dict is part of this method's
        # contract (the caller replaces it with the returned dict).
        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)

            # Undo to_hf's multimodal naming before any pattern below sees the key.
            # to_hf strips the wrapper child prefix and hands the rest to hf_key_map,
            # so a multimodal export is "language_model.model.layers.N..." for text and
            # "vision_tower.*"/"mm_projector.*" for vision. Every regex here is anchored
            # at "model.layers.", so without this NOTHING matched and _hf_key_to_tt
            # returned (None, value) -- which is a skip, not an error. Measured before
            # the fix: 526 of 526 text tensors of a kimi_k3_mini_vl round trip were
            # dropped in silence, i.e. an official multimodal shard loaded as
            # near-empty.
            mm_prefix = ""
            official_key = key  # hf_key_map's inverse expects the ORIGINAL export name
            if key.startswith(_MM_TEXT_PREFIX):
                mm_prefix = _MM_TEXT_PREFIX
                key = key[len(mm_prefix) :]
            elif key.startswith(("vision_tower.", "mm_projector.")):
                # hf_key_map owns the vision naming in both directions.
                tt_key, _kind = official_to_titan(key, kda_layers=set())
                state_dict[tt_key] = value
                continue

            expert_m = re.match(
                r"model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\."
                r"(\d+)\.(\w+)\.weight",
                key,
            )
            if expert_m is not None:
                layer_num, expert_num, proj = expert_m.groups()
                w_tag = _HF_TO_W.get(proj, proj)  # w1/w2/w3 or gate_proj-style
                if w_tag not in ("w1", "w2", "w3"):
                    raise ValueError(f"Unknown expert projection in {key!r}")
                titan_abstract_key = (
                    "layers.{}.moe._moe.routed_experts.inner_experts."
                    + _EXPERT_W_SUFFIXED[w_tag]
                )
                new_key = mm_prefix + titan_abstract_key.format(layer_num)
                titan_abstract_key = mm_prefix + titan_abstract_key

                layer_bucket = expert_weights_by_layer.setdefault(layer_num, {})
                layer_bucket.setdefault(titan_abstract_key, {})[int(expert_num)] = value

                if titan_abstract_key in self.local_experts_indices:
                    # Online path: to_hf() ran first and recorded shards.
                    stacked = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer, titan_abstract_key, layer_num
                    )
                else:
                    stacked = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked
                continue

            # Table first, hf_key_map for the two keys it cannot decide (module docstring).
            #
            # TODO: the mm_prefix test below is a proxy for "does this model use the
            # latent MoE layout". It holds for every flavor here because the K3 layouts
            # arrived with the multimodal ones, and it breaks the day a text-only latent
            # flavor exists.
            tt_key = None
            if "g_proj" in key or (mm_prefix and "shared_experts" in key):
                try:
                    tt_key, _kind = official_to_titan(
                        official_key,
                        kda_layers=kda_zero_based,
                    )
                except UnmappedKey:
                    tt_key = None
            if tt_key is None:
                tt_key, value = self._hf_key_to_tt(key, value)
            if tt_key is None:
                try:
                    # official_key, not the prefix-stripped one: hf_key_map keys off the
                    # full released name and returns an unprefixed tt key, which
                    # mm_prefix re-attaches below.
                    tt_key, _kind = official_to_titan(
                        official_key,
                        kda_layers=kda_zero_based,
                    )
                except UnmappedKey:
                    tt_key = None
            tt_key = self._add_lora_base(tt_key) if tt_key else tt_key
            if tt_key is not None:
                state_dict[mm_prefix + tt_key] = value

        mm = (
            _MM_TEXT_PREFIX
            if any(k.startswith(_MM_TEXT_PREFIX) for k in state_dict)
            else ""
        )
        if (
            f"{mm}lm_head.weight" not in state_dict
            and f"{mm}embed_tokens.weight" in state_dict
        ):
            # Kimi scaling-law configs tie lm_head to the embedding and the
            # HF export omits the alias. For a genuinely untied model with a
            # missing head this is wrong -- warn loudly either way.
            logger.warning(
                "HF checkpoint has no lm_head.weight; aliasing "
                "embed_tokens.weight (Kimi tied-embedding convention)."
            )
            state_dict[f"{mm}lm_head.weight"] = state_dict[f"{mm}embed_tokens.weight"]

        return state_dict

    def _hf_key_to_tt(self, key: str, value: Any) -> tuple[str | None, Any]:
        """Single-tensor HF -> tt key mapping (experts handled separately).

        Returns (None, value) for HF keys with no tt destination (e.g.
        vision tower tensors in a multimodal export).
        """
        if key in _DIRECT_MAP_FROM_HF:
            return _DIRECT_MAP_FROM_HF[key], value

        m = re.match(r"model\.layers\.(\d+)\.(.+)", key)
        if m is None:
            return None, value
        idx_s, sub = m.groups()
        tt_prefix = f"layers.{idx_s}"

        if sub in _PASSTHROUGH_LAYER_TAGS:
            return f"{tt_prefix}.{sub}", value
        if sub.startswith("self_attn."):
            if (
                sub == "self_attn.A_log"
                and isinstance(value, torch.Tensor)
                and value.dim() == 4
            ):
                value = value.reshape(-1)
            # The file spells both attention types self_attn; we hold MLA under
            # attention and KDA under delta_attention, so the layer index picks
            # the attribute. No leaf name can do it: o_proj exists on both.
            # Through is_kda_layer, the SAME predicate the constructor used --
            # kda_layers_zero_based answers a different question (it renumbers
            # for CHECKPOINT key indices) and disagrees on layer 0.
            attr = (
                "delta_attention"
                if self.kimi_config.is_kda_layer(int(idx_s))
                else "attention"
            )
            leaf = sub[len("self_attn.") :]
            return f"{tt_prefix}.{attr}.{leaf}", value

        # Dense MLP (both HF prefixes)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if sub == f"mlp.{proj}.weight":
                return f"{tt_prefix}.feed_forward.{proj}.weight", value

        # Router / bias (both HF prefixes)
        router_m = re.match(
            r"(?:mlp|block_sparse_moe)\.gate\.(weight|e_score_correction_bias)",
            sub,
        )
        if router_m is not None:
            tail = router_m.group(1)
            if tail == "weight":
                return f"{tt_prefix}.moe._moe.router.gate.weight", value
            return f"{tt_prefix}.moe._moe.expert_bias_E", value

        # Shared experts (both HF prefixes, both naming styles)
        shared_m = re.match(
            r"(?:mlp|block_sparse_moe)\.shared_experts\.(\w+)\.(.+)", sub
        )
        if shared_m is not None:
            proj, suff = shared_m.groups()
            w_tag = _HF_TO_W.get(proj, proj)
            if w_tag not in ("w1", "w2", "w3"):
                raise ValueError(f"Unknown shared-expert projection in {key!r}")
            # Kimi Linear's layout. K3's latent MoE names the same tensor
            # moe.shared_experts.gate_proj, so from_hf asks hf_key_map FIRST for these
            # and uses this answer only when hf_key_map declines -- see the
            # shared-expert branch there. Returning it unconditionally wrote a key that
            # exists in another layout, which no fallback could detect.
            return f"{tt_prefix}.moe._moe.shared_experts.{w_tag}.{suff}", value

        # Unknown per-layer key: skip with a debug note rather than failing
        # (multimodal exports carry vision/projector keys the LM ignores).
        logger.debug("KimiLinearStateDictAdapter: skipping HF key %s", key)
        return None, value
