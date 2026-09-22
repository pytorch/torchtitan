# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Nemotron-3 Nano / Nemotron-H State Dict Adapter

import dataclasses
import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.tools.logging import logger

from .model import Nemotron3Model

__all__ = ["NemotronStateDictAdapter"]


# ---------------------------------------------------------------------------
# HF key grammar (verified against NVIDIA-Nemotron-3-Nano-4B-BF16, 263 tensors)
#
#   backbone.embeddings.weight
#   backbone.layers.{N}.norm.weight              per-layer input norm, EVERY layer
#   backbone.layers.{N}.mixer.A_log              mamba
#   backbone.layers.{N}.mixer.D                  mamba
#   backbone.layers.{N}.mixer.conv1d.weight      mamba [conv_dim, 1, conv_kernel]
#   backbone.layers.{N}.mixer.conv1d.bias        mamba
#   backbone.layers.{N}.mixer.dt_bias            mamba
#   backbone.layers.{N}.mixer.in_proj.weight     mamba, fused z | x,B,C | dt
#   backbone.layers.{N}.mixer.norm.weight        mamba gated RMSNorm
#   backbone.layers.{N}.mixer.out_proj.weight    mamba
#   backbone.layers.{N}.mixer.q_proj.weight      attention
#   backbone.layers.{N}.mixer.k_proj.weight      attention
#   backbone.layers.{N}.mixer.v_proj.weight      attention
#   backbone.layers.{N}.mixer.o_proj.weight      attention
#   backbone.layers.{N}.mixer.up_proj.weight     MLP (ungated, relu^2)
#   backbone.layers.{N}.mixer.down_proj.weight   MLP (ungated, relu^2)
#   backbone.norm_f.weight
#   lm_head.weight
#
# MoE additions (verified against NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, 6243
# tensors across 13 shards; 52 layers = 23 mamba / 23 moe / 6 attention):
#
#   backbone.layers.{N}.mixer.gate.weight                    [E, dim]
#   backbone.layers.{N}.mixer.gate.e_score_correction_bias   [E]
#   backbone.layers.{N}.mixer.experts.{i}.up_proj.weight     PER-EXPERT, E of them
#   backbone.layers.{N}.mixer.experts.{i}.down_proj.weight   PER-EXPERT, E of them
#   backbone.layers.{N}.mixer.shared_experts.up_proj.weight
#   backbone.layers.{N}.mixer.shared_experts.down_proj.weight
#
# Nemotron-H layers hold a SINGLE mixer each, so `backbone.layers.{N}.norm.weight`
# maps to a DIFFERENT torchtitan FQN depending on the layer's type. The type is
# read from `model_config.layers[N]`; nothing here is pattern-matched on N.
# ---------------------------------------------------------------------------

_HF_EMBEDDING = "backbone.embeddings.weight"
_HF_FINAL_NORM = "backbone.norm_f.weight"
_HF_LM_HEAD = "lm_head.weight"

_TT_EMBEDDING = "tok_embeddings.weight"
_TT_FINAL_NORM = "norm.weight"
_TT_LM_HEAD = "lm_head.weight"

_HF_LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.(.+)$")
_TT_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

# HF mixer suffix -> torchtitan per-layer suffix, for Mamba blocks.
_MAMBA_SUFFIX_MAP = {
    "mixer.in_proj.weight": "in_proj.weight",
    "mixer.conv1d.weight": "conv1d.weight",
    "mixer.conv1d.bias": "conv1d.bias",
    "mixer.dt_bias": "dt_bias",
    "mixer.A_log": "A_log",
    "mixer.D": "D",
    "mixer.norm.weight": "mamba_norm.weight",
    "mixer.out_proj.weight": "out_proj.weight",
}

# Per-layer input norm, by layer kind.
_NORM_SUFFIX_BY_KIND = {
    "mamba": "norm.weight",
    "attention": "attention_norm.weight",
    "mlp": "ffn_norm.weight",
    "moe": "ffn_norm.weight",
}

# HF attention projections that participate in the fused QKV buffer.
_HF_MOE_ROUTER = "mixer.gate.weight"
_HF_MOE_EXPERT_BIAS = "mixer.gate.e_score_correction_bias"

_TT_MOE_ROUTER = "moe.router.gate.weight"
# Registered with `register_buffer(..., persistent=True)`, so it is part of the
# model state dict and must round-trip. The trailing `_E` is the real attribute
# name -- verified via `named_buffers()` on a meta-device build of nemotron_31b.
_TT_MOE_EXPERT_BIAS = "moe.expert_bias_E"

# `tokens_per_expert_E` is a torchtitan-only load-balancing counter with no HF
# counterpart. It is registered non-persistent (so it does not normally reach a
# state dict at all), but `to_hf` skips it explicitly rather than reporting it as
# unmapped if a caller hands over `named_buffers()` instead of `state_dict()`.
_TT_MOE_LOCAL_ONLY = frozenset({"moe.tokens_per_expert_E"})

# HF shared-expert suffix -> torchtitan per-layer suffix. The shared expert is a
# plain ungated NemotronMLP, so the projection names carry over verbatim.
_MOE_SHARED_SUFFIX_MAP = {
    "mixer.shared_experts.up_proj.weight": "moe.shared_experts.up_proj.weight",
    "mixer.shared_experts.down_proj.weight": "moe.shared_experts.down_proj.weight",
}

# Doubly-indexed per-expert keys: backbone.layers.{N}.mixer.experts.{i}.{proj}.weight
_HF_EXPERT_RE = re.compile(r"^mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$")

# HF per-expert projection -> the torchtitan STACKED parameter it lands in.
#   up_proj   [F, D] -> w1_EFD [E, F, D]
#   down_proj [D, F] -> w2_EDF [E, D, F]
# Stacking is along dim 0 at the expert's own index; no transpose is involved.
_MOE_EXPERT_STACK_SUFFIX = {
    "up_proj": "moe.routed_experts.inner_experts.w1_EFD",
    "down_proj": "moe.routed_experts.inner_experts.w2_EDF",
}

_QKV_HF_SUFFIXES = {
    "mixer.q_proj.weight": "q",
    "mixer.k_proj.weight": "k",
    "mixer.v_proj.weight": "v",
}

# torchtitan stock (split) QKV FQN suffixes. FusedQKVLinear checkpoints in this
# layout via its state_dict hooks, so this is what a real state dict contains
# whether or not `fuse_qkv` is enabled.
_TT_QKV_SPLIT_SUFFIXES = {
    "q": "attention.qkv_linear.wq.weight",
    "k": "attention.qkv_linear.wk.weight",
    "v": "attention.qkv_linear.wv.weight",
}
_TT_QKV_FUSED_SUFFIX = "attention.qkv_linear.wqkv.weight"
_TT_WO_SUFFIX = "attention.wo.weight"

# Keys that are safe to drop with an explicit warning rather than an error:
# derived RoPE caches and framework bookkeeping, never learned weights.
_IGNORABLE_HF_PATTERNS = (
    re.compile(r"rotary_emb\.inv_freq$"),
    re.compile(r"\.rotary_emb\."),
    re.compile(r"_extra_state$"),
    re.compile(r"\.masked_bias$"),
    re.compile(r"\.(attn|attention)\.bias$"),
)


class NemotronStateDictAdapter(MoEStateDictAdapter):
    """Convert Nemotron-H (Nemotron-3 Nano) checkpoints between HF and torchtitan.

    Design rules, in response to how the previous adapter failed:

    * **Nothing is dropped silently.** Every input key is either mapped, or
      matched against an explicit ignore list (and warned about), or reported in
      a ``ValueError``. A mismatch is loud.
    * **Layer type comes from the config, not from a key pattern.** Nemotron-H
      interleaves Mamba, attention and MLP layers in an arbitrary order
      (``hybrid_override_pattern``), and all three spell their input norm
      ``backbone.layers.{N}.norm.weight``. The torchtitan name for that tensor is
      resolved via ``model_config.layers[N]``.
    * **No llama-style Q/K permutation.** Nemotron-H uses transformers' default
      (non-interleaved) RoPE; the ``_permute``/``_reverse_permute`` dance is a
      llama-conversion artifact and applying it here would corrupt weights.
    * **No ``_validate_hf_rope_config``.** It dereferences ``layer.attention.rope``
      for every layer, and Nemotron-H layer 0 is a Mamba block with
      ``attention=None``. Attention config is looked up via
      :meth:`_attention_config`, which finds a layer that actually has one.

    QKV layout
    ----------
    ``FusedQKVLinear`` registers ``_split_qkv_on_save`` / ``_merge_qkv_on_load``
    hooks, so its *state dict* is always the stock ``wq``/``wk``/``wv`` layout
    even though its *parameter* is a single fused ``wqkv``. That fused parameter
    is interleaved as ``(n_kv_heads, heads_per_kv + 2, head_dim, dim)``, **not**
    ``cat([q, k, v], dim=0)``.

    So ``from_hf`` emits ``wq``/``wk``/``wv`` by default -- the form a real state
    dict actually uses, and the form ``_merge_qkv_on_load`` expects. Q, K and V
    are still buffered until all three of a layer have been seen, and
    ``emit_fused_qkv=True`` produces a single ``wqkv`` tensor in the correct
    interleaved layout. ``to_hf`` accepts either form.

    MoE layout
    ----------
    HF stores routed experts as ``num_experts`` separate tensors per layer
    (``mixer.experts.{i}.up_proj.weight``), while torchtitan uses one stacked
    tensor per projection (``routed_experts.inner_experts.w1_EFD``, shape
    ``[E, F, D]``). ``from_hf`` buffers the per-expert tensors keyed by the
    expert index *parsed from the key* and stacks them only once all ``E`` have
    arrived, so the result does not depend on shard or dict ordering; an
    incomplete set at the end of a call is a ``ValueError`` naming the layer and
    the missing indices. ``to_hf`` unbinds dim 0 back into per-expert keys.

    The experts are UNGATED (``down(relu(up(x))**2)``) -- there is no ``w3_EFD``
    and no HF ``gate_proj``. The router's ``e_score_correction_bias`` maps to the
    persistent ``moe.expert_bias_E`` buffer; ``moe.tokens_per_expert_E`` is a
    torchtitan-only counter (non-persistent) with no HF counterpart.
    """

    def __init__(
        self,
        model_config: Nemotron3Model.Config,
        hf_assets_path: str | None,
        *,
        emit_fused_qkv: bool = False,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.emit_fused_qkv = emit_fused_qkv

        # Partial Q/K/V sets, keyed by layer id. Persisted on the instance so a
        # sharded / multi-call load can complete a layer across calls.
        self._qkv_buffer: dict[int, dict[str, Any]] = {}

        # Partial per-expert sets: layer_id -> proj ("up_proj"/"down_proj") ->
        # {expert_index: tensor}. Experts arrive as `num_experts` separate HF
        # tensors, spread arbitrarily across shards, and must be stacked into a
        # single tensor indexed by expert id along dim 0. Keying by the index
        # parsed out of the key -- never by arrival order -- is what makes the
        # result independent of shard/dict iteration order.
        self._expert_buffer: dict[int, dict[str, dict[int, Any]]] = {}

    # -- layer introspection -------------------------------------------------

    @property
    def _layer_configs(self) -> list[Any]:
        layers = getattr(self.model_config, "layers", None)
        if not layers:
            raise ValueError(
                "model_config.layers is empty; the Nemotron adapter needs the "
                "per-layer configs to resolve layer types."
            )
        return layers

    def _layer_config(self, layer_id: int) -> Any:
        layers = self._layer_configs
        if layer_id >= len(layers):
            raise ValueError(
                f"Checkpoint references layer {layer_id} but model_config only "
                f"defines {len(layers)} layers."
            )
        return layers[layer_id]

    def _layer_kind(self, layer_id: int) -> str:
        """One of ``mamba`` | ``attention`` | ``mlp`` | ``moe``."""
        cfg = self._layer_config(layer_id)

        # Prefer an explicit block_type, but never trust a default "mamba" on a
        # config whose is_mamba_block says otherwise.
        block_type = getattr(cfg, "block_type", None)
        if block_type in _NORM_SUFFIX_BY_KIND:
            if block_type != "mamba" or getattr(cfg, "is_mamba_block", True):
                return block_type

        if getattr(cfg, "is_mamba_block", False):
            return "mamba"
        if getattr(cfg, "attention", None) is not None:
            return "attention"
        if getattr(cfg, "moe", None) is not None:
            return "moe"
        if getattr(cfg, "feed_forward", None) is not None:
            return "mlp"
        raise ValueError(
            f"Cannot determine the layer type of layer {layer_id}: its config has "
            "no block_type, is_mamba_block, attention, moe or feed_forward set."
        )

    def _attention_config(self, layer_id: int | None = None) -> Any:
        """Attention config for ``layer_id``, else the first layer that has one.

        Never assumes layer 0 is an attention layer -- in Nemotron-H it is a
        Mamba block, which is what made the old ``_validate_hf_rope_config`` call
        crash.
        """
        if layer_id is not None:
            attn = getattr(self._layer_config(layer_id), "attention", None)
            if attn is not None:
                return attn
        for cfg in self._layer_configs:
            attn = getattr(cfg, "attention", None)
            if attn is not None:
                return attn
        raise ValueError(
            "Checkpoint contains attention weights but no layer in model_config "
            "defines an attention config."
        )

    def _attention_dims(self, layer_id: int | None = None) -> tuple[int, int, int]:
        """``(n_heads, n_kv_heads, head_dim)`` for the QKV fuse/split."""
        attn = self._attention_config(layer_id)
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        head_dim = attn.head_dim
        if head_dim is None:
            head_dim = self.model_config.dim // n_heads
        return n_heads, n_kv_heads, head_dim

    def _mlp_param_names(self, layer_id: int) -> tuple[str, str]:
        """``(up_name, down_name)`` inside ``layers.{N}.feed_forward``.

        Nemotron-H's MLP is UNGATED (``down_proj(relu(up_proj(x)) ** 2)``), so
        only two projections exist. The names are read off the feed-forward
        config's dataclass fields so this keeps working whether the module spells
        them ``w1``/``w2`` (torchtitan's shared ``FeedForward``) or ``up_proj``/
        ``down_proj`` (a Nemotron-specific module).
        """
        ff = getattr(self._layer_config(layer_id), "feed_forward", None)
        names: set[str] = set()
        if ff is not None and dataclasses.is_dataclass(ff):
            names = {f.name for f in dataclasses.fields(ff)}

        if {"up_proj", "down_proj"} <= names:
            return "up_proj", "down_proj"
        # torchtitan's FeedForward is w2(silu(w1(x)) * w3(x)); an ungated relu^2
        # MLP drops w3, leaving w1 = up projection, w2 = down projection.
        return "w1", "w2"

    def _num_experts(self, layer_id: int) -> int:
        """Routed-expert count for a MoE layer, read from the config.

        The stacked ``w1_EFD``/``w2_EDF`` tensors are sized by this, and it is
        also the completeness criterion for the expert buffer: a layer is only
        emitted once every index in ``range(num_experts)`` has arrived.
        """
        moe = getattr(self._layer_config(layer_id), "moe", None)
        for holder, attr in (
            (moe, "num_experts"),
            (getattr(moe, "routed_experts", None), "num_experts"),
            (
                getattr(getattr(moe, "routed_experts", None), "inner_experts", None),
                "num_experts",
            ),
            (self.model_config, "num_experts"),
        ):
            count = getattr(holder, attr, None)
            if isinstance(count, int) and count > 0:
                return count
        raise ValueError(
            f"Layer {layer_id} is a MoE layer but its config does not declare a "
            "positive num_experts; cannot size the stacked expert tensors."
        )

    # -- expert stack / unstack ----------------------------------------------

    def _drain_experts(self, state_dict: dict[str, Any]) -> None:
        """Emit stacked expert tensors for every layer/projection now complete.

        ``torch.stack`` is fed the per-expert tensors in *expert-index* order,
        so index ``i`` of dim 0 is expert ``i`` regardless of the order the
        shards produced them in.
        """
        for layer_id in sorted(self._expert_buffer):
            by_proj = self._expert_buffer[layer_id]
            num_experts = self._num_experts(layer_id)
            for proj in sorted(by_proj):
                shards = by_proj[proj]
                expected = self._expected_expert_indices(proj, num_experts)
                if set(shards) != expected:
                    continue
                suffix = _MOE_EXPERT_STACK_SUFFIX[proj]
                state_dict[f"layers.{layer_id}.{suffix}"] = self._stack_experts(
                    proj, [shards[i] for i in sorted(expected)]
                )
                by_proj[proj] = {}
            self._expert_buffer[layer_id] = {
                proj: shards for proj, shards in by_proj.items() if shards
            }
            if not self._expert_buffer[layer_id]:
                del self._expert_buffer[layer_id]

    def _expert_key(self, proj: str) -> str:
        """Layer-independent metadata key, matching :meth:`_unstack_experts`.

        Every MoE layer has the same expert count and the same EP sharding, so
        one key per projection is enough.
        """
        return f"layers.{{}}.moe.routed_experts.{proj}"

    def _expected_expert_indices(self, proj: str, num_experts: int) -> set[int]:
        """Expert indices this rank must see before it can restack.

        Under expert parallelism a rank only ever receives its own slice --
        ``to_hf`` emitted just the local experts -- so demanding the full set
        would reject a load that is actually complete. Offline conversion has
        no recorded indices and still requires every expert.
        """
        local = self.local_experts_indices.get(self._expert_key(proj))
        if local is None:
            return set(range(num_experts))
        return set(range(local[0], local[1]))

    def _stack_experts(self, proj: str, ordered: list[Any]) -> Any:
        """Stack per-expert tensors back into the grouped dim-0 weight.

        On the EP path the result must be rebuilt as a DTensor from the local
        shard, using the mesh/placements recorded by :meth:`_unstack_experts`.
        """
        stacked = torch.stack(ordered, dim=0)
        key = self._expert_key(proj)
        if key not in self.grouped_expert_weight_mesh:
            return stacked
        local = stacked._local_tensor if isinstance(stacked, DTensor) else stacked
        return DTensor.from_local(
            local,
            self.grouped_expert_weight_mesh[key],
            self.grouped_expert_weight_placements[key],
            run_check=False,
        )

    def _assert_experts_drained(self) -> None:
        """Raise if any layer is still holding an incomplete expert set.

        Unlike the Q/K/V buffer -- three tensors that a caller may reasonably
        stream in over several calls -- a missing expert means the stacked
        tensor would either be short or silently contain a garbage slice, so
        this is an error, naming the layer and the exact missing indices.
        """
        if not self._expert_buffer:
            return
        details: list[str] = []
        for layer_id in sorted(self._expert_buffer):
            num_experts = self._num_experts(layer_id)
            for proj in sorted(self._expert_buffer[layer_id]):
                present = set(self._expert_buffer[layer_id][proj])
                expected = self._expected_expert_indices(proj, num_experts)
                missing = sorted(expected - present)
                extra = sorted(present - expected)
                details.append(
                    f"layer {layer_id} {proj}: {len(present)}/{len(expected)} experts "
                    f"present, missing indices {missing}"
                    + (f", out-of-range indices {extra}" if extra else "")
                )
        self._expert_buffer.clear()
        raise ValueError(
            "Nemotron from_hf: incomplete routed-expert set(s). Refusing to emit a "
            "stacked expert tensor with missing slices.\n  " + "\n  ".join(details)
        )

    def _unstack_experts(
        self, stacked: Any, layer_id: int, proj: str, prefix: str
    ) -> dict[str, Any]:
        """Inverse of :meth:`_drain_experts`: dim-0 slice ``i`` -> HF expert ``i``."""
        num_experts = self._num_experts(layer_id)
        if stacked.shape[0] != num_experts:
            raise ValueError(
                f"Nemotron to_hf: layer {layer_id} expert tensor for {proj} has "
                f"{stacked.shape[0]} experts on dim 0 but the config declares "
                f"{num_experts}. Refusing to write a mismatched checkpoint."
            )
        if isinstance(stacked, DTensor):
            # Under EP the stacked weight is sharded on dim 0 -- the expert
            # dim -- so torch.unbind(dim=0) raises "Attempted to unbind along
            # the sharded dimension". Emit only this rank's local experts, on a
            # sub-mesh with the expert-dim sharding removed, exactly as the
            # other MoE adapters do. The metadata recorded here is what
            # from_hf() uses to restack.
            titan_abstract_key = f"layers.{{}}.moe.routed_experts.{proj}"
            self.grouped_expert_weight_placements[
                titan_abstract_key
            ] = stacked.placements
            self.grouped_expert_weight_shape[titan_abstract_key] = stacked.shape
            self.grouped_expert_weight_mesh[titan_abstract_key] = stacked.device_mesh
            # `prefix` already has the layer baked in, but the helper formats
            # with (layer_id, expert_id). Use the explicit positional index
            # {1} so the EXPERT id lands here and the layer arg is ignored.
            return self._get_local_experts_weights(
                f"{prefix}.mixer.experts.{{1}}.{proj}.weight",
                titan_abstract_key,
                str(layer_id),
                stacked,
            )

        return {
            f"{prefix}.mixer.experts.{i}.{proj}.weight": tensor
            for i, tensor in enumerate(torch.unbind(stacked, dim=0))
        }

    # -- QKV fuse / split ----------------------------------------------------

    def _fuse_qkv(self, q: Any, k: Any, v: Any, layer_id: int) -> Any:
        """Build the interleaved ``(n_kv, R, head_dim, dim)`` fused wqkv."""
        _, n_kv_heads, head_dim = self._attention_dims(layer_id)
        heads_per_kv = q.shape[0] // head_dim // n_kv_heads
        tail = tuple(q.shape[1:])
        return torch.cat(
            [
                q.reshape(n_kv_heads, heads_per_kv, head_dim, *tail),
                k.reshape(n_kv_heads, 1, head_dim, *tail),
                v.reshape(n_kv_heads, 1, head_dim, *tail),
            ],
            dim=1,
        ).reshape(-1, *tail)

    def _split_qkv(self, wqkv: Any, layer_id: int) -> tuple[Any, Any, Any]:
        """Inverse of :meth:`_fuse_qkv`."""
        n_heads, n_kv_heads, head_dim = self._attention_dims(layer_id)
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        tail = tuple(wqkv.shape[1:])
        w = wqkv.reshape(n_kv_heads, r_dim, head_dim, *tail)
        return (
            w[:, :heads_per_kv].reshape(-1, *tail),
            w[:, heads_per_kv].reshape(-1, *tail),
            w[:, heads_per_kv + 1].reshape(-1, *tail),
        )

    def _drain_qkv(self, state_dict: dict[str, Any]) -> None:
        """Emit torchtitan QKV entries for every layer whose q, k and v arrived."""
        for layer_id in sorted(self._qkv_buffer):
            parts = self._qkv_buffer[layer_id]
            if not {"q", "k", "v"} <= parts.keys():
                continue
            q, k, v = parts["q"], parts["k"], parts["v"]
            if self.emit_fused_qkv:
                state_dict[f"layers.{layer_id}.{_TT_QKV_FUSED_SUFFIX}"] = self._fuse_qkv(
                    q, k, v, layer_id
                )
            else:
                for part, tensor in (("q", q), ("k", k), ("v", v)):
                    suffix = _TT_QKV_SPLIT_SUFFIXES[part]
                    state_dict[f"layers.{layer_id}.{suffix}"] = tensor
            del self._qkv_buffer[layer_id]

    # -- HF -> torchtitan ----------------------------------------------------

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a HuggingFace Nemotron-H state dict to torchtitan format.

        Raises ``ValueError`` listing every key it could not map. Nothing is
        dropped without either a mapping or an explicit warning.
        """
        state_dict: dict[str, Any] = {}
        unmapped: list[str] = []
        ignored: list[str] = []

        for key, value in hf_state_dict.items():
            if key == _HF_EMBEDDING:
                state_dict[_TT_EMBEDDING] = value
                continue
            if key == _HF_FINAL_NORM:
                state_dict[_TT_FINAL_NORM] = value
                continue
            if key == _HF_LM_HEAD:
                state_dict[_TT_LM_HEAD] = value
                continue

            match = _HF_LAYER_RE.match(key)
            if match is None:
                if any(p.search(key) for p in _IGNORABLE_HF_PATTERNS):
                    ignored.append(key)
                else:
                    unmapped.append(key)
                continue

            layer_id = int(match.group(1))
            suffix = match.group(2)
            kind = self._layer_kind(layer_id)

            # Per-layer input norm: same HF name for every layer type, different
            # torchtitan name depending on what the layer actually is.
            if suffix == "norm.weight":
                state_dict[f"layers.{layer_id}.{_NORM_SUFFIX_BY_KIND[kind]}"] = value
                continue

            if any(p.search(key) for p in _IGNORABLE_HF_PATTERNS):
                ignored.append(key)
                continue

            if kind == "mamba":
                tt_suffix = _MAMBA_SUFFIX_MAP.get(suffix)
                if tt_suffix is None:
                    unmapped.append(key)
                    continue
                state_dict[f"layers.{layer_id}.{tt_suffix}"] = value
                continue

            if kind == "attention":
                part = _QKV_HF_SUFFIXES.get(suffix)
                if part is not None:
                    self._qkv_buffer.setdefault(layer_id, {})[part] = value
                    continue
                if suffix == "mixer.o_proj.weight":
                    state_dict[f"layers.{layer_id}.{_TT_WO_SUFFIX}"] = value
                    continue
                unmapped.append(key)
                continue

            if kind == "mlp":
                up_name, down_name = self._mlp_param_names(layer_id)
                if suffix == "mixer.up_proj.weight":
                    state_dict[f"layers.{layer_id}.feed_forward.{up_name}.weight"] = value
                    continue
                if suffix == "mixer.down_proj.weight":
                    state_dict[
                        f"layers.{layer_id}.feed_forward.{down_name}.weight"
                    ] = value
                    continue
                unmapped.append(key)
                continue

            if kind == "moe":
                expert_match = _HF_EXPERT_RE.match(suffix)
                if expert_match is not None:
                    expert_id = int(expert_match.group(1))
                    proj = expert_match.group(2)
                    self._expert_buffer.setdefault(layer_id, {}).setdefault(proj, {})[
                        expert_id
                    ] = value
                    continue
                if suffix == _HF_MOE_ROUTER:
                    state_dict[f"layers.{layer_id}.{_TT_MOE_ROUTER}"] = value
                    continue
                if suffix == _HF_MOE_EXPERT_BIAS:
                    state_dict[f"layers.{layer_id}.{_TT_MOE_EXPERT_BIAS}"] = value
                    continue
                shared_suffix = _MOE_SHARED_SUFFIX_MAP.get(suffix)
                if shared_suffix is not None:
                    state_dict[f"layers.{layer_id}.{shared_suffix}"] = value
                    continue
                unmapped.append(key)
                continue

            unmapped.append(key)

        self._drain_qkv(state_dict)
        self._drain_experts(state_dict)

        if ignored:
            logger.warning(
                "Nemotron from_hf: ignoring %d non-weight key(s) (derived caches / "
                "framework bookkeeping): %s",
                len(ignored),
                ", ".join(sorted(ignored)[:10]),
            )

        if unmapped:
            raise ValueError(
                f"Nemotron from_hf: {len(unmapped)} HuggingFace key(s) have no "
                "torchtitan mapping. Refusing to load a partial checkpoint.\n  "
                + "\n  ".join(sorted(unmapped))
            )

        if self._qkv_buffer:
            logger.warning(
                "Nemotron from_hf: incomplete Q/K/V set(s) still buffered for "
                "layer(s) %s; they will be emitted once the remaining projections "
                "arrive.",
                sorted(self._qkv_buffer),
            )

        # Every expert must be accounted for: a partially-filled stack is never
        # emitted, and never silently tolerated.
        self._assert_experts_drained()

        return state_dict

    # -- torchtitan -> HF ----------------------------------------------------

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a torchtitan Nemotron state dict back to HuggingFace format.

        ``to_hf(from_hf(x))`` reproduces the input keys and shapes.
        """
        hf_state_dict: dict[str, Any] = {}
        unmapped: list[str] = []

        # torchtitan per-layer suffix -> HF mixer suffix, for Mamba blocks.
        mamba_to_hf = {v: k for k, v in _MAMBA_SUFFIX_MAP.items()}
        split_to_part = {v: k for k, v in _TT_QKV_SPLIT_SUFFIXES.items()}
        shared_to_hf = {v: k for k, v in _MOE_SHARED_SUFFIX_MAP.items()}
        norm_suffixes = set(_NORM_SUFFIX_BY_KIND.values())

        for key, value in state_dict.items():
            if key == _TT_EMBEDDING:
                hf_state_dict[_HF_EMBEDDING] = value
                continue
            if key == _TT_FINAL_NORM:
                hf_state_dict[_HF_FINAL_NORM] = value
                continue
            if key == _TT_LM_HEAD:
                hf_state_dict[_HF_LM_HEAD] = value
                continue

            match = _TT_LAYER_RE.match(key)
            if match is None:
                unmapped.append(key)
                continue

            layer_id = int(match.group(1))
            suffix = match.group(2)
            kind = self._layer_kind(layer_id)
            prefix = f"backbone.layers.{layer_id}"

            # Every layer's input norm collapses back to the same HF name. Note
            # `norm.weight` is the Mamba input norm while `mamba_norm.weight` is
            # the gated SSM norm -- they must not be confused.
            if suffix in norm_suffixes and suffix == _NORM_SUFFIX_BY_KIND[kind]:
                hf_state_dict[f"{prefix}.norm.weight"] = value
                continue

            if kind == "mamba":
                hf_suffix = mamba_to_hf.get(suffix)
                if hf_suffix is None:
                    unmapped.append(key)
                    continue
                hf_state_dict[f"{prefix}.{hf_suffix}"] = value
                continue

            if kind == "attention":
                part = split_to_part.get(suffix)
                if part is not None:
                    hf_state_dict[f"{prefix}.mixer.{part}_proj.weight"] = value
                    continue
                if suffix == _TT_QKV_FUSED_SUFFIX:
                    q, k, v = self._split_qkv(value, layer_id)
                    hf_state_dict[f"{prefix}.mixer.q_proj.weight"] = q
                    hf_state_dict[f"{prefix}.mixer.k_proj.weight"] = k
                    hf_state_dict[f"{prefix}.mixer.v_proj.weight"] = v
                    continue
                if suffix == _TT_WO_SUFFIX:
                    hf_state_dict[f"{prefix}.mixer.o_proj.weight"] = value
                    continue
                unmapped.append(key)
                continue

            if kind == "mlp":
                up_name, down_name = self._mlp_param_names(layer_id)
                if suffix == f"feed_forward.{up_name}.weight":
                    hf_state_dict[f"{prefix}.mixer.up_proj.weight"] = value
                    continue
                if suffix == f"feed_forward.{down_name}.weight":
                    hf_state_dict[f"{prefix}.mixer.down_proj.weight"] = value
                    continue
                unmapped.append(key)
                continue

            if kind == "moe":
                # torchtitan-only load-balancing counter: no HF counterpart, so
                # it is dropped deliberately (and explicitly) rather than being
                # reported as unmapped. It is a non-persistent buffer, so a real
                # state dict will not contain it in the first place.
                if suffix in _TT_MOE_LOCAL_ONLY:
                    continue
                if suffix == _TT_MOE_ROUTER:
                    hf_state_dict[f"{prefix}.{_HF_MOE_ROUTER}"] = value
                    continue
                if suffix == _TT_MOE_EXPERT_BIAS:
                    hf_state_dict[f"{prefix}.{_HF_MOE_EXPERT_BIAS}"] = value
                    continue
                if suffix == _MOE_EXPERT_STACK_SUFFIX["up_proj"]:
                    hf_state_dict.update(
                        self._unstack_experts(value, layer_id, "up_proj", prefix)
                    )
                    continue
                if suffix == _MOE_EXPERT_STACK_SUFFIX["down_proj"]:
                    hf_state_dict.update(
                        self._unstack_experts(value, layer_id, "down_proj", prefix)
                    )
                    continue
                hf_shared = shared_to_hf.get(suffix)
                if hf_shared is not None:
                    hf_state_dict[f"{prefix}.{hf_shared}"] = value
                    continue
                unmapped.append(key)
                continue

            unmapped.append(key)

        if unmapped:
            raise ValueError(
                f"Nemotron to_hf: {len(unmapped)} torchtitan key(s) have no "
                "HuggingFace mapping. Refusing to write a partial checkpoint.\n  "
                + "\n  ".join(sorted(unmapped))
            )

        return hf_state_dict
