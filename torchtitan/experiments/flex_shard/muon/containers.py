# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Optimizer containers (placement -> optimizer routing) for FlexShard Muon.

Layer-2 of the distributed-Muon stack: each container reads a model's FlexShard
placements and routes every parameter to one of the layer-3 optimizer
implementations (``owned``'s dense Muon via ``torch.optim.Muon``, ``grouped_muon.GroupedMuon``,
``gather_muon.GatherMuon`` / ``gather_muon.GatherGroupedMuon``, ``dtensor_muon.DTensorMuon``) or AdamW,
then presents them as one ``OptimizersContainer``. Sits downstream of all the
implementation modules, so the routing helper and every container live here together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig

from ..example.ragged_shard import RaggedShard
from ..example.shard import Shard
from .bench import _BenchMixin
from .bucketing import _DEFAULT_EXPERT_MARKER
from .dtensor_muon import DTensorMuon
from .gather_muon import GatherGroupedMuon, GatherMuon
from .grouped_muon import GroupedMuon
from .owned import is_owned_2d
from .qkclip import QKClip


_GATHER_PLACEMENTS = (Shard, RaggedShard)


def _is_dense_gather_bucket(infos: list) -> bool:
    """Whether a bucket holds sharded 2D dense-layer matrices (for :class:`GatherMuon`).

    True iff every param is a 2D matrix under ``layers.<i>.`` with a gatherable placement
    (``Shard`` / ``RaggedShard`` / ``GroupedRaggedShard``). The ``layers.`` check excludes
    embeddings / output head (2D but not Muon by convention); the 2D check excludes norms
    (1D) and MoE expert stacks (>=3D).
    """
    if not infos:
        return False
    if not all(len(info.global_shape) == 2 for info in infos):
        return False
    if not all(info.fqn.startswith("layers.") for info in infos):
        return False
    return isinstance(infos[0].placement, _GATHER_PLACEMENTS)


def _is_expert_bucket(infos: list) -> bool:
    """Whether a bucket holds 3D ``[E, m, n]`` MoE expert stacks.

    Two semantic signals, both required: a 3D ``global_shape`` *and* an FQN under the
    expert marker (``_DEFAULT_EXPERT_MARKER`` = ``.moe.experts.``, the same marker the
    partition layer uses to bucket experts). Routing then reads the placement: ``Shard(0)``
    (whole experts per rank) -> :class:`GroupedMuon`; ``Shard(dim>=1)`` ->
    :class:`GatherGroupedMuon`. Gating on the marker (not raw 3D ``ndim``) means a future
    3D non-expert param falls to AdamW rather than silently routing to expert Muon.
    """
    if not infos:
        return False
    if not all(len(info.global_shape) == 3 for info in infos):
        return False
    return all(_DEFAULT_EXPERT_MARKER in info.fqn for info in infos)


def _discover_buckets(model: nn.Module, predicate) -> list[dict[str, Any]]:
    """This rank's FlexShard buckets whose ``param_infos`` match ``predicate(infos)``.

    Bucket-to-parameter mapping is by **canonical** FQN (wrapper segments such as
    ``_checkpoint_wrapped_module`` stripped via :func:`canonical_fqn`), so it survives
    reshard-after-forward composed with activation checkpointing (which inserts
    ``CheckpointWrapper`` segments into ``model.named_parameters()`` names). The optimizers
    hold the persistent sharded params, so RAF -- which only frees/recomputes the unsharded
    forward views -- does not affect these references.
    """
    param_by_fqn = {
        canonical_fqn(name): param for name, param in model.named_parameters()
    }
    buckets: list[dict[str, Any]] = []
    for storage in getattr(model, "sharded_bucket_storages", []):
        infos = list(storage.param_infos.values())
        if not predicate(infos):
            continue
        params = [param_by_fqn.get(canonical_fqn(info.fqn)) for info in infos]
        if any(p is None for p in params):
            raise ValueError(
                "Could not map a FlexShard bucket param FQN to a model parameter "
                f"(canonical FQNs: {[canonical_fqn(i.fqn) for i in infos]})."
            )
        buckets.append(
            {
                "placement": infos[0].placement,
                "infos": infos,
                "params": params,
                "storage": storage,
            }
        )
    return buckets


def _discover_dense_gather_buckets(model: nn.Module) -> list[dict[str, Any]]:
    """This rank's sharded 2D dense-matrix buckets (for :class:`GatherMuon`)."""
    return _discover_buckets(model, _is_dense_gather_bucket)


def _discover_expert_buckets(model: nn.Module) -> list[dict[str, Any]]:
    """This rank's 3D MoE expert buckets (for GroupedMuon / GatherGroupedMuon)."""
    return _discover_buckets(model, _is_expert_bucket)


def _build_grouped_expert_optimizers(
    container,
    model: nn.Module,
    part_idx: int,
    muon_kwargs: dict[str, Any],
    claimed_ids: set[int],
    all_params: list[nn.Parameter],
) -> set[int]:
    """Route this model part's 3D MoE expert buckets to Muon, by FlexShard placement.

    Experts are identified by FlexShard bucket metadata -- the 3D ``[E, m, n]`` buckets the
    parallelizer placed (see :func:`_discover_expert_buckets`), not raw ``ndim`` over every
    parameter -- and routed by placement:
    * ``Shard(dim>=1)`` (sharded within the matrix, ``world_size > num_experts``) ->
      :class:`GatherGroupedMuon` (all-gather over the efsdp mesh + batched per-expert NS);
    * ``Shard(0)`` (whole experts per rank, ``world_size <= num_experts``) -> comm-efficient
      :class:`GroupedMuon` (local batched per-expert NS).

    ``claimed_ids`` are params already assigned (e.g. dense). Extends ``all_params`` and
    returns the set of expert param ids handled (so the caller routes the rest to AdamW).
    """
    handled: set[int] = set()
    expert_buckets = _discover_expert_buckets(model)

    # Within-matrix-sharded experts (Shard(dim>=1)) -> gather-before-NS.
    gather_buckets = [
        b
        for b in expert_buckets
        if isinstance(b["placement"], Shard) and b["placement"].dim >= 1
    ]
    if gather_buckets:
        mesh = gather_buckets[0]["storage"]._mesh
        gather_grouped = GatherGroupedMuon(gather_buckets, mesh, **muon_kwargs)
        container.optimizers.append(gather_grouped)
        container._log_optimizer(
            gather_grouped, part_idx, ["<3D experts Shard(>=1) (gathered)>"]
        )
        all_params.extend(gather_grouped.param_groups[0]["params"])
        handled |= {id(p) for b in gather_buckets for p in b["params"] if p.numel() > 0}

    # Whole-per-rank experts (Shard(0)) -> comm-efficient local batched NS.
    grouped_params: list[nn.Parameter] = []
    grouped_names: list[str] = []
    for bucket in expert_buckets:
        pl = bucket["placement"]
        if isinstance(pl, Shard) and pl.dim >= 1:
            continue  # handled by GatherGroupedMuon above
        for info, param in zip(bucket["infos"], bucket["params"], strict=True):
            if param.numel() == 0 or id(param) in claimed_ids:
                continue
            grouped_params.append(param)
            grouped_names.append(canonical_fqn(info.fqn))
    if grouped_params:
        grouped = GroupedMuon(
            [{"params": grouped_params, "param_names": grouped_names, **muon_kwargs}]
        )
        container.optimizers.append(grouped)
        container._log_optimizer(grouped, part_idx, ["<3D MoE expert stacks (whole)>"])
        all_params.extend(grouped_params)
        handled |= {id(p) for p in grouped_params}
    return handled


class FlexShardMuonOptimizers(_BenchMixin, OptimizersContainer):
    """Full comm-efficient Muon for FlexShard: 2D dense -> Muon, 3D experts -> GroupedMuon.

    Routes by FlexShard placement (not regex), reusing core's container for
    ``step``/``zero_grad``/``state_dict``/checkpoint integration:

    * owned 2D matrices -> ``torch.optim.Muon`` (comm-efficient on the owner; empty
      ``Owned`` shards on non-owner ranks, ``numel == 0``, are skipped so Muon never
      sees a zero-size matrix);
    * 3D MoE expert stacks -> :class:`GroupedMuon` (whole experts, ``Shard(0)``) or
      :class:`GatherGroupedMuon` (experts sharded within the matrix, ``Shard(dim>=1)``),
      via :func:`_build_grouped_expert_optimizers` -- experts always get Muon;
    * everything else (norms, embeddings, LM head) -> AdamW.

    ``Config.param_groups`` supplies only the optimizer kwargs per name (one entry with
    ``optimizer_name="Muon"`` -- reused for the experts -- and one ``"AdamW"``); their
    regex ``pattern`` is ignored, since routing is by FlexShard placement, not FQN.

    ``Config.qk_clip_tau`` (optional) enables MuonClip's QK-clip (:class:`QKClip`):
    after each step, per-head q/k projection rows are rescaled so the max attention
    logit stays ``<= tau``. ``None`` (default) -> off.

    On a dense model (no 3D experts) this reduces to 2D dense -> Muon, rest -> AdamW.
    Subclasses can swap the dense Muon class by overriding :meth:`_resolve_optimizer_cls`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        qk_clip_tau: float | None = None

    def __init__(self, config: OptimizersContainer.Config, *, model_parts) -> None:
        impl_kwargs = self._build_impl_kwargs(config)
        kwargs_by_name = {
            pg.optimizer_name: pg.optimizer_kwargs for pg in config.param_groups
        }
        if "Muon" not in kwargs_by_name or "AdamW" not in kwargs_by_name:
            raise ValueError(
                "FlexShardMuonOptimizers requires param_groups with optimizer_name "
                "'Muon' and 'AdamW'."
            )
        self.model_parts = model_parts
        self.optimizers = []
        all_params: list[nn.Parameter] = []
        muon_cls = self._resolve_optimizer_cls("Muon")

        for part_idx, model in enumerate(model_parts):
            # Dense 2D Owned matrices -> comm-efficient Muon. numel == 0 -> an Owned
            # shard belonging to another rank; nothing local.
            muon_params: list[nn.Parameter] = []
            muon_names: list[str] = []
            for name, param in model.named_parameters():
                if not param.requires_grad or param.numel() == 0:
                    continue
                if is_owned_2d(param):
                    muon_params.append(param)
                    muon_names.append(canonical_fqn(name))
            claimed: set[int] = {id(p) for p in muon_params}
            if muon_params:
                # Muon does not accept fused/foreach impl_kwargs; pass only its kwargs.
                muon = muon_cls(
                    [
                        {
                            "params": muon_params,
                            "param_names": muon_names,
                            **kwargs_by_name["Muon"],
                        }
                    ]
                )
                self.optimizers.append(muon)
                self._log_optimizer(muon, part_idx, ["<owned 2D matrices>"])
                all_params.extend(muon_params)

            # 3D MoE expert stacks -> GroupedMuon (whole) / GatherGroupedMuon (Shard(>=1)).
            claimed |= _build_grouped_expert_optimizers(
                self, model, part_idx, kwargs_by_name["Muon"], claimed, all_params
            )

            # Everything else (norms, embeddings, LM head) -> AdamW.
            adamw_params: list[nn.Parameter] = []
            adamw_names: list[str] = []
            for name, param in model.named_parameters():
                if not param.requires_grad or param.numel() == 0:
                    continue
                if id(param) in claimed:
                    continue
                adamw_params.append(param)
                adamw_names.append(canonical_fqn(name))
            if adamw_params:
                adamw = torch.optim.AdamW(
                    [
                        {
                            "params": adamw_params,
                            "param_names": adamw_names,
                            **impl_kwargs,
                            **kwargs_by_name["AdamW"],
                        }
                    ]
                )
                self.optimizers.append(adamw)
                self._log_optimizer(adamw, part_idx, ["<rest>"])
                all_params.extend(adamw_params)

        self._validate_params(all_params)
        # Honor implementation=fused_opt_states_bf16 (bf16 Adam states) for the AdamW
        # group, like the base container does -- this custom __init__ bypasses it.
        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params)
        tau = getattr(config, "qk_clip_tau", None)
        self._qkclip = QKClip(self.model_parts, tau=tau) if tau is not None else None

    def _post_step(self) -> None:
        if self._qkclip is not None:
            self._qkclip.step()

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "Muon":
            return torch.optim.Muon
        return OptimizersContainer._resolve_optimizer_cls(name)

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Every *non-empty* trainable param must be assigned to one optimizer.

        Overrides the base check to exclude empty ``Owned`` shards (``numel == 0``)
        that this rank does not own -- they are intentionally skipped, not optimized.
        """
        expected = {
            id(p)
            for model in self.model_parts
            for p in model.parameters()
            if p.requires_grad and p.numel() > 0
        }
        actual = {id(p) for p in all_params}
        assert expected == actual, (
            f"Parameter mismatch: {len(expected)} non-empty trainable params in "
            f"model, {len(actual)} in optimizers"
        )


class BenchInstrumentedOptimizers(_BenchMixin, OptimizersContainer):
    """Plain core ``OptimizersContainer`` plus the benchmark instrumentation.

    No Muon / placement-aware routing -- it is the stock container (regex param
    groups, standard optimizers) with :class:`_BenchMixin` so the vanilla
    **FSDP2 + AdamW** reference reports the same ``[muon-bench]`` line (total_iter /
    opt_step / fwd_bwd / comm) as the FlexShard paths. Used to measure the cost of
    *adopting* Owned/Muon over the production FSDP2 + AdamW setup.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass


def build_comm_efficient_muon_optimizer(
    model: nn.Module,
    mesh: DeviceMesh | None = None,
    *,
    muon_kwargs: dict[str, Any] | None = None,
    adamw_kwargs: dict[str, Any] | None = None,
    implementation: str = "for-loop",
) -> FlexShardMuonOptimizers:
    """Convenience builder: a :class:`FlexShardMuonOptimizers` for one model.

    For standalone use (tests, notebooks) without the trainer's config machinery.
    Routes owned 2D matrices to Muon and the rest to AdamW. ``mesh`` is unused
    (ownership is read from FlexShard param metadata) and kept for call-site
    compatibility.
    """
    del mesh  # ownership is derived from param placement, not the mesh
    config = FlexShardMuonOptimizers.Config(
        param_groups=[
            ParamGroupConfig(
                pattern="<owned 2D matrices>",
                optimizer_name="Muon",
                optimizer_kwargs=dict(muon_kwargs or {}),
            ),
            ParamGroupConfig(
                pattern="<rest>",
                optimizer_name="AdamW",
                optimizer_kwargs=dict(adamw_kwargs or {}),
            ),
        ],
        implementation=implementation,
    )
    return FlexShardMuonOptimizers(config, model_parts=[model])


class FlexShardGatherMuonOptimizers(_BenchMixin, OptimizersContainer):
    """Full Muon with the *gather* dense distribution (FSDP/RaggedShard backend) + AdamW.

    The gather counterpart to :class:`FlexShardMuonOptimizers` (Owned): sharded 2D dense
    matrices go to :class:`GatherMuon` (all-gather + NS), 3D MoE experts go to
    :class:`GroupedMuon` (whole, ``Shard(0)``) / :class:`GatherGroupedMuon` (sharded,
    ``Shard(>=1)``), and everything else (norms, embeddings, output) to AdamW. So it
    differs from Owned full Muon *only* in how the dense 2D matrices are distributed
    (gather vs Owned). On a dense model (no experts) it is just gather-Muon + AdamW.
    ``Config.param_groups`` supplies the per-optimizer kwargs (one ``optimizer_name="Muon"``,
    one ``"AdamW"``); routing is by FlexShard placement/shape, not regex.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        qk_clip_tau: float | None = None

    def __init__(self, config: OptimizersContainer.Config, *, model_parts) -> None:
        impl_kwargs = self._build_impl_kwargs(config)
        kwargs_by_name = {
            pg.optimizer_name: pg.optimizer_kwargs for pg in config.param_groups
        }
        if "Muon" not in kwargs_by_name or "AdamW" not in kwargs_by_name:
            raise ValueError(
                "FlexShardGatherMuonOptimizers requires param_groups with "
                "optimizer_name 'Muon' and 'AdamW'."
            )
        self.model_parts = model_parts
        self.optimizers = []
        all_params: list[nn.Parameter] = []
        all_buckets: list[dict[str, Any]] = []

        for part_idx, model in enumerate(model_parts):
            buckets = _discover_dense_gather_buckets(model)
            all_buckets.extend(buckets)
            muon_param_ids = {
                id(p) for b in buckets for p in b["params"] if p.numel() > 0
            }
            if buckets:
                mesh = buckets[0]["storage"]._mesh
                muon = GatherMuon(buckets, mesh, **kwargs_by_name["Muon"])
                self.optimizers.append(muon)
                self._log_optimizer(muon, part_idx, ["<sharded 2D matrices>"])
                all_params.extend(muon.param_groups[0]["params"])

            # 3D MoE experts -> GroupedMuon (whole) / GatherGroupedMuon (Shard(>=1)).
            claimed = muon_param_ids | _build_grouped_expert_optimizers(
                self,
                model,
                part_idx,
                kwargs_by_name["Muon"],
                muon_param_ids,
                all_params,
            )

            adamw_params: list[nn.Parameter] = []
            adamw_names: list[str] = []
            for name, param in model.named_parameters():
                if not param.requires_grad or param.numel() == 0:
                    continue
                if id(param) in claimed:
                    continue
                adamw_params.append(param)
                adamw_names.append(canonical_fqn(name))
            if adamw_params:
                adamw = torch.optim.AdamW(
                    [
                        {
                            "params": adamw_params,
                            "param_names": adamw_names,
                            **impl_kwargs,
                            **kwargs_by_name["AdamW"],
                        }
                    ]
                )
                self.optimizers.append(adamw)
                self._log_optimizer(adamw, part_idx, ["<rest>"])
                all_params.extend(adamw_params)

        self._validate_params(all_params)
        # Honor implementation=fused_opt_states_bf16 (bf16 Adam states) for the AdamW
        # group, like the base container does -- this custom __init__ bypasses it.
        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params)
        self._setup_qkclip(config, all_buckets)

    def _setup_qkclip(
        self, config: OptimizersContainer.Config, buckets: list[dict[str, Any]]
    ) -> None:
        """Build :class:`QKClip` with a shard_map from the gather buckets (MuonClip).

        Maps each gathered 2D param to its ``(placement, info, mesh)`` so QK-clip can
        scale per-head rows of the *sharded* q/k projections (``Shard(0)`` /
        ``GroupedRaggedShard``); ``None`` tau -> off.
        """
        tau = getattr(config, "qk_clip_tau", None)
        if tau is None:
            self._qkclip = None
            return
        shard_map: dict[int, tuple] = {}
        for bucket in buckets:
            placement = bucket["placement"]
            mesh = bucket["storage"]._mesh
            for info, param in zip(bucket["infos"], bucket["params"], strict=True):
                shard_map[id(param)] = (placement, info, mesh)
        self._qkclip = QKClip(self.model_parts, tau=tau, shard_map=shard_map)

    def _post_step(self) -> None:
        if self._qkclip is not None:
            self._qkclip.step()

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Every *non-empty* trainable param must be assigned to one optimizer.

        Skips empty shards (``numel == 0``), which a rank may hold but not optimize.
        """
        expected = {
            id(p)
            for model in self.model_parts
            for p in model.parameters()
            if p.requires_grad and p.numel() > 0
        }
        actual = {id(p) for p in all_params}
        assert expected == actual, (
            f"Parameter mismatch: {len(expected)} non-empty trainable params in "
            f"model, {len(actual)} in optimizers"
        )


class FSDP2MuonOptimizers(_BenchMixin, OptimizersContainer):
    """FSDP2 (core ``fully_shard``) full-Muon baseline: DTensorMuon on dense 2D + experts.

    The "DTensor does the all-gather in opt.step" counterpart to comm-efficient ``Owned``
    (:class:`FlexShardMuonOptimizers`). The model is sharded by core ``fully_shard`` (params
    are ``Shard`` DTensors); 2D transformer-body matrices (under ``layers.``) and 3D MoE
    expert stacks both go to :class:`DTensorMuon` (which all-gathers and runs single /
    batched-per-expert NS); embeddings, LM head and 1D norms -> AdamW. So it matches the
    Owned / gather full-Muon recipe, differing only in that DTensor does the gather in the
    step. ``Config.param_groups`` supplies the per-optimizer kwargs; routing is by FQN + ndim.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: OptimizersContainer.Config, *, model_parts) -> None:
        impl_kwargs = self._build_impl_kwargs(config)
        kwargs_by_name = {
            pg.optimizer_name: pg.optimizer_kwargs for pg in config.param_groups
        }
        if "Muon" not in kwargs_by_name or "AdamW" not in kwargs_by_name:
            raise ValueError(
                "FSDP2MuonOptimizers requires param_groups with optimizer_name "
                "'Muon' and 'AdamW'."
            )
        self.model_parts = model_parts
        self.optimizers = []
        all_params: list[nn.Parameter] = []

        for part_idx, model in enumerate(model_parts):
            muon_params: list[nn.Parameter] = []
            adamw_params: list[nn.Parameter] = []
            adamw_names: list[str] = []
            # TODO(checkpoint): we collect canonical FQNs for the AdamW group only. The
            # Muon group below is built flat (no "param_names"), so DCP save/load fails --
            # checkpoint_utils.py:140 (_optim_state_dict_to_fqn_keys) raises on any group
            # without "param_names". To make the fsdp2 Muon baseline checkpointable, also
            # accumulate a muon_names list here and pass it through to DTensorMuon.
            for name, param in model.named_parameters():
                if not param.requires_grad or param.numel() == 0:
                    continue
                cfqn = canonical_fqn(name)
                # 2D transformer-body matrices and 3D MoE expert stacks (under the expert
                # marker) -> DTensorMuon (it all-gathers via full_tensor() and runs single /
                # batched-per-expert NS); embeddings/LM head (outside ``layers.``) and 1D
                # norms -> AdamW. Gating experts on the marker (not raw 3D ndim) keeps a
                # future 3D non-expert param off expert Muon.
                is_body_2d = (
                    param.ndim == 2
                    and cfqn.startswith("layers.")
                    and _DEFAULT_EXPERT_MARKER not in cfqn
                )
                is_expert = param.ndim == 3 and _DEFAULT_EXPERT_MARKER in cfqn
                if is_body_2d or is_expert:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)
                    adamw_names.append(cfqn)
            if muon_params:
                muon = DTensorMuon(muon_params, **kwargs_by_name["Muon"])
                self.optimizers.append(muon)
                self._log_optimizer(
                    muon, part_idx, ["<dense 2D body + 3D experts (DTensor)>"]
                )
                all_params.extend(muon_params)
            if adamw_params:
                adamw = torch.optim.AdamW(
                    [
                        {
                            "params": adamw_params,
                            "param_names": adamw_names,
                            **impl_kwargs,
                            **kwargs_by_name["AdamW"],
                        }
                    ]
                )
                self.optimizers.append(adamw)
                self._log_optimizer(adamw, part_idx, ["<rest>"])
                all_params.extend(adamw_params)

        self._validate_params(all_params)
        # Honor implementation=fused_opt_states_bf16 (bf16 Adam states) for the AdamW
        # group, like the base container does -- this custom __init__ bypasses it.
        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params)

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Every non-empty trainable param must be assigned to exactly one optimizer."""
        expected = {
            id(p)
            for model in self.model_parts
            for p in model.parameters()
            if p.requires_grad and p.numel() > 0
        }
        actual = {id(p) for p in all_params}
        assert expected == actual, (
            f"Parameter mismatch: {len(expected)} non-empty trainable params in "
            f"model, {len(actual)} in optimizers"
        )
