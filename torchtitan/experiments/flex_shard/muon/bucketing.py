# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-agnostic FlexShard bucket builders and parallelizers for distributed Muon.

Shared by the per-model FlexShard test beds (``deepseek_v3``, ``qwen3``, ...). All
torchtitan decoder models name the same top-level modules in
``models/common/decoder.py`` (``tok_embeddings``, ``layers``, ``norm``,
``lm_head``), and partition a layer's params by fully-qualified name, so the
bucketing here is model-agnostic:

- Each transformer layer's dense params form one communication-efficient ``Owned``
  bucket on the dp mesh (Muon on the 2D matrices; the owner runs Newton-Schulz
  locally, owners balanced by Newton-Schulz FLOPs over the layer's 2D matrices).
- MoE expert stacks, identified by an FQN marker, form ``Shard`` buckets on the
  expert mesh (the optimizer routes them to GroupedMuon / GatherGroupedMuon). Dense
  models simply have none.
- Embeddings, final norm, and the LM head are ``Shard`` buckets (AdamW).

The gather-for-NS Muon baselines instead *shard* the dense 2D matrices and the
optimizer all-gathers them before Newton-Schulz; see ``build_gather_muon_buckets``.

Mental model -- bucket vs placement (the two distinct concerns this file wires up):

- A *bucket* is a **communication unit**: all of its params are unsharded (and their
  grads reduced) in one collective. ``BucketSpec.patterns`` decide which params group
  together, so bucketing partitions the *parameter set* -- it does NOT pick ranks.
- A bucket's ``placement_fn`` gives the bucket **one** cross-rank layout. (It returns a
  placement per param, but a bucket is a single collective, so those placements must be
  jointly collective-able -- effectively uniform within the bucket.) Two layouts:
    * ``Owned(r)``: the *whole* bucket lives on rank ``r`` (every other rank holds an
      empty ``(0, 0)`` shard). The owner runs Newton-Schulz locally -> the step issues
      no collective.
    * ``Shard(0)`` / ``GroupedRaggedShard``: the bucket is *split across all ranks*
      (each holds ~``1/world_size``); the optimizer all-gathers it before Newton-Schulz.
- **Load balancing for ``Owned`` happens across buckets, never within one.** A bucket
  has exactly one owner (one broadcast root), so the only way to use more ranks is to
  make more buckets. The builders therefore: (1) pick a *granularity* -- whole-layer /
  per-matrix / two-level -- which sets how finely params are cut into buckets; then
  (2) run greedy LPT over Newton-Schulz FLOPs to assign each bucket an owner, and bake
  that owner into the bucket's ``placement_fn`` via ``make_owned_placement_fn(owner)``.
  So the balancing decision is computed *before* the ``BucketSpec`` list and merely
  replayed by the placement fn. Finer granularity => more, smaller buckets => the LPT
  fills more ranks / balances tighter (and is what lets per-matrix run when
  ``num_layers < world_size``, where whole-layer would leave ranks idle).

Scope: 1D FSDP (+ optional expert parallelism for the expert stacks), eager only.
The whole-layer ``Owned`` allocation (the default) requires ``num_layers >= dp_shard``;
the per-matrix / two-level / auto allocations relax that to
``num_2D_dense_matrices >= dp_shard`` so the comm-efficient path still fills every rank
when ``num_layers < world_size``. No PP/TP/CP/HSDP/compile/offload/full_dtensor.
"""

from __future__ import annotations

import heapq
import os
from collections.abc import Callable

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example.ragged_shard import (
    make_grouped_ragged_placement_fn,
)
from torchtitan.experiments.flex_shard.example.shard import per_param_placements, Shard
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
)
from torchtitan.tools.logging import logger
from .newton_schulz import (
    assign_layer_owners_lpt,
    layer_newton_schulz_cost,
    newton_schulz_flops,
)
from .owned import make_owned_placement_fn


# torchtitan decoder models name these top-level modules (models/common/decoder.py).
_REST_PATTERNS = ("tok_embeddings.*", "norm.*", "lm_head.*")
# MoE expert stacks (3D: [num_experts, out, in]) live under this FQN marker. They are
# placed off the dense Owned bucket (on the EP/efsdp mesh) and run GroupedMuon /
# GatherGroupedMuon -- not the dense per-matrix Owned Muon.
_DEFAULT_EXPERT_MARKER = ".moe.experts."


def shard0_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Shard, ...]]:
    """Plain FSDP ``Shard(0)`` for every param (embeddings, norms, LM head)."""
    return {fqn: (Shard(0),) for fqn, _ in named_params}


def expert_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Shard, ...]]:
    """Shard a 3D expert stack ``[E, F, D]`` across the expert mesh.

    Splits the expert (leading) dim with ``Shard(0)`` when there are at least as many
    experts as ranks (``E >= mesh.size()``), so each rank holds whole ``(F, D)`` expert
    matrices -- the comm-efficient :class:`GroupedMuon` regime. When there are more ranks
    than experts (``mesh.size() > E``), it falls back to ``Shard(1)``, which splits
    *within* each matrix: every rank then holds only a ``(F/n, D)`` slice of every expert,
    not whole matrices -- the incomplete-tensor regime that :class:`GatherGroupedMuon`
    all-gathers before Newton-Schulz.
    """
    if not named_params:
        return {}
    num_local_experts = named_params[0][1].shape[0]
    dim = 1 if mesh.size() > num_local_experts else 0
    return {fqn: (Shard(dim),) for fqn, _ in named_params}


def build_muon_buckets(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    expert_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    reshard_last: bool,
    expert_marker: str = _DEFAULT_EXPERT_MARKER,
    allow_idle_ranks: bool = False,
) -> list[BucketSpec]:
    """Comm-efficient Muon buckets: dense layers ``Owned`` (Muon), experts ``Shard``.

    Per transformer layer, the dense params (everything not under ``expert_marker``)
    form one ``Owned`` bucket on ``dp_mesh`` so the owner runs Newton-Schulz locally;
    the MoE expert stack, if any, is a separate ``Shard`` bucket on ``expert_mesh``.
    Embeddings, final norm, and the LM head are ``Shard`` buckets (AdamW). Layer
    owners are balanced by Newton-Schulz FLOPs over each layer's dense 2D matrices.

    Dense models (no expert params) simply produce no expert buckets, and
    ``expert_mesh`` is unused. By default requires ``num_layers >= dp_mesh.size()`` so
    every dp rank owns a layer; ``allow_idle_ranks=True`` lifts that (LPT then leaves
    ``world_size - num_layers`` ranks idle in the step) -- used only to *measure* the
    whole-layer cost in the ``num_layers < world_size`` regime that per-matrix
    ownership (:func:`build_permatrix_muon_buckets`) is meant to fix.
    """
    layer_ids = sorted(model.layers.keys(), key=int)
    world_size = dp_mesh.size()
    if len(layer_ids) < world_size and not allow_idle_ranks:
        raise ValueError(
            f"FlexShard Muon requires num_layers >= dp_shard, but got "
            f"{len(layer_ids)} layers and dp_shard {world_size}. Use a smaller "
            "data_parallel_shard_degree, a deeper model, or per-matrix ownership."
        )

    # Partition each layer's params into dense (Owned -> Muon) vs expert (Shard ->
    # GroupedMuon) by fully-qualified name -- model-agnostic, no per-model glob patterns.
    dense_named: dict[str, list[tuple[str, nn.Parameter]]] = {}
    expert_fqns: dict[str, list[str]] = {}
    for lid in layer_ids:
        dense: list[tuple[str, nn.Parameter]] = []
        experts: list[str] = []
        for name, param in model.layers[lid].named_parameters():
            fqn = f"layers.{lid}.{name}"
            if expert_marker in fqn:
                experts.append(fqn)
            else:
                dense.append((fqn, param))
        dense_named[lid] = dense
        expert_fqns[lid] = experts

    costs = [layer_newton_schulz_cost(dense_named[lid]) for lid in layer_ids]
    owners = assign_layer_owners_lpt(costs, world_size)

    buckets: list[BucketSpec] = [
        BucketSpec(
            ["tok_embeddings.*"],
            placement_fn=shard0_placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    ]
    for lid, owner in zip(layer_ids, owners, strict=True):
        # Dense params of this layer -> one Owned bucket (Muon for the 2D matrices).
        buckets.append(
            BucketSpec(
                [fqn for fqn, _ in dense_named[lid]],
                placement_fn=make_owned_placement_fn(owner),
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        )
        # MoE expert stack (if this layer has one) -> Shard on the expert mesh.
        if expert_fqns[lid]:
            buckets.append(
                BucketSpec(
                    expert_fqns[lid],
                    placement_fn=expert_placement_fn,
                    mesh=expert_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            )
    for pattern in ("norm.*", "lm_head.*"):
        buckets.append(
            BucketSpec(
                [pattern],
                placement_fn=shard0_placement_fn,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_last,
            )
        )
    return buckets


def build_permatrix_muon_buckets(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    reshard_last: bool,
    expert_mesh: DeviceMesh | None = None,
    expert_reshard_after_forward: bool | None = None,
    expert_marker: str = _DEFAULT_EXPERT_MARKER,
) -> list[BucketSpec]:
    """Per-2D-tensor comm-efficient Muon: each dense matrix is its own ``Owned`` bucket.

    Generalizes :func:`build_muon_buckets` (whole-layer) by allocating *individual*
    2D dense tensors to owners via greedy LPT over Newton-Schulz FLOPs across **all**
    such matrices in the model. This balances the optimizer-step makespan at matrix
    granularity, so it (a) fills every rank when ``num_layers < world_size`` (which
    whole-layer cannot), and (b) balances better even when ``num_layers`` is not a
    multiple of ``world_size``. Still comm-efficient (each matrix lives wholly on one
    owner). Requires ``num_2D_dense_matrices >= world_size``.

    MoE expert stacks (identified by ``expert_marker``) are kept off the Owned
    allocation and placed as ``Shard`` buckets on ``expert_mesh`` (routed to GroupedMuon),
    exactly as :func:`build_muon_buckets` does -- so this composes with expert parallelism. The
    Owned dense buckets use ``reshard_after_forward`` (forced ``False`` by the caller --
    the fine-Owned RAF engine bug), while the experts use
    ``expert_reshard_after_forward`` (the configured policy), so the large expert state
    can still reshard and the resident footprint stays bounded. Each layer's non-2D
    dense params (norms) and embeddings / final norm / LM head are ``Shard(0)`` (AdamW).
    Dense models have no expert params and ``expert_mesh`` is unused.

    Cost vs whole-layer: more broadcasts (one per owned matrix vs one per layer);
    per-(layer, owner) grouping (:func:`build_twolevel_muon_buckets`) reduces that.
    """
    if expert_mesh is None:
        expert_mesh = dp_mesh
    if expert_reshard_after_forward is None:
        expert_reshard_after_forward = reshard_after_forward
    world_size = dp_mesh.size()
    layer_ids = sorted(model.layers.keys(), key=int)
    matrices: list[tuple[str, nn.Parameter]] = []
    other_fqns_by_layer: dict[str, list[str]] = {}
    expert_fqns_by_layer: dict[str, list[str]] = {}
    for lid in layer_ids:
        others: list[str] = []
        experts: list[str] = []
        for name, param in model.layers[lid].named_parameters():
            fqn = f"layers.{lid}.{name}"
            if expert_marker in fqn:
                experts.append(fqn)
            elif param.ndim == 2:
                matrices.append((fqn, param))
            else:
                others.append(fqn)
        other_fqns_by_layer[lid] = others
        expert_fqns_by_layer[lid] = experts

    if len(matrices) < world_size:
        raise ValueError(
            f"Per-matrix Muon requires num_2D_dense_matrices >= dp_shard, but got "
            f"{len(matrices)} matrices and dp_shard {world_size}."
        )

    costs = [newton_schulz_flops(param.shape) for _, param in matrices]
    owners = assign_layer_owners_lpt(costs, world_size)

    buckets: list[BucketSpec] = [
        BucketSpec(
            ["tok_embeddings.*"],
            placement_fn=shard0_placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    ]
    # One Owned bucket per 2D dense matrix (owner from LPT); each layer's non-2D dense
    # params in their own Shard(0) bucket; each MoE expert stack Shard on expert_mesh.
    for (fqn, _), owner in zip(matrices, owners, strict=True):
        buckets.append(
            BucketSpec(
                [fqn],
                placement_fn=make_owned_placement_fn(owner),
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        )
    for lid in layer_ids:
        # Each non-2D param in its own bucket (a bucket spanning different submodules'
        # params crosses forward-unshard units and corrupts the unsharded-param slots).
        for fqn in other_fqns_by_layer[lid]:
            buckets.append(
                BucketSpec(
                    [fqn],
                    placement_fn=shard0_placement_fn,
                    mesh=dp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        if expert_fqns_by_layer[lid]:
            buckets.append(
                BucketSpec(
                    expert_fqns_by_layer[lid],
                    placement_fn=expert_placement_fn,
                    mesh=expert_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=expert_reshard_after_forward,
                )
            )
    for pattern in ("norm.*", "lm_head.*"):
        buckets.append(
            BucketSpec(
                [pattern],
                placement_fn=shard0_placement_fn,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_last,
            )
        )
    return buckets


def _cost_weighted_group_sizes(
    layer_costs: list[int], layer_caps: list[int], world_size: int
) -> list[int]:
    """Apportion ``world_size`` ranks to layers (heterogeneous two-level level 1).

    Each layer with >=1 matrix is seeded with 1 rank, then every remaining rank goes
    greedily to the layer with the highest per-rank NS load (``cost / group_size``),
    **capped at the layer's 2D-matrix count** (a layer cannot use more comm-efficient ranks
    than it has whole tensors). Deterministic (ties by layer index) so every rank
    computes the identical sizing. Returns sizes summing to ``min(world_size,
    sum(caps))``; any shortfall (``world_size > num_matrices``) leaves ranks idle in the
    step -- the L3 (slice) regime. Homogeneous layers -> ~uniform sizes (== the old
    two-level); a heavy MoE layer with many matrices gets proportionally more ranks.
    """
    n = len(layer_costs)
    sizes = [1 if layer_caps[i] >= 1 else 0 for i in range(n)]
    placed = sum(sizes)
    heap = [
        (-(layer_costs[i] / sizes[i]), i)
        for i in range(n)
        if 0 < sizes[i] < layer_caps[i] and layer_costs[i] > 0
    ]
    heapq.heapify(heap)
    while placed < world_size and heap:
        _, i = heapq.heappop(heap)
        sizes[i] += 1
        placed += 1
        if sizes[i] < layer_caps[i]:
            heapq.heappush(heap, (-(layer_costs[i] / sizes[i]), i))
    return sizes


def build_twolevel_muon_buckets(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    reshard_last: bool,
    expert_mesh: DeviceMesh | None = None,
    expert_reshard_after_forward: bool | None = None,
    expert_marker: str = _DEFAULT_EXPERT_MARKER,
) -> list[BucketSpec]:
    """Two-level comm-efficient Muon allocation (for ``world_size >= num_layers``).

    Level 1: partition the ``world_size`` ranks into one contiguous group per layer,
    **sized by the layer's NS-cost** (capped at its 2D-matrix count) via
    :func:`_cost_weighted_group_sizes` -- so a heavy MoE layer gets more ranks than a
    light dense one, and each layer independently lands at whole-layer (group size 1) or
    per-tensor (group > 1). Level 2: within each layer, LPT the layer's 2D dense matrices
    over its group. Homogeneous layers give ~uniform groups (the plain two-level).

    This matches per-tensor LPT's makespan but **confines each layer's matrices to its
    rank group**, so each rank owns matrices from only one layer -> one Owned broadcast
    per rank (~``world_size`` total) instead of per-tensor's ~``num_matrices``. The
    grouped, heterogeneous-aware middle ground between whole-layer (case 1) and flat
    per-tensor (case 2). Falls back to whole-layer when ``world_size < num_layers``; when
    ``world_size > num_matrices`` some ranks stay idle (the L3 / slice regime, warned).

    MoE expert stacks (``expert_marker``) are kept off the Owned allocation and placed as
    ``Shard`` buckets on ``expert_mesh`` (routed to GroupedMuon) with
    ``expert_reshard_after_forward`` -- so this composes with expert parallelism, and the
    large expert state can reshard while
    the fine Owned dense buckets stay at ``reshard_after_forward`` (forced ``False``).
    Other non-2D dense params and embeddings / final norm / LM head are ``Shard(0)``.
    """
    if expert_mesh is None:
        expert_mesh = dp_mesh
    if expert_reshard_after_forward is None:
        expert_reshard_after_forward = reshard_after_forward
    world_size = dp_mesh.size()
    layer_ids = sorted(model.layers.keys(), key=int)
    num_layers = len(layer_ids)
    if world_size < num_layers:
        # Fewer ranks than layers: two-level degenerates to whole-layer (each rank
        # owns >=1 whole layer), which is exactly build_muon_buckets.
        return build_muon_buckets(
            model,
            dp_mesh=dp_mesh,
            expert_mesh=expert_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            reshard_last=reshard_last,
            expert_marker=expert_marker,
        )

    # Per-layer dense matrices / non-2D / experts / NS-cost / cap (= matrix count), then
    # size each layer's rank group by cost (capped at its matrix count) -- the
    # heterogeneous level-1 split (a heavy layer gets more ranks than a light one).
    layer_matrices: dict[str, list[tuple[str, nn.Parameter]]] = {}
    layer_others: dict[str, list[str]] = {}
    layer_experts: dict[str, list[str]] = {}
    costs: list[int] = []
    caps: list[int] = []
    for lid in layer_ids:
        matrices: list[tuple[str, nn.Parameter]] = []
        others: list[str] = []
        experts: list[str] = []
        for name, param in model.layers[lid].named_parameters():
            fqn = f"layers.{lid}.{name}"
            if expert_marker in fqn:
                experts.append(fqn)
            elif param.ndim == 2:
                matrices.append((fqn, param))
            else:
                others.append(fqn)
        layer_matrices[lid] = matrices
        layer_others[lid] = others
        layer_experts[lid] = experts
        costs.append(sum(newton_schulz_flops(p.shape) for _, p in matrices))
        caps.append(len(matrices))

    group_sizes = _cost_weighted_group_sizes(costs, caps, world_size)
    if sum(group_sizes) < world_size:
        logger.warning(
            "Two-level Muon: %d ranks > %d total 2D matrices; %d ranks will be idle "
            "in the optimizer step (the L3 / slice regime is not yet implemented).",
            world_size,
            sum(caps),
            world_size - sum(group_sizes),
        )

    buckets: list[BucketSpec] = [
        BucketSpec(
            ["tok_embeddings.*"],
            placement_fn=shard0_placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    ]
    next_rank = 0
    for li, lid in enumerate(layer_ids):
        group = list(range(next_rank, next_rank + group_sizes[li]))
        next_rank += group_sizes[li]

        # Level 2: LPT this layer's matrices over its rank group, by NS-FLOPs.
        loads = [(0, g) for g in group]
        heapq.heapify(loads)
        owner_fqns: dict[int, list[str]] = {g: [] for g in group}
        for fqn, param in sorted(
            layer_matrices[lid],
            key=lambda fp: newton_schulz_flops(fp[1].shape),
            reverse=True,
        ):
            load, g = heapq.heappop(loads)
            owner_fqns[g].append(fqn)
            heapq.heappush(loads, (load + newton_schulz_flops(param.shape), g))

        # One Owned bucket per rank-in-group that owns matrices of this layer.
        for g in group:
            if owner_fqns[g]:
                buckets.append(
                    BucketSpec(
                        owner_fqns[g],
                        placement_fn=make_owned_placement_fn(g),
                        mesh=dp_mesh,
                        mp_policy=mp_policy,
                        reshard_after_forward=reshard_after_forward,
                    )
                )
        for fqn in layer_others[lid]:
            buckets.append(
                BucketSpec(
                    [fqn],
                    placement_fn=shard0_placement_fn,
                    mesh=dp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        if layer_experts[lid]:
            buckets.append(
                BucketSpec(
                    layer_experts[lid],
                    placement_fn=expert_placement_fn,
                    mesh=expert_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=expert_reshard_after_forward,
                )
            )
    for pattern in ("norm.*", "lm_head.*"):
        buckets.append(
            BucketSpec(
                [pattern],
                placement_fn=shard0_placement_fn,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_last,
            )
        )
    return buckets


def build_gather_muon_buckets(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    dense_placement_fn: Callable,
    expert_mesh: DeviceMesh | None = None,
    expert_reshard_after_forward: bool | None = None,
    expert_marker: str = _DEFAULT_EXPERT_MARKER,
) -> list[BucketSpec]:
    """Buckets for gather-for-NS Muon: dense 2D matrices sharded, gathered at step.

    Per layer, the dense 2D matrices (everything 2D not under ``expert_marker``) share one
    bucket placed by ``dense_placement_fn`` (``Shard(0)`` or ``GroupedRaggedShard``) ->
    ``GatherMuon`` all-gathers + NS. MoE expert stacks (``expert_marker``) go to ``Shard``
    on ``expert_mesh`` (the EP mesh) -> GroupedMuon, exactly as the comm-efficient Owned path -- so the
    gather baselines compose with EP and the experts are handled *identically* across
    methods (only the dense-matrix distribution differs). Other non-2D dense params (1D
    norms) and embeddings / final norm / LM head are ``Shard(0)`` -> AdamW. Dense models
    have no expert params and ``expert_mesh`` is unused.
    """
    if expert_mesh is None:
        expert_mesh = dp_mesh
    if expert_reshard_after_forward is None:
        expert_reshard_after_forward = reshard_after_forward
    buckets: list[BucketSpec] = []
    for lid in sorted(model.layers.keys(), key=int):
        matrices: list[str] = []
        other: list[str] = []
        experts: list[str] = []
        for name, param in model.layers[lid].named_parameters():
            fqn = f"layers.{lid}.{name}"
            if expert_marker in fqn:
                experts.append(fqn)
            elif param.ndim == 2:
                matrices.append(fqn)
            else:
                other.append(fqn)
        if matrices:
            buckets.append(
                BucketSpec(
                    matrices,
                    placement_fn=dense_placement_fn,
                    mesh=dp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        for fqn in other:
            buckets.append(
                BucketSpec(
                    [fqn],
                    placement_fn=shard0_placement_fn,
                    mesh=dp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        if experts:
            buckets.append(
                BucketSpec(
                    experts,
                    placement_fn=expert_placement_fn,
                    mesh=expert_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=expert_reshard_after_forward,
                )
            )
    buckets += [
        BucketSpec(
            [pattern],
            placement_fn=shard0_placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        for pattern in _REST_PATTERNS
    ]
    return buckets


def validate_supported_parallelisms(
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    training: TrainingConfig,
    compile_config: CompileConfig,
) -> None:
    """Reject parallelism configs the FlexShard Muon test beds do not support."""
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError(
            "FlexShard Muon does not support the full_dtensor backend."
        )
    if parallel_dims.pp_enabled:
        raise NotImplementedError("FlexShard Muon does not support PP.")
    if parallel_dims.tp_enabled:
        raise NotImplementedError("FlexShard Muon does not support TP.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("FlexShard Muon does not support CP.")
    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError("FlexShard Muon does not support HSDP.")
    if training.enable_cpu_offload:
        raise NotImplementedError("FlexShard Muon does not support CPU offload.")
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError(
            "FlexShard Muon is eager-only; disable model compile."
        )


def _mp_policy_from(training: TrainingConfig) -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )


def resolve_auto_granularity(model: nn.Module, world_size: int) -> str:
    """Pick the shallowest comm-efficient allocation level for the recursive descent.

    The unified system (see docs/muon.md "unified allocation system"):
    ``layer`` when ``W <= num_layers`` (whole layers fill the ranks, fewest broadcasts);
    ``matrix`` when ``num_layers < W < 2*num_layers`` (two-level groups would be size 1
    and degrade, so flat per-tensor balances better); ``twolevel`` when ``2*num_layers <=
    W <= num_matrices`` (clean rank groups: per-tensor balance at ~W broadcasts). Beyond
    ``num_matrices`` needs L3 group-local slicing (not yet implemented). ``num_matrices``
    counts the actual 2D dense matrices across all layers (heterogeneous-safe).
    """
    layer_ids = sorted(model.layers.keys(), key=int)
    num_layers = len(layer_ids)
    # Count the actual 2D dense matrices across all layers (heterogeneous-safe -- e.g. a
    # dense first layer followed by MoE layers, as in DeepSeek V3), not a layer-0 multiplier.
    num_matrices = sum(
        1
        for lid in layer_ids
        for _, p in model.layers[lid].named_parameters()
        if p.ndim == 2
    )
    if world_size <= num_layers:
        return "layer"
    if world_size < 2 * num_layers:
        return "matrix"
    if world_size <= num_matrices:
        return "twolevel"
    raise NotImplementedError(
        f"world_size {world_size} > num_matrices ~{num_matrices}: needs L3 group-local "
        "slicing (not yet implemented). Use a gather-for-NS config for this regime."
    )


def _install_mem_debug_hooks(model: nn.Module) -> None:
    """Env-gated (``FLEX_SHARD_MEM_DEBUG=1``) per-layer CUDA memory trace.

    Logs ``torch.cuda.memory_allocated/reserved`` before each transformer layer's forward
    (rank 0 only). If the comm-efficient unshard frees per-layer, ``alloc`` stays ~flat across
    layers; if the broadcast/unshard accumulates (RAF not freeing, or prefetch holding all
    layers), ``alloc`` grows monotonically with layer index toward the full model size.
    """
    import torch

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return
    layer_ids = sorted(model.layers.keys(), key=int)

    def make_hook(tag: str):
        def hook(_mod, _args):
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(
                f"[mem-debug] {tag}: alloc={alloc:.2f} GB reserved={reserved:.2f} GB"
            )

        return hook

    for lid in layer_ids:
        model.layers[lid].register_forward_pre_hook(make_hook(f"layer {lid} pre-fwd"))


def make_muon_parallelize_fn(
    *,
    model_name: str,
    support_ep: bool,
    granularity: str = "layer",
    allow_idle_ranks: bool = False,
):
    """Return a parallelize_fn for the comm-efficient ``Owned`` Muon path.

    ``support_ep`` enables expert parallelism (the experts become DTensors that the
    engine unwraps to local EP shards, then FlexShard FSDP-shards on the ``efsdp``
    mesh); dense-only models pass ``support_ep=False`` and reject EP.

    ``granularity`` selects the comm-efficient allocation unit: ``"layer"`` (whole-layer
    Owned), ``"matrix"`` (per-2D-tensor Owned), ``"twolevel"`` (per-layer rank groups +
    within-layer LPT), or ``"auto"`` -- the unified selector that picks the shallowest
    level for the regime (see :func:`resolve_auto_granularity`). ``allow_idle_ranks``
    lifts the whole-layer ``num_layers >= world_size`` guard. All granularities compose
    with EP when ``support_ep`` is set: the dense 2D matrices run comm-efficient Owned on the
    dp mesh at the chosen granularity, and MoE expert stacks Shard on the EP mesh. This
    is what lets MoE models run comm-efficient Muon at ``world_size > num_layers`` (where
    whole-layer cannot fill the ranks but per-tensor / two-level can).
    """
    if granularity not in ("layer", "matrix", "twolevel", "auto"):
        raise ValueError(
            "granularity must be 'layer', 'matrix', 'twolevel', or 'auto', "
            f"got {granularity!r}"
        )

    def parallelize_fn(
        model: nn.Module,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: CompileConfig,
        ac_config: ActivationCheckpointingConfig,
        dump_folder: str,
    ):
        validate_supported_parallelisms(
            parallel_dims, parallelism, training, compile_config
        )
        if parallel_dims.ep_enabled:
            if not support_ep:
                raise NotImplementedError(
                    f"FlexShard Muon {model_name} ({granularity}) is dense-only; "
                    "set expert_parallel_degree=1."
                )
            # EP is a permanent partition (each rank owns its expert subset; tokens
            # are routed to it). Apply it first so the experts become DTensors that
            # flex_shard() unwraps to each rank's local EP shard. The dense 2D matrices
            # still run comm-efficient Owned on the full dp mesh at the selected granularity;
            # only the expert stacks live on the EP (efsdp) mesh.
            model.parallelize(parallel_dims)

        if ac_config is not None:
            ac_config.build(dump_folder=dump_folder).apply(model)

        dp_mesh = parallel_dims.get_mesh("fsdp")
        expert_mesh = dp_mesh
        if support_ep and parallel_dims.ep_enabled:
            expert_mesh = parallel_dims.get_optional_mesh("efsdp") or dp_mesh

        # "auto" resolves to the shallowest level that fills the ranks (the system).
        gran = (
            resolve_auto_granularity(model, dp_mesh.size())
            if granularity == "auto"
            else granularity
        )

        mp_policy = _mp_policy_from(training)
        reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
        )
        reshard_last = parallelism.fsdp_reshard_after_forward == "always"
        if gran in ("matrix", "twolevel"):
            # Fine Owned buckets + RAF=True hit the engine's saved-tensor re-view bug
            # (mis-materialize on backward -- same limitation as the gather RAF=True
            # path), so the *Owned* dense buckets run RAF=False. opt_step (what the
            # allocation controls) is RAF-independent, so the makespan comparison stays
            # valid. MoE expert stacks are plain Shard buckets on the EP mesh, so they
            # keep the configured RAF policy -- letting the large expert state reshard
            # so the resident footprint stays bounded at scale.
            builder = (
                build_permatrix_muon_buckets
                if gran == "matrix"
                else build_twolevel_muon_buckets
            )
            buckets = builder(
                model,
                dp_mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=False,
                reshard_last=False,
                expert_mesh=expert_mesh,
                expert_reshard_after_forward=reshard_after_forward,
            )
        else:
            buckets = build_muon_buckets(
                model,
                dp_mesh=dp_mesh,
                expert_mesh=expert_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                reshard_last=reshard_last,
                allow_idle_ranks=allow_idle_ranks,
            )
        flex_shard(model, buckets=buckets)
        if os.environ.get("FLEX_SHARD_MEM_DEBUG"):
            _install_mem_debug_hooks(model)
        label = gran if granularity != "auto" else f"auto->{gran}"
        logger.info(
            f"Applied FlexShard + comm-efficient Muon ({label}) to {model_name}"
        )
        return model

    return parallelize_fn


def make_gather_muon_parallelize_fn(
    dense_kind: str, *, model_name: str, reshard_after_forward: bool = False
):
    """Return a parallelize_fn for gather-for-NS Muon with the given dense placement.

    ``dense_kind`` selects how dense 2D matrices are sharded: ``"shard"`` (plain FSDP
    ``Shard(0)``) or ``"grouped"`` (byte-perfect ``GroupedRaggedShard``). EP-capable
    (experts -> EP mesh, like Owned); no TP/PP/HSDP. Unlike the comm-efficient ``Owned`` path
    there is no ``num_layers >= dp_shard`` constraint (sharding is within-tensor, so any
    dp_shard works).

    ``reshard_after_forward`` enables RAF for the sharded gather buckets. The engine
    supports RAF with ``Shard(0)`` (and the GatherMuon optimizer now maps bucket params by
    *canonical* FQN, so the discovery survives the ``CheckpointWrapper`` FQN renaming that
    RAF-with-AC introduces). When ``True`` the configured ``fsdp_reshard_after_forward``
    policy is used; default ``False`` preserves the original full-model-resident behavior.
    RAF reshards the sharded matrices after forward (recomputed in backward), so the gather
    baselines no longer hold the full model resident -- the prerequisite for a fair
    memory-vs-comm comparison vs comm-efficient ``Owned``.
    """
    if dense_kind not in ("shard", "grouped"):
        raise ValueError(f"dense_kind must be 'shard' or 'grouped', got {dense_kind!r}")

    def parallelize_fn(
        model: nn.Module,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: CompileConfig,
        ac_config: ActivationCheckpointingConfig,
        dump_folder: str,
    ):
        validate_supported_parallelisms(
            parallel_dims, parallelism, training, compile_config
        )
        if parallel_dims.ep_enabled:
            # EP: apply it first so the experts become DTensors the engine unwraps to each
            # rank's EP shard. The gather baselines then gather only the DENSE 2D matrices
            # on the dp mesh; experts go to Shard on the EP mesh -- handled *identically* to
            # the comm-efficient Owned path, so the only difference vs Owned is the dense-matrix
            # distribution (the variable under test). This is what makes a fair fixed-config
            # (dp/ep) comparison across all methods possible.
            model.parallelize(parallel_dims)
        if ac_config is not None:
            ac_config.build(dump_folder=dump_folder).apply(model)

        dp_mesh = parallel_dims.get_mesh("fsdp")
        expert_mesh = dp_mesh
        if parallel_dims.ep_enabled:
            expert_mesh = parallel_dims.get_optional_mesh("efsdp") or dp_mesh
        if dense_kind == "grouped":
            dense_placement_fn = make_grouped_ragged_placement_fn(
                dims=(0,), local_units=(1,) * dp_mesh.size()
            )
        else:
            dense_placement_fn = per_param_placements  # Shard(0)

        raf = (
            get_fsdp_reshard_after_forward_policy(
                parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
            )
            if reshard_after_forward
            else False
        )
        buckets = build_gather_muon_buckets(
            model,
            dp_mesh=dp_mesh,
            mp_policy=_mp_policy_from(training),
            reshard_after_forward=raf,
            dense_placement_fn=dense_placement_fn,
            expert_mesh=expert_mesh,
            expert_reshard_after_forward=get_fsdp_reshard_after_forward_policy(
                parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
            ),
        )
        flex_shard(model, buckets=buckets)
        logger.info(
            f"Applied FlexShard + gather-for-NS Muon ({dense_kind}, "
            f"reshard_after_forward={raf}) to {model_name}"
        )
        return model

    return parallelize_fn
