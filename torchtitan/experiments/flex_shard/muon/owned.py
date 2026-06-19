# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Owned placement helpers + bucket builder for communication-efficient Muon.

The ``Owned`` distribution for the dense 2D matrices: each whole matrix lives on one
owner rank (so its Newton-Schulz runs locally, no all-gather in the step), with owners
balanced across ranks by Newton-Schulz FLOPs. The dense Muon optimizer itself is
``torch.optim.Muon`` (external); the routing that sends owned 2D matrices to it lives
in :mod:`containers`.
"""

from __future__ import annotations

import re
from collections import defaultdict

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..example.owned import Owned
from ..example.shard import per_param_placements

from ..flex_shard.bucket_storage import BucketSpec, PlacementFn
from ..flex_shard.sharded_param import get_global_shape, get_placements
from .newton_schulz import assign_layer_owners_lpt, layer_newton_schulz_cost


_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def make_owned_placement_fn(owner_rank: int) -> PlacementFn:
    """Return a ``placement_fn`` assigning every param in the bucket to one owner.

    FlexShard requires all params in a bucket to share one placement, so a whole
    layer owned by one rank is a single uniform ``Owned`` bucket. (Contrast with
    ``param_boundary_placements``, which assigns each param an independent owner
    and therefore needs one-param buckets.)
    """

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Owned, ...]]:
        return {fqn: (Owned(owner_rank),) for fqn, _ in named_params}

    return placement_fn


def _group_params_by_layer(
    model: nn.Module,
) -> tuple[dict[int, list[tuple[str, nn.Parameter]]], list[str]]:
    """Split named params into per-layer groups and the rest (by top-level name).

    Returns ``(layer_params, rest_top_names)`` where ``layer_params[i]`` lists the
    ``(fqn, param)`` of layer ``i``, and ``rest_top_names`` is the sorted set of
    top-level module names for params not under ``layers.<i>.`` (e.g.
    ``tok_embeddings``, ``norm``, ``output``).
    """
    layer_params: dict[int, list[tuple[str, nn.Parameter]]] = defaultdict(list)
    rest_top_names: set[str] = set()
    for fqn, param in model.named_parameters():
        match = _LAYER_RE.match(fqn)
        if match:
            layer_params[int(match.group(1))].append((fqn, param))
        else:
            rest_top_names.add(fqn.split(".")[0])
    return layer_params, sorted(rest_top_names)


def comm_efficient_muon_buckets(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    reshard_after_forward: bool = True,
) -> list[BucketSpec]:
    """Build FlexShard buckets for communication-efficient Muon.

    Produces one ``Owned`` bucket per transformer layer (owner balanced by
    Newton-Schulz FLOPs via :func:`assign_layer_owners_lpt`), plus one
    ``Shard(0)`` bucket per non-layer top-level module (embeddings, final norm,
    output head) -- those go to AdamW.

    Must be called before :func:`flex_shard` (parameters are still full). Requires
    ``num_layers >= world_size`` so every rank owns at least one layer.
    """
    world_size = mesh.size()
    layer_params, rest_top_names = _group_params_by_layer(model)
    if not layer_params:
        raise ValueError(
            "comm_efficient_muon_buckets found no `layers.<i>.*` parameters; the model "
            "does not match the expected transformer layout."
        )

    layer_indices = sorted(layer_params)
    num_layers = len(layer_indices)
    if num_layers < world_size:
        raise ValueError(
            f"comm_efficient_muon_buckets requires num_layers >= world_size, but got "
            f"{num_layers} layers and world_size {world_size}. Below this "
            "threshold some ranks would own no layer and idle during the step; "
            "finer-grained (per-matrix) ownership is needed there (out of scope)."
        )

    layer_costs = [layer_newton_schulz_cost(layer_params[i]) for i in layer_indices]
    owners = assign_layer_owners_lpt(layer_costs, world_size)

    buckets = [
        BucketSpec(
            [f"layers.{idx}.*"],
            placement_fn=make_owned_placement_fn(owner),
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )
        for idx, owner in zip(layer_indices, owners, strict=True)
    ]
    # Non-layer params (embeddings / final norm / output head) stay Shard(0) and
    # are optimized by AdamW. One bucket per top-level module keeps each bucket's
    # forward hook on a real execution unit (a grouped root-level bucket is
    # rejected when reshard-after-forward composes with activation checkpointing).
    for top in rest_top_names:
        buckets.append(
            BucketSpec(
                [f"{top}.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
        )
    return buckets


def is_owned_2d(param: nn.Parameter) -> bool:
    """Whether this rank owns the full 2D matrix backing ``param`` (Muon-eligible).

    Under ``Owned`` a non-owner rank holds an empty ``(0, 0)`` shard, so
    "owned-by-this-rank" is exactly "``Owned`` placement and ``numel > 0``" -- no
    rank lookup needed. Sharded params (embeddings/output) and this rank's owned
    non-2D params (norms) return False and fall through to AdamW.
    """
    if param.numel() == 0:
        return False
    placements = get_placements(param)
    global_shape = get_global_shape(param)
    if placements is None or global_shape is None:
        return False
    if len(placements) != 1 or not isinstance(placements[0], Owned):
        return False
    return len(global_shape) == 2
