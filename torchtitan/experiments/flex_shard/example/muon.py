# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Communication-free Muon for FlexShard models.

Muon orthogonalizes each 2D weight's momentum with a Newton-Schulz iteration that
needs the *full* matrix: it forms the Gram matrix ``G @ G.T``, a reduction over all
columns. Under row-sharding that would require an all-gather inside
``optimizer.step()``. FlexShard's ``Owned`` placement instead keeps each chosen
matrix whole on a single rank. After the backward reduce-to-owner, that rank
already holds the full parameter, the full (batch-averaged) gradient, and the full
momentum buffer, so it runs Newton-Schulz locally with **no collective in the
optimizer step** -- and exactly, matching single-device Muon. The forward broadcast
and backward reduce that FlexShard performs anyway are the only communication, so
Muon adds none.

This module wires that up for the example Transformer:

* :func:`comm_free_muon_buckets` builds the FlexShard buckets: each transformer
  layer is one ``Owned`` bucket (balanced across ranks); embeddings, the LM head,
  and the final norm stay ``Shard`` (FSDP) for AdamW.
* :func:`build_muon_param_groups` partitions a FlexSharded model's local
  parameters into the Muon group (this rank's owned 2D matrices) and the rest.
* :func:`build_comm_free_muon_optimizers` constructs this rank's Muon + AdamW
  optimizers from those groups.

See ``communication_free_muon_plan.md`` for the design and trade-offs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from ..flex_shard.bucket_storage import BucketSpec
from ..flex_shard.sharded_param import get_global_shape, get_placements
from .owned import assign_layer_owners_lpt, make_owned_placement_fn, Owned
from .shard import per_param_placements

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = [
    "build_comm_free_muon_optimizers",
    "build_muon_param_groups",
    "CombinedOptimizer",
    "comm_free_muon_buckets",
]


# Non-Muon parameters that stay evenly Shard(0)-ed (FSDP) and are optimized by
# AdamW on the local shard. Embeddings and the LM head are 2D but excluded from
# Muon by convention; the final norm is 1D.
_DEFAULT_REST_PATTERNS = [
    "tok_embeddings.*",
    "pos_embeddings.*",
    "norm.*",
    "output.*",
]


def comm_free_muon_buckets(
    model: nn.Module,
    world_size: int,
    *,
    reshard_after_forward: bool = True,
    balance: str = "lpt",
    rest_patterns: list[str] | None = None,
) -> list[BucketSpec]:
    """Build FlexShard buckets for communication-free Muon.

    Each transformer layer (``model.layers[i]``) becomes one ``Owned`` bucket owned
    by a single rank, so the owner runs Muon on that layer's 2D matrices with no
    collective in the step. Embeddings, the LM head, and the final norm stay
    ``Shard(0)`` (FSDP) and go to AdamW on the local shard.

    Each rest pattern gets its own bucket (not one grouped bucket) so that, with
    ``reshard_after_forward=True``, every bucket maps to a single execution-unit
    module whose forward hook is replayable during activation-checkpoint recompute.
    A grouped embeddings/LM-head/norm bucket resolves to the root module and is
    rejected by the reshard-after-forward hook installer.

    Args:
        model: A FlexShard-compatible Transformer exposing ``model.layers`` as an
            indexable sequence of per-layer modules.
        world_size: Number of ranks in the 1D FlexShard mesh.
        reshard_after_forward: Free unsharded params after forward and recompute
            them in backward (defaults to ``True``). The ``Owned`` broadcast and the
            ``Shard`` all-gather are both tagged for recompute, so this composes
            with activation checkpointing.
        balance: ``"lpt"`` (default) balances whole layers across ranks with greedy
            Longest-Processing-Time; ``"roundrobin"`` assigns layer ``i`` to
            ``i % world_size`` (optimal only for homogeneous, evenly divisible
            stacks).
        rest_patterns: FQN globs for the non-Muon (Shard + AdamW) region; each
            becomes its own bucket. Defaults to embeddings, LM head, and final norm.

    Returns:
        A list of ``BucketSpec`` for ``flex_shard(model, mesh, buckets)``.
    """
    num_layers = len(model.layers)
    if balance == "lpt":
        layer_numels = [
            sum(p.numel() for _, p in model.layers[i].named_parameters())
            for i in range(num_layers)
        ]
        owners = assign_layer_owners_lpt(layer_numels, world_size)
    elif balance == "roundrobin":
        owners = [i % world_size for i in range(num_layers)]
    else:
        raise ValueError(f"balance must be 'lpt' or 'roundrobin', but got {balance!r}.")

    layer_buckets = [
        BucketSpec(
            [f"layers.{i}.*"],
            placement_fn=make_owned_placement_fn(owners[i]),
            reshard_after_forward=reshard_after_forward,
        )
        for i in range(num_layers)
    ]
    patterns = _DEFAULT_REST_PATTERNS if rest_patterns is None else rest_patterns
    rest_buckets = [
        BucketSpec(
            [pattern],
            placement_fn=per_param_placements,
            reshard_after_forward=reshard_after_forward,
        )
        for pattern in patterns
    ]
    return [*layer_buckets, *rest_buckets]


MuonParamPredicate = Callable[[str, "torch.Size | None"], bool]


def _default_muon_predicate(fqn: str, global_shape: torch.Size | None) -> bool:
    """Muon-eligible iff the parameter is a 2D matrix."""
    return global_shape is not None and len(global_shape) == 2


def build_muon_param_groups(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    muon_param_predicate: MuonParamPredicate | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition a FlexSharded model's local params into ``(muon, other)`` groups.

    On each rank the Muon group holds only the **2D matrices this rank owns** via
    ``Owned`` placement (full, non-empty tensors). Everything else with non-empty
    local storage -- ``Shard`` embeddings / LM head, owned norms and biases -- goes
    to the other (AdamW) group. Non-owned ``Owned`` params are empty ``(0, 0)`` on
    this rank and are skipped; their owner rank optimizes them.

    Eligibility keys off placement (``Owned``) *and* shape (2D), so parameters
    deliberately kept on ``Shard`` (embeddings, LM head) never enter the Muon group
    even though they are 2D -- no FQN exclusion list is needed.

    Args:
        model: A model already wrapped by ``flex_shard``.
        mesh: The 1D FlexShard mesh, used for this rank's index.
        muon_param_predicate: Optional override ``(fqn, global_shape) -> bool`` for
            Muon eligibility. Defaults to "is a 2D matrix".

    Returns:
        ``(muon_params, other_params)`` for this rank.
    """
    predicate = muon_param_predicate or _default_muon_predicate
    rank = mesh.get_local_rank()
    muon_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for fqn, param in model.named_parameters():
        if param.numel() == 0:
            # Non-owned Owned param on this rank: the owner rank optimizes it.
            continue
        placements = get_placements(param)
        global_shape = get_global_shape(param)
        owned_here = (
            placements is not None
            and len(placements) == 1
            and isinstance(placements[0], Owned)
            and placements[0].owner_rank == rank
        )
        if owned_here and predicate(fqn, global_shape):
            muon_params.append(param)
        else:
            other_params.append(param)
    return muon_params, other_params


class CombinedOptimizer:
    """Step several optimizers together (e.g. this rank's Muon + AdamW).

    Each rank owns a different subset of parameters, so this just forwards
    ``step`` / ``zero_grad`` / state-dict to the sub-optimizers it was given. An
    empty sub-optimizer list is allowed (a rank may own no Muon matrices).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = list(optimizers)

    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None if closure is None else closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for opt in self.optimizers for group in opt.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        sub_state_dicts = state_dict["optimizers"]
        if len(sub_state_dicts) != len(self.optimizers):
            raise ValueError(
                f"Expected {len(self.optimizers)} sub-optimizer state dicts, "
                f"but got {len(sub_state_dicts)}."
            )
        for optimizer, sub_state in zip(self.optimizers, sub_state_dicts, strict=True):
            optimizer.load_state_dict(sub_state)


def build_comm_free_muon_optimizers(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    muon_kwargs: dict[str, Any] | None = None,
    adamw_kwargs: dict[str, Any] | None = None,
    muon_param_predicate: MuonParamPredicate | None = None,
) -> CombinedOptimizer:
    """Build this rank's Muon + AdamW optimizers for communication-free Muon.

    Muon optimizes this rank's owned 2D matrices (full tensors, so Newton-Schulz is
    exact and local); AdamW optimizes the rest of this rank's local parameters.
    Sub-optimizers with no parameters on this rank are omitted.

    Args:
        model: A model already wrapped by ``flex_shard`` (see
            :func:`comm_free_muon_buckets`).
        mesh: The 1D FlexShard mesh.
        muon_kwargs: Keyword args forwarded to ``torch.optim.Muon``.
        adamw_kwargs: Keyword args forwarded to ``torch.optim.AdamW``.
        muon_param_predicate: Optional Muon-eligibility override; see
            :func:`build_muon_param_groups`.

    Returns:
        A :class:`CombinedOptimizer` over the constructed optimizers.
    """
    muon_params, other_params = build_muon_param_groups(
        model,
        mesh,
        muon_param_predicate=muon_param_predicate,
    )
    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(torch.optim.Muon(muon_params, **(muon_kwargs or {})))
    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, **(adamw_kwargs or {})))
    return CombinedOptimizer(optimizers)
