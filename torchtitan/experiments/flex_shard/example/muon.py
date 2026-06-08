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

For the memory-balanced (not comm-free) alternative, :func:`grouped_ragged_shard_muon_buckets`
+ :class:`RaggedShardMuon` evenly shard each Muon matrix across ranks and all-gather it
inside the step to run Newton-Schulz -- byte-perfect memory balance and no idle ranks, at
the cost of one collective per bucket (instead of ``Owned``'s zero). The step stays
bit-identical to single-device Muon either way.

See ``muon_flex_shard_placement_strategies.md`` for the design and trade-offs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim._muon import (
    _adjust_lr,
    _zeropower_via_newtonschulz,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
)

from ..flex_shard.bucket_storage import BucketSpec
from ..flex_shard.sharded_param import get_global_shape, get_placements
from .owned import (
    assign_layer_owners_lpt,
    assign_matrix_owners_per_layer_balanced,
    make_owned_placement_fn,
    Owned,
)
from .ragged_shard import (
    GroupedRaggedShard,
    make_grouped_ragged_placement_fn,
    RaggedShard,
)
from .shard import per_param_placements, Shard

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = [
    "build_comm_free_muon_optimizers",
    "build_muon_param_groups",
    "build_ragged_shard_muon_optimizers",
    "CombinedOptimizer",
    "comm_free_muon_buckets",
    "GroupedMuon",
    "grouped_ragged_shard_muon_buckets",
    "RaggedShardMuon",
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


def _per_matrix_layer_buckets(
    model: nn.Module,
    num_layers: int,
    mesh: DeviceMesh,
    reshard_after_forward: bool,
) -> list[BucketSpec]:
    """One ``Owned`` bucket per 2D matrix (per-layer balanced); non-2D -> ``Shard(0)``.

    Within each layer the 2D matrices (attention / dense FFN / shared experts) get
    one ``Owned`` bucket each, with owners from
    :func:`assign_matrix_owners_per_layer_balanced` so every rank holds a balanced
    share of the layer. Each non-2D param gets its own ``Shard(0)`` bucket: >= 3D
    grouped experts (-> ``GroupedMuon``) and 1D norms/biases (-> AdamW).
    """
    world_size = mesh.size()
    per_layer_matrices: list[list[tuple[str, int]]] = []
    per_layer_other: list[list[str]] = []
    for i in range(num_layers):
        matrices: list[tuple[str, int]] = []
        other: list[str] = []
        for name, param in model.layers[i].named_parameters():
            fqn = f"layers.{i}.{name}"
            if param.ndim == 2:
                matrices.append((fqn, param.numel()))
            else:
                other.append(fqn)
        per_layer_matrices.append(matrices)
        per_layer_other.append(other)

    owners = assign_matrix_owners_per_layer_balanced(
        [[numel for _, numel in matrices] for matrices in per_layer_matrices],
        world_size,
    )

    buckets: list[BucketSpec] = []
    for i in range(num_layers):
        for (fqn, _), owner in zip(per_layer_matrices[i], owners[i], strict=True):
            buckets.append(
                BucketSpec(
                    [fqn],
                    placement_fn=make_owned_placement_fn(owner),
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        for fqn in per_layer_other[i]:
            buckets.append(
                BucketSpec(
                    [fqn],
                    placement_fn=per_param_placements,  # Shard(0)
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
            )
    return buckets


def comm_free_muon_buckets(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    reshard_after_forward: bool = True,
    balance: str = "lpt",
    rest_patterns: list[str] | None = None,
) -> list[BucketSpec]:
    """Build FlexShard buckets for communication-free Muon.

    With ``balance="lpt"`` / ``"roundrobin"`` each transformer layer
    (``model.layers[i]``) becomes one ``Owned`` bucket on a single rank; with
    ``balance="per-matrix"`` each 2D matrix becomes its own ``Owned`` bucket and the
    layer's experts / norms are ``Shard``ed (see ``balance``). Either way the owner
    runs Muon locally with no collective in the step, and embeddings, the LM head,
    and the final norm stay ``Shard(0)`` (FSDP) for AdamW.

    Each rest pattern gets its own bucket (not one grouped bucket) so that, with
    ``reshard_after_forward=True``, every bucket maps to a single execution-unit
    module whose forward hook is replayable during activation-checkpoint recompute.
    A grouped embeddings/LM-head/norm bucket resolves to the root module and is
    rejected by the reshard-after-forward hook installer.

    Args:
        model: A FlexShard-compatible Transformer exposing ``model.layers`` as an
            indexable sequence of per-layer modules.
        mesh: The 1D CUDA FlexShard mesh; every bucket runs its collective on it
            and ``world_size`` is taken from ``mesh.size()``.
        reshard_after_forward: Free unsharded params after forward and recompute
            them in backward (defaults to ``True``). The ``Owned`` broadcast and the
            ``Shard`` all-gather are both tagged for recompute, so this composes
            with activation checkpointing.
        balance: how layer parameters are placed. ``"lpt"`` (default) /
            ``"roundrobin"`` put each *whole layer* on one rank (greedy
            Longest-Processing-Time, or ``i % world_size``). ``"per-matrix"`` gives
            every 2D matrix its own ``Owned`` bucket, balanced *within each layer*
            (:func:`assign_matrix_owners_per_layer_balanced`), and puts each non-2D
            param in its own ``Shard(0)`` bucket (>= 3D experts -> ``GroupedMuon``,
            1D norms -> AdamW) -- smaller, balanced per-collective messages.
        rest_patterns: FQN globs for the non-Muon (Shard + AdamW) region; each
            becomes its own bucket. Defaults to embeddings, LM head, and final norm.

    Returns:
        A list of ``BucketSpec`` for ``flex_shard(model, buckets)``.
    """
    world_size = mesh.size()
    num_layers = len(model.layers)
    if balance == "per-matrix":
        layer_buckets = _per_matrix_layer_buckets(
            model, num_layers, mesh, reshard_after_forward
        )
    elif balance in ("lpt", "roundrobin"):
        if balance == "lpt":
            layer_numels = [
                sum(p.numel() for _, p in model.layers[i].named_parameters())
                for i in range(num_layers)
            ]
            owners = assign_layer_owners_lpt(layer_numels, world_size)
        else:
            owners = [i % world_size for i in range(num_layers)]
        layer_buckets = [
            BucketSpec(
                [f"layers.{i}.*"],
                placement_fn=make_owned_placement_fn(owners[i]),
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
            for i in range(num_layers)
        ]
    else:
        raise ValueError(
            f"balance must be 'lpt', 'roundrobin', or 'per-matrix', but got {balance!r}."
        )

    patterns = _DEFAULT_REST_PATTERNS if rest_patterns is None else rest_patterns
    rest_buckets = [
        BucketSpec(
            [pattern],
            placement_fn=per_param_placements,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )
        for pattern in patterns
    ]
    return [*layer_buckets, *rest_buckets]


def grouped_ragged_shard_muon_buckets(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    reshard_after_forward: bool = False,
    rest_patterns: list[str] | None = None,
) -> list[BucketSpec]:
    """Build FlexShard buckets for memory-balanced (gather-based) Muon.

    Each transformer layer's 2D matrices share **one** ``GroupedRaggedShard`` bucket:
    the bucket is flattened param-major and cut into ``world_size`` byte-balanced ranges
    that cross matrix boundaries, so every rank holds an exactly ``1/world_size`` slice
    of the layer's matrices -- no whole-matrix hotspot, and no idle ranks even when a
    layer has fewer matrices than ranks. :class:`RaggedShardMuon` all-gathers each
    bucket inside the step to run Newton-Schulz on the full matrix. Each non-2D param
    gets its own ``Shard(0)`` bucket (>= 3D experts -> ``GroupedMuon``, 1D norms ->
    AdamW), and embeddings / LM head / final norm stay ``Shard(0)`` for AdamW.

    Unlike :func:`comm_free_muon_buckets` this is **not communication-free**: it spends
    one all-gather per bucket to buy byte-perfect memory balance. See
    ``muon_flex_shard_placement_strategies.md`` for the trade-off.

    Args:
        model: A FlexShard-compatible Transformer exposing ``model.layers``.
        mesh: The 1D CUDA FlexShard mesh; every bucket runs its collective on it
            and ``world_size`` is taken from ``mesh.size()``.
        reshard_after_forward: Free unsharded params after forward (defaults to
            ``False``; ``RaggedShardMuon`` maps bucket params by FQN, and the
            activation-checkpoint wrappers that reshard-after-forward inserts would
            rename them).
        rest_patterns: FQN globs for the non-Muon (Shard + AdamW) region; each becomes
            its own bucket. Defaults to embeddings, LM head, and final norm.

    Returns:
        A list of ``BucketSpec`` for ``flex_shard(model, buckets)``.
    """
    world_size = mesh.size()
    grouped_ragged = make_grouped_ragged_placement_fn(
        dims=(0,), local_units=(1,) * world_size
    )
    buckets: list[BucketSpec] = []
    for i in range(len(model.layers)):
        matrices: list[str] = []
        other: list[str] = []
        for name, param in model.layers[i].named_parameters():
            fqn = f"layers.{i}.{name}"
            (matrices if param.ndim == 2 else other).append(fqn)
        if matrices:
            buckets.append(
                BucketSpec(
                    matrices,
                    placement_fn=grouped_ragged,
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
            )
        for fqn in other:
            buckets.append(
                BucketSpec(
                    [fqn],
                    placement_fn=per_param_placements,  # Shard(0)
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
            )

    patterns = _DEFAULT_REST_PATTERNS if rest_patterns is None else rest_patterns
    buckets += [
        BucketSpec(
            [pattern],
            placement_fn=per_param_placements,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )
        for pattern in patterns
    ]
    return buckets


MuonParamPredicate = Callable[[str, "torch.Size | None"], bool]


def _default_muon_predicate(fqn: str, global_shape: torch.Size | None) -> bool:
    """Muon-eligible iff a 2D matrix or a stack of matrices (>= 3D, e.g. MoE experts).

    build_comm_free_muon_optimizers routes 2D params to ``torch.optim.Muon`` and
    >= 3D stacked params (grouped experts) to :class:`GroupedMuon`, which runs a
    batched Newton-Schulz over the leading dim(s).
    """
    return global_shape is not None and len(global_shape) >= 2


def build_muon_param_groups(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    muon_param_predicate: MuonParamPredicate | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition a FlexSharded model's local params into ``(muon, other)`` groups.

    A parameter is comm-free Muon-eligible on this rank when its local shard keeps
    whole matrices, i.e. either:

    * it is ``Owned`` by this rank (the full matrix / stack lives here), or
    * it is ``Shard(0)`` and ``>= 3D`` -- sharding the leading (expert/batch) dim of
      a stack keeps every ``(m, n)`` matrix whole on every rank (e.g.
      expert-parallel MoE experts), so the local ``(E/N, m, n)`` shard is still
      grouped-Muon-able with no collective.

    Everything else with non-empty local storage goes to the other (AdamW) group:
    ``Shard`` embeddings / LM head (2D, where sharding would split the matrix),
    owned norms and biases, etc. Empty local shards (e.g. a non-owned ``Owned``
    param, or a rank with no experts) are skipped; another rank optimizes them.
    ``build_comm_free_muon_optimizers`` then sends 2D Muon params to
    ``torch.optim.Muon`` and ``>= 3D`` ones to ``GroupedMuon``.

    Args:
        model: A model already wrapped by ``flex_shard``.
        mesh: The 1D FlexShard mesh, used for this rank's index.
        muon_param_predicate: Optional override ``(fqn, global_shape) -> bool`` for
            Muon eligibility. Defaults to "is a matrix or stack of matrices"
            (``ndim >= 2``).

    Returns:
        ``(muon_params, other_params)`` for this rank.
    """
    predicate = muon_param_predicate or _default_muon_predicate
    rank = mesh.get_local_rank()
    muon_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for fqn, param in model.named_parameters():
        if param.numel() == 0:
            # Empty local shard (e.g. non-owned Owned param): another rank owns it.
            continue
        placements = get_placements(param)
        global_shape = get_global_shape(param)
        placement = (
            placements[0] if placements is not None and len(placements) == 1 else None
        )
        # Owned: this rank holds the whole matrix / stack -> comm-free Muon.
        owned_here = isinstance(placement, Owned) and placement.owner_rank == rank
        # Shard(0) of a >= 3D stack: only the expert/batch dim is split, so the
        # local shard keeps whole (m, n) matrices -> still comm-free grouped Muon.
        expert_dim_sharded = (
            isinstance(placement, Shard)
            and placement.dim == 0
            and global_shape is not None
            and len(global_shape) >= 3
        )
        if (owned_here or expert_dim_sharded) and predicate(fqn, global_shape):
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


def _grouped_zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """Batched Newton-Schulz over the leading dim(s) of a ``>= 3D`` tensor.

    Treats ``grad`` of shape ``(*lead, m, n)`` as a batch of ``(m, n)`` matrices and
    orthogonalizes each independently. Mirrors ``torch.optim``'s
    ``_zeropower_via_newtonschulz`` exactly, but with ``baddbmm`` / ``transpose(-2,
    -1)`` and a per-matrix norm, so the result equals running the 2D iteration on
    each matrix separately. Returns bfloat16 (like the upstream helper).
    """
    if grad.ndim < 3:
        raise ValueError("grouped Newton-Schulz expects a >= 3D (batched) tensor")
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    lead = grad.shape[:-2]
    m, n = grad.shape[-2], grad.shape[-1]
    ortho = grad.reshape(-1, m, n).bfloat16()
    transposed = m > n
    if transposed:
        ortho = ortho.transpose(-2, -1)
    # Per-matrix spectral-norm proxy (Frobenius norm is transpose-invariant).
    ortho = ortho.div_(ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps))
    for _ in range(ns_steps):
        gram = torch.bmm(ortho, ortho.transpose(-2, -1))
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.transpose(-2, -1)
    return ortho.reshape(*lead, m, n)


class GroupedMuon(torch.optim.Optimizer):
    """Muon for stacked weight matrices (``ndim >= 3``), e.g. MoE grouped experts.

    A parameter of shape ``(*lead, m, n)`` is treated as a batch of ``(m, n)``
    matrices: Newton-Schulz runs batched over the leading dim(s), orthogonalizing
    each matrix independently. This is numerically identical to running
    ``torch.optim.Muon`` on each ``(m, n)`` sub-matrix separately (same momentum,
    Nesterov, decoupled weight decay, LR adjustment, and NS coefficients). Use
    ``torch.optim.Muon`` for plain 2D matrices.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
        ):
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 3:
                    raise ValueError(
                        "GroupedMuon expects stacked matrices with ndim >= 3 "
                        f"(e.g. grouped experts), but found shape {tuple(p.size())}. "
                        "Use torch.optim.Muon for 2D parameters."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step (batched Muon over leading dims)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GroupedMuon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                update = _grouped_zeropower_via_newtonschulz(
                    update, ns_coefficients, ns_steps, eps
                )
                # LR adjustment uses the per-matrix shape (m, n), not the batch dim.
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape[-2:])
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-adjusted_lr)
        return loss


def _discover_ragged_muon_buckets(
    model: nn.Module,
) -> list[dict[str, Any]]:
    """Find this rank's ``RaggedShard`` / ``GroupedRaggedShard`` 2D-matrix buckets.

    Returns one entry per Muon bucket with its placement, ``ParamInfo`` list (bucket
    order), the matching local parameter objects (by FQN), and the bucket storage.
    Buckets whose placement is not a (subclass of) ``RaggedShard``, or that hold any
    non-2D param, are skipped -- experts / norms / embeddings are optimized by
    GroupedMuon / AdamW instead.
    """
    param_by_fqn = dict(model.named_parameters())
    buckets: list[dict[str, Any]] = []
    for storage in getattr(model, "sharded_bucket_storages", []):
        infos = list(storage.param_infos.values())
        if not infos:
            continue
        placement = infos[0].placement
        # GroupedRaggedShard is a subclass of RaggedShard, so this covers both.
        if not isinstance(placement, RaggedShard):
            continue
        if not all(len(info.global_shape) == 2 for info in infos):
            continue
        params = [param_by_fqn.get(info.fqn) for info in infos]
        if any(p is None for p in params):
            raise ValueError(
                "RaggedShardMuon could not map bucket parameters to model parameters "
                "by FQN; reshard-after-forward renames them, which the ragged Muon "
                "recipe does not yet support. Use reshard_after_forward=False."
            )
        buckets.append(
            {
                "placement": placement,
                "infos": infos,
                "params": params,
                "storage": storage,
            }
        )
    return buckets


class RaggedShardMuon(torch.optim.Optimizer):
    """Memory-balanced Muon for ``RaggedShard`` / ``GroupedRaggedShard`` 2D matrices.

    Each Muon matrix is evenly sharded across ranks -- byte-perfect across the bucket
    with ``GroupedRaggedShard`` (the cut crosses parameter boundaries) or row-even per
    matrix with ``RaggedShard`` -- so parameters, gradients, and the momentum buffer are
    all balanced ``1/N``. Newton-Schulz needs the *full* matrix, so the step all-gathers
    each bucket's matrices (**one collective per bucket**), runs Newton-Schulz on the
    full matrix on every rank, and writes back only this rank's shard of the update. The
    momentum buffer stays sharded (its update is element-wise); only the pre-NS update is
    gathered.

    The result is **bit-identical to single-device** ``torch.optim.Muon`` (NS runs on
    the genuine full matrix with the batch-averaged gradient). Unlike ``Owned`` +
    ``torch.optim.Muon`` this is **not communication-free**: it trades the per-bucket
    all-gather for perfect memory balance and zero idle ranks. Take the model after
    ``flex_shard`` (with :func:`grouped_ragged_shard_muon_buckets`) and the 1D mesh; the
    optimizer discovers its ragged buckets from ``model.sharded_bucket_storages``.
    """

    def __init__(
        self,
        model: nn.Module,
        mesh: DeviceMesh,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
        ):
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        self._mesh = mesh
        self._buckets = _discover_ragged_muon_buckets(model)
        params: list[nn.Parameter] = []
        seen: set[int] = set()
        for bucket in self._buckets:
            for param in bucket["params"]:
                if param.numel() > 0 and id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
        if not params:
            raise ValueError(
                "RaggedShardMuon found no RaggedShard/GroupedRaggedShard 2D matrices on "
                "this rank. Place Muon matrices with grouped_ragged_shard_muon_buckets()."
            )
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _local_update_shard(
        full_update: torch.Tensor,
        placement: RaggedShard,
        info: Any,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        """This rank's shard of the full Newton-Schulz update."""
        if isinstance(placement, GroupedRaggedShard):
            # Byte-balanced cut crosses matrix boundaries: slice the flat full matrix
            # at this rank's offset within the param (mirrors copy_param_to_storage).
            layout = info.bucket_layout.param_layouts[info.fqn]
            start = layout.local_global_offset - layout.param_offset
            return full_update.reshape(-1)[start : start + info.local_numel].view(
                info.local_shape
            )
        return placement.extract_local_shard(full_update, rank, world_size)

    @torch.no_grad()
    def step(self, closure=None):
        """All-gather each bucket, run NS on the full matrix, write back the local shard."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_coefficients = group["ns_coefficients"]
        eps = group["eps"]
        ns_steps = group["ns_steps"]
        adjust_lr_fn = group["adjust_lr_fn"]
        rank = self._mesh.get_local_rank()
        world_size = self._mesh.size()

        for bucket in self._buckets:
            placement = bucket["placement"]
            infos = bucket["infos"]
            params = bucket["params"]
            storage = bucket["storage"]

            # Stage each local shard's pre-NS update into one bucket-shaped scratch
            # buffer so GroupedRaggedShard can view it as a contiguous bucket slice;
            # the momentum buffer stays sharded (its update is element-wise).
            scratch = torch.zeros(
                storage.total_bytes,
                dtype=torch.uint8,
                device=storage.byte_storage.device,
            )
            pre_views = [
                placement.make_local_storage_view(scratch, info) for info in infos
            ]
            active: list[bool] = []
            for info, param, pre_view in zip(infos, params, pre_views, strict=True):
                has_update = info.local_numel > 0 and param.grad is not None
                active.append(has_update)
                if not has_update:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "RaggedShardMuon does not support sparse gradients"
                    )
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                pre_view.copy_(grad.lerp(buf, momentum) if nesterov else buf)

            # One collective per bucket: all-gather the staged pre-NS shards and
            # reconstruct each full matrix on every rank.
            prepared = placement.prepare_unshard_bucket(
                pre_views, infos, self._mesh, None
            )
            placement.run_prepared_unshard(prepared)
            full_pre_updates = placement.finish_prepared_unshard(prepared).full_params

            for info, param, full_pre, has_update in zip(
                infos, params, full_pre_updates, active, strict=True
            ):
                if not has_update:
                    continue
                full_update = _zeropower_via_newtonschulz(
                    full_pre, ns_coefficients, ns_steps, eps
                )
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, info.global_shape)
                update_shard = self._local_update_shard(
                    full_update, placement, info, rank, world_size
                )
                param.mul_(1 - lr * weight_decay)
                param.add_(update_shard, alpha=-adjusted_lr)
        return loss


def build_comm_free_muon_optimizers(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    muon_kwargs: dict[str, Any] | None = None,
    adamw_kwargs: dict[str, Any] | None = None,
    muon_param_predicate: MuonParamPredicate | None = None,
) -> CombinedOptimizer:
    """Build this rank's Muon + AdamW optimizers for communication-free Muon.

    ``torch.optim.Muon`` optimizes this rank's owned 2D matrices and
    :class:`GroupedMuon` optimizes its owned ``>= 3D`` stacked matrices (e.g. MoE
    grouped experts) via batched Newton-Schulz -- both on full tensors, so the
    iteration is exact and local. AdamW optimizes the rest of this rank's local
    parameters. Sub-optimizers with no parameters on this rank are omitted.

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
    # 2D matrices -> torch.optim.Muon; >= 3D stacked matrices (e.g. MoE grouped
    # experts) -> GroupedMuon (batched Newton-Schulz over the leading dim(s)).
    muon_2d = [p for p in muon_params if p.ndim == 2]
    muon_grouped = [p for p in muon_params if p.ndim >= 3]
    optimizers: list[torch.optim.Optimizer] = []
    if muon_2d:
        optimizers.append(torch.optim.Muon(muon_2d, **(muon_kwargs or {})))
    if muon_grouped:
        optimizers.append(GroupedMuon(muon_grouped, **(muon_kwargs or {})))
    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, **(adamw_kwargs or {})))
    return CombinedOptimizer(optimizers)


def build_ragged_shard_muon_optimizers(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    muon_kwargs: dict[str, Any] | None = None,
    adamw_kwargs: dict[str, Any] | None = None,
) -> CombinedOptimizer:
    """Build this rank's optimizers for the memory-balanced ragged Muon recipe.

    :class:`RaggedShardMuon` optimizes the ragged-sharded 2D matrices (all-gather +
    Newton-Schulz, byte-balanced), :class:`GroupedMuon` optimizes ``Shard(0)`` >= 3D
    grouped experts (comm-free), and AdamW optimizes the rest of this rank's local
    parameters. Sub-optimizers with no parameters on this rank are omitted.

    Args:
        model: A model already wrapped by ``flex_shard`` (see
            :func:`grouped_ragged_shard_muon_buckets`).
        mesh: The 1D FlexShard mesh.
        muon_kwargs: Keyword args forwarded to ``RaggedShardMuon`` and ``GroupedMuon``.
        adamw_kwargs: Keyword args forwarded to ``torch.optim.AdamW``.

    Returns:
        A :class:`CombinedOptimizer` over the constructed optimizers.
    """
    optimizers: list[torch.optim.Optimizer] = []

    # Ragged 2D matrices -> RaggedShardMuon (discovers its own buckets). Only build it
    # when this rank actually holds ragged Muon shards (it always does under an even
    # ragged split, but guard so an all-AdamW rank does not error).
    has_ragged = any(
        param.numel() > 0
        for bucket in _discover_ragged_muon_buckets(model)
        for param in bucket["params"]
    )
    if has_ragged:
        optimizers.append(RaggedShardMuon(model, mesh, **(muon_kwargs or {})))

    grouped_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for _, param in model.named_parameters():
        if param.numel() == 0:
            continue
        placements = get_placements(param)
        global_shape = get_global_shape(param)
        placement = (
            placements[0] if placements is not None and len(placements) == 1 else None
        )
        # Ragged 2D matrices are handled by RaggedShardMuon above (GroupedRaggedShard
        # is a RaggedShard subclass, so this also skips it).
        if (
            isinstance(placement, RaggedShard)
            and global_shape is not None
            and len(global_shape) == 2
        ):
            continue
        # Shard(0) of a >= 3D stack keeps whole (m, n) matrices -> comm-free GroupedMuon.
        if (
            isinstance(placement, Shard)
            and placement.dim == 0
            and global_shape is not None
            and len(global_shape) >= 3
        ):
            grouped_params.append(param)
        else:
            other_params.append(param)
    if grouped_params:
        optimizers.append(GroupedMuon(grouped_params, **(muon_kwargs or {})))
    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, **(adamw_kwargs or {})))
    return CombinedOptimizer(optimizers)
