# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .bucket_runtime import _create_param_accessor_states, _install_bucket_unshard_hooks
from .bucket_storage import (
    _assign_params_to_buckets,
    _materialize_bucket_storages,
    BucketSpec,
)
from .reshard_after_forward import _apply_reshard_after_forward
from .unsharded_param_access import (
    _attach_flex_shard_module_state,
    _check_not_already_flex_sharded,
    _register_module_param_accessors,
    FlexShardModule,
)
from .utils import (
    _get_device_from_mesh,
    _get_managed_named_params,
    _validate_bucket_placements,
    _validate_eager_params,
    _validate_flex_shard_mesh,
    _validate_placements,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .placement_contract import Placement


__all__ = [
    "flex_shard",
]


@dataclass(frozen=True)
class PreparedFlexShardInputs:
    """Validated inputs and derived setup state for flex_shard()."""

    named_params: list[tuple[str, nn.Parameter]]
    shard_mesh: DeviceMesh
    device: torch.device
    param_placements: dict[str, tuple[Placement, ...]]
    bucket_assignments: list[list[str]]


def _resolve_bucket_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    shard_mesh: DeviceMesh,
    bucket_assignments: list[list[str]],
    buckets: list[BucketSpec],
) -> dict[str, tuple[Placement, ...]]:
    """Call each bucket's placement function for the params assigned to it."""
    named_params_dict = dict(named_params)
    param_placements: dict[str, tuple[Placement, ...]] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue
        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_param_placements = buckets[bucket_idx].placement_fn(
            bucket_named_params,
            shard_mesh,
        )
        _validate_placements(bucket_param_placements, bucket_named_params, shard_mesh)
        param_placements.update(bucket_param_placements)

    _validate_placements(param_placements, named_params, shard_mesh)
    return param_placements


def _prepare_flex_shard_inputs(
    module: nn.Module,
    mesh: DeviceMesh,
    buckets: list[BucketSpec],
) -> PreparedFlexShardInputs:
    """Validate inputs and derive setup state for flex_shard()."""
    _check_not_already_flex_sharded(module)
    _validate_flex_shard_mesh(mesh)
    shard_mesh = mesh

    if not buckets:
        raise ValueError("flex_shard requires at least one BucketSpec in buckets.")
    if not all(isinstance(bucket, BucketSpec) for bucket in buckets):
        raise TypeError("flex_shard buckets must be a list of BucketSpec objects.")
    for bucket in buckets:
        if not callable(bucket.placement_fn):
            raise TypeError("BucketSpec.placement_fn must be callable.")
        if bucket.offload_policy is not None:
            raise NotImplementedError(
                "FlexShard eager mode does not yet support BucketSpec.offload_policy."
            )

    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )

    all_params_meta = all(param.device.type == "meta" for _, param in named_params)
    device = (
        torch.device("meta") if all_params_meta else _get_device_from_mesh(shard_mesh)
    )
    if not all_params_meta and all(
        param.device.type == "cpu" for _, param in named_params
    ):
        module.to(device)
        named_params = _get_managed_named_params(module)

    _validate_eager_params(
        named_params,
        expected_device=None if all_params_meta else device,
    )

    param_fqns = [fqn for fqn, _ in named_params]
    bucket_assignments = _assign_params_to_buckets(param_fqns, buckets)
    param_placements = _resolve_bucket_param_placements(
        named_params,
        shard_mesh,
        bucket_assignments,
        buckets,
    )
    _validate_bucket_placements(
        bucket_assignments,
        param_placements,
        buckets,
        named_params,
    )

    return PreparedFlexShardInputs(
        named_params=named_params,
        shard_mesh=shard_mesh,
        device=device,
        param_placements=param_placements,
        bucket_assignments=bucket_assignments,
    )


def flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    buckets: list[BucketSpec],
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Registers property-based accessors for eager parameter access
    6. Stores DStorages on the module (accessible via module.dstorages)

    Each bucket gets its own byte buffer and DStorage, enabling independent
    unshard operations per bucket.

    Args:
        module: The module to shard. CPU modules are moved to the mesh's CUDA
            device before sharding. Meta parameters keep uninitialized storage.
        mesh: The 1D device mesh for sharding.
        buckets: Required list of bucket specifications. Each bucket owns its
            placement function. A single whole-module bucket can be expressed as
            ``[BucketSpec(["*"], placement_fn=per_param_placements)]``.
            When ``reshard_after_forward=True``, FlexShard raises if bucket
            hooks cannot run in both the original forward and activation
            checkpoint recomputation.

    Returns:
        The module (mutated in-place). Use module.dstorages to access internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        >>> model = Transformer(args)
        >>> # Single bucket without reshard-after-forward:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     buckets=[
        ...         BucketSpec(
        ...             ["*"],
        ...             placement_fn=per_param_placements,
        ...             reshard_after_forward=False,
        ...         )
        ...     ],
        ... )
        >>> # Explicit buckets:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     buckets=[
        ...         BucketSpec(["attn.*"], placement_fn=per_param_placements),
        ...         BucketSpec(["ffn.*"], placement_fn=per_param_placements),
        ...     ],
        ... )
    Note:
        - Each parameter must have exactly one placement.
        - Each bucket must contain one original parameter dtype. Split mixed
          dtype parameters into separate buckets.
        - Parameters on meta device will have uninitialized storage
    """
    inputs = _prepare_flex_shard_inputs(
        module,
        mesh,
        buckets,
    )

    storages, fqn_to_bucket_spec = _materialize_bucket_storages(
        module,
        inputs.named_params,
        inputs.bucket_assignments,
        buckets,
        inputs.param_placements,
        inputs.shard_mesh,
        inputs.device,
    )

    _attach_flex_shard_module_state(module, storages)

    module_param_map = _create_param_accessor_states(
        module,
        storages,
        fqn_to_bucket_spec,
        inputs.device,
    )
    _register_module_param_accessors(module_param_map)

    # Reshard-after-forward: in eager mode, wrap each layer in checkpoint with
    # a selective policy that recomputes only collective ops (all-gather,
    # broadcast), saving compute ops to avoid redundant work.
    reshard_storages = [s for s in storages if s._reshard_after_forward]
    if reshard_storages:
        _apply_reshard_after_forward(module, reshard_storages)

    # Install bucket unshard hooks for eager mode when the storage layout
    # supports one collective per bucket.
    _install_bucket_unshard_hooks(storages, module_param_map)

    return module
