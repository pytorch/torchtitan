# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Mixed precision policy for FlexShard buckets.

    Args:
        param_dtype: Dtype for forward compute. Parameters are all-gathered
            in storage dtype, then cast to param_dtype. If None, no cast.
        reduce_dtype: Dtype for gradient reduction. Gradients are cast to
            this dtype before reduce-scatter. If None, uses param_dtype
            (or storage dtype if param_dtype is also None).
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class OffloadPolicy:
    """CPU offload policy for FlexShard buckets.

    When set on a BucketSpec, the bucket's byte storage is allocated on
    CPU (optionally pinned). The parametrization handles H2D transfer
    before all-gather; backward autograd handles D2H automatically.

    Args:
        pin_memory: Whether to pin CPU memory for faster H2D/D2H
            transfers via DMA. Set to False if insufficient CPU memory.
            Default True.
    """

    pin_memory: bool = True


@dataclass(frozen=True)
class BucketSpec:
    """Specification for a parameter communication bucket.

    Args:
        patterns: fnmatch glob patterns matched against parameter FQNs.
            A parameter matches this bucket if its FQN matches any pattern.
        mp_policy: Mixed precision policy for this bucket.
        offload_policy: CPU offload policy for this bucket.
        reshard_after_forward: Whether to free this bucket's unsharded
            parameters after forward and recompute them in backward.
    """

    patterns: list[str]
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: OffloadPolicy | None = None
    reshard_after_forward: bool = True


def _assign_params_to_buckets(
    param_fqns: list[str],
    buckets: list[BucketSpec],
) -> list[list[str]]:
    """Assign each param FQN to exactly one bucket via fnmatch.

    Returns:
        List of lists: assignments[i] = [fqn, ...] for bucket i.

    Raises:
        ValueError: if any param matches zero or multiple buckets.
    """
    param_to_buckets: dict[str, list[int]] = {fqn: [] for fqn in param_fqns}
    for bucket_idx, bucket in enumerate(buckets):
        for fqn in param_fqns:
            for pattern in bucket.patterns:
                if fnmatch.fnmatch(fqn, pattern):
                    param_to_buckets[fqn].append(bucket_idx)
                    break  # one match per bucket is enough

    # Check for orphans
    orphans = [fqn for fqn, idxs in param_to_buckets.items() if len(idxs) == 0]
    if orphans:
        orphan_list = "\n  ".join(orphans)
        raise ValueError(
            f"flex_shard: {len(orphans)} parameters not covered by any bucket:\n"
            f"  {orphan_list}\n"
            'Add these to an existing bucket or add a catch-all bucket: ["*"]'
        )

    # Check for overlaps
    overlaps = {fqn: idxs for fqn, idxs in param_to_buckets.items() if len(idxs) > 1}
    if overlaps:
        lines = []
        for fqn, idxs in overlaps.items():
            bucket_descs = ", ".join(f"bucket {i} {buckets[i].patterns}" for i in idxs)
            lines.append(f"  {fqn} -> {bucket_descs}")
        overlap_list = "\n".join(lines)
        raise ValueError(
            f"flex_shard: {len(overlaps)} parameters matched multiple buckets:\n"
            f"{overlap_list}\n"
            "Ensure each parameter matches exactly one bucket."
        )

    # Build assignments
    assignments: list[list[str]] = [[] for _ in buckets]
    for fqn, idxs in param_to_buckets.items():
        assignments[idxs[0]].append(fqn)

    return assignments


def auto_buckets(module: nn.Module) -> list[BucketSpec]:
    """Generate one bucket per direct child module.

    Returns a list of ``BucketSpec`` objects suitable for the ``buckets``
    parameter of :func:`flex_shard`. Each bucket contains a single
    ``"child_name.*"`` pattern matching all parameters under that child.

    Example::

        >>> buckets = auto_buckets(model)
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     dp_mesh_dims,
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=buckets,
        ... )
    """
    children = list(module.named_children())
    if not children:
        return [BucketSpec(["*"])]
    return [BucketSpec([f"{name}.*"]) for name, _ in children]
