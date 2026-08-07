# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from collections.abc import Mapping

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

__all__ = [
    "collect_dtensor_metrics",
    "distribute_rank_local_metric",
    "merge_dtensor_metrics",
]


def distribute_rank_local_metric(
    local_metric: torch.Tensor,
    mesh: DeviceMesh,
) -> DTensor:
    """Represent one scalar contribution per mesh rank as a sharded DTensor.

    This function only describes the data layout. The caller applies the
    tensor operation that defines the metric, such as ``sum`` or ``amax``;
    DTensor propagation then produces the corresponding ``Partial`` placement.

    Args:
        local_metric: Detached scalar contribution owned by the current rank.
        mesh: One-dimensional mesh across which contributions are distributed.

    Returns:
        A one-dimensional DTensor with one element per mesh rank and a
        ``Shard(0)`` placement.

    Raises:
        ValueError: If the mesh is not one-dimensional or the contribution is
            not a detached scalar plain tensor.
    """
    if isinstance(local_metric, DTensor):
        raise ValueError(
            "local_metric must be a plain tensor; call DTensor.to_local() explicitly"
        )
    if local_metric.ndim != 0:
        raise ValueError(
            f"local_metric must be scalar, got shape {tuple(local_metric.shape)}"
        )
    if local_metric.requires_grad:
        raise ValueError("local_metric must be detached from autograd")
    if mesh.ndim != 1:
        raise ValueError(
            f"Rank-local metrics require a one-dimensional mesh, got {mesh.ndim}D"
        )

    return DTensor.from_local(
        local_metric.reshape(1),
        device_mesh=mesh,
        placements=(Shard(0),),
        run_check=False,
        shape=torch.Size((mesh.size(),)),
        stride=(1,),
    )


def _validate_dtensor_metric(name: str, metric: DTensor) -> None:
    if metric.ndim != 0:
        raise ValueError(
            f"Metric {name!r} must be scalar, got shape {tuple(metric.shape)}"
        )
    if metric.requires_grad:
        raise ValueError(f"Metric {name!r} must be detached from autograd")
    if metric.dtype.is_complex:
        raise ValueError(
            f"Metric {name!r} must have a real-valued dtype, got {metric.dtype}"
        )

    unsupported = [
        placement
        for placement in metric.placements
        if not isinstance(placement, (Replicate, Partial))
    ]
    if unsupported:
        raise ValueError(
            f"Metric {name!r} has unsupported placements {unsupported}; "
            "scalar metrics may only be Replicate or Partial"
        )


def merge_dtensor_metrics(
    current: Mapping[str, DTensor],
    values: Mapping[str, DTensor],
) -> dict[str, DTensor]:
    """Merge scalar metric contributions using their propagated placements."""
    merged = dict(current)
    for name, value in values.items():
        if not isinstance(value, DTensor):
            raise ValueError(
                f"Metric {name!r} must be a DTensor, got {type(value).__name__}"
            )
        _validate_dtensor_metric(name, value)

        previous = merged.get(name)
        if previous is None:
            merged[name] = value
            continue

        _validate_dtensor_metric(name, previous)
        if previous.device_mesh != value.device_mesh:
            raise ValueError(f"Metric {name!r} contributions must use the same mesh")
        if previous.placements != value.placements:
            raise ValueError(
                f"Metric {name!r} contributions must use the same placements, got "
                f"{previous.placements} and {value.placements}"
            )
        if previous.dtype != value.dtype:
            raise ValueError(
                f"Metric {name!r} contributions must use the same dtype, got "
                f"{previous.dtype} and {value.dtype}"
            )

        reduce_ops = {
            placement.reduce_op
            for placement in value.placements
            if isinstance(placement, Partial)
        }
        if len(reduce_ops) != 1:
            raise ValueError(
                f"Metric {name!r} must have exactly one Partial reduction operation "
                f"to merge contributions, got {value.placements}"
            )
        reduce_op = reduce_ops.pop()
        if reduce_op == "sum":
            result = previous + value
        elif reduce_op == "max":
            result = torch.maximum(previous, value)
        elif reduce_op == "min":
            result = torch.minimum(previous, value)
        else:
            raise ValueError(
                f"Metric {name!r} has unsupported merge operation {reduce_op!r}"
            )
        assert isinstance(result, DTensor)
        merged[name] = result

    return merged


def collect_dtensor_metrics(metrics: Mapping[str, DTensor]) -> dict[str, float]:
    """Materialize scalar DTensor metrics as replicated host floats.

    The metric computation determines collective semantics through DTensor
    shard propagation. Consequently, every input must already be a scalar with
    only ``Replicate`` or ``Partial`` placements. This function resolves each
    partial placement without interpreting the metric name or accepting a
    separate reduction operation.

    Args:
        metrics: Metrics keyed by their public logging names.

    Returns:
        Metrics with the same names and replicated scalar values converted to
        host floats.

    Raises:
        ValueError: If a metric is not a detached, real-valued scalar DTensor
            with only ``Replicate`` or ``Partial`` placements.
    """
    buckets: dict[tuple[object, ...], list[tuple[str, DTensor]]] = defaultdict(list)
    for name, metric in metrics.items():
        if not isinstance(metric, DTensor):
            raise ValueError(
                f"Metric {name!r} must be a DTensor, got {type(metric).__name__}"
            )
        _validate_dtensor_metric(name, metric)
        bucket_key = (
            metric.device_mesh,
            metric.placements,
            metric.dtype,
            metric.device,
        )
        buckets[bucket_key].append((name, metric))

    collected: dict[str, float] = {}
    for bucket in buckets.values():
        names, metrics_to_stack = zip(*bucket)
        stacked = torch.stack(metrics_to_stack)
        assert isinstance(stacked, DTensor)
        replicated = stacked.redistribute(
            placements=(Replicate(),) * stacked.device_mesh.ndim
        )
        values = replicated.to_local().tolist()
        collected.update(zip(names, map(float, values)))

    return collected
