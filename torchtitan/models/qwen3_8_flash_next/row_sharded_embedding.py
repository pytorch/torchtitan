# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route embedding requests to persistent row owners using PyTorch collectives."""

# L = local requests, C = request slots per owner, R = table rows,
# D = embedding features, P = ranks in the row mesh.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.functional_collectives import all_to_all_single, wait_tensor
from torch.distributed.tensor import DTensor
from torch.export import register_dataclass


@dataclass(kw_only=True, frozen=True, slots=True)
class RowShardedEmbeddingMetadata:
    """Prepared request routing for one fixed set of input IDs.

    Shapes depend only on the request count and ``max_ids_per_owner``, so the
    routing can be refreshed in place for accelerator-graph replay.
    ``local_row_ids`` holds ``P * C`` received requests in requester order;
    unused slots read local row 0 and their vectors are never gathered.
    """

    row_mesh: DeviceMesh
    num_local_rows: int
    input_ids: torch.Tensor
    local_row_ids: torch.Tensor
    inverse_indices: torch.Tensor


# Expose the routing tensors so graph input copying refreshes them on replay.
register_dataclass(RowShardedEmbeddingMetadata)


@torch.no_grad()
def prepare_row_sharded_embedding(
    input_ids: torch.Tensor,
    *,
    num_embeddings: int,
    row_mesh: DeviceMesh,
    max_ids_per_owner: int,
) -> RowShardedEmbeddingMetadata:
    """Route IDs to their row owners outside the model forward.

    The named one-dimensional mesh owns equal contiguous row shards, so
    ``num_embeddings`` must be divisible by the mesh size. Each requester sends
    every owner a fixed slot of ``max_ids_per_owner`` distinct IDs, an upper
    bound the caller derives from its ID layout, so no count reaches the host.
    IDs outside ``[0, num_embeddings)`` or more distinct IDs than the bound for
    any owner fail an asynchronous assertion on every owner, after the
    exchange. Reuse the resulting metadata during activation recomputation.
    Every rank in the row mesh must participate, including ranks with no IDs.
    """
    if row_mesh.ndim != 1 or row_mesh.mesh_dim_names is None:
        raise ValueError("row_mesh must have one named axis")
    if num_embeddings < 1:
        raise ValueError("num_embeddings must be positive")
    if max_ids_per_owner < 0:
        raise ValueError("max_ids_per_owner must be non-negative")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have dtype int32 or int64")
    if input_ids.device.type != row_mesh.device_type:
        raise ValueError("input_ids and row_mesh must use the same device type")

    num_owners = row_mesh.size()
    if num_embeddings % num_owners:
        raise ValueError("num_embeddings must be divisible by the row mesh size")
    rows_per_owner = num_embeddings // num_owners
    row_start = row_mesh.get_local_rank() * rows_per_owner
    num_slots = max_ids_per_owner
    # Use the same wire dtype when requesters supply different integer dtypes.
    ids_L = input_ids.reshape(-1).to(torch.int64)
    sorted_ids_L, order_L = ids_L.sort(stable=True)
    first_L = torch.ones_like(sorted_ids_L, dtype=torch.bool)
    first_L[1:] = sorted_ids_L[1:] != sorted_ids_L[:-1]
    # Invalid IDs may map outside [0, num_owners); clamp only to keep routing
    # well-formed until the validity assertion fails on every owner.
    owners_L = sorted_ids_L.div(rows_per_owner, rounding_mode="floor").clamp(
        0, num_owners - 1
    )
    # Sorted distinct IDs are grouped by their contiguous row owner.
    counts_P = torch.zeros(num_owners, dtype=torch.int64, device=ids_L.device)
    counts_P.scatter_add_(0, owners_L, first_L.long())
    slots_L = first_L.long().cumsum(0) - 1 - (counts_P.cumsum(0) - counts_P)[owners_L]
    valid = ((ids_L >= 0) & (ids_L < num_embeddings)).all() & (
        counts_P <= num_slots
    ).all()
    # Each owner's row ends with the requester's validity flag; the extra
    # trailing element absorbs duplicate and overflowing IDs.
    requests_P1 = torch.full(
        (num_owners * (num_slots + 1) + 1,), -1, dtype=torch.int64, device=ids_L.device
    )
    requests_P1.scatter_(
        0,
        torch.where(
            first_L & (slots_L < num_slots),
            owners_L * (num_slots + 1) + slots_L,
            requests_P1.numel() - 1,
        ),
        sorted_ids_L,
    )
    requests_P1 = requests_P1[:-1].view(num_owners, num_slots + 1)
    requests_P1[:, num_slots] = valid.long()
    received_P1 = wait_tensor(
        all_to_all_single(requests_P1.flatten(), None, None, row_mesh)
    ).view(num_owners, num_slots + 1)
    torch._assert_async(
        received_P1[:, num_slots].bool().all(),
        "input_ids must be in [0, num_embeddings) with at most max_ids_per_owner "
        "distinct IDs per owner on every requester",
    )
    local_row_ids = (received_P1[:, :num_slots].flatten() - row_start).clamp(
        0, rows_per_owner - 1
    )
    inverse_indices = torch.empty_like(ids_L).scatter_(
        0, order_L, owners_L * num_slots + slots_L.clamp(0, max(num_slots - 1, 0))
    )
    return RowShardedEmbeddingMetadata(
        row_mesh=row_mesh,
        num_local_rows=rows_per_owner,
        input_ids=input_ids,
        local_row_ids=local_row_ids,
        inverse_indices=inverse_indices,
    )


def row_sharded_embedding(
    weight_RD: DTensor,
    metadata: RowShardedEmbeddingMetadata,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Look up owned rows and return vectors to the original requesters.

    ``weight_RD`` is a ``Shard(0)`` DTensor over ``metadata.row_mesh``. The
    output has shape ``[*metadata.input_ids.shape, D]`` and uses ``dtype``
    (the weight's dtype by default). Duplicate IDs are expanded on the requester
    with ``F.embedding`` so their gradients accumulate in the compute dtype,
    as in an ordinary mixed-precision embedding. Owners then sum the requester
    contributions in the master weight's dtype, without mesh-size division.
    Unused request slots carry zero gradient.

    The caller owns parameter sharding and FSDP exclusion. The gradient keeps
    the weight's placements. Weights remain fixed until the ordinary optimizer
    step.
    """
    local_weight_RD = weight_RD.to_local(grad_placements=weight_RD.placements)
    if local_weight_RD.ndim != 2 or local_weight_RD.shape[0] != metadata.num_local_rows:
        raise ValueError("weight_RD must contain exactly this owner's row shard")
    if local_weight_RD.device != metadata.local_row_ids.device:
        raise ValueError("weight_RD and routed IDs must be on the same device")
    local_vectors_CD = F.embedding(metadata.local_row_ids, local_weight_RD).to(
        dtype=dtype or local_weight_RD.dtype
    )
    returned_vectors_CD = wait_tensor(
        all_to_all_single(local_vectors_CD, None, None, metadata.row_mesh)
    )
    return F.embedding(metadata.inverse_indices, returned_vectors_CD).reshape(
        *metadata.input_ids.shape, local_weight_RD.shape[1]
    )
