# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route embedding requests to persistent row owners using PyTorch collectives."""

# L = local requests, U = unique local requests, M = requests received by an owner, R = table rows,
# D = embedding features, P = ranks in the row mesh.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.functional_collectives import all_to_all_single, wait_tensor
from torch.distributed.tensor import DTensor


@dataclass(kw_only=True, frozen=True, slots=True)
class RowShardedEmbeddingMetadata:
    """Prepared request routing for one fixed set of input IDs."""

    row_mesh: DeviceMesh
    num_local_rows: int
    input_ids: torch.Tensor
    local_row_ids: torch.Tensor
    inverse_indices: torch.Tensor
    input_splits: list[int]
    output_splits: list[int]


@torch.no_grad()
def prepare_row_sharded_embedding(
    input_ids: torch.Tensor, *, num_embeddings: int, row_mesh: DeviceMesh
) -> RowShardedEmbeddingMetadata:
    """Route IDs and prepare splits outside the model forward.

    The named one-dimensional mesh owns equal contiguous row shards, so
    ``num_embeddings`` must be divisible by the mesh size.
    Counts depend on the IDs and are copied to CPU here for variable-size
    all-to-all. Reuse the resulting metadata during activation recomputation.
    Every rank in the row mesh must participate, including ranks with no IDs.
    """
    if row_mesh.ndim != 1 or row_mesh.mesh_dim_names is None:
        raise ValueError("row_mesh must have one named axis")
    if num_embeddings < 1:
        raise ValueError("num_embeddings must be positive")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have dtype int32 or int64")
    if input_ids.device.type != row_mesh.device_type:
        raise ValueError("input_ids and row_mesh must use the same device type")

    num_owners = row_mesh.size()
    if num_embeddings % num_owners:
        raise ValueError("num_embeddings must be divisible by the row mesh size")
    rows_per_owner = num_embeddings // num_owners
    row_start = row_mesh.get_local_rank() * rows_per_owner
    # Use the same wire dtype when requesters supply different integer dtypes.
    ids_L = input_ids.reshape(-1).to(torch.int64)
    ids_U, inverse_L = torch.unique(ids_L, sorted=True, return_inverse=True)
    # Invalid IDs may map outside [0, num_owners); clamp only to keep bincount
    # well-formed until the validity exchange below raises on every owner.
    owners_U = ids_U.div(rows_per_owner, rounding_mode="floor").clamp(0, num_owners - 1)
    # Sorted unique IDs are already grouped by their contiguous row owner.
    counts_P = torch.bincount(owners_U, minlength=num_owners)
    valid_ids = ((ids_L >= 0) & (ids_L < num_embeddings)).all().to(torch.int64)
    # Include validity in the count exchange so invalid IDs fail on all owners
    # before any rank starts the request/response collectives.
    counts_and_validity_P2 = torch.stack((counts_P, valid_ids.expand(num_owners)), 1)
    received_counts_P2 = wait_tensor(
        all_to_all_single(counts_and_validity_P2, None, None, row_mesh)
    )
    splits = torch.cat((counts_P, received_counts_P2[:, 0], received_counts_P2[:, 1]))
    host_splits = splits.cpu().tolist()
    if not all(host_splits[2 * num_owners :]):
        raise ValueError("input_ids must be in [0, num_embeddings) on every owner")
    input_splits = host_splits[:num_owners]
    output_splits = host_splits[num_owners : 2 * num_owners]
    routed_ids_M = wait_tensor(
        all_to_all_single(ids_U, output_splits, input_splits, row_mesh)
    )
    return RowShardedEmbeddingMetadata(
        row_mesh=row_mesh,
        num_local_rows=rows_per_owner,
        input_ids=input_ids,
        local_row_ids=routed_ids_M - row_start,
        inverse_indices=inverse_L,
        input_splits=input_splits,
        output_splits=output_splits,
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

    The caller owns parameter sharding and FSDP exclusion. The gradient keeps
    the weight's placements. Weights remain fixed until the ordinary optimizer
    step.
    """
    local_weight_RD = weight_RD.to_local(grad_placements=weight_RD.placements)
    if local_weight_RD.ndim != 2 or local_weight_RD.shape[0] != metadata.num_local_rows:
        raise ValueError("weight_RD must contain exactly this owner's row shard")
    if local_weight_RD.device != metadata.local_row_ids.device:
        raise ValueError("weight_RD and routed IDs must be on the same device")
    local_vectors_MD = F.embedding(metadata.local_row_ids, local_weight_RD).to(
        dtype=dtype or local_weight_RD.dtype
    )
    returned_vectors_UD = wait_tensor(
        all_to_all_single(
            local_vectors_MD,
            metadata.input_splits,
            metadata.output_splits,
            metadata.row_mesh,
        )
    )
    return F.embedding(metadata.inverse_indices, returned_vectors_UD).reshape(
        *metadata.input_ids.shape, local_weight_RD.shape[1]
    )
