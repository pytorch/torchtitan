# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route embedding requests to persistent row owners using PyTorch collectives."""

# L = local requests, C = request slots per owner, S = C + 1 (slots plus the
# requester's validity flag), Q = P * C (slots across all owners),
# R = table rows, D = embedding features, P = ranks in the row mesh.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types import rules

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
    unused slots read local row 0, and ``inverse_indices`` never selects them.
    """

    row_mesh: DeviceMesh
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
    The all-to-alls use equal splits sized from the local ``max_ids_per_owner``,
    so it must be equal on every rank of ``row_mesh``; derive it from
    rank-uniform quantities such as the table layout and the configured tokens
    per microbatch, never from the local IDs. A mismatch mis-sizes the
    exchange instead of raising. IDs outside ``[0, num_embeddings)`` or more
    distinct IDs than the bound for any owner fail an asynchronous assertion
    on every owner, after the exchange. Reuse the resulting metadata during
    activation recomputation.
    Every rank in the row mesh must participate, including ranks with no IDs.
    """
    if row_mesh.ndim != 1 or row_mesh.mesh_dim_names is None:
        raise ValueError("row_mesh must have one named axis")
    if num_embeddings < 1:
        raise ValueError("num_embeddings must be positive")
    if max_ids_per_owner < 1:
        raise ValueError("max_ids_per_owner must be positive")
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
        counts_P <= max_ids_per_owner
    ).all()
    # Each owner's row ends with the requester's validity flag; the extra
    # trailing element absorbs duplicate and overflowing IDs.
    request_buffer = torch.full(
        (num_owners * (max_ids_per_owner + 1) + 1,),
        -1,
        dtype=torch.int64,
        device=ids_L.device,
    )
    request_buffer.scatter_(
        0,
        torch.where(
            first_L & (slots_L < max_ids_per_owner),
            owners_L * (max_ids_per_owner + 1) + slots_L,
            request_buffer.numel() - 1,
        ),
        sorted_ids_L,
    )
    requests_PS = request_buffer[:-1].view(num_owners, max_ids_per_owner + 1)
    requests_PS[:, max_ids_per_owner] = valid.long()
    received_PS = wait_tensor(
        all_to_all_single(requests_PS.flatten(), None, None, row_mesh)
    ).view(num_owners, max_ids_per_owner + 1)
    torch._assert_async(
        received_PS[:, max_ids_per_owner].bool().all(),
        "input_ids must be in [0, num_embeddings) with at most max_ids_per_owner "
        "distinct IDs per owner on every requester",
    )
    local_row_ids = (received_PS[:, :max_ids_per_owner].flatten() - row_start).clamp(
        0, rows_per_owner - 1
    )
    inverse_indices = torch.empty_like(ids_L).scatter_(
        0,
        order_L,
        owners_L * max_ids_per_owner + slots_L.clamp(0, max_ids_per_owner - 1),
    )
    return RowShardedEmbeddingMetadata(
        row_mesh=row_mesh,
        input_ids=input_ids,
        local_row_ids=local_row_ids,
        inverse_indices=inverse_indices,
    )


class _OwnerLookup(torch.autograd.Function):
    """Owner lookup plus the response all-to-all, with an explicit backward.

    Forward: each owner reads the requested rows of its local shard and returns
    them to the requesters with one all-to-all over the row mesh. Backward: the
    reverse all-to-all returns the vector gradients to their owners, which sum
    them into their rows in the weight's dtype. The weight gradient is a DTensor
    with the weight's placements, built locally: no other collective runs, and
    DTensor's autograd is not involved.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        weight_RD: DTensor,
        local_row_ids_Q: torch.Tensor,
        row_mesh: DeviceMesh,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        local_weight_RD = weight_RD.to_local()
        ctx.save_for_backward(local_row_ids_Q)
        ctx.row_mesh = row_mesh
        ctx.num_local_rows = local_weight_RD.shape[0]
        ctx.weight_dtype = local_weight_RD.dtype
        ctx.weight_mesh = weight_RD.device_mesh
        ctx.weight_placements = weight_RD.placements
        ctx.weight_shape = weight_RD.shape
        ctx.weight_stride = weight_RD.stride()
        local_vectors_QD = F.embedding(local_row_ids_Q, local_weight_RD).to(
            dtype=dtype or local_weight_RD.dtype
        )
        # TODO: unlike FSDP's ring collectives, this all-to-all keeps NCCL buffers
        # for every DP peer, so its memory grows with DP. Measured only on 16 GPUs.
        return wait_tensor(all_to_all_single(local_vectors_QD, None, None, row_mesh))

    @staticmethod
    def spmd_typecheck(outputs, *, weight_RD, local_row_ids_Q):
        # The weight is a DTensor; the returned vectors differ on every rank.
        rules.ignore(weight_RD)
        rules.ignore(local_row_ids_Q)
        spmd.assert_local_type(
            outputs, {axis: spmd.V for axis in spmd.get_local_type(local_row_ids_Q)}
        )

    @staticmethod
    def backward(ctx, grad_returned_QD: torch.Tensor):  # pyrefly: ignore [bad-override]
        (local_row_ids_Q,) = ctx.saved_tensors
        grad_local_vectors_QD = wait_tensor(
            all_to_all_single(grad_returned_QD.contiguous(), None, None, ctx.row_mesh)
        )
        # The same reduction F.embedding's backward runs, in the weight's dtype.
        # Unused request slots carry zero gradient.
        grad_local_weight_RD = torch.ops.aten.embedding_backward(
            grad_local_vectors_QD.to(ctx.weight_dtype),
            local_row_ids_Q,
            ctx.num_local_rows,
            -1,
            False,
            False,
        )
        grad_weight_RD = DTensor.from_local(
            grad_local_weight_RD,
            ctx.weight_mesh,
            ctx.weight_placements,
            shape=ctx.weight_shape,
            stride=ctx.weight_stride,
            run_check=False,
        )
        return grad_weight_RD, None, None, None


def row_sharded_embedding(
    weight_RD: DTensor,
    metadata: RowShardedEmbeddingMetadata,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Look up owned rows and return vectors to the original requesters.

    ``weight_RD`` is a DTensor sharded ``Shard(0)`` on the row axis that
    ``metadata.row_mesh`` routes over; its other mesh axes (e.g. TP feature
    columns) stay local, so D is the local feature width. The output has shape
    ``[*metadata.input_ids.shape, D]`` and uses ``dtype``
    (the weight's dtype by default). Duplicate IDs are expanded on the requester
    with ``F.embedding`` so their gradients accumulate in the compute dtype,
    as in an ordinary mixed-precision embedding. Owners then sum the requester
    contributions in the master weight's dtype, without mesh-size division.
    ``_OwnerLookup`` spells out the only backward collective.

    The caller owns parameter sharding and FSDP exclusion. The gradient keeps
    the weight's placements. Weights remain fixed until the ordinary optimizer
    step.
    """
    returned_vectors_QD = _OwnerLookup.apply(
        weight_RD, metadata.local_row_ids, metadata.row_mesh, dtype
    )
    return F.embedding(metadata.inverse_indices, returned_vectors_QD).reshape(
        *metadata.input_ids.shape, returned_vectors_QD.shape[-1]
    )
