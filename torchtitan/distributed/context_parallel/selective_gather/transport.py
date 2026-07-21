# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Transport context for the selective K/V gather.

Resolves the CP ``ProcessGroup`` (or ``DeviceMesh`` + axis) and holds the static
config the backends need. The ``"p2p"`` backend uses only these plain attributes
-- it sends over ``batch_isend_irecv`` on the group and needs no registered
buffers.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from .backend import select_backend, SUPPORTED_BACKENDS


def _resolve_group(group, mesh_axis):
    """Normalize a ProcessGroup or (DeviceMesh, axis) to a single ProcessGroup.

    Per the distributed rules, never assume a 1D mesh: require the axis name
    explicitly and assert it exists.
    """
    if isinstance(group, DeviceMesh):
        if mesh_axis is None:
            raise ValueError(
                "A DeviceMesh needs an explicit mesh_axis naming the CP axis "
                f"(available: {group.mesh_dim_names})."
            )
        if group.mesh_dim_names is None or mesh_axis not in group.mesh_dim_names:
            raise ValueError(
                f"mesh_axis {mesh_axis!r} not in mesh axes {group.mesh_dim_names}."
            )
        return group.get_group(mesh_axis)
    if isinstance(group, dist.ProcessGroup):
        if mesh_axis is not None:
            raise ValueError("mesh_axis is only valid when passing a DeviceMesh.")
        return group
    raise TypeError(
        f"group must be a ProcessGroup or DeviceMesh, got {type(group).__name__}."
    )


def _resolve_device(device: torch.device) -> torch.device:
    """Fill in the index of an indexless accelerator device.

    ``torch.device("cuda")`` and ``torch.device("cuda:0")`` do not compare equal,
    but a tensor built with the former reports the latter. The backends compare
    buffers against this device, so pin it to the current one here.
    """
    if device.index is None and device.type != "cpu":
        return torch.device(device.type, torch.accelerator.current_device_index())
    return device


class SelectiveGatherContext:
    """CP group + static config for a selective K/V gather.

    Args:
        group: the CP ``ProcessGroup``, or a ``DeviceMesh`` (pass ``mesh_axis``).
        mesh_axis: CP axis name; required iff ``group`` is a ``DeviceMesh``.
        shard_numel: elements in this rank's local K/V shard (fixed per run).
        block_numel: elements per gather block (transport granularity).
        dtype: K/V element dtype.
        device: device the shard lives on; an indexless one resolves to the
            current device.
        batch_size: batch elements packed into the shard (default 1).
        max_consumers: staging depth for the backward (from
            ``backward_staging_map``); unused by the p2p backend.
        backend: force one of ``SUPPORTED_BACKENDS``; ``None`` auto-selects via
            ``select_backend``.
    """

    def __init__(
        self,
        group,
        *,
        shard_numel: int,
        block_numel: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int = 1,
        max_consumers: int = 1,
        backend: str | None = None,
        mesh_axis: str | None = None,
    ):
        # shard_numel is the TOTAL per-rank shard (batch_size * per-batch shard).
        if shard_numel % (batch_size * block_numel) != 0:
            raise ValueError(
                f"shard_numel ({shard_numel}) must be a multiple of "
                f"batch_size*block_numel ({batch_size * block_numel})."
            )
        if backend is not None and backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported selective-gather backend {backend!r}; "
                f"supported: {', '.join(SUPPORTED_BACKENDS)}."
            )
        self.pg = _resolve_group(group, mesh_axis)
        self.device = _resolve_device(device)
        self.dtype = dtype
        self.block_numel = block_numel
        self.shard_numel = shard_numel
        self.batch_size = batch_size
        self.max_consumers = max_consumers
        self.cp_size = self.pg.size()
        self.cp_rank = self.pg.rank()
        # Plans hold group-local rank ids, so metadata is only valid against the
        # group it was gathered on. Global ranks identify that group.
        self.group_ranks = tuple(dist.get_process_group_ranks(self.pg))
        # Per-batch blocks this rank owns (B==1 -> shard_numel // block_numel).
        self.blocks_per_rank = shard_numel // (batch_size * block_numel)
        self.backend = backend or select_backend(self.device, self.pg)
        # The p2p backend uses only the plain attrs above (it sends over
        # batch_isend_irecv on self.pg); no registered windows / dev-comm.

    def close(self) -> None:
        """Release backend resources; the p2p backend holds none."""
        return
