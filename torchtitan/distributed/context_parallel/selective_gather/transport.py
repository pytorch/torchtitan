# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Transport context for selective K/V gather over the NCCL device API.

Owns the NCCL resources a selective gather needs -- the adopted communicator and
the registered windows -- and exposes a small interface the kernel drives. The
context is built from a user-supplied ``ProcessGroup`` or ``DeviceMesh`` (+ axis),
and reuses PyTorch's own ``ncclComm_t`` rather than standing up a second
communicator.

Memory model: persistent, fixed-size, NCCL-registered windows per rank, allocated
once and reused every step:

  * ``shard_window`` -- this rank's local K/V shard; peers read from it.
  * ``gathered_window`` -- full-sequence destination, registered as GIN-prep; the
    LSA forward writes the caller's plain ``out`` buffer instead.
  * ``signal_window`` -- readiness signal pad (forward epoch + backward slots).
  * ``grad_stage_window`` -- per-consumer backward grad staging.

``gathered_window`` is registered even though the LSA forward writes plain local
memory, because GIN pushes (put) between registered windows -- registering it now
lets the same window set serve the intra-node (LSA) and inter-node (GIN) legs.

The LSA/GIN transport is not PyTorch-native (nccl4py + CuTeDSL); the portable
``"p2p"`` backend keeps the package usable on hosts without them, using only
the plain attributes above and no registered buffers.
"""

import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from .backend import select_backend, SUPPORTED_BACKENDS
from .topology import PlanMetadata

# nccl4py (nccl.core) is imported lazily inside __init__ only for the "lsa"
# backend, so the package stays importable on hosts without nccl4py/CuTeDSL
# (which fall back to the portable "p2p" backend).


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


def _register_window(comm, buf, what: str):
    """Register a window, rejecting a platform that cannot provide one.

    ``register_window`` returns None where device windows are unsupported, so
    reading ``.is_valid`` straight off it would raise ``AttributeError``.
    """
    window = comm.register_window(buf)
    if window is None or not window.is_valid:
        raise RuntimeError(f"NCCL window registration failed ({what}).")
    return window


def _resolve_device(device: torch.device) -> torch.device:
    """Fill in the index of an indexless accelerator device.

    ``torch.device("cuda")`` and ``torch.device("cuda:0")`` do not compare equal,
    but a tensor built with the former reports the latter. The backends compare
    buffers against this device, so pin it to the current one here.
    """
    if device.index is None and device.type != "cpu":
        return torch.device(device.type, torch.accelerator.current_device_index())
    return device


def check_ctx_meta(
    ctx, meta: PlanMetadata, shard: torch.Tensor, full: torch.Tensor
) -> None:
    """Guard the affine reshapes and the buffers the P2P ops are posted on.

    Shared by every backend, so they enforce one contract. ``shard`` holds one
    rank's blocks and ``full`` the whole gathered sequence (in the backward,
    their gradients). A context that disagrees with the plan
    would silently misalign the gather. In particular ``blocks_per_rank`` bounds
    what ``build_plan_metadata`` accepted as a block id, so a context with fewer
    blocks would index past its own shard. A dtype or device disagreement would
    post receives that do not match what the peers send, which fails or hangs
    the job somewhere else instead of raising here.
    """
    plan = meta.plan
    if meta.group_ranks != ctx.group_ranks:
        raise ValueError(
            f"metadata was gathered on ranks {meta.group_ranks} but ctx runs on "
            f"{ctx.group_ranks}; its local rank ids name different peers."
        )
    if meta.blocks_per_rank != ctx.blocks_per_rank:
        raise ValueError(
            f"metadata was validated for {meta.blocks_per_rank} blocks per rank "
            f"but ctx.blocks_per_rank is {ctx.blocks_per_rank}."
        )
    if plan.block_numel != ctx.block_numel:
        raise ValueError(
            f"plan.block_numel {plan.block_numel} != ctx.block_numel "
            f"{ctx.block_numel}."
        )
    if plan.batch_size != ctx.batch_size:
        raise ValueError(
            f"plan.batch_size {plan.batch_size} != ctx.batch_size " f"{ctx.batch_size}."
        )
    if shard.dtype != ctx.dtype or full.dtype != ctx.dtype:
        raise ValueError(
            f"buffer dtypes {shard.dtype}/{full.dtype} != ctx.dtype {ctx.dtype}."
        )
    if shard.device != ctx.device or full.device != ctx.device:
        raise ValueError(
            f"buffers are on {shard.device}/{full.device}, not ctx.device "
            f"{ctx.device}."
        )
    if shard.numel() != ctx.shard_numel:
        raise ValueError(
            f"shard has {shard.numel()} elements, not ctx.shard_numel "
            f"{ctx.shard_numel}."
        )
    expected_full = ctx.cp_size * ctx.shard_numel
    if full.numel() != expected_full:
        raise ValueError(
            f"gathered buffer has {full.numel()} elements, not "
            f"cp_size*shard_numel {expected_full}."
        )


class SelectiveGatherContext:
    """NCCL resources + windows for a selective K/V gather over a CP group.

    Args:
        group: the CP ``ProcessGroup``, or a ``DeviceMesh`` (pass ``mesh_axis``).
        mesh_axis: CP axis name; required iff ``group`` is a ``DeviceMesh``.
        shard_numel: elements in this rank's local K/V shard (fixed per run).
        block_numel: elements per gather block (transport granularity).
        dtype: K/V element dtype.
        device: device the shard and any registered windows live on; an
            indexless one resolves to the current device.
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

        # Pick the backend; the "p2p" fallback needs none of the nccl4py
        # resources below (it uses batch_isend_irecv over the plain group).
        self.backend = backend or select_backend(
            self.device, self.pg, dtype=dtype, block_numel=block_numel
        )
        if self.backend == "p2p":
            return
        import nccl.core as nccl

        # The device kernels index registered buffers with int32 offsets, so a
        # buffer whose byte extent exceeds ~2GB would silently overflow. Fail
        # fast here (setup-time, no per-step cost); int64 offsets in the kernels
        # are the follow-up for genuinely >2GB long-context shards.
        itemsize = torch.empty((), dtype=dtype).element_size()
        largest_words = max(
            2 * shard_numel, self.cp_size * shard_numel, max_consumers * shard_numel
        )
        if largest_words * itemsize > 2**31 - 1:
            raise ValueError(
                f"a registered buffer is {largest_words * itemsize} bytes, over "
                "the ~2GB int32-offset limit of the CuTeDSL kernels; reduce "
                "shard_numel or block size (int64 offsets are a follow-up)."
            )

        # Adopt PyTorch's ncclComm_t for this group. It is created lazily on
        # first collective use, so force it, then wrap the raw pointer without
        # taking ownership (PyTorch still creates/destroys it).
        dist.all_reduce(torch.zeros(1, device=self.device), group=self.pg)
        torch.cuda.synchronize()
        comm_ptr = self.pg._get_backend(self.device)._comm_ptr()
        if comm_ptr == 0:
            raise RuntimeError("PyTorch NCCL communicator not created for this group.")
        comm = nccl.Communicator(comm_ptr)

        # The hostname probe in select_backend is only a pre-filter. The
        # communicator is what actually decides: lsa_pointer addresses a peer
        # inside the local LSA team, so the whole group has to be one team, and
        # without device-API support there are no device windows at all. Both
        # come from the shared communicator, so every rank reads the same
        # values and falls back together.
        unusable = None
        if not comm.device_api_support:
            unusable = "the platform has no NCCL device-API support"
        elif comm.n_lsa_teams != 1:
            unusable = f"the group spans {comm.n_lsa_teams} LSA teams, not 1"
        if unusable is not None:
            if backend is not None:
                raise ValueError(f"backend 'lsa' was forced but {unusable}.")
            warnings.warn(
                f"selective gather: falling back to 'p2p' because {unusable}.",
                stacklevel=2,
            )
            self.backend = "p2p"
            return
        self.comm = comm

        # Persistent, fixed-size, registered windows (see module docstring).
        # Window registration is collective, so every rank registers all of them
        # in the same order.
        # shard_buf is double-sized: the forward alternates halves by step parity
        # so a peer still reading last step's half is not overwritten.
        self.shard_buf = nccl.torch.empty(2 * shard_numel, dtype=dtype)
        self.gathered_buf = nccl.torch.empty(self.cp_size * shard_numel, dtype=dtype)
        self.shard_window = _register_window(self.comm, self.shard_buf, "shard")
        self.gathered_window = _register_window(
            self.comm, self.gathered_buf, "gathered"
        )

        # Per-peer readiness signal pad. Slot 0 holds this rank's current epoch;
        # peers read it via lsa_pointer to know this rank's shard is written for a
        # given step, replacing a host barrier. A one-time barrier below makes the
        # zeroed pad visible cross-GPU before any gather.
        # Signal slots (int32): 0 = forward epoch (sig_write / reverse-ack),
        # 1 = backward "staging ready", 2 = backward "grads pushed, done".
        self.signal_buf = nccl.torch.empty(1024, dtype=torch.int32)  # 4KB
        self.signal_buf.zero_()
        self.signal_window = _register_window(self.comm, self.signal_buf, "signal")
        self._signal_step = 0
        self._bwd_step = 0

        # Backward staging: each remote consumer writes (plain vectorized copy, no
        # atomics) its grad contribution for this rank's shard into its own slot;
        # the owner then reduces the slots in FP32 registers. Same K/V dtype as
        # the shard -- FP32 accumulation happens in the reduce kernel.
        self.grad_stage = nccl.torch.empty(max_consumers * shard_numel, dtype=dtype)
        self.grad_stage_window = _register_window(
            self.comm, self.grad_stage, "grad staging"
        )

        # One-time: make the zeroed signal pad globally visible before any gather
        # reads a peer's epoch. This barrier is NOT on the per-gather path.
        torch.cuda.synchronize()
        dist.barrier(group=self.pg)

    def next_signal_step(self) -> int:
        """Monotonic epoch for the signal pad; call once per gather."""
        self._signal_step += 1
        return self._signal_step

    def next_bwd_step(self) -> int:
        """Monotonic epoch for the backward signal slots; once per backward."""
        self._bwd_step += 1
        return self._bwd_step

    def load_shard_half(self, kv_shard: torch.Tensor, half: int) -> None:
        """Copy-in into shard-buffer half ``half`` (0 or 1) for double-buffering."""
        flat = kv_shard.reshape(-1)
        if flat.numel() != self.shard_numel:
            raise ValueError(
                f"kv_shard has {flat.numel()} elements, expected {self.shard_numel}."
            )
        base = half * self.shard_numel
        self.shard_buf[base : base + self.shard_numel].copy_(flat)

    def close(self) -> None:
        """Release the resources we created; the ncclComm_t belongs to PyTorch."""
        if self.backend == "p2p":
            return  # no nccl4py resources were created
        self.shard_window.close()
        self.gathered_window.close()
        self.signal_window.close()
        self.grad_stage_window.close()
