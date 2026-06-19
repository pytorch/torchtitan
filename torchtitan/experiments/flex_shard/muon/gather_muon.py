# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Gather-for-NS Muon optimizers (2D dense + 3D Shard(>=1) experts).

The Muon optimizers for the "this rank holds only a slice -> all-gather the full tensor
before Newton-Schulz" case: :class:`GatherMuon` (sharded 2D dense matrices) and its
subclass :class:`GatherGroupedMuon` (3D expert stacks sharded within the matrix). The
routing that selects them lives in :mod:`containers`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim._muon import (
    _adjust_lr,
    _zeropower_via_newtonschulz,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
)

from ..example.ragged_shard import GroupedRaggedShard
from .newton_schulz import _zeropower_via_newtonschulz_batched


class GatherMuon(torch.optim.Optimizer):
    """Muon for FSDP/RaggedShard-sharded 2D matrices via all-gather before NS.

    Each step, per bucket: update the sharded momentum (element-wise), all-gather the
    pre-NS update to reconstruct each full matrix, run Newton-Schulz on the full
    matrix, and write back only this rank's shard. Bit-exact with single-device Muon;
    cost is one all-gather/bucket plus redundant NS on every rank.
    """

    def __init__(
        self,
        buckets: list[dict[str, Any]],
        mesh: DeviceMesh,
        *,
        lr: float = 0.02,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        self._mesh = mesh
        self._buckets = buckets
        params: list[nn.Parameter] = []
        seen: set[int] = set()
        for bucket in buckets:
            for param in bucket["params"]:
                if param.numel() > 0 and id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
        if not params:
            raise ValueError("GatherMuon found no sharded 2D matrices on this rank.")
        # TODO(checkpoint): this builds a flat param group with no "param_names" key, so
        # DCP save/load fails -- torchtitan/components/checkpoint_utils.py:140
        # (_optim_state_dict_to_fqn_keys) raises when a group lacks "param_names". To make
        # GatherMuon checkpointable, thread the per-param canonical FQNs through here and
        # add them as a "param_names" entry (see how FlexShardGatherMuonOptimizers builds
        # adamw_names for the AdamW group).
        super().__init__(
            params,
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_coefficients": ns_coefficients,
                "eps": eps,
                "ns_steps": ns_steps,
                "adjust_lr_fn": adjust_lr_fn,
            },
        )

    @staticmethod
    def _local_update_shard(full_update, placement, info, rank, world_size):
        """This rank's shard of the full Newton-Schulz update."""
        if isinstance(placement, GroupedRaggedShard):
            # Byte-balanced cut crosses param boundaries: slice the flat full matrix
            # at this rank's offset within the param.
            layout = info.bucket_layout.param_layouts[info.fqn]
            start = layout.local_global_offset - layout.param_offset
            return full_update.reshape(-1)[start : start + info.local_numel].view(
                info.local_shape
            )
        return placement.extract_local_shard(full_update, rank, world_size)

    @staticmethod
    def _orthogonalize_full(full_pre, ns_coefficients, ns_steps, eps, global_shape):
        """Orthogonalize one reconstructed full 2D matrix; returns (ortho, lr_shape).

        Overridden by :class:`GatherGroupedMuon` for 3D ``[E, m, n]`` expert stacks
        (batched per-expert NS, lr scaled by the per-expert ``(m, n)`` shape).
        """
        return (
            _zeropower_via_newtonschulz(full_pre, ns_coefficients, ns_steps, eps),
            global_shape,
        )

    @torch.no_grad()
    def step(self, closure=None):
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
            # buffer so the placement can view it as a contiguous bucket slice; the
            # momentum buffer stays sharded (its update is element-wise).
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
                    raise RuntimeError("GatherMuon does not support sparse gradients")
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
                full_update, lr_shape = self._orthogonalize_full(
                    full_pre, ns_coefficients, ns_steps, eps, info.global_shape
                )
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)
                update_shard = self._local_update_shard(
                    full_update, placement, info, rank, world_size
                )
                param.mul_(1 - lr * weight_decay)
                param.add_(update_shard, alpha=-adjusted_lr)
        return loss


class GatherGroupedMuon(GatherMuon):
    """GatherMuon for 3D MoE expert stacks sharded *within* the matrix (``Shard(dim>=1)``).

    The ``Shard(1)`` counterpart to comm-efficient :class:`GroupedMuon`: when ``world_size >
    num_experts`` each rank holds only a slice of every expert matrix, so per-expert
    Newton-Schulz needs the full matrix. Reuses GatherMuon's gather/scatter unchanged
    (all-gather each expert stack to ``[E, m, n]`` over the efsdp mesh, write back this
    rank's ``[E, m_local, n]`` shard); only the orthogonalization differs -- one *batched*
    NS over the reconstructed ``[E, m, n]`` (per-expert), with lr scaled by the per-expert
    ``(m, n)`` shape. Cost: one all-gather/bucket + redundant batched NS per rank -- the
    communication comm-efficient GroupedMuon avoids when experts are whole (``Shard(0)``).
    """

    @staticmethod
    def _orthogonalize_full(full_pre, ns_coefficients, ns_steps, eps, global_shape):
        # full_pre is the reconstructed [E, m, n]; batched per-expert NS; lr uses (m, n).
        return (
            _zeropower_via_newtonschulz_batched(
                full_pre, ns_coefficients, ns_steps, eps
            ),
            global_shape[1:],
        )
