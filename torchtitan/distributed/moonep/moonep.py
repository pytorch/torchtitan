# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP primitives for MoE expert parallelism: one ``Buffer`` and one set of
NVLink expert pools per process, shared by every MoE layer."""

import inspect

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

try:
    import moonep
    from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
except ImportError as e:
    raise ImportError(
        "MoonEP (33327eb or later) is required for this module. Install from: "
        "https://github.com/MoonshotAI/MoonEP"
    ) from e

if (
    "local_gate_weight"
    not in inspect.signature(moonep.Buffer.prefetch_weight).parameters
):
    raise ImportError(
        "MoonEP 33327eb or later is required; this install's "
        "Buffer.prefetch_weight takes no local_gate_weight."
    )

_PROJECTIONS = ("gate", "up", "down")

_buffer: "moonep.Buffer | None" = None
_buffer_key: tuple | None = None
_pools: dict[tuple, torch.Tensor] = {}


def get_buffer(
    group: ProcessGroup,
    *,
    num_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
) -> "moonep.Buffer":
    """Get or create the process-global MoonEP ``Buffer``."""
    global _buffer, _buffer_key
    key = (group, num_tokens_per_rank, hidden, num_topk, num_experts)
    if _buffer is not None:
        if _buffer_key != key:
            raise ValueError(
                "MoonEP sizes its buffer statically; every MoE layer must share "
                "one token count, hidden size, top-k, expert count and EP group."
            )
        return _buffer
    # Never destroyed: destroy() syncs the device and barriers, which a CUDA-graph capture cannot take.
    _buffer = moonep.Buffer(
        S=num_tokens_per_rank,
        H=hidden,
        K=num_topk,
        E=num_experts,
        num_ep_ranks=group.size(),
        group=group,
        explicitly_destroy=True,
    )
    _buffer_key = key
    return _buffer


def current_buffer() -> tuple["moonep.Buffer", ProcessGroup]:
    """Return the ``Buffer`` that ``get_buffer`` created and its EP group."""
    if _buffer is None or _buffer_key is None:
        raise RuntimeError("get_buffer() must run before the first MoonEP dispatch.")
    return _buffer, _buffer_key[0]


def _pool(
    name: str,
    rows: int,
    row_shape: tuple[int, ...],
    dtype: torch.dtype,
    group: ProcessGroup,
) -> torch.Tensor:
    key = (name, rows, row_shape, dtype, group)
    pool = _pools.get(key)
    if pool is None:
        padded = pad_dim0_for_alignment([rows, *row_shape], dtype)
        size = group.size()
        pool = create_nvl_dist_tensor(
            [padded, *row_shape], dtype, dist.get_rank(group), size, group=group
        ).view(size, padded, *row_shape)[:, :rows]
        _pools[key] = pool
    return pool


def _pools_for(
    group: ProcessGroup,
    rows: int,
    shapes: dict[str, tuple[int, ...]],
    dtype: torch.dtype,
    kind: str,
) -> dict[str, torch.Tensor]:
    return {
        name: _pool(f"{kind}.{name}", rows, shapes[name], dtype, group)
        for name in _PROJECTIONS
    }


def allocate_pools(
    group: ProcessGroup, num_local_experts: int, shapes: dict[str, tuple[int, ...]]
) -> None:
    """Create the weight pools and the slot-gradient pools, a collective, before the first step."""
    _pools_for(group, 2 * num_local_experts, shapes, torch.bfloat16, "weight")
    _pools_for(group, num_local_experts, shapes, torch.float32, "grad")


def _projections(
    w13_e2FD: torch.Tensor, w2_eDF: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {"gate": w13_e2FD[:, 0], "up": w13_e2FD[:, 1], "down": w2_eDF}


def prefetch_rows(
    buffer: "moonep.Buffer",
    plan: object,
    group: ProcessGroup,
    w13_e2FD: torch.Tensor,
    w2_eDF: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fill this rank's rows and slots for ``plan``; return the ``[2e]`` GEMM rows."""
    num_local_experts = w2_eDF.shape[0]
    local = _projections(w13_e2FD, w2_eDF)
    shapes = {name: tuple(w.shape[1:]) for name, w in local.items()}
    pools = _pools_for(group, 2 * num_local_experts, shapes, torch.bfloat16, "weight")
    rank = dist.get_rank(group)
    with torch.no_grad():
        for name in _PROJECTIONS:
            pools[name][rank, :num_local_experts].copy_(local[name])
    # Prefetch has only a trailing barrier: a dispatch or combine must separate it from the last GEMM over the pools.
    buffer.prefetch_weight(
        plan=plan,
        **{
            f"local_{name}_weight": pools[name][rank, :num_local_experts]
            for name in _PROJECTIONS
        },
        **{
            f"{name}_prefetch_buffer": pools[name][:, num_local_experts:]
            for name in _PROJECTIONS
        },
    )
    return {name: pools[name][rank] for name in _PROJECTIONS}


def reduce_rows(
    buffer: "moonep.Buffer",
    plan: object,
    group: ProcessGroup,
    row_grads: dict[str, torch.Tensor],
) -> tuple[torch.cuda.Event, dict[str, torch.Tensor]]:
    """Start sending slot gradients home on MoonEP's stream; return its event and this
    rank's fp32 expert gradients, which are complete once the event is."""
    num_local_experts = row_grads["down"].shape[0] // 2
    shapes = {name: tuple(g.shape[1:]) for name, g in row_grads.items()}
    slots = _pools_for(group, num_local_experts, shapes, torch.float32, "grad")
    rank = dist.get_rank(group)
    local = {}
    for name in _PROJECTIONS:
        slots[name][rank].copy_(row_grads[name][num_local_experts:])
        local[name] = row_grads[name][:num_local_experts].float()
    # reduce_grad reads the peers' slot gradients without a barrier; every rank's writes must land first.
    dist.all_reduce(torch.zeros(1, device=row_grads["down"].device), group=group)
    done = buffer.reduce_grad(
        plan=plan,
        async_finish=True,
        **{f"local_{name}_grad": local[name] for name in _PROJECTIONS},
        **{f"{name}_reduce_buffer": slots[name] for name in _PROJECTIONS},
    )
    return done, local
