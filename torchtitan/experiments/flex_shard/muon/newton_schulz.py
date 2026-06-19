# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Newton-Schulz primitives + ownership cost model (NS FLOPs, LPT balancing)."""

from __future__ import annotations

import heapq

import torch
import torch.nn as nn


def newton_schulz_flops(global_shape: torch.Size) -> int:
    """Relative Newton-Schulz cost of one 2D matrix: ``m * n * min(m, n)``.

    Each Newton-Schulz iteration forms the Gram matrix ``X @ X.T`` (an ``[m, m]``
    product contracting all ``n`` columns), so cost scales with
    ``m * n * min(m, n)`` rather than the parameter count ``m * n``. This is the
    quantity to balance across owner ranks, since an owner runs NS only on the
    matrices it owns. The constant factor (iteration count) is shared by all
    matrices and irrelevant to balancing, so it is dropped.
    """
    m, n = int(global_shape[-2]), int(global_shape[-1])
    return m * n * min(m, n)


def layer_newton_schulz_cost(named_params: list[tuple[str, nn.Parameter]]) -> int:
    """Total Newton-Schulz cost of a layer's 2D (Muon-eligible) matrices.

    Non-2D params (norms, biases) carry no NS cost; they ride along on the
    owner under AdamW and contribute negligible step time.
    """
    return sum(
        newton_schulz_flops(param.shape) for _, param in named_params if param.ndim == 2
    )


def assign_layer_owners_lpt(layer_costs: list[int], world_size: int) -> list[int]:
    """Assign one owner rank per layer, balancing total cost via greedy LPT.

    Sorts layers by cost descending and repeatedly gives the heaviest remaining
    layer to the least-loaded rank (a min-heap over ``(load, rank)``; ties broken
    by rank index, so every rank computes the identical assignment -- required so
    broadcast sources agree). The makespan is within 4/3 of optimal; residual
    imbalance is bounded by the largest single layer, which cannot be split while
    it stays one ``Owned`` bucket.

    Returns ``owners`` where ``owners[i]`` is the owner rank of the ``i``-th layer
    (same order as ``layer_costs``).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    loads = [(0, rank) for rank in range(world_size)]
    heapq.heapify(loads)
    owners = [0] * len(layer_costs)
    for layer_idx in sorted(
        range(len(layer_costs)),
        key=lambda i: layer_costs[i],
        reverse=True,
    ):
        load, rank = heapq.heappop(loads)
        owners[layer_idx] = rank
        heapq.heappush(loads, (load + layer_costs[layer_idx], rank))
    return owners


def _zeropower_via_newtonschulz_batched(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """Batched Newton-Schulz over a stack of same-shape matrices ``[K, m, n]``.

    Mathematically identical to applying :func:`_zeropower_via_newtonschulz` to each
    ``[m, n]`` matrix independently, but replaces the per-matrix ``addmm`` with one
    ``baddbmm`` over the batch -- the same FLOPs in far fewer kernel launches. Since
    the batch dimension is independent, each slice's result matches the per-matrix
    computation (same bf16 path, same coefficients).
    """
    if grad.ndim != 3:
        raise ValueError("Batched Newton-Schulz expects a 3D [K, m, n] stack.")
    a, b, c = ns_coefficients
    ortho = grad.bfloat16()
    # Same-shape group -> the transpose decision is uniform across the batch.
    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.transpose(-2, -1)
    # Per-matrix spectral-norm bound (Frobenius >= spectral norm).
    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm
    for _ in range(ns_steps):
        gram = ortho @ ortho.transpose(-2, -1)
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.transpose(-2, -1)
    return ortho
