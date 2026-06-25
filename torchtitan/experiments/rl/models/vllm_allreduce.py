# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Swap the tensor-parallel all-reduce in the generator forward from DTensor's
NCCL ring redistribute to vLLM's custom one-shot/multimem all-reduce.

Background. The unified TorchTitan model shards the attention output projection
``wo`` and the FFN output projection ``w2`` ``RowwiseParallel``: the weight is
sharded on the contraction dim and the input activation arrives sharded on its
last dim, so the local matmul produces a ``Partial`` (per-rank partial sum) that
DTensor reduces to ``Replicate`` with an NCCL ring all-reduce. vLLM's native
model instead reduces the same partial with a custom one-shot/multimem
all-reduce, which is markedly lower latency for the small messages decode emits.
Profiling the generator put that NCCL redistribute at a large fraction of TP
inference GPU time, and it is also a cross-rank arrival-spin source.

This module replaces the ``Partial -> Replicate`` step on every attention
``wo`` and dense feed-forward ``w2`` instance with
``vllm.distributed.tensor_model_parallel_all_reduce`` while leaving the rest of
the model on DTensor. MoE routed experts (EP all-to-all combine) and MoE
``shared_experts`` (whose ``w2`` keeps a deferred ``Partial`` reduced at the MoE
boundary) are left untouched. It binds the replacement forward per-instance
(``module.forward = MethodType(...)``); it does NOT monkeypatch the shared
``BaseAttention`` / ``FeedForward`` / ``Linear`` classes, so training and other
model paths are unaffected. Generator-only and inference-only (no autograd, so
``to_local`` needs no ``grad_placements``).
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger


logger = init_logger(__name__)


def _vllm_allreduce_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Drop-in ``nn.Linear.forward`` for a row-parallel (contraction-sharded)
    output projection: local matmul -> vLLM all-reduce -> ``Replicate``.

    The input ``x`` is sharded on its last (contraction) dim, so the per-rank
    ``F.linear`` is a partial sum; ``tensor_model_parallel_all_reduce`` reduces
    it across the TP group (vLLM's custom AR rather than DTensor's NCCL ring),
    and the replicated result is rewrapped as a ``Replicate`` DTensor so the
    residual stream stays a DTensor for the next op.
    """
    x_local = x.to_local() if isinstance(x, DTensor) else x
    w = self.weight
    w_local = w.to_local() if isinstance(w, DTensor) else w
    y = F.linear(x_local, w_local)
    y = tensor_model_parallel_all_reduce(y)
    if self.bias is not None:
        b = self.bias
        y = y + (b.to_local() if isinstance(b, DTensor) else b)
    if isinstance(x, DTensor):
        mesh = x.device_mesh
        y = DTensor.from_local(y, mesh, [Replicate()] * mesh.ndim)
    return y


def apply_vllm_allreduce(model: nn.Module) -> int:
    """Route every attention ``wo`` and dense feed-forward ``w2`` output
    projection through vLLM's custom all-reduce instead of DTensor's NCCL
    ``Partial -> Replicate`` redistribute.

    Attention is matched on the shared :class:`BaseAttention` base, so it covers
    every decoder model -- ``GQAttention`` (qwen3) plus the per-model
    ``Attention`` subclasses (deepseek_v3 / gpt_oss / qwen3_5). All route their
    ``wo`` / dense ``w2`` through the shared ``rowwise_config`` (weight
    contraction-sharded, output reduced to ``Replicate`` at the projection), so
    the per-instance swap preserves that reduction contract.

    Correct under the generator's attention-DP + expert-EP layout
    (``ep == dp * tp``): ``wo`` / ``w2`` are sharded only on the TP axis, so the
    swap reduces over vLLM's TP group exactly as the NCCL path did. The EP-folded
    routed experts use a separate all-to-all combine and are never matched here.

    MoE ``shared_experts`` are also ``FeedForward`` instances, but their rowwise
    ``w2`` output is intentionally kept ``Partial`` and reduced together with the
    routed-expert output at the MoE boundary (see ``_shared_expert_rowwise_config``).
    All-reducing it here would break that deferred-reduction contract, so they
    are excluded.

    Binds :func:`_vllm_allreduce_linear_forward` on each target instance; the
    shared module classes are left untouched. Returns the number of projections
    patched so the caller can sanity-check it matched the layer count.
    """
    # Imported here (not at module top) to avoid importing core model classes
    # before the generator has selected/replaced its layers.
    from torchtitan.models.common.attention import BaseAttention
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.common.moe import MoE

    # MoE shared experts are dense FeedForwards whose rowwise w2 output stays
    # Partial (reduced at the MoE boundary); exclude them from the swap.
    shared_expert_ids = {
        id(module.shared_experts)
        for module in model.modules()
        if isinstance(module, MoE) and module.shared_experts is not None
    }

    patched = 0
    for module in model.modules():
        if isinstance(module, BaseAttention):
            target = module.wo
        elif isinstance(module, FeedForward) and id(module) not in shared_expert_ids:
            target = module.w2
        else:
            continue
        target.forward = types.MethodType(_vllm_allreduce_linear_forward, target)
        patched += 1

    logger.info(
        "vllm_allreduce: routed %d output projections (attention wo + dense FFN "
        "w2) through vLLM custom all-reduce",
        patched,
    )
    return patched
