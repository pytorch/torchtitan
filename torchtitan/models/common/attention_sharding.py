# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Context-parallel sharding for the inner attention module.

``set_inner_attention_config`` installs the ``ShardingConfig`` on an attention
block's inner attention, dispatched on the context-parallel method:

  - all-gather (default): K/V redistribute to Replicate@CP, so each rank's
    seq-sharded Q attends to the full-length K/V.
  - ulysses: an all-to-all turns seq-sharded q/k/v into head-sharded,
    full-sequence tensors around the kernel (and the output back), so the kernel
    runs standard attention over the full sequence for this rank's heads.

Both only change the kernel's ShardingConfig; the kernel forward is unchanged.

Inner attention is also a local_map region because most attention kernels don't
have the sharding propagation rules implemented.

Tensor legend (see attention.py):
q/k/v are ``_BLNH`` -- (batch, seq_len, num_heads, head_dim).
"""

import spmd_types as spmd

from torchtitan.distributed.context_parallel import ContextParallelMethod
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout
from torchtitan.tools.logging import logger

DP, CP, TP = MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP


def _allgather_inner_sharding() -> ShardingConfig:
    """CP with all-gathered K/V. Q stays seq-sharded.

    Correct attention needs the full K and V, so the sharding config
    redistributes them (spmd.S(1)@CP -> spmd.R@CP). Their gradients are Partial.
    """

    q = dense_activation_placement(tp=spmd.S(2))
    kv_dst = dense_activation_placement(tp=spmd.S(2), cp=spmd.R)
    kv_grad = dense_activation_placement(tp=spmd.S(2), cp=spmd.P)
    return ShardingConfig(
        in_src_shardings={"q_BLNH": q, "k_BLNH": q, "v_BLNH": q},
        in_dst_shardings={"q_BLNH": q, "k_BLNH": kv_dst, "v_BLNH": kv_dst},
        out_src_shardings=q,
        local_map=LocalMapConfig(in_grad_placements=(q, kv_grad, kv_grad)),
    )


def _ulysses_inner_sharding() -> ShardingConfig:
    """DeepSpeed-Ulysses style CP. seq-sharding -> head-sharding.

    The sharding config reshards Q, K, and V from sequence-sharded to
    head-sharded (spmd.S(1)@CP -> spmd.S(2)@CP). partition_spec is required
    because TP also uses spmd.S(2).
    """

    seq = SpmdLayout(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=(DP, CP, TP, None),
    )
    head = SpmdLayout(
        {DP: spmd.V, CP: spmd.V, TP: spmd.V},
        partition_spec=(DP, None, (TP, CP), None),
    )
    return ShardingConfig(
        in_src_shardings={"q_BLNH": seq, "k_BLNH": seq, "v_BLNH": seq},
        in_dst_shardings={"q_BLNH": head, "k_BLNH": head, "v_BLNH": head},
        out_src_shardings=head,
        out_dst_shardings=seq,
        local_map=LocalMapConfig(in_grad_placements=(head, head, head)),
    )


_INNER_SHARDING_BUILDERS = {
    ContextParallelMethod.ALLGATHER: _allgather_inner_sharding,
    ContextParallelMethod.ULYSSES: _ulysses_inner_sharding,
}


def set_inner_attention_config(attention_cfg, cp_method: str = "") -> None:
    """Install the CP ShardingConfig on ``attention_cfg.inner_attention``.

    ``cp_method`` defaults to all-gather when empty.
    """
    method = ContextParallelMethod(cp_method or "allgather")
    attention_cfg.inner_attention.sharding_config = _INNER_SHARDING_BUILDERS[method]()


def _validate_context_parallel(
    attention_cfg,
    cp_method: str,
    *,
    tp: int,
    cp: int,
    spmd_backend: str,
    load_balancer: str | None,
) -> None:
    """Validate the context-parallel method against config and mesh degrees.

    Only ``ulysses`` has constraints; other methods return early. Ulysses:
      - takes effect only under ``spmd_backend`` ``full_dtensor``/``spmd_types``;
        the default backend ignores the ShardingConfig it relies on.
      - shards the head dim on the TP and CP axes, so ``n_heads`` and
        ``n_kv_heads`` must both be divisible by ``tp * cp``.
      - ignores ``load_balancer``; a non-default balancer other than ``None``
        triggers a warning.
    """
    if not cp_method:
        return
    if ContextParallelMethod(cp_method) is not ContextParallelMethod.ULYSSES:
        return

    if spmd_backend not in ("full_dtensor", "spmd_types"):
        raise ValueError(
            "context_parallel_method='ulysses' takes effect only under "
            "parallelism.spmd_backend 'full_dtensor' or 'spmd_types'. The "
            f"{spmd_backend} backend ignores it: SDPA silently falls back to "
            "all-gather CP and FlexAttention sees a mask whose length does not "
            "match the sharded query."
        )

    div = tp * cp
    n_heads = attention_cfg.n_heads
    # MLA-style attention (e.g. DeepSeek V3) has no n_kv_heads; fall back to
    # n_heads (GQA with n_kv_heads=None means MHA, also n_heads).
    n_kv_heads = getattr(attention_cfg, "n_kv_heads", None)
    if n_kv_heads is None:
        n_kv_heads = n_heads
    if n_heads % div or n_kv_heads % div:
        raise ValueError(
            "Ulysses context parallel shards the head dim on the TP and CP axes, "
            f"so n_heads ({n_heads}) and n_kv_heads ({n_kv_heads}) must both be "
            f"divisible by tensor_parallel_degree * context_parallel_degree "
            f"({tp} * {cp} = {div})."
        )

    # headtail is the config default; only an explicit non-default balancer,
    # which the user clearly intended, is worth a warning.
    if load_balancer is not None and load_balancer != "headtail":
        logger.warning(
            "context_parallel_load_balancer='%s' is ignored when "
            "context_parallel_method='ulysses': head-sharded attention has no "
            "per-rank sequence imbalance to balance.",
            load_balancer,
        )


def validate_context_parallel(model, parallel_dims, parallelism) -> None:
    """Validate the context-parallel config for a decoder model.

    No-op unless CP is enabled and the model has attention; otherwise checks the
    first attention config against the parallelism settings.
    """
    if not parallel_dims.cp_enabled or model.config.first_attention is None:
        return
    _validate_context_parallel(
        model.config.first_attention,
        parallelism.context_parallel_method,
        tp=parallelism.tensor_parallel_degree,
        cp=parallelism.context_parallel_degree,
        spmd_backend=parallelism.spmd_backend,
        load_balancer=parallelism.context_parallel_load_balancer,
    )
