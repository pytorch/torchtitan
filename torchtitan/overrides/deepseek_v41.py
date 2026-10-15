# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in V4.1 reference replacements for the TorchTitan config tree.

The ``reference_attention`` override swaps ``CompressedSparseInnerAttention2``
for ``ReferenceCSA2`` (dense eager attention with stop-gradient indexer
teacher). Since V4.1's ``CompressedSparseInnerAttention2.forward`` was
already rewritten to use pure PyTorch einsum (no ``gather_attn`` dependency),
this override is currently an identity -- it exists so that future fused
implementations can replace it.

Future overrides (not yet implemented):
  - ``dense_engram``: single-rank dense Engram lookup
  - ``sinkhorn_optimizer``: SinkhornMomentum optimizer container
"""

from torchtitan.config import derive, override
from torchtitan.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2


@override(
    target=CompressedSparseInnerAttention2.Config,
    exact=True,
    fqns=["model.layers.*.attention.inner_attention"],
    description="Reference CSA2 with stop-gradient indexer teacher (identity for now)",
)
def reference_attention(
    cfg: CompressedSparseInnerAttention2.Config,
) -> CompressedSparseInnerAttention2.Config:
    """Identity override: the base class already uses the reference path."""
    return cfg
