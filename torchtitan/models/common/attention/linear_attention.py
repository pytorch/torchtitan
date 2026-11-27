# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pieces shared by the linear-attention families (Gated DeltaNet and KDA).

Family modules (``gdn.py``, ``kda.py``) import from here and never from each
other; their context-parallel variants share ``cp_linear_attention.py``.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.export import register_dataclass

from torchtitan.models.common.attn_gym import (
    _ConfiguredShortConv,
    _L2Norm,
    _ShortConv,
    is_missing_kernel,
    KernelOptions,
)

from .attention import Attention, VarlenAttentionMetadata

# The shared Attention Gym kernels run on rank-local heads inside each family's
# local SPMD region. Family modules register their family-specific kernels.
for _kernel_function in (_ShortConv, _ConfiguredShortConv, _L2Norm):
    if not is_missing_kernel(_kernel_function):
        spmd.register_local_autograd_function(_kernel_function)


class LinearAttention(Attention):
    """Base class for recurrent linear attention layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        pass


@dataclass(frozen=True, slots=True)
class LinearAttentionMetadata:
    """Sequence metadata shared by convolutional linear-attention backends."""

    varlen: VarlenAttentionMetadata | None
    num_conv_history_tokens: int

    def annotate_spmd_types(self) -> None:
        """Annotate sequence offsets."""
        if self.varlen is not None:
            self.varlen.annotate_spmd_types()


# Expose tensor fields to pytree so CUDA graph replay copies fresh offsets.
register_dataclass(LinearAttentionMetadata)


def chunk_kernel_options(q: torch.Tensor) -> KernelOptions:
    """Attention Gym chunk-kernel options for `q`: the fused kernels, for every input."""
    # TODO: bring back backend="warpdrive" once it is causally consistent at long packed lengths. Under its old name
    # "cudnn" it was not (Qwen3.8-27B, 78k tokens, pos 562: full -1.91, prefix -10.35; fused -0.0146), though its GDN
    # fwd+bwd was 1.9-2.1x faster than fused on GB300. KDA's is unchecked. No Attention Gym issue yet: file one with
    # this repro and link it here.
    del q
    return KernelOptions(backend="fused")
