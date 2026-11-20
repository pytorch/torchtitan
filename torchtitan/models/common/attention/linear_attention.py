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
    """Attention Gym chunk-kernel options: the fused kernels for every input. Takes `q` for the TODO's cuDNN choice."""
    # TODO: use backend="cudnn" again for packed fp16/bf16 on SM100/SM103 (GDN fwd+bwd ~2x faster on GB300) once a
    # model using it scores a token the same in a sequence and in that token's own prefix: one 78k-token sequence
    # broke this, random inputs don't. No Attention Gym issue filed yet. KDA's cuDNN kernel shares the vendored code.
    return KernelOptions(backend="fused")
