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
from torch.utils import _pytree as pytree

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
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


# Linear-attention metadata crosses compiler and whole-step CUDA graph
# boundaries. Expose its tensor fields so fresh sequence offsets are copied
# on replay.
pytree.register_dataclass(LinearAttentionMetadata)


def chunk_kernel_options(q: torch.Tensor) -> KernelOptions:
    """Attention Gym chunk-kernel options: cuDNN where it runs, fused elsewhere.

    The cuDNN chunk kernels take packed (batch 1) fp16/bf16 inputs on SM100/SM103, where
    they are 2.3-3.9x faster than the fused ones (Qwen3.5-27B GDN fwd+bwd on GB300).
    Fused stays in deterministic mode (cuDNN's backward is not checked for
    determinism) and in batch-invariant mode (paged KDA replay requires fused).
    """
    use_cudnn = (
        q.is_cuda
        and not torch.are_deterministic_algorithms_enabled()
        and not is_in_batch_invariant_mode()
        and q.shape[0] == 1
        and q.dtype in (torch.float16, torch.bfloat16)
        and torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
    )
    return KernelOptions(backend="cudnn" if use_cudnn else "fused")
