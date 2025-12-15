# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _ContextParallel,
    _enable_context_parallel_dispatcher,
)
from torch.distributed.tensor.parallel import parallelize_module

from torchtitan.tools.logging import logger


def apply_cp(
    model: nn.Module,
    cp_mesh: DeviceMesh,
    use_flex_attn: bool,
) -> None:
    """
    Apply context parallelism to the model.

    Context Parallelism (CP) splits the sequence dimension across devices to enable
    training with longer sequences. This function applies CP to the attention modules
    of all transformer blocks in the model.

    Args:
        model: The transformer model with layers containing attention modules
        cp_mesh: Device mesh for context parallel dimension
        use_flex_attn: Whether the model uses FlexAttention (True) or SDPA (False)

    Note:
        - For FlexAttention: CP plan uses FLEX attention type
        - For SDPA: Enables CP dispatcher and uses SDPA attention type
        - Applies to transformer_block.attention.inner_attention for each layer
    """
    # Apply context parallelism to every transformer block
    # TODO: make seq_sim configurable once the implementation doesn't assume 2
    # internally.
    if use_flex_attn:
        cp_plan = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
        )
    else:
        # This is currently required as DTensor dispatcher is not enabled to
        # dispatch SDPA to CP implementation. We don't disable the CP
        # dispatching in TorchTitan as it is not needed. But there is a
        # corresponding API, _disable_context_parallel_dispatcher to do
        # that if users have this use case.
        _enable_context_parallel_dispatcher()
        cp_plan = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
        )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.attention.inner_attention,
            device_mesh=cp_mesh,
            parallelize_plan=cp_plan,
        )

    logger.info("Applied Context Parallel to the model")
