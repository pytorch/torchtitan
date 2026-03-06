# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RL-specific parallelize function for vLLM-compatible TP plan.
# Applies tensor parallelism so vLLM's flash attention kernels receive
# local tensors (DTensor unwrap/wrap is handled in the attention modules),
# and optionally FSDP for the trainer.

import logging

import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims

logger = logging.getLogger(__name__)


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    has_position_id: bool = False,
    enable_sp: bool = True,
):
    """
    Apply tensor parallelism to the Qwen3 dense model for RL training/inference.

    NOTE: The function signature is intentionally simpler than core torchtitan's
    parallelize_qwen3 — it only accepts the configs needed for TP.
    TODO: Change to core torchtitan's Qwen3 parallel plan when full DTensor is ready

    Args:
        has_position_id: Whether position IDs are passed as an explicit argument
            to the attention module. True for vLLM inference (generator),
            False for training (trainer).
        enable_sp: Whether to enable sequence parallelism on top of tensor
            parallelism. When False, only tensor parallelism is applied
            (activations are Replicate across TP ranks instead of Shard on
            the sequence dimension). Defaults to True.
    """

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        if enable_sp:
            _apply_non_moe_tp_sp(
                model,
                tp_mesh,
                has_position_id=has_position_id,
            )
        else:
            _apply_tp_only(
                model,
                tp_mesh,
                has_position_id=has_position_id,
            )

    return model


def _apply_non_moe_tp_sp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    has_position_id: bool = False,
):
    """Apply tensor parallelism with sequence parallelism."""

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "norm": SequenceParallel(
                use_local_output=False,
            ),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    if has_position_id:
        attention_module_plan = PrepareModuleInput(
            input_layouts=(Shard(1), Replicate(), None, Replicate()),
            desired_input_layouts=(Replicate(), Replicate(), None, Replicate()),
        )
    else:
        attention_module_plan = PrepareModuleInput(
            input_layouts=(Shard(1), Replicate(), None, None),
            desired_input_layouts=(Replicate(), Replicate(), None, None),
        )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(use_local_output=False),
            "attention": attention_module_plan,
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(
                sequence_dim=2,
                use_local_output=False,
            ),
            "attention.k_norm": SequenceParallel(
                sequence_dim=2,
                use_local_output=False,
            ),
            "attention.wo": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "ffn_norm": SequenceParallel(use_local_output=False),
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": ColwiseParallel(use_local_output=False),
                    "feed_forward.w2": RowwiseParallel(
                        output_layouts=Shard(1), use_local_output=False
                    ),
                    "feed_forward.w3": ColwiseParallel(use_local_output=False),
                }
            )
        else:
            raise ValueError(
                "Running vLLM inference with torchtitan Qwen3 MoE model is not supported yet."
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism with Sequence Parallelism to the model")


def _apply_tp_only(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    has_position_id: bool = False,
):
    """Apply tensor parallelism without sequence parallelism.

    Activations stay Replicate across TP ranks. RowwiseParallel performs
    all-reduce (instead of reduce-scatter) so the output is Replicate.
    """

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=False,
            ),
            "output": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
                use_local_output=True,
            ),
        },
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.q_norm": SequenceParallel(
                sequence_dim=2,
                use_local_output=False,
            ),
            "attention.k_norm": SequenceParallel(
                sequence_dim=2,
                use_local_output=False,
            ),
            "attention.wo": RowwiseParallel(
                output_layouts=Replicate(),
                use_local_output=False,
            ),
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward.w1": ColwiseParallel(use_local_output=False),
                    "feed_forward.w2": RowwiseParallel(
                        output_layouts=Replicate(), use_local_output=False
                    ),
                    "feed_forward.w3": ColwiseParallel(use_local_output=False),
                }
            )
        else:
            raise ValueError(
                "Running vLLM inference with torchtitan Qwen3 MoE model is not supported yet."
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism (TP-only, no SP) to the model")
