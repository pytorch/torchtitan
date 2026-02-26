# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RL-specific parallelize function for vLLM-compatible TP plan.
# Applies tensor parallelism with DTensor-aware attention hooks
# (PrepareModuleInputOutput) so vLLM's flash attention kernels receive
# local tensors, and optionally FSDP for the trainer.

import logging

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
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
):
    """
    Apply tensor parallelism to the Qwen3 dense model for RL training/inference.

    NOTE: The function signature is intentionally simpler than core torchtitan's
    parallelize_qwen3 — it only accepts the configs needed for TP.

    Args:
        has_position_id: Whether position IDs are passed as an explicit argument
            to the attention module. True for vLLM inference (generator),
            False for training (trainer).
    """

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
            has_position_id=has_position_id,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
    has_position_id: bool = False,
):
    """Apply tensor parallelism to the Qwen3 dense model.

    This is a temporary TP plan used while we resolve composability issues in the
    main torchtitan codebase. Once DTensor is fully supported across the TP
    region, this separate plan should be removed.
    """

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
                use_local_output=True,  # return logits and plain tensor
            ),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
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
            "attention_norm": SequenceParallel(
                use_local_output=False,
            ),
            # NOTE: when the fourth argument (positions) is not None, its input layout
            # and desired input layout should be Replicate()
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
            # Apply on vllm.Attention() module to use local tensor
            "attention.inner_attention": PrepareModuleInputOutput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),  # xq, xk, xv
                desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
                use_local_input=True,  # use local tensor for attention calculation
                output_layouts=(Shard(1)),  # output
                desired_output_layouts=(Shard(1)),
                use_local_output=False,
            ),
            "attention.wo": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "ffn_norm": SequenceParallel(
                use_local_output=False,
            ),
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


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the Qwen3 model.

    Args:
        model: The model to apply data parallelism to.
        dp_mesh: The device mesh to use for data parallelism.
        param_dtype: The data type to use for model parameters.
        reduce_dtype: The data type to use for reduction operations.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to offload model parameters to CPU.
        reshard_after_forward_policy: The policy to use for resharding after forward pass.
            "default" applies default resharding behavior.
            "always" enables reshard_after_forward for all forward passes.
            "never" disables reshard_after_forward for all forward passes.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        # pyrefly: ignore[bad-typed-dict-key]
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.items():
        # pyrefly: ignore[no-matching-overload]
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    # pyrefly: ignore[no-matching-overload]
    fully_shard(model, **fsdp_config)
