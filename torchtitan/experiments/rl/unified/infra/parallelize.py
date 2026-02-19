# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.


import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConvertersContainer


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Temporary helper to apply tensor parallelism to the Qwen3 dense model so vLLM can run the torchtitan model.
    """

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
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
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(
                use_local_output=False,
            ),
            # NOTE: when the fourth argument (positions) is not None, its input layout
            # and desired input layout should be Replicate()
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), Replicate(), None, Replicate()),
                desired_input_layouts=(Replicate(), Replicate(), None, Replicate()),
            ),
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
                desired_input_layouts=(None, None, None),
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
