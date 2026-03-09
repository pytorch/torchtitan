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
import types

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
)

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims

logger = logging.getLogger(__name__)


def _vllm_allreduce(dtensor: DTensor) -> DTensor:
    """Replace DTensor's NCCL allreduce with vLLM's custom allreduce.

    Takes a Partial DTensor, extracts the local tensor, runs vLLM's
    custom allreduce (P2P shared-memory on same-node GPUs), and wraps
    the result back as a Replicate DTensor.
    """
    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    local = dtensor.to_local()
    reduced = tp_group.all_reduce(local)
    return DTensor.from_local(
        reduced,
        device_mesh=dtensor.device_mesh,
        placements=[Replicate()],
    )


def _replicate_norm_weights(module: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Convert norm parameters to Replicate DTensors.

    This avoids the mixed Tensor/DTensor error when norm receives DTensor
    activations, without using SequenceParallel (which would also reshard
    activations on the sequence dimension and introduce alltoall collectives).
    """
    for name, param in module.named_parameters():
        if not isinstance(param, DTensor):
            dtensor_param = nn.Parameter(
                DTensor.from_local(
                    param.data, device_mesh=tp_mesh, placements=[Replicate()]
                ),
                requires_grad=param.requires_grad,
            )
            # Walk to the owning submodule and set the leaf parameter
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = module.get_submodule(parts[0])
                setattr(parent, parts[1], dtensor_param)
            else:
                setattr(module, name, dtensor_param)


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
    TODO: Change to core torchtitan's Qwen3 parallel plan when full DTensor is ready

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

    Uses Replicate() layouts for activations (no sequence parallelism) so that
    norms and other non-sharded ops stay in DTensor land consistently.
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
                use_local_output=False,
            ),
        },
    )

    # Apply tensor parallelism to every transformer block.
    # Activations flow as Replicate() DTensors between layers (no SP).
    #
    # When has_position_id=True (vLLM inference), wo/w2 output Partial() and
    # we manually call vLLM's custom P2P allreduce instead of DTensor's NCCL
    # allreduce. For training (has_position_id=False), wo/w2 output Replicate()
    # which triggers DTensor's built-in NCCL allreduce.
    if has_position_id:
        attention_module_plan = PrepareModuleInput(
            input_layouts=(Replicate(), Replicate(), None, Replicate()),
            desired_input_layouts=(Replicate(), Replicate(), None, Replicate()),
        )
        wo_output_layouts = Partial()
        w2_output_layouts = Partial()
    else:
        attention_module_plan = PrepareModuleInput(
            input_layouts=(Replicate(), Replicate(), None, None),
            desired_input_layouts=(Replicate(), Replicate(), None, None),
        )
        wo_output_layouts = Replicate()
        w2_output_layouts = Replicate()

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention": attention_module_plan,
            "attention.wq": ColwiseParallel(use_local_output=False),
            "attention.wk": ColwiseParallel(use_local_output=False),
            "attention.wv": ColwiseParallel(use_local_output=False),
            "attention.wo": RowwiseParallel(
                output_layouts=wo_output_layouts,
                use_local_output=False,
            ),
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": PrepareModuleInput(
                        input_layouts=(Replicate(),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": ColwiseParallel(use_local_output=False),
                    "feed_forward.w2": RowwiseParallel(
                        output_layouts=w2_output_layouts, use_local_output=False
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

    # When using vLLM inference, monkey-patch TransformerBlock.forward to
    # use vLLM's custom P2P allreduce on the Partial DTensor outputs from
    # wo and w2, instead of DTensor's NCCL allreduce.
    if has_position_id:
        for transformer_block in model.layers.values():

            def _patched_block_forward(
                self, x, freqs_cis, attention_masks, positions=None
            ):
                attn_out = self.attention(
                    self.attention_norm(x),
                    freqs_cis,
                    attention_masks,
                    positions,
                )
                x = x + _vllm_allreduce(attn_out)
                if self.moe_enabled:
                    x = x + _vllm_allreduce(self.moe(self.ffn_norm(x)))
                else:
                    x = x + _vllm_allreduce(
                        self.feed_forward(self.ffn_norm(x))
                    )
                return x

            transformer_block.forward = types.MethodType(
                _patched_block_forward, transformer_block
            )

        logger.info(
            "Patched TransformerBlock.forward to use vLLM custom allreduce"
        )

    # Convert any remaining plain tensor params (norms, etc.) to Replicate
    # DTensors so they're compatible with DTensor activations.
    _replicate_norm_weights(model, tp_mesh)
