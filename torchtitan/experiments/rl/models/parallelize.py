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
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import ParallelismConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import apply_compile_sparse
from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import (
    ColwiseParallelWithGradPlacement,
    NoParallel,
)

logger = logging.getLogger(__name__)


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig | None = None,
    has_position_id: bool = False,
):
    """
    Apply tensor parallelism to the Qwen3 model (dense and MoE) for RL training/inference.

    NOTE: The function signature is intentionally simpler than core torchtitan's
    parallelize_qwen3 — it only accepts the configs needed for TP/EP.

    Args:
        compile_config: If provided and enabled, applies per-layer torch.compile
            after TP (matching the pattern in torchtitan/models/llama3/parallelize.py).
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

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
        )

    if (
        compile_config is not None
        and compile_config.enable
        and "model" in compile_config.components
    ):
        apply_compile_sparse(model, compile_config, parallel_dims.ep_enabled)

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
        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
):
    """Apply MoE parallelism for the RL experiment.

    This is a variant of llama4's apply_moe_ep_tp with use_local_output=False on
    the "moe" PrepareModuleInputOutput so the output stays as DTensor. This is
    required because the RL TP plan keeps all intermediate activations as DTensors
    (use_local_output=False), so `x + self.moe(self.ffn_norm(x))` needs both
    operands to be DTensors.
    """
    assert ep_mesh is not None or tp_mesh is not None

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                # use_local_output=False: keep output as DTensor to match
                # the residual x which is also a DTensor in the RL TP plan.
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                    use_local_output=False,
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            if ep_mesh is not None and etp_mesh is None:
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial(),
                        ),
                        "moe.shared_experts.w3": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                    }
                )
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_etp_mesh
            experts_plan = ExpertTensorParallel()

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
