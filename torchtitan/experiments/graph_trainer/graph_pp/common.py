# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common graph PP building blocks shared by autoparallel and graph_trainer."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model import BaseModel


class ModelWithLoss(nn.Module):
    """Wraps a stage model with a loss function so that loss computation
    is included in the compiled graph callables for GraphPP."""

    def __init__(self, model: nn.Module, loss_fn: Callable) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        output = self.model(h)
        return self.loss_fn(output, labels)

    def init_weights(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self.model, "init_weights"):
            self.model.init_weights(*args, **kwargs)


def get_input_generating_fns(
    model_config: BaseModel.Config,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
) -> tuple[Callable, Callable, Callable, Callable]:
    """Create tracing input functions for each pipeline stage type.

    Returns:
        Tuple of (first_stage_fn, intermediate_stage_fn, last_stage_fn, target_fn)
    """

    def make_input_fn(
        batch_size: int,
        inp_type: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Callable:
        def input_fn() -> torch.Tensor:
            if inp_type == "tokens":
                return torch.randint(
                    0,
                    model_config.vocab_size,
                    (batch_size, training.seq_len),
                    dtype=dtype,
                    device=device,
                )
            elif inp_type == "embeddings":
                return torch.randn(
                    (batch_size, training.seq_len, model_config.dim),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            elif inp_type == "logits":
                return torch.randn(
                    (batch_size, training.seq_len, model_config.vocab_size),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            elif inp_type == "loss":
                return torch.scalar_tensor(
                    1.0,
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                )
            else:
                raise ValueError(f"Unknown input type: {inp_type}")

        return input_fn

    dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    spmd_batch_size = microbatch_size * dp_degree

    device = torch.device("cuda")

    tracing_target_fn = make_input_fn(spmd_batch_size, "tokens", torch.int64, device)
    tracing_input_fn_first_stage = make_input_fn(
        spmd_batch_size, "tokens", torch.int64, device
    )
    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    tracing_input_fn_intermediate_stage = make_input_fn(
        spmd_batch_size, "embeddings", param_dtype, device
    )

    def tracing_input_fn_last_stage():
        return (
            tracing_input_fn_intermediate_stage(),
            tracing_target_fn(),
        )

    return (
        tracing_input_fn_first_stage,
        tracing_input_fn_intermediate_stage,
        tracing_input_fn_last_stage,
        tracing_target_fn,
    )


def get_shape_inference_fns(
    model_config: BaseModel.Config,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    has_loss: bool,
) -> tuple[Callable, Callable, Callable]:
    """Create shape inference functions returning meta-device tensors.

    Used by GraphPipelineStage constructor for input_args / output_args
    to infer inter-stage activation shapes.

    Returns:
        Tuple of (first_stage_input_fn, intermediate_fn, last_stage_output_fn)
    """
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    meta_device = torch.device("meta")

    def first_stage_input():
        return torch.randint(
            0,
            model_config.vocab_size,
            (microbatch_size, training.seq_len),
            device=meta_device,
        )

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]

    def intermediate_stage():
        return torch.randn(
            (microbatch_size, training.seq_len, model_config.dim),
            device=meta_device,
            dtype=param_dtype,
        )

    def last_stage_output():
        if has_loss:
            return torch.scalar_tensor(
                1.0,
                dtype=torch.float32,
                device=meta_device,
            )
        else:
            return torch.randn(
                (microbatch_size, training.seq_len, model_config.vocab_size),
                device=meta_device,
                dtype=param_dtype,
            )

    return first_stage_input, intermediate_stage, last_stage_output
