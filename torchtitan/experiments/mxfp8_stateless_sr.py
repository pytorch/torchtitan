# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A single-microbatch CUDA-graph demonstration of stateless MXFP8 rounding."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.func._random as stateless_random

from torchtitan.distributed.cuda_graph import (
    NUM_CUDA_GRAPH_WARMUP_STEPS,
    wrap_fwd_bwd_with_cuda_graph,
)
from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan.trainer import Trainer
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


class StatelessSRTrainingEngine(TrainingEngine):
    """Give each MXFP8 linear a new stateless SR key on every graph invocation."""

    def _initialize_forward_backward(self) -> None:
        self._stateless_sr_linears = [
            module
            for model_part in self.model_parts
            for module in model_part.modules()
            if isinstance(module, MXFP8Linear)
            and module.grad_output_qdata_rounding_mode == "stochastic"
        ]
        self._sr_key_cursor = stateless_random.key(
            self.config.debug.seed + torch.distributed.get_rank(), device=self.device
        )
        graph_forward_backward = wrap_fwd_bwd_with_cuda_graph(
            partial(
                self._forward_backward_with_key,
                defer_fsdp_gradient_reduction=(
                    self.config.parallelism.fsdp_defer_gradient_reduction
                ),
            ),
            parameters=(
                parameter
                for model_part in self.model_parts
                for parameter in model_part.parameters()
            ),
            num_warmup_iterations=NUM_CUDA_GRAPH_WARMUP_STEPS,
        )

        def run_forward_backward(
            microbatch_groups: list[tuple[Any, ...]],
            global_loss_token_counts: torch.Tensor,
        ) -> ForwardBackwardResult:
            # This runs outside the main CUDA graph on every call and replay.
            self._sr_key_cursor, graph_key = stateless_random.split(self._sr_key_cursor)
            return graph_forward_backward(
                microbatch_groups, global_loss_token_counts, graph_key
            )

        self._run_forward_backward = run_forward_backward

    def _forward_backward_with_key(
        self,
        microbatch_groups: list[tuple[Any, ...]],
        global_loss_token_counts: torch.Tensor,
        graph_key: torch.Tensor,
        *,
        defer_fsdp_gradient_reduction: bool,
    ) -> ForwardBackwardResult:
        linear_keys = stateless_random.split(graph_key, len(self._stateless_sr_linears))
        for linear, linear_key in zip(
            self._stateless_sr_linears, linear_keys, strict=True
        ):
            linear.grad_output_random_key = linear_key
        return self._forward_backward_body(
            microbatch_groups,
            global_loss_token_counts,
            defer_fsdp_gradient_reduction=defer_fsdp_gradient_reduction,
        )


class StatelessSRTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    engine_cls = StatelessSRTrainingEngine
