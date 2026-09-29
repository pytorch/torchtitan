# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig

from .common_utils import annotate_graph_trainer_model, apply_simple_fsdp
from .compile import apply_compile
from .configs import GraphTrainerCompileConfig
from .ep_eager_chunk import maybe_apply_ep_overlap_eager_chunking
from .simple_fsdp import disable_active_parametrization


class GraphTrainerModel:
    """Model behavior shared by GraphTrainer model implementations."""

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: GraphTrainerCompileConfig,
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
        skip_dp: bool = False,
    ):
        del ac_config
        if skip_dp:
            raise ValueError("GraphTrainer models do not support skip_dp=True.")
        if (
            training.num_tokens_per_microbatch_per_dp_rank
            % parallelism_context.seq_len_divisor
            != 0
        ):
            raise ValueError(
                "Token count "
                f"{training.num_tokens_per_microbatch_per_dp_rank} must be "
                "divisible by the sequence sharding degree "
                f"{parallelism_context.seq_len_divisor}."
            )

        annotate_graph_trainer_model(self)
        self._parallelize(parallelism_context)
        model = apply_simple_fsdp(
            self,
            parallelism_context=parallelism_context,
            training=training,
        )
        maybe_apply_ep_overlap_eager_chunking(model, compile_config)
        return apply_compile(
            model,
            compile_config=compile_config,
            parallelism_context=parallelism_context,
        )

    def pipeline(self, **kwargs: Any):
        from .graph_pp.pipeline import graph_pipeline_llm

        return graph_pipeline_llm(self, **kwargs)
