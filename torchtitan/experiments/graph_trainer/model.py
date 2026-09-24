# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
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
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: GraphTrainerCompileConfig,
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ):
        del ac_config
        if (
            training.num_tokens_per_microbatch_per_dp_rank
            % parallel_dims.seq_len_divisor
            != 0
        ):
            raise ValueError(
                "Token count "
                f"{training.num_tokens_per_microbatch_per_dp_rank} must be "
                "divisible by the sequence sharding degree "
                f"{parallel_dims.seq_len_divisor}."
            )

        annotate_graph_trainer_model(self)
        self._parallelize(parallel_dims)
        model = apply_simple_fsdp(
            self,
            parallel_dims=parallel_dims,
            training=training,
        )
        maybe_apply_ep_overlap_eager_chunking(model, compile_config)
        return apply_compile(
            model,
            compile_config=compile_config,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
        )

    def pipeline(self, **kwargs: Any):
        compile_config = kwargs["compile_config"]
        if compile_config.mode is None:
            return super().pipeline(**kwargs)

        from .graph_pp.pipeline import graph_pipeline_llm

        return graph_pipeline_llm(self, **kwargs)
