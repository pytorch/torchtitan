# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.loss import LossFunction
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.llama3.config_registry import llama3_debugmodel


class _PipelineTestModel(nn.Module):
    def __init__(self, *, num_layers: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            vocab_size,
            hidden_dim,
            dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleDict(
            {
                str(i): nn.Linear(
                    hidden_dim,
                    hidden_dim,
                    dtype=torch.bfloat16,
                )
                for i in range(num_layers)
            }
        )
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(
            hidden_dim,
            vocab_size,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tok_embeddings is not None:
            x = self.tok_embeddings(x)
        for layer in self.layers.values():
            x = layer(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.lm_head is not None:
            x = self.lm_head(x)
        return x


def _loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = output.float().square().mean() + target.float().sum() * 0
    return loss, {}


def _run_pipeline_rank(rank: int) -> None:
    config = llama3_debugmodel()
    config.training.local_batch_size = 4
    config.training.seq_len = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.pipeline_parallel_microbatch_size = 1
    assert config.model_spec is not None
    model_config = config.model_spec.model
    assert isinstance(model_config, Decoder.Config)

    parallel_dims = ParallelDims.from_config(
        config.parallelism,
        world_size=dist.get_world_size(),
    )
    with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
        parallel_dims.build_mesh()
    model = _PipelineTestModel(
        num_layers=len(model_config.layers),
        hidden_dim=model_config.dim,
        vocab_size=model_config.vocab_size,
    )

    schedule, _, has_first_stage, has_last_stage = pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=config.compile,
        ac_config=config.activation_checkpoint,
        dump_folder=config.dump_folder,
        device=torch.device("cpu"),
        model_config=model_config,
        parallelize_fn=lambda model, **_: model,
        loss_fn=cast(LossFunction, _loss_fn),
    )
    num_microbatches = (
        config.training.local_batch_size
        // config.parallelism.pipeline_parallel_microbatch_size
    )
    arg_mbs = (
        [
            (
                torch.randint(
                    model_config.vocab_size,
                    (1, config.training.seq_len),
                ),
            )
            for _ in range(num_microbatches)
        ]
        if has_first_stage
        else None
    )
    target_mbs = (
        [
            torch.zeros(
                (1, config.training.seq_len),
                dtype=torch.int64,
            )
            for _ in range(num_microbatches)
        ]
        if has_last_stage
        else None
    )
    losses = [] if has_last_stage else None

    schedule.step(
        arg_mbs=arg_mbs,
        target_mbs=target_mbs,
        losses=losses,
        return_outputs=False,
    )

    assert has_first_stage is (rank == 0)
    assert has_last_stage is (rank == 1)
    if losses is not None:
        assert len(losses) == num_microbatches


def test_fake_pipeline_llm_per_rank() -> None:
    for rank in (0, 1):
        try:
            dist.init_process_group("fake", rank=rank, world_size=2)
            _run_pipeline_rank(rank)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
