# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port

from torchtitan.distributed.pipeline_parallel import (
    PipelineSharedParameter,
    SharedParameterPipelineRuntime,
)


pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
]


class _EmbeddingStage(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(4, 8, device="cuda")
        with torch.no_grad():
            self.tok_embeddings.weight.fill_(value)


def _run_shared_parameter_lifecycle(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        pp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("pp",))
        stage = _EmbeddingStage(value=float(rank + 2))
        parameter = stage.tok_embeddings.weight
        runtime = SharedParameterPipelineRuntime(
            model_parts=[stage],
            stage_indices=(rank,),
            pp_mesh=pp_mesh,
            pp_schedule="1F1B",
            num_stages=2,
            shared_parameters=(
                PipelineSharedParameter(
                    fqn="tok_embeddings.weight",
                    stage_indices=(0, 1),
                ),
            ),
        )

        runtime.synchronize_parameters()
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2.0))

        parameter.grad = torch.full_like(parameter, 3.0 + 2.0 * rank)
        runtime.finalize_gradients()
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 8.0))

        norm_parameters = runtime.parameters_for_grad_norm((parameter,))
        assert len(norm_parameters) == (1 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


def test_pipeline_shared_parameter_real_collectives() -> None:
    mp.spawn(
        _run_shared_parameter_lifecycle,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )
