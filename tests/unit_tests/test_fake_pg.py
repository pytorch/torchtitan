# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    ScheduleInterleaved1F1B,
)

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed._pipeline_compat import (
    ensure_fake_pg_static_metadata_support,
)
from torchtitan.distributed.pipeline_parallel import _build_fake_stage_args
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.llama3.config_registry import llama3_debugmodel


def test_fake_pipeline_stage_args() -> None:
    config = llama3_debugmodel()
    config.training.local_batch_size = 4
    config.training.seq_len = 16
    config.parallelism.pipeline_parallel_microbatch_size = 2
    assert config.model_spec is not None
    model_config = config.model_spec.model
    assert isinstance(model_config, Decoder.Config)
    loss_fn = config.loss.build(compile_config=config.compile)

    first_input, first_output = _build_fake_stage_args(
        0,
        4,
        training=config.training,
        parallelism=config.parallelism,
        model_config=model_config,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )
    assert first_input.shape == (2, 16)
    assert first_input.dtype == torch.int64
    assert first_output.shape == (2, 16, model_config.dim)
    assert first_output.dtype == torch.bfloat16

    last_input, last_output = _build_fake_stage_args(
        3,
        4,
        training=config.training,
        parallelism=config.parallelism,
        model_config=model_config,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )
    assert last_input.shape == (2, 16, model_config.dim)
    assert last_input.requires_grad
    assert last_output.shape == (2, 16, model_config.dim)
    assert last_output.requires_grad

    cross_entropy = CrossEntropyLoss.Config(
        global_vocab_size=model_config.vocab_size
    ).build(compile_config=config.compile)
    _, last_logits = _build_fake_stage_args(
        3,
        4,
        training=config.training,
        parallelism=config.parallelism,
        model_config=model_config,
        loss_fn=cross_entropy,
        device=torch.device("cpu"),
    )
    assert last_logits.shape == (2, 16, model_config.vocab_size)


def test_fake_pipeline_stage_args_reject_distributed_boundaries() -> None:
    config = llama3_debugmodel()
    config.parallelism.tensor_parallel_degree = 2
    assert config.model_spec is not None
    model_config = config.model_spec.model
    loss_fn = config.loss.build(compile_config=config.compile)

    with pytest.raises(
        ValueError,
        match="Fake pipeline execution currently requires",
    ):
        _build_fake_stage_args(
            0,
            4,
            training=config.training,
            parallelism=config.parallelism,
            model_config=model_config,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )


def test_fake_pipeline_backport_is_inert_with_upstream_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def upstream_warmup_p2p(*args):
        return (
            "Fake process group detected; set inference_mode=static "
            "for %d stage(s) without voting"
        )

    monkeypatch.setattr(_PipelineSchedule, "_warmup_p2p", upstream_warmup_p2p)

    ensure_fake_pg_static_metadata_support()

    assert _PipelineSchedule._warmup_p2p is upstream_warmup_p2p


@pytest.mark.parametrize(
    ("rank", "stage_indices"),
    [(0, (0, 2)), (1, (1, 3))],
)
def test_fake_pipeline_schedule_per_rank(
    rank: int,
    stage_indices: tuple[int, int],
) -> None:
    original_warmup_p2p = _PipelineSchedule._warmup_p2p
    try:
        ensure_fake_pg_static_metadata_support()
        patched_warmup_p2p = _PipelineSchedule._warmup_p2p
        ensure_fake_pg_static_metadata_support()
        assert _PipelineSchedule._warmup_p2p is patched_warmup_p2p

        dist.init_process_group("fake", rank=rank, world_size=2)
        stages = []
        for stage_idx in stage_indices:
            input_args = torch.empty(1, 4, requires_grad=True)
            output_args = torch.empty(1, 4, requires_grad=True)
            stages.append(
                PipelineStage(
                    torch.nn.Linear(4, 4),
                    stage_idx,
                    4,
                    torch.device("cpu"),
                    input_args=input_args,
                    output_args=output_args,
                    group=dist.group.WORLD,
                )
            )

        schedule = ScheduleInterleaved1F1B(
            stages,
            n_microbatches=4,
            loss_fn=lambda output, target: ((output - target) ** 2).sum(),
        )
        if rank == 0:
            schedule.step(torch.randn(4, 4), return_outputs=False)
        else:
            schedule.step(target=torch.randn(4, 4), return_outputs=False)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        _PipelineSchedule._warmup_p2p = original_warmup_p2p
