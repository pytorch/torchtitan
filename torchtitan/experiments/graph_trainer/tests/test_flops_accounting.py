# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

import torchtitan.experiments.graph_trainer.trainer as graph_trainer_module

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.trainer import (
    GraphTrainer,
    GraphTrainingEngine,
)
from torchtitan.protocols import BaseModel


class _GraphBody(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.weight

    def _apply_fsdp(self, **kwargs) -> None:
        pass

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return input_dict["input"].float(), input_dict["labels"].float(), {}


def _loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    return (prediction - labels).square().sum() / global_valid_tokens


def _batch(index: int) -> TokenizedTrainingMicrobatch:
    return TokenizedTrainingMicrobatch(
        input=torch.full((1,), index + 1, dtype=torch.int64),
        labels=torch.ones(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        padding_mask=torch.zeros(1, dtype=torch.bool),
        num_valid_tokens=1,
        model_kwargs={"batch_index": index},
    )


def test_graph_trainer_estimates_raw_batches_outside_trace_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    replay_compute_calls = 0

    def flops_estimator(raw_batch) -> int:
        batch_index = raw_batch["batch_index"]
        assert raw_batch["input"].device.type == "cpu"
        events.append(("estimate", batch_index))
        return batch_index + 1

    model = _GraphBody()
    engine = cast(GraphTrainingEngine, object.__new__(GraphTrainingEngine))
    engine.model_parts = [model]
    engine.loss_fn = _loss
    engine.flops_estimator = flops_estimator
    engine.num_completed_steps = 0
    engine.ntokens_seen = 0
    engine.num_accumulation_steps = 2
    engine.device = torch.device("cpu")
    engine.parallel_dims = SimpleNamespace(
        pp_enabled=False,
        dp_enabled=False,
        dp_cp_enabled=False,
        dp_replicate_enabled=False,
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        ep_enabled=False,
        get_optional_mesh=lambda name: None,
    )
    engine.train_context = nullcontext
    engine.prepare_step = MagicMock(side_effect=lambda value, **kwargs: value)

    def replay_forward_backward(compute_forward_backward, **kwargs):
        nonlocal replay_compute_calls
        compute_forward_backward()
        replay_compute_calls += 1
        loss = compute_forward_backward()
        replay_compute_calls += 1
        return loss

    engine.sdc_replayer = SimpleNamespace(
        run_fwd_bwd=MagicMock(side_effect=replay_forward_backward)
    )
    engine.lr_schedulers = SimpleNamespace(get_metrics=MagicMock(return_value={}))
    engine.optimizer_step = MagicMock(return_value=torch.tensor(2.0))
    engine._traced_step = None
    engine._graph_runner = None
    engine._trainable_params = None
    engine._graph_gradient_state = None
    engine.estimate_flops = MagicMock(wraps=engine.estimate_flops)

    compile_config = GraphTrainerCompileConfig(
        mode="aot_fx_trace",
        enable_passes=False,
    )
    metrics_processor = SimpleNamespace(
        should_log=MagicMock(return_value=False),
        log=MagicMock(),
    )
    trainer = cast(GraphTrainer, object.__new__(GraphTrainer))
    trainer.engine = engine
    trainer.config = SimpleNamespace(
        compile=compile_config,
        training=SimpleNamespace(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=17,
        ),
        parallelism=SimpleNamespace(),
        debug=SimpleNamespace(spmd_typechecking=False),
    )
    engine.config = trainer.config
    trainer.gradient_accumulation_steps = 2
    trainer.num_pp_microbatches = 1
    trainer.metrics_processor = metrics_processor
    trainer._local_num_flops_since_last_log = 0

    trace_builder = MagicMock(wraps=graph_trainer_module.minimal_fx_tracer)
    monkeypatch.setattr(
        graph_trainer_module.dist_utils,
        "get_spmd_context",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(graph_trainer_module, "minimal_fx_tracer", trace_builder)

    GraphTrainer.train_step(trainer, iter([_batch(0), _batch(1)]))

    assert engine.estimate_flops.call_count == 2
    assert events == [("estimate", 0), ("estimate", 1)]
    assert replay_compute_calls == 2
    engine.sdc_replayer.run_fwd_bwd.assert_called_once()
    trace_builder.assert_called_once()
    assert trainer._local_num_flops_since_last_log == 3

    assert engine._traced_step is not None
    generated_graph = engine._traced_step.gm
    assert "estimate_flops" not in generated_graph.code
    assert all(
        "estimate_flops" not in str(node.target) for node in generated_graph.graph.nodes
    )
