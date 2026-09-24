# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import override
from torchtitan.config.transform import LinearLoRAHandler, LoRATransform
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3 import model_registry
from torchtitan.training_engine import TrainingEngine


def _microbatch(num_valid_tokens: int = 1) -> TokenizedTrainingMicrobatch:
    return TokenizedTrainingMicrobatch(
        input=torch.ones(1),
        labels=torch.ones(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        padding_mask=torch.zeros(1, dtype=torch.bool),
        num_valid_tokens=num_valid_tokens,
    )


def _ft_metric_boundary_trainer(
    *,
    should_log: bool = False,
    gradient_accumulation_steps: int = 2,
    num_pp_microbatches: int = 3,
    num_tokens_per_microbatch: int = 11,
    dp_cp_enabled: bool = False,
    dp_degree: int = 1,
    cp_degree: int = 1,
    loss_sync_pg=None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    metrics_processor = SimpleNamespace(
        should_log=MagicMock(return_value=should_log),
        log=MagicMock(),
        ntokens_since_last_log=0,
        data_loading_times=[],
    )
    engine = SimpleNamespace(
        num_completed_steps=0,
        ntokens_seen=0,
        lr_schedulers=SimpleNamespace(
            schedulers=[SimpleNamespace(get_last_lr=MagicMock(return_value=[0.1]))]
        ),
        parallel_dims=SimpleNamespace(
            dp_enabled=False,
            dp_cp_enabled=dp_cp_enabled,
            dp_replicate=dp_degree,
            dp_shard=1,
            cp=cp_degree,
            tp=17,
            pp=19,
            get_optional_mesh=lambda name: None,
        ),
        device=torch.device("cpu"),
        estimate_flops=MagicMock(return_value=7),
        prepare_step=MagicMock(side_effect=lambda value, **kwargs: value),
        forward_backward_microbatch=MagicMock(return_value=torch.tensor(1.0)),
        optimizer_step=MagicMock(return_value=torch.tensor(2.0)),
        ft_manager=SimpleNamespace(loss_sync_pg=loss_sync_pg, group_size=23),
    )
    trainer = SimpleNamespace(
        engine=engine,
        config=SimpleNamespace(
            training=SimpleNamespace(
                num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            )
        ),
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_pp_microbatches=num_pp_microbatches,
        metrics_processor=metrics_processor,
        _local_num_flops_since_last_log=0,
    )
    return trainer, engine


def test_ft_applies_ffn_lora_override_before_model_build(monkeypatch):
    # Restore shared registration and distributed state after this test.
    monkeypatch.setattr(import_module("torchtitan.config.override"), "_REGISTRY", {})

    @override(target=FeedForward.Config)
    def ffn_lora(config):
        return LoRATransform(
            handlers=(LinearLoRAHandler(),), rank=1, alpha=1.0
        ).transform(config)

    config = ft.FaultTolerantTrainer.Config(
        model=model_registry("debugmodel"),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
    )
    config.override.imports = [f"{__name__}.ffn_lora"]

    def initialize_distributed_runtime(engine):
        engine.device = torch.device("cpu")
        engine.parallel_dims = ParallelDims.from_config(
            config.parallelism, world_size=1
        )
        engine.ft_manager = config.fault_tolerance.build()
        engine.gc_handler = None
        engine.device_memory_monitor = SimpleNamespace()

    class ModelBuildReachedError(Exception):
        """Stop FT initialization at the model-build boundary."""

    built_ffns = []

    def build_model(model_config):
        built_ffns.append(model_config.layers[0].feed_forward.build())
        raise ModelBuildReachedError

    # Bypass hardware/data setup, but keep config updates and FFN building real.
    monkeypatch.setattr(
        ft.FaultTolerantTrainingEngine,
        "_initialize_distributed_runtime",
        initialize_distributed_runtime,
    )
    monkeypatch.setattr(
        type(config.dataloader),
        "build",
        lambda self, **kwargs: SimpleNamespace(max_num_documents=None),
    )
    monkeypatch.setattr(
        type(config.metrics),
        "build",
        lambda self, **kwargs: SimpleNamespace(color=""),
    )
    monkeypatch.setattr(type(config.model), "build", build_model)

    with pytest.raises(ModelBuildReachedError):
        ft.FaultTolerantTrainer(config)

    assert hasattr(built_ffns[0].w13, "lora_a"), "FT ignored the FFN LoRA override"


def test_ft_trainer_composes_specialized_training_engine() -> None:
    assert not issubclass(ft.FaultTolerantTrainer, TrainingEngine)
    assert issubclass(ft.FaultTolerantTrainingEngine, TrainingEngine)


def test_ft_averages_logged_loss_and_flops_by_active_replica_count(monkeypatch):
    ft_pg = Mock(size=lambda: 2)
    trainer, engine = _ft_metric_boundary_trainer(
        should_log=True,
        gradient_accumulation_steps=1,
        num_pp_microbatches=2,
        dp_cp_enabled=True,
        loss_sync_pg=ft_pg,
    )
    trainer._local_num_flops_since_last_log = 100
    engine.estimate_flops.side_effect = [13, 17]
    engine.ntokens_seen = 4
    loss_mesh = object()
    engine.parallel_dims.get_optional_mesh = MagicMock(return_value=loss_mesh)
    engine.forward_backward_microbatch.return_value = torch.tensor(2.0)
    # Two active replicas contribute a loss sum of 4.0, despite group_size=23.
    monkeypatch.setattr(ft.dist_utils, "dist_sum", Mock(side_effect=[4.0, 8]))
    monkeypatch.setattr(ft.dist_utils, "dist_max", Mock(return_value=2.0))
    mean_flops = Mock(return_value=130.0)
    monkeypatch.setattr(ft.dist_utils, "dist_mean", mean_flops)
    monkeypatch.setattr(ft, "collect_aux_loss_metrics", Mock(return_value={}))

    ft.FaultTolerantTrainer.train_step(trainer, iter([_microbatch(4), _microbatch(4)]))

    trainer.metrics_processor.log.assert_called_once()
    _, logged_loss, *_ = trainer.metrics_processor.log.call_args.args
    assert logged_loss == 2.0
    assert trainer.metrics_processor.log.call_args.kwargs["num_flops"] == 130.0
    mean_flops.assert_called_once()
    local_num_flops_tensor = mean_flops.call_args.args[0]
    torch.testing.assert_close(
        local_num_flops_tensor,
        torch.tensor(130.0, dtype=torch.float64),
    )
    assert mean_flops.call_args.kwargs == {"mesh": loss_mesh, "extra_pg": ft_pg}
    assert trainer._local_num_flops_since_last_log == 0


def test_ft_engine_installs_all_reduce_hook_after_model_initialization() -> None:
    engine = object.__new__(ft.FaultTolerantTrainingEngine)
    engine.model_parts = [object()]
    engine.ft_manager = SimpleNamespace(maybe_set_all_reduce_hook=MagicMock())

    with patch.object(TrainingEngine, "_initialize_model") as initialize_model:
        ft.FaultTolerantTrainingEngine._initialize_model(
            engine,
            compile_config=SimpleNamespace(),
            hf_assets_path="",
        )

    initialize_model.assert_called_once()
    engine.ft_manager.maybe_set_all_reduce_hook.assert_called_once_with(
        engine.model_parts
    )
