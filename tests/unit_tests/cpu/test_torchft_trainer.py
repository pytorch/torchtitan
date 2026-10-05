# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import override
from torchtitan.config.transform import LinearLoRAHandler, LoRATransform
from torchtitan.distributed import DistributedTopology, ParallelismContext
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3 import build_model_config
from torchtitan.training_engine import ForwardBackwardResult, TrainingEngine


def test_ft_applies_ffn_lora_override_before_model_build(monkeypatch):
    # Restore shared registration and distributed state after this test.
    monkeypatch.setattr(import_module("torchtitan.config.override"), "_REGISTRY", {})

    @override(target=FeedForward.Config)
    def ffn_lora(config):
        return LoRATransform(
            handlers=(LinearLoRAHandler(),), rank=1, alpha=1.0
        ).transform(config)

    config = ft.FaultTolerantTrainer.Config(
        model=build_model_config("debugmodel", seq_len=2048),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
    )
    config.override.imports = [f"{__name__}.ffn_lora"]

    def initialize_distributed_runtime(engine):
        engine.device = torch.device("cpu")
        engine.parallelism_context = ParallelismContext.from_config(
            config.parallelism, DistributedTopology(world_size=1)
        )
        engine.ft_manager = config.fault_tolerance.build()
        engine.garbage_collector = None
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
        ParallelismContext,
        "activate_spmd",
        lambda self: nullcontext(),
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


def test_ft_rejects_cuda_graphed_fsdp_gradient_accumulation(monkeypatch) -> None:
    config = ft.FaultTolerantTrainer.Config(
        model=build_model_config("debugmodel", seq_len=2048),
        tokenizer=None,
        loss=CrossEntropyLoss.Config(),
    )
    config.fault_tolerance.enable = True
    config.training.disable_cuda_graphs = False
    config.training.num_tokens_per_train_step = (
        2 * config.training.num_tokens_per_microbatch_per_dp_rank
    )
    engine = SimpleNamespace(
        parallelism_context=SimpleNamespace(
            dp_enabled=False,
            pp_enabled=False,
            fsdp_enabled=True,
        ),
        ft_manager=SimpleNamespace(get_dp_info=lambda degree, rank: (degree, rank)),
        device_memory_monitor=SimpleNamespace(),
    )

    monkeypatch.setattr(
        ft, "FaultTolerantTrainingEngine", lambda *args, **kwargs: engine
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
    monkeypatch.setattr(ft, "cuda_graphs_supported", lambda: True)

    with pytest.raises(
        ValueError, match="does not support CUDA-graphed FSDP gradient accumulation"
    ):
        ft.FaultTolerantTrainer(config)


def test_ft_training_engine_rejects_optimizer_cuda_graph() -> None:
    config = SimpleNamespace(
        optim=SimpleNamespace(enable_cuda_graph=True),
    )

    with (
        patch.object(TrainingEngine, "__init__") as init,
        pytest.raises(ValueError, match="not supported with TorchFT"),
    ):
        ft.FaultTolerantTrainingEngine(
            config,
            model_config=MagicMock(),
            max_num_documents=None,
            output_dir="",
            fault_tolerance=MagicMock(),
        )

    init.assert_not_called()


def test_ft_averages_logged_loss_by_active_replica_count(monkeypatch):
    engine = Mock(
        spec=ft.FaultTolerantTrainingEngine,
        config=Mock(training=Mock(disable_cuda_graphs=True)),
        device=torch.device("cpu"),
        parallelism_context=Mock(
            dp_enabled=False, dp_cp_enabled=True, pp_enabled=False, ep_enabled=False
        ),
        ft_manager=Mock(loss_sync_pg=Mock(size=lambda: 2), group_size=4),
        optim=Mock(lr_schedulers=Mock(schedulers=[Mock(get_last_lr=lambda: [0.1])])),
        num_completed_steps=1,
        ntokens_seen=4,
        model_config=SimpleNamespace(mtp_layers=None),
        forward_backward=Mock(
            return_value=ForwardBackwardResult(torch.tensor(2.0), [])
        ),
        optim_step=Mock(return_value=torch.tensor(0.0)),
    )
    trainer = Mock(
        spec=ft.FaultTolerantTrainer,
        engine=engine,
        metrics_processor=Mock(should_log=Mock(return_value=True)),
        config=Mock(
            dataloader=Mock(num_mtp_layers=0),
            model=Mock(traverse=lambda _: iter(())),
        ),
        gradient_accumulation_steps=1,
        num_pp_microbatches=1,
    )
    # Two active replicas contribute a loss sum of 4.0, despite group_size=4.
    monkeypatch.setattr(ft.dist_utils, "dist_sum", Mock(side_effect=[4.0, 8]))
    monkeypatch.setattr(ft.dist_utils, "dist_max", Mock(return_value=2.0))
    monkeypatch.setattr(ft, "collect_aux_loss_metrics", Mock(return_value={}))

    microbatch = SimpleNamespace(
        num_loss_tokens=4,
        labels=torch.arange(4),
        as_input_dict=lambda: {
            "positions": torch.arange(4),
            "padding_mask": torch.zeros(4, dtype=torch.bool),
        },
    )
    ft.FaultTolerantTrainer.train_step(trainer, iter([microbatch]))

    engine.forward_backward.assert_called_once()
    forward_backward_args = engine.forward_backward.call_args.kwargs
    assert forward_backward_args["microbatch_groups"] == [[microbatch]]
    torch.testing.assert_close(
        forward_backward_args["global_loss_token_counts"], torch.tensor([4])
    )
    torch.testing.assert_close(
        forward_backward_args["global_routing_token_counts"], torch.tensor([4])
    )
    trainer.metrics_processor.log.assert_called_once()
    _, logged_loss, *_ = trainer.metrics_processor.log.call_args.args
    assert logged_loss == 2.0


def test_ft_engine_installs_all_reduce_hook_after_model_initialization() -> None:
    engine = object.__new__(ft.FaultTolerantTrainingEngine)
    engine.model_parts = [object()]
    engine.ft_manager = SimpleNamespace(maybe_set_all_reduce_hook=MagicMock())

    with patch.object(TrainingEngine, "_initialize_model") as initialize_model:
        ft.FaultTolerantTrainingEngine._initialize_model(
            engine,
            hf_assets_path="",
        )

    initialize_model.assert_called_once()
    engine.ft_manager.maybe_set_all_reduce_hook.assert_called_once_with(
        engine.model_parts
    )
