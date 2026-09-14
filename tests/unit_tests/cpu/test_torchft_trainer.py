# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import override
from torchtitan.config.transform import LoRAConverter
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3 import model_registry
from torchtitan.training_engine import TrainingEngine


def test_ft_applies_ffn_lora_override_before_model_build(monkeypatch):
    # Restore shared registration and distributed state after this test.
    monkeypatch.setattr(import_module("torchtitan.config.override"), "_REGISTRY", {})

    @override(target=FeedForward.Config)
    def ffn_lora(config):
        return LoRAConverter(LoRAConverter.Config(rank=1, alpha=1.0)).convert(config)

    config = ft.FaultTolerantTrainer.Config(
        model_spec=model_registry("debugmodel"),
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

    class ModelBuildReachedError(Exception):
        """Stop FT initialization at the model-build boundary."""

    built_ffns = []

    def build_model(model_config):
        built_ffns.append(model_config.layers[0].feed_forward.build())
        raise ModelBuildReachedError

    # Bypass hardware/data setup, but keep config updates and FFN building real.
    monkeypatch.setattr(
        ft.FaultTolerantTrainingEngine,
        "initialize_distributed_runtime",
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
    monkeypatch.setattr(type(config.model_spec.model), "build", build_model)

    with pytest.raises(ModelBuildReachedError):
        ft.FaultTolerantTrainer(config)

    assert hasattr(built_ffns[0].w13, "lora_a"), "FT ignored the FFN LoRA override"


def test_ft_trainer_composes_specialized_training_engine() -> None:
    assert not issubclass(ft.FaultTolerantTrainer, TrainingEngine)
    assert issubclass(ft.FaultTolerantTrainingEngine, TrainingEngine)


def test_ft_engine_installs_all_reduce_hook_after_model_initialization() -> None:
    engine = object.__new__(ft.FaultTolerantTrainingEngine)
    engine.model_parts = [object()]
    engine.ft_manager = SimpleNamespace(maybe_set_all_reduce_hook=MagicMock())

    with patch.object(TrainingEngine, "initialize_model") as initialize_model:
        ft.FaultTolerantTrainingEngine.initialize_model(
            engine,
            SimpleNamespace(),
            compile_config=SimpleNamespace(),
        )

    initialize_model.assert_called_once()
    engine.ft_manager.maybe_set_all_reduce_hook.assert_called_once_with(
        engine.model_parts
    )
