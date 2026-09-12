# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import override
from torchtitan.config.transform import LoRAConverter
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3 import model_registry


def test_ft_applies_ffn_lora_override_before_model_build(monkeypatch):
    # Restore shared registration and distributed state after this test.
    monkeypatch.setattr(import_module("torchtitan.config.override"), "_REGISTRY", {})

    @override(target=FeedForward.Config)
    def ffn_lora(config):
        return LoRAConverter(LoRAConverter.Config(rank=1, alpha=1.0)).convert(config)

    config = ft.FaultTolerantTrainer.Config(
        model_spec=model_registry("debugmodel"),
        tokenizer=None,
    )
    config.override.imports = [f"{__name__}.ffn_lora"]

    def init_distributed(trainer):
        trainer.ft_manager = config.fault_tolerance.build()
        return ParallelDims.from_config(config.parallelism, world_size=1)

    class ModelBuildReachedError(Exception):
        """Stop FT initialization at the model-build boundary."""

    built_ffns = []

    def build_model(model_config):
        built_ffns.append(model_config.layers[0].feed_forward.build())
        raise ModelBuildReachedError

    # Bypass hardware/data setup, but keep config updates and FFN building real.
    monkeypatch.setattr(ft.FaultTolerantTrainer, "init_distributed", init_distributed)
    monkeypatch.setattr(ft.utils, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ft.utils, "device_module", torch.cpu)
    monkeypatch.setattr(ft.utils, "GarbageCollection", lambda **kwargs: None)
    monkeypatch.setattr(type(config.dataloader), "build", lambda self, **kwargs: None)
    monkeypatch.setattr(type(config.model_spec.model), "build", build_model)

    with pytest.raises(ModelBuildReachedError):
        ft.FaultTolerantTrainer(config)

    assert hasattr(built_ffns[0].w1, "lora_a"), "FT ignored the FFN LoRA override"


def test_ft_chunked_loss_without_pp_binds_lm_head_and_skips_model_projection():
    model = SimpleNamespace(lm_head=torch.nn.Linear(4, 8), _skip_lm_head=False)
    trainer = object.__new__(ft.FaultTolerantTrainer)
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.model_parts = [model]
    trainer.loss_fn = ChunkedLossWrapper.Config(num_chunks=2).build()

    trainer._configure_chunked_loss()

    assert trainer.loss_fn.lm_head is model.lm_head
    assert model._skip_lm_head is True
