# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module

import pytest
import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.lora import LoRAConverter
from torchtitan.config import override
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3 import model_registry


def test_ft_applies_ffn_lora_override_before_model_build(monkeypatch):
    # Restore shared registration and distributed state after this test.
    monkeypatch.setattr(import_module("torchtitan.config.override"), "_REGISTRY", {})
    monkeypatch.setattr(ft.dist_utils, "_spmd_backend", "spmd_types")

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
