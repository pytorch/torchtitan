# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from torchtitan.components import validate as validate_module
from torchtitan.components.validate import Validator
from torchtitan.models.flux import validate as flux_validate_module
from torchtitan.models.flux.validate import FluxValidator


class _ClosableLoader:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def __iter__(self):
        return iter(self.rows)

    def close(self):
        self.closed = True


class _FailingModel(nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("validation failed")


class _FluxModel(nn.Module):
    def forward(self, **kwargs):
        return torch.zeros_like(kwargs["img"])


def _generic_validator(loader):
    validator = object.__new__(Validator)
    validator.config = SimpleNamespace(steps=1)
    validator.parallel_dims = SimpleNamespace(
        dp_enabled=False,
        pp_enabled=False,
        dp_cp_enabled=False,
    )
    validator.dl_config = SimpleNamespace(build=mock.Mock(return_value=loader))
    validator.dp_world_size = 1
    validator.dp_rank = 0
    validator.tokenizer = mock.Mock()
    validator.seq_len = 4
    validator.local_batch_size = 1
    validator.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        log_validation=mock.Mock(),
    )
    validator.validation_context = nullcontext
    validator.loss_fn = lambda predictions, labels: (predictions.sum(), None)
    validator.post_dataloading_process = lambda input_dict, labels, model_parts: (
        input_dict["input"],
        labels,
        {},
    )
    return validator


@pytest.mark.parametrize("raises", [False, True])
def test_generic_validator_closes_temporary_loader(monkeypatch, raises):
    row = ({"input": torch.ones(1, 1)}, torch.ones(1, 1, dtype=torch.long))
    loader = _ClosableLoader([row, row])
    validator = _generic_validator(loader)
    model = _FailingModel() if raises else nn.Identity()
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")

    if raises:
        with pytest.raises(RuntimeError, match="validation failed"):
            validator.validate([model], step=1)
    else:
        validator.validate([model], step=1)

    assert loader.closed


def _flux_validator(loader):
    validator = object.__new__(FluxValidator)
    validator.config = FluxValidator.Config(
        dataloader=mock.Mock(),
        steps=1,
        save_img_count=0,
    )
    validator.parallel_dims = SimpleNamespace(
        cp_enabled=False,
        dp_cp_enabled=False,
    )
    validator.dl_config = SimpleNamespace(
        build=mock.Mock(return_value=loader),
        dataset=mock.Mock(),
    )
    validator.dp_world_size = 1
    validator.dp_rank = 0
    validator.tokenizer = mock.Mock()
    validator.seq_len = 4
    validator.local_batch_size = 1
    validator.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        log_validation=mock.Mock(),
    )
    validator.validation_context = nullcontext
    validator.loss_fn = lambda predictions, labels: (predictions.sum(), None)
    validator.all_timesteps = False
    validator.device = torch.device("cpu")
    validator._dtype = torch.float32
    validator.autoencoder = None
    validator.clip_encoder = None
    validator.t5_encoder = None
    validator.dump_folder = "."
    return validator


@pytest.mark.parametrize("raises", [False, True])
def test_flux_validator_closes_temporary_loader(monkeypatch, raises):
    row = (
        {
            "prompt": "test",
            "timestep": torch.tensor([0.5]),
        },
        torch.zeros(1, 1, 2, 2),
    )
    loader = _ClosableLoader([row, row])
    validator = _flux_validator(loader)

    def preprocess_data(**kwargs):
        if raises:
            raise RuntimeError("validation failed")
        batch = kwargs["batch"]
        return {
            **batch,
            "img_encodings": torch.zeros(1, 1, 2, 2),
            "clip_encodings": torch.zeros(1, 1),
            "t5_encodings": torch.zeros(1, 1, 1),
        }

    monkeypatch.setattr(flux_validate_module, "preprocess_data", preprocess_data)
    monkeypatch.setattr(flux_validate_module, "pack_latents", lambda value: value)
    monkeypatch.setattr(
        flux_validate_module,
        "create_position_encoding_for_latents",
        lambda *args: torch.zeros(1, 1, 3),
    )
    monkeypatch.setattr(flux_validate_module.dist_utils, "device_type", "cpu")

    if raises:
        with pytest.raises(RuntimeError, match="validation failed"):
            validator.validate([_FluxModel()], step=1)
    else:
        validator.validate([_FluxModel()], step=1)

    assert loader.closed


def test_flux_validator_generates_at_batch_image_dimensions(monkeypatch):
    labels = torch.zeros(1, 3, 6, 10)
    loader = _ClosableLoader(
        [
            (
                {
                    "prompt": "test",
                    "timestep": torch.tensor([0.5]),
                },
                labels,
            )
        ]
    )
    validator = _flux_validator(loader)
    validator.config.save_img_count = 1
    generated = {}

    def generate_image(**kwargs):
        generated.update(kwargs)
        return torch.zeros(3, kwargs["img_height"], kwargs["img_width"])

    monkeypatch.setattr(flux_validate_module, "generate_image", generate_image)
    monkeypatch.setattr(flux_validate_module, "save_image", lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        flux_validate_module,
        "preprocess_data",
        lambda **kwargs: {
            **kwargs["batch"],
            "img_encodings": torch.zeros(1, 1, 2, 2),
            "clip_encodings": torch.zeros(1, 1),
            "t5_encodings": torch.zeros(1, 1, 1),
        },
    )
    monkeypatch.setattr(flux_validate_module, "pack_latents", lambda value: value)
    monkeypatch.setattr(
        flux_validate_module,
        "create_position_encoding_for_latents",
        lambda *args: torch.zeros(1, 1, 3),
    )
    monkeypatch.setattr(flux_validate_module.dist_utils, "device_type", "cpu")

    validator.validate([_FluxModel()], step=1)

    assert generated["img_height"] == 6
    assert generated["img_width"] == 10
