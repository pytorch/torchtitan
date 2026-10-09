# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from torchtitan.components import validate as validate_module
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.validate import Validator
from torchtitan.models.flux import validate as flux_validate_module
from torchtitan.models.flux.flux_datasets import FluxTrainingMicrobatch
from torchtitan.models.flux.validate import FluxValidator


class _ClosableLoader:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def __iter__(self):
        return iter(self.rows)

    def close(self):
        self.closed = True


class _EchoModel(nn.Module):
    def preprocess_inputs(self, input_dict, **kwargs):
        del kwargs
        return input_dict["input"], input_dict["labels"], {}

    def forward(self, inputs, **kwargs):
        del kwargs
        return inputs


class _FailingModel(_EchoModel):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("validation failed")


class _FluxModel(nn.Module):
    def forward(self, **kwargs):
        return torch.zeros_like(kwargs["img"])


def _generic_validator(loader):
    validator = object.__new__(Validator)
    validator.config = SimpleNamespace(steps=1)
    validator._cuda_graph_enabled = False
    validator._run_eval = None
    validator.parallelism_context = SimpleNamespace(
        dp_enabled=False,
        pp_enabled=False,
        dp_cp_enabled=False,
        activate_spmd=lambda **kwargs: nullcontext(),
    )
    validator.dl_config = SimpleNamespace(
        build=mock.Mock(return_value=loader), max_num_documents=None
    )
    validator.dp_world_size = 1
    validator.dp_rank = 0
    validator.tokenizer = mock.Mock()
    validator.seq_len = 4
    validator.num_tokens_per_microbatch = 4
    validator.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        reset=mock.Mock(),
        log_validation=mock.Mock(),
    )
    validator.loss_fn = lambda predictions, labels: (predictions.sum(), None)
    validator.parallelism = SimpleNamespace()
    return validator


def test_eval_wrapper_returns_each_real_run(monkeypatch):
    from torchtitan.distributed.cuda_graph import (
        is_cuda_graph_capture_enabled,
        set_cuda_graph_capture_enabled,
    )

    row = TokenizedTrainingMicrobatch(
        input=torch.ones(1, 1),
        labels=torch.ones(1, 1, dtype=torch.long),
        positions=torch.zeros(1, 1, dtype=torch.long),
        padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        loss_token_counts=torch.tensor(1),
        routing_token_counts=torch.tensor([1]),
    )
    validator = _generic_validator(_ClosableLoader([row]))
    validator._cuda_graph_enabled = True
    set_cuda_graph_capture_enabled(False)
    model = _EchoModel()
    model.register_buffer("counter", torch.tensor(0))
    calls = []

    def body(parts, prepared):
        assert not parts[0].training
        assert not torch.is_grad_enabled()
        calls.append(prepared)
        parts[0].counter.add_(1)
        return torch.tensor(float(parts[0].counter))

    validator._evaluate_body = body
    wrap = mock.Mock(side_effect=lambda fn, **kwargs: fn)
    monkeypatch.setattr(validate_module, "wrap_with_cuda_graph", wrap)
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")
    validator.validate([model], step=1)
    assert len(calls) == 1
    validator.metrics_processor.log_validation.assert_called_once_with(loss=1.0, step=1)
    assert not is_cuda_graph_capture_enabled()
    validator.validate([model], step=2)
    assert len(calls) == 2
    validator.metrics_processor.log_validation.assert_called_with(loss=2.0, step=2)
    torch.testing.assert_close(calls[-1][0][0], row.input, rtol=0, atol=0)
    assert model.training
    assert model.counter.item() == 2
    wrap.assert_called_once()
    assert wrap.call_args.kwargs == {
        "num_warmup_iterations": 1,
        "name": "validation",
    }
    set_cuda_graph_capture_enabled(True)


def test_validation_preprocessing_uses_validation_document_limit(monkeypatch):
    row = TokenizedTrainingMicrobatch(
        input=torch.ones(1, 1),
        labels=torch.ones(1, 1, dtype=torch.long),
        positions=torch.zeros(1, 1, dtype=torch.long),
        padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        loss_token_counts=torch.tensor(1),
        routing_token_counts=torch.tensor([1]),
    )
    validator = _generic_validator(_ClosableLoader([row]))
    validator.dl_config.max_num_documents = 4
    model = _EchoModel()
    preprocess = mock.Mock(wraps=model.preprocess_inputs)
    monkeypatch.setattr(model, "preprocess_inputs", preprocess)
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")
    validator.validate([model], step=1)
    preprocess.assert_called_once()
    assert preprocess.call_args.kwargs == {
        "parallelism_context": validator.parallelism_context,
        "parallelism": validator.parallelism,
        "max_num_documents": 4,
        "max_context_length": validator.seq_len,
    }
    torch.testing.assert_close(preprocess.call_args.args[0]["input"], row.input)


def test_real_eval_keeps_buffers_and_rng_and_reports_loss(monkeypatch):
    rows = [
        TokenizedTrainingMicrobatch(
            input=torch.ones(1, 1),
            labels=torch.ones(1, 1, dtype=torch.long),
            positions=torch.zeros(1, 1, dtype=torch.long),
            padding_mask=torch.zeros(1, 1, dtype=torch.bool),
            loss_token_counts=torch.tensor(1),
            routing_token_counts=torch.tensor([1]),
        )
    ]
    loader = _ClosableLoader(rows)
    validator = _generic_validator(loader)
    validator._cuda_graph_enabled = True
    monkeypatch.setattr(
        validate_module, "wrap_with_cuda_graph", lambda fn, **kwargs: fn
    )
    model = _EchoModel()
    model.register_buffer("counter", torch.tensor(0))

    def forward(inputs):
        model.counter.add_(1)
        torch.rand(())
        return inputs

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")
    rng_before = torch.random.get_rng_state()
    validator.validate([model], step=2)
    assert loader.closed
    assert model.training
    assert model.counter.item() == 1
    assert not torch.equal(torch.random.get_rng_state(), rng_before)
    validator.metrics_processor.reset.assert_called_once()
    validator.metrics_processor.log_validation.assert_called_once_with(loss=1.0, step=2)


@pytest.mark.parametrize("raises", [False, True])
def test_generic_validator_closes_temporary_loader(monkeypatch, raises):
    def microbatch():
        return TokenizedTrainingMicrobatch(
            input=torch.ones(1, 1),
            labels=torch.ones(1, 1, dtype=torch.long),
            positions=torch.zeros(1, 1, dtype=torch.long),
            padding_mask=torch.zeros(1, 1, dtype=torch.bool),
            loss_token_counts=torch.tensor(1),
            routing_token_counts=torch.tensor([1]),
        )

    loader = _ClosableLoader([microbatch(), microbatch()])
    validator = _generic_validator(loader)
    model = _FailingModel() if raises else _EchoModel()
    model.add_module("dropout", nn.Dropout())
    model.dropout.eval()
    modes = [module.training for module in model.modules()]
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")
    if raises:
        with pytest.raises(RuntimeError, match="validation failed"):
            validator.validate([model], step=1)
    else:
        validator.validate([model], step=1)

    assert loader.closed
    assert [module.training for module in model.modules()] == modes
    validator.metrics_processor.reset.assert_called_once()


def test_flux_validator_rejects_unimplemented_graph_capture():
    with pytest.raises(ValueError, match="not implemented for FluxValidator"):
        FluxValidator.Config(dataloader=mock.Mock(), steps=1, enable_cuda_graphs=True)


def _flux_validator(loader):
    validator = object.__new__(FluxValidator)
    validator.config = FluxValidator.Config(
        dataloader=mock.Mock(),
        steps=1,
        save_img_count=0,
    )
    validator.parallelism_context = SimpleNamespace(
        dp_enabled=False,
        cp_enabled=False,
        dp_cp_enabled=False,
        activate_spmd=lambda **kwargs: nullcontext(),
    )
    validator.dl_config = SimpleNamespace(
        build=mock.Mock(return_value=loader),
        dataset=mock.Mock(),
    )
    validator.dp_world_size = 1
    validator.dp_rank = 0
    validator.tokenizer = mock.Mock()
    validator.seq_len = 4
    validator.num_tokens_per_microbatch = 4
    validator.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        reset=mock.Mock(),
        log_validation=mock.Mock(),
    )
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
    microbatch = FluxTrainingMicrobatch(
        prompt=["test"],
        timestep=torch.tensor([0.5]),
        labels=torch.zeros(1, 1, 2, 2),
        t5=torch.zeros(1, 1),
        clip=torch.zeros(1, 1),
        loss_token_counts=torch.tensor(4),
        routing_token_counts=torch.tensor([4]),
    )
    loader = _ClosableLoader([microbatch, microbatch])
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
    validator.metrics_processor.reset.assert_called_once()


def test_flux_validator_generates_at_batch_image_dimensions(monkeypatch):
    labels = torch.zeros(1, 3, 6, 10)
    loader = _ClosableLoader(
        [
            FluxTrainingMicrobatch(
                prompt=["test"],
                timestep=torch.tensor([0.5]),
                labels=labels,
                t5=torch.zeros(1, 1),
                clip=torch.zeros(1, 1),
                loss_token_counts=torch.tensor(labels.numel()),
                routing_token_counts=torch.tensor([labels.numel()]),
            )
        ]
    )
    validator = _flux_validator(loader)
    validator.config.save_img_count = 1
    generated = {}
    spmd_context_entries = 0
    spmd_context_active = False

    @contextmanager
    def spmd_context():
        nonlocal spmd_context_active, spmd_context_entries
        assert not spmd_context_active
        spmd_context_active = True
        spmd_context_entries += 1
        try:
            yield
        finally:
            spmd_context_active = False

    validator.parallelism_context.activate_spmd = lambda **kwargs: spmd_context()

    def generate_image(**kwargs):
        assert spmd_context_active
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
    assert spmd_context_entries == 2


def test_generic_validator_raises_on_zero_validation_batches(monkeypatch):
    loader = _ClosableLoader([])
    validator = _generic_validator(loader)
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")
    with pytest.raises(ValueError, match="zero batches"):
        validator.validate([_EchoModel()], step=1)

    assert loader.closed


def test_generic_validator_raises_on_zero_valid_tokens(monkeypatch):
    microbatch = TokenizedTrainingMicrobatch(
        input=torch.ones(1, 1),
        labels=torch.full((1, 1), IGNORE_INDEX, dtype=torch.long),
        positions=torch.zeros(1, 1, dtype=torch.long),
        padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        loss_token_counts=torch.tensor(0),
        routing_token_counts=torch.tensor([1]),
    )
    loader = _ClosableLoader([microbatch])
    validator = _generic_validator(loader)
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")
    with pytest.raises(ValueError, match="zero valid tokens"):
        validator.validate([_EchoModel()], step=1)

    assert loader.closed


def _validator_from_init(*, steps: int, dp_world_size: int) -> Validator:
    return Validator(
        Validator.Config(steps=steps),
        parallelism=mock.Mock(),
        dp_world_size=dp_world_size,
        dp_rank=0,
        tokenizer=mock.Mock(),
        parallelism_context=mock.Mock(),
        loss_fn=mock.Mock(),
        validation_context=nullcontext,
        metrics_processor=mock.Mock(),
        seq_len=4,
        num_tokens_per_microbatch=4,
    )


def test_validator_rejects_steps_neg1_when_dp_gt_1():
    with pytest.raises(ValueError, match="hang on validation collectives"):
        _validator_from_init(steps=-1, dp_world_size=2)


def test_validator_accepts_finite_pass_or_positive_steps():
    _validator_from_init(steps=-1, dp_world_size=1)
    _validator_from_init(steps=10, dp_world_size=8)
