# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import mock

from torchtitan.experiments.torchft import trainer as trainer_module
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer


class _ChunkedLoss:
    def __init__(self) -> None:
        self.lm_head = None

    def set_lm_head(self, lm_head) -> None:
        self.lm_head = lm_head


def _new_trainer_for_chunked_loss(*, pp_enabled: bool, pp_has_last_stage: bool):
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.loss_fn = _ChunkedLoss()
    trainer.parallel_dims = SimpleNamespace(pp_enabled=pp_enabled)
    trainer.pp_has_last_stage = pp_has_last_stage
    return trainer


def test_model_config_applies_overrides_after_training_config():
    calls = []
    trainer = object.__new__(FaultTolerantTrainer)
    trainer.config = SimpleNamespace(override=SimpleNamespace(imports=["module"]))
    model_config = mock.Mock()
    model_config.update_from_config.side_effect = lambda **kwargs: calls.append(
        "update"
    )

    with mock.patch.object(
        trainer_module,
        "apply_overrides",
        side_effect=lambda *args: calls.append("override"),
    ) as apply_overrides:
        trainer._update_model_config(model_config)

    assert calls == ["update", "override"]
    model_config.update_from_config.assert_called_once_with(config=trainer.config)
    apply_overrides.assert_called_once_with(
        trainer.config.override, trainer.config
    )


def test_chunked_loss_uses_non_pipeline_model_lm_head():
    trainer = _new_trainer_for_chunked_loss(
        pp_enabled=False, pp_has_last_stage=False
    )
    model_part = SimpleNamespace(lm_head=mock.sentinel.lm_head)
    trainer.model_parts = [model_part]

    with mock.patch.object(trainer_module, "ChunkedLossWrapper", _ChunkedLoss):
        trainer._set_chunked_loss_lm_head()

    assert trainer.loss_fn.lm_head is mock.sentinel.lm_head
    assert model_part._skip_lm_head is True


def test_chunked_loss_uses_last_pipeline_stage_lm_head():
    trainer = _new_trainer_for_chunked_loss(pp_enabled=True, pp_has_last_stage=True)
    model_part = SimpleNamespace(lm_head=mock.sentinel.lm_head)
    trainer.model_parts = [model_part]

    with mock.patch.object(trainer_module, "ChunkedLossWrapper", _ChunkedLoss):
        trainer._set_chunked_loss_lm_head()

    assert trainer.loss_fn.lm_head is mock.sentinel.lm_head
    assert model_part._skip_lm_head is True


def test_chunked_loss_skips_non_last_pipeline_stage():
    trainer = _new_trainer_for_chunked_loss(pp_enabled=True, pp_has_last_stage=False)
    model_part = SimpleNamespace()
    trainer.model_parts = [model_part]

    with mock.patch.object(trainer_module, "ChunkedLossWrapper", _ChunkedLoss):
        trainer._set_chunked_loss_lm_head()

    assert trainer.loss_fn.lm_head is None
    assert not hasattr(model_part, "_skip_lm_head")
