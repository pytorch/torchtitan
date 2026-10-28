# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from unittest.mock import MagicMock, call

import pytest

from torchtitan.observability.metrics import WandBLogger


@pytest.mark.parametrize(
    "env_value, expected",
    [
        (None, None),
        ("", None),
        ("exp1", ["exp1"]),
        ("exp1, baseline,,", ["exp1", "baseline"]),
        (" , ", None),
    ],
)
def test_wandb_run_tags_are_split(tmp_path, monkeypatch, env_value, expected):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    if env_value is None:
        monkeypatch.delenv("WANDB_RUN_TAGS", raising=False)
    else:
        monkeypatch.setenv("WANDB_RUN_TAGS", env_value)

    WandBLogger(log_dir=str(tmp_path))

    assert fake_wandb.init.call_args.kwargs["tags"] == expected


def test_wandb_default_logs_with_explicit_step(tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandBLogger(log_dir=str(tmp_path))
    logger.log({"loss/mean": 0.25}, step=3)

    fake_wandb.define_metric.assert_not_called()
    fake_wandb.log.assert_called_once_with({"loss/mean": 0.25}, step=3, commit=True)


def test_wandb_custom_step_logs_validation_after_training(tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandBLogger(log_dir=str(tmp_path), step_metric="trainer/step")
    logger.log({"validation_reward/_mean": 0.1}, step=0)
    logger.log({"rollout_reward/_mean": 0.5}, step=150)
    logger.log({"validation_reward/_mean": 0.17}, step=150)

    assert fake_wandb.define_metric.call_args_list == [
        call("trainer/step"),
        call("*", step_metric="trainer/step"),
    ]
    assert fake_wandb.log.call_args_list == [
        call({"validation_reward/_mean": 0.1, "trainer/step": 0}, commit=True),
        call({"rollout_reward/_mean": 0.5, "trainer/step": 150}, commit=True),
        call({"validation_reward/_mean": 0.17, "trainer/step": 150}, commit=True),
    ]
