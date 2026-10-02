# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from unittest.mock import MagicMock

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
