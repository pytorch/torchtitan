# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import pytest

from torchtitan import train


@pytest.fixture
def error_file(tmp_path, monkeypatch):
    path = tmp_path / "error.json"
    monkeypatch.setenv("TORCHELASTIC_ERROR_FILE", str(path))
    # init_logger reconfigures the root logger for the rest of the session.
    monkeypatch.setattr(train, "init_logger", lambda: None)
    # init_structured_logger sets module globals that are never reset.
    monkeypatch.setattr(train.sl, "init_structured_logger", lambda **_: None)
    return path


def _recorded_message(path) -> str:
    return json.loads(path.read_text())["message"]["message"]


def test_main_records_config_error(error_file, monkeypatch):
    """Config errors raised before Trainer construction reach the error file."""

    def load(self):
        raise ValueError("bad config")

    monkeypatch.setattr(train.ConfigLoader, "load", load)

    with pytest.raises(ValueError, match="bad config"):
        train.main()

    assert "ValueError: bad config" in _recorded_message(error_file)


def test_main_records_train_error_when_close_fails(error_file, tmp_path, monkeypatch):
    """A close() failure during cleanup does not replace the original error."""

    def fail_train():
        raise RuntimeError("train failed")

    def fail_close():
        raise OSError("close failed")

    trainer = SimpleNamespace(train=fail_train, close=fail_close)
    config = SimpleNamespace(
        dump_folder=str(tmp_path),
        debug=SimpleNamespace(enable_structured_logging=False),
        create_seed_checkpoint=False,
        build=lambda: trainer,
    )
    monkeypatch.setattr(train.ConfigLoader, "load", lambda self: config)

    with pytest.raises(RuntimeError, match="train failed"):
        train.main()

    message = _recorded_message(error_file)
    assert "RuntimeError: train failed" in message
    assert "close failed" not in message
