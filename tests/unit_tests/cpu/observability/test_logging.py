# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import sys

import pytest

from torchtitan.observability.logging import init_logger


@pytest.fixture
def restore_root_logger():
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]
    yield root_logger
    root_logger.setLevel(original_level)
    root_logger.handlers[:] = original_handlers


def test_init_logger_configures_named_loggers(
    monkeypatch: pytest.MonkeyPatch, restore_root_logger: logging.Logger
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    init_logger()
    logger = logging.getLogger("torchtitan.example")
    logger.info("message")

    assert "torchtitan.example - INFO - message" in output.getvalue()


def test_init_logger_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, restore_root_logger: logging.Logger
) -> None:
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    init_logger()
    init_logger()

    assert len(restore_root_logger.handlers) == 1
