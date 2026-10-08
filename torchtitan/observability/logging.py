# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Console logging setup and helpers.

Library modules should create a named logger with ``logging.getLogger(__name__)``.
Process entry points should call :func:`init_logger` once before doing work.

The console log level defaults to ``INFO`` and can be overridden with the
``TITAN_LOG_LEVEL`` env var (e.g. ``TITAN_LOG_LEVEL=DEBUG``).
"""

import logging
import os
import sys

__all__ = ["init_logger", "warn_once"]


def _get_log_level() -> int:
    level_name = os.environ.get("TITAN_LOG_LEVEL", "INFO").upper()
    level_names = logging.getLevelNamesMapping()
    if level_name not in level_names:
        raise ValueError(
            f"Invalid TITAN_LOG_LEVEL={level_name!r}; "
            f"expected one of {sorted(level_names)}"
        )
    return level_names[level_name]


def init_logger() -> None:
    """Configure the process-wide root logger for TorchTitan applications.

    The level is read from the ``TITAN_LOG_LEVEL`` env var (default ``INFO``).
    """
    level = _get_log_level()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


_logged: set[str] = set()


def warn_once(logger: logging.Logger, msg: str) -> None:
    """Log a warning message only once per unique message.

    Uses a global set to track messages that have already been logged
    to prevent duplicate warning messages from cluttering the output.

    Args:
        logger (logging.Logger): The logger instance to use for warning.
        msg (str): The warning message to log.
    """
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)
