# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward structured records from external libraries into TorchTitan logs."""

import logging

from torchtitan.observability.structured_logger.structured_logging import (
    _structured_logger,
    _structured_logger_disabled,
    ExtraFields,
    LogType,
)


class _ForwardingStructuredLoggingHandler(logging.Handler):
    def __init__(self, destination_logger: logging.Logger) -> None:
        super().__init__()
        self.destination_logger = destination_logger

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == self.destination_logger.name:
            return
        if getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is None:
            return
        if getattr(record, str(ExtraFields.LOG_TYPE), None) is None:
            record = logging.makeLogRecord(record.__dict__.copy())
            setattr(record, str(ExtraFields.LOG_TYPE), str(LogType.EVENT))

        self.destination_logger.handle(record)


def install_forwarding_structured_logging_handler(
    source_logger_name: str,
) -> bool:
    """Forward structured records from ``source_logger_name`` to TorchTitan."""

    if _structured_logger_disabled() or not _structured_logger.handlers:
        return False

    source_logger = logging.getLogger(source_logger_name)
    for handler in source_logger.handlers:
        if (
            isinstance(handler, _ForwardingStructuredLoggingHandler)
            and handler.destination_logger is _structured_logger
        ):
            return True

    source_logger.addHandler(_ForwardingStructuredLoggingHandler(_structured_logger))
    return True
