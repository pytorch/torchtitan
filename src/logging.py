"""Logging facade: rich console via logfire + plain-text file via stdlib logging."""

import logging as _stdlib_logging
import os
from typing import Any

import logfire

logfire.configure(
    service_name="moe-training",
    send_to_logfire="if-token-present",
    inspect_arguments=False,
)

_stdlib_logger = _stdlib_logging.getLogger("moe_training")
_stdlib_logger.setLevel(_stdlib_logging.INFO)
_stdlib_logger.propagate = False


class _Logger:
    """Forwards calls to logfire (rich console) and optionally a stdlib FileHandler."""

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logfire.info(msg, *args, **kwargs)
        _stdlib_logger.info(msg)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logfire.warning(msg, *args, **kwargs)
        _stdlib_logger.warning(msg)

    def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.warning(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logfire.debug(msg, *args, **kwargs)
        _stdlib_logger.debug(msg)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        logfire.error(msg, *args, **kwargs)
        _stdlib_logger.error(msg)


logger = _Logger()


def init_logger(log_dir: str | None = None) -> None:
    """Attach a file handler writing to `log_dir/train.log` if `log_dir` is set."""
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    # Avoid attaching duplicate handlers on repeat calls
    for h in _stdlib_logger.handlers:
        if isinstance(
            h, _stdlib_logging.FileHandler
        ) and h.baseFilename == os.path.abspath(log_path):
            return

    handler = _stdlib_logging.FileHandler(log_path)
    handler.setFormatter(
        _stdlib_logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    )
    _stdlib_logger.addHandler(handler)
