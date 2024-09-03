# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum
import argparse


logger = logging.getLogger()



def init_logger(log_level):
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError(f"Invalid log level: '{value}'. "
                                             f"Choose from {', '.join([lvl.name for lvl in cls])}.")

def validate_log_level(value):
    return LogLevel.from_string(value)

