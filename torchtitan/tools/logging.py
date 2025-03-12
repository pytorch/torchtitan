# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os

logger = logging.getLogger()


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    Emit a warning message only once for unique arguments.

    This method is similar to `logger.warning()`, but will emit the warning
    with the same message only once for a given set of arguments.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
