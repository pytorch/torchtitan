# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from torchtitan.config import JobConfig


@contextmanager
def disable_compile(job_config: JobConfig):
    """Context manager to temporarily disable compilation."""
    original_value = job_config.compile.enable
    job_config.compile.enable = False
    try:
        yield
    finally:
        job_config.compile.enable = original_value
