# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable


__all__ = ["DatasetConfig"]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable
