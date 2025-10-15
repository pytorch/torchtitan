# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class CustomConfig:
    how_is_your_day: str = "good"
    """Just an example helptext"""

    num_days: int = 7
    """Number of days in a week"""


@dataclass
class Training:
    steps: int = 99
    my_custom_steps: int = 32


@dataclass
class JobConfig:
    """
    This is an example of how to extend the tyro parser with custom config classes.
    """

    custom_config: CustomConfig = field(default_factory=CustomConfig)
    training: Training = field(default_factory=Training)
