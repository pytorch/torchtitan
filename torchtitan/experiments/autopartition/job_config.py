# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class AutoPipeConfig:
    auto_partition: bool = True
    """Whether to use autopartition method to split module, default False"""

@dataclass
class JobConfig:
    autopipe_config: AutoPipeConfig = field(default_factory=AutoPipeConfig)
