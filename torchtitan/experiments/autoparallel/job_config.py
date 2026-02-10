# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


"""
Use a Python config file with ConfigManager._merge_configs() to include
autoparallel-specific config fields. See docs/extension.md for details.
"""


@dataclass
class Experimental:
    # "aten" (default), "inductor", "none"
    comms_bucket_reorder_strategy: str = "aten"

    autop_force_bf16: bool = False


@dataclass
class JobConfig:
    experimental: Experimental = field(default_factory=Experimental)
