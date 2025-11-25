# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


"""
Use --job.custom_config_module=torchtitan.experiments.auto_parallel.job_config
"""


@dataclass
class Experimental:
    # "aten" (default), "inductor", "none"
    comms_bucket_reorder_strategy: str = "aten"

    autop_force_bf16: bool = False


@dataclass
class JobConfig:
    experimental: Experimental = field(default_factory=Experimental)
