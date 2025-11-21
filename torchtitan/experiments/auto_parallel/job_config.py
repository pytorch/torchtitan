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
    custom_import: str = ""
    """
    This option enables the importation of external modules.
    Currently, it only supports dotted import modules (e.g., some_package.model_x).
    It is the user's responsibility to ensure that the specified path can be
    successfully imported. One method to achieve this, you can place your module
    inside the ``torchtitan/torchtitan`` folder and execute ``pip install -e .`` to
    make it available for import.
    """

    custom_args_module: str = ""
    """
    DEPRECATED (moved to Job.custom_config_module). Will be removed soon.

    This option allows users to extend TorchTitan's existing JobConfig by extending
    a user defined JobConfig dataclass. Similar to ``--experimental.custom_import``, the user
    needs to ensure that the path can be imported.
    """

    # "aten" (default), "inductor", "none"
    comms_bucket_reorder_strategy: str = "aten"

    autop_force_bf16: bool = False


@dataclass
class JobConfig:
    experimental: Experimental = field(default_factory=Experimental)
