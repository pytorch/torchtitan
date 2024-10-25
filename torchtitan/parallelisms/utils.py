# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchtitan.logging import logger


def check_strided_sharding_enabled() -> None:
    # Correct 2D/3D DCP usage requires DTensor's strided sharding in PR
    # https://github.com/pytorch/pytorch/pull/130760. This function checks if users'
    # PyTorch nightly-build version is newer than 2024-08-09 to make sure this PR is
    # included when 2D/3D DCP is used.
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the commit hash is newer than 2024-08-09
        logger.warning(
            "detected that the pytorch is built from source. Please make sure the PR "
            "(https://github.com/pytorch/pytorch/pull/130760) is included in pytorch "
            "for correct 2D/3D DCP usage."
        )
    elif torch.__version__ < "2.5.0.dev20240809":
        # the nightly build pytorch was built before 2024-08-09
        logger.warning(
            f"detected that the pytorch version {torch.__version__} is older than "
            "2.5.0.dev20240809. Please upgrade a newer version to include the change "
            "made in https://github.com/pytorch/pytorch/pull/130760 for correct 2D/3D "
            "DCP usage."
        )
