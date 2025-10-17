# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed.expert_parallel import ExpertParallel


class MXExpertParallel(ExpertParallel):
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchao.prototype.moe_training.kernels.mxfp8.comms import (
                to_mxfp8_a2a_dequant,
            )
        except ImportError as err:
            raise ImportError(
                "Please install torchao v0.14+ to use MXExpertParallel"
            ) from err
        self._a2a_dispatch_impl = to_mxfp8_a2a_dequant
        self._a2a_combine_impl = to_mxfp8_a2a_dequant
