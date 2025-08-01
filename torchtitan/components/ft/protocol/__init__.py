# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import to register quantization modules.
from torchtitan.components.ft.protocol.model import (
    FaultTolerantModelArgs,
    FaultTolerantTrainSpec,
)


__all__ = [
    "FaultTolerantModelArgs",
    "FaultTolerantTrainSpec",
]
