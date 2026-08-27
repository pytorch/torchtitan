# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 constants and types shared by the linear, grouped-expert, and
converter modules.

This module deliberately has no TorchAO dependency so the converter can
describe its configuration even when the MXFP8 kernels are unavailable.
"""

from typing import Literal


# One E8M0 scale per 32 elements along the scaled axis, per the OCP
# microscaling spec. Weight quantization uses a *square* 32x32 tile -- side
# equal to the block size is the only shape that is a valid MX group along
# both axes at once, which is what makes it transpose-invariant and lets
# FPROP and DGRAD share one qdata allocation.
_MXFP8_BLOCK_SIZE = 32

# Activation and gradient quantization takes a scaling mode; the 32x32 weight
# cast hardcodes RCEIL. Pin the two to match, so both operands of a GEMM round
# their E8M0 scales the same way. This is TorchAO's current default too, but
# relying on that would let a default change silently desync them.
_MXFP8_SCALING_MODE = "rceil"

_MXFP8_SCALE_GROUP_ALIGNMENT = 128
"""Row alignment each token group needs in the blocked grouped-GEMM scale layout."""

InputActivationFormatForBackward = Literal["bf16", "mxfp8"]
_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD = ("bf16", "mxfp8")


__all__ = ["InputActivationFormatForBackward"]
