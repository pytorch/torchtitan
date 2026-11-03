# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 multi-token prediction (NextN) training components."""

from torchtitan.models.deepseek_v3.mtp import (
    MTPLoss,
    MTPTransformerBlock as Glm5MTPTransformerBlock,
    roll_mtp_sequence,
)

__all__ = ["Glm5MTPTransformerBlock", "MTPLoss", "roll_mtp_sequence"]
