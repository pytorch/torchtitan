# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 aliases for the shared KDA implementation."""

from torchtitan.models.common.attention import KDA, KDAInnerAttention


KimiDeltaAttention = KDA
KimiKDAKernel = KDAInnerAttention

__all__ = ["KimiDeltaAttention", "KimiKDAKernel"]
