# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Models for deterministic vLLM RL training.
"""

from .attention import VLLMCompatibleFlashAttention

__all__ = ["VLLMCompatibleFlashAttention"]
