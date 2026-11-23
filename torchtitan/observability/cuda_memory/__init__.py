# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA memory consumer of PyTorch's private Cuspy callback API."""

from .capture import MemoryCapture

__all__ = ["MemoryCapture"]
