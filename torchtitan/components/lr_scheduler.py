# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility imports for the relocated learning-rate scheduler module."""

from .optimizer.lr_scheduler import LRSchedulersContainer

__all__ = [
    "LRSchedulersContainer",
]
