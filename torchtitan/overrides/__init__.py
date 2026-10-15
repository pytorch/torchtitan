# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchtitan overrides registry. External packages may follow the same pattern."""

from torchtitan.config import override as _override

__all__ = ["_override"]
