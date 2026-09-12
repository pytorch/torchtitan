# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Context Parallel APIs

``cp_shard`` is the low-level API for jointly sharding tensors and BlockMasks.
It remains available for Flux, whose input preparation differs from decoders.
"""

from .api import cp_shard

__all__ = ["cp_shard"]
