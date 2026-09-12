# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Context Parallel APIs

``cp_shard_inputs`` shards named model input tensors without modifying
attention metadata.
"""

from .api import cp_shard_inputs

__all__ = ["cp_shard_inputs"]
