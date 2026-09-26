# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a pipeline rank keeps what its backward needs."""

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class PPMemoryConfig:
    """How a pipeline rank stores the tensors its backward reads."""

    manager: bool = False
    """Route every tensor autograd saves through one activation storage that holds the rank store's blocks pinned."""
