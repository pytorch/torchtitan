# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hook for autoresearch agents to insert optimized kernels.

Autoresearch agents implement ``autoresearch_kernel_pass`` to apply
custom kernel replacements to the FX graph.  This is a placeholder
that agents will fill in with their optimized transformations.
"""

from __future__ import annotations

import torch


def autoresearch_kernel_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Placeholder for autoresearch agent kernel optimizations."""
    return gm
