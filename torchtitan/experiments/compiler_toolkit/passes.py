# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for the compiler toolkit.

This module provides various compiler passes that can be applied to graph modules
during compilation. Passes can be selected and configured via job config.
"""

import torch
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch.fx.passes.regional_inductor import regional_inductor


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize communication patterns in distributed training.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply regional inductor compilation based on user annotation.
    """
    return regional_inductor(gm, example_inputs)


# Registry mapping pass names to pass functions
AVAILABLE_PASSES = {
    "autobucketing_reordering": autobucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
}
