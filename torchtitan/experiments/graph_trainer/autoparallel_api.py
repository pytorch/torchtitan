# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallelGraph: solver-based SPMD sharding that returns a parallelized
joint graph (JointWithDescriptors) for graph_trainer's compilation pipeline.

The key difference from AutoParallel.apply_placement() is that this stops
after sharding instead of going all the way to compilation — the returned
JointWithDescriptors feeds into graph_trainer's joint_graph_builder (non-PP)
or GraphPPRunner (PP).
"""

import logging
from typing import Any

from autoparallel.api import AutoParallel
from autoparallel.module_construction import make_parallel_module

logger = logging.getLogger(__name__)


class AutoParallelGraph(AutoParallel):
    """AutoParallel variant that returns the parallelized joint graph.

    Usage::

        with AutoParallelGraph(model, input_fn, mesh, ...) as autop:
            autop.add_input_constraints(...)
            placement = autop.optimize_placement()
            result = autop.apply_placement(placement)

        jwd = result["joint_with_descriptors"]
        parallel_model = result["parallel_model"]
    """

    def apply_placement(self, sharding_placement=None) -> dict[str, Any]:
        """Apply SPMD sharding and return the parallelized joint graph.

        Unlike AutoParallel.apply_placement() which compiles the graph,
        this returns early with the JointWithDescriptors so graph_trainer's
        compilation pipeline can take over.
        """
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )

        parallel_model = make_parallel_module(
            self.model, sharded_param_dict, sharded_buffer_dict
        )

        return {
            "joint_with_descriptors": self.joint_with_descriptors,
            "parallel_model": parallel_model,
            "sharded_param_dict": sharded_param_dict,
            "sharded_buffer_dict": sharded_buffer_dict,
        }
