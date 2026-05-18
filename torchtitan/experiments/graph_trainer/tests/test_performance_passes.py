# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.performance_passes import (
    annotate_rmsnorm_for_regional_inductor_pass,
)


class TestAnnotateRMSNormForRegionalInductorPass(TestCase):
    """Unit tests for annotate_rmsnorm_for_regional_inductor_pass."""

    def _build_rmsnorm_gm(self, node_specs):
        """Build a GraphModule with fused RMSNorm ops and other ops.

        Args:
            node_specs: List of op targets. For ``_fused_rms_norm`` and
                ``_fused_rms_norm_backward`` nodes, ``getitem`` users are
                automatically appended (mirroring the real traced graph
                structure).
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        w = graph.placeholder("w")
        last = x

        _FUSED_TARGETS = {
            torch.ops.aten._fused_rms_norm.default,
            torch.ops.aten._fused_rms_norm_backward.default,
        }

        for target in node_specs:
            if target in _FUSED_TARGETS:
                # Mimic the real graph: fused op returns a tuple,
                # followed by getitem nodes extracting elements.
                if target == torch.ops.aten._fused_rms_norm.default:
                    fused = graph.call_function(target, args=(last, [256], w, 1e-5))
                else:
                    fused = graph.call_function(
                        target, args=(last, w, last, [256], 1e-5)
                    )
                gi0 = graph.call_function(operator.getitem, args=(fused, 0))
                gi1 = graph.call_function(operator.getitem, args=(fused, 1))
                last = gi0
            else:
                last = graph.call_function(target, args=(last,))

        graph.output(last)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def _count_tagged_nodes(self, gm):
        """Count nodes that have compile_with_inductor in their custom metadata."""
        count = 0
        for node in gm.graph.nodes:
            custom = node.meta.get("custom", {})
            if "compile_with_inductor" in custom:
                count += 1
        return count

    def test_tags_fused_rmsnorm_and_getitems(self):
        """_fused_rms_norm nodes and their getitem users are tagged."""
        gm = self._build_rmsnorm_gm(
            [
                torch.ops.aten._fused_rms_norm.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.add.Tensor,
            ]
        )

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        # 1 fused node + 2 getitem users = 3 tagged nodes
        self.assertEqual(self._count_tagged_nodes(gm), 3)

    def test_does_not_tag_non_rmsnorm_nodes(self):
        """Nodes that are not _fused_rms_norm targets are not tagged."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        n1 = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, x))
        n2 = graph.call_function(torch.ops.aten.add.Tensor, args=(n1, x))
        graph.output(n2)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        self.assertEqual(self._count_tagged_nodes(gm), 0)

    def test_fwd_and_bwd_both_tagged(self):
        """Forward and backward fused norms and their getitems are all tagged."""
        gm = self._build_rmsnorm_gm(
            [
                torch.ops.aten._fused_rms_norm.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten._fused_rms_norm_backward.default,
            ]
        )

        annotate_rmsnorm_for_regional_inductor_pass(gm)

        # 2 fused nodes + 2*2 getitem users = 6 tagged nodes
        self.assertEqual(self._count_tagged_nodes(gm), 6)

    def test_custom_compile_config_propagated(self):
        """A custom compile config is wrapped under inductor_configs."""
        gm = self._build_rmsnorm_gm([torch.ops.aten._fused_rms_norm.default])

        config = {"max_autotune": True, "coordinate_descent_tuning": True}
        annotate_rmsnorm_for_regional_inductor_pass(gm, rmsnorm_compile_config=config)

        for node in gm.graph.nodes:
            annotation = node.meta.get("custom", {}).get("compile_with_inductor")
            if annotation is not None:
                self.assertEqual(annotation["inductor_configs"], config)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
