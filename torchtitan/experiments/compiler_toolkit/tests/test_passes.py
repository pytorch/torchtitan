# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor as is_all_gather,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.utils.checkpoint import checkpoint

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.compiler_toolkit.graph_utils import export_joint
from torchtitan.experiments.compiler_toolkit.passes import reassign_to_pg_pass
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel


class ToyModel(nn.Module):
    """A small toy model with multiple linear layers and activation
    checkpointing so that the backward graph recomputes the forward
    all-gathers."""

    def __init__(self, dim=16, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(
                lambda m, inp: torch.relu(m(inp)),
                layer,
                x,
                use_reentrant=False,
            )
        return x


class TestReassignToPgPass(FSDPTest):
    """Integration tests: toy model + simple_fsdp + export_joint + reassign_to_pg_pass."""

    def _setup(self):
        """Set up ParallelDims and device mesh for FSDP."""
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

    def _make_fsdp_model(self, dim=16, n_layers=3):
        """Create a toy model and apply simple_fsdp data_parallel."""
        model = ToyModel(dim, n_layers).cuda()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = data_parallel(model, device_mesh=fsdp_mesh, mode="fully_shard")
        return model

    def _get_fsdp_pg_name(self):
        """Get the FSDP process group name from the mesh."""
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        return fsdp_mesh.get_group().group_name

    def _export_and_get_bw_graph(self, model, inputs):
        """Export the joint graph and capture the backward graph via
        aot_compile_joint_with_descriptors with a custom bw_compiler."""
        joint_with_descriptors, tracing_context = export_joint(model, (inputs,))

        captured_bw_gm = {}

        def capture_bw_compiler(gm, example_inputs):
            captured_bw_gm["gm"] = gm
            captured_bw_gm["example_inputs"] = example_inputs
            return gm

        with tracing(tracing_context):
            aot_compile_joint_with_descriptors(
                joint_with_descriptors,
                bw_compiler=capture_bw_compiler,
            )

        return captured_bw_gm["gm"], captured_bw_gm["example_inputs"]

    def _count_ag_nodes_with_pg(self, gm, pg_name):
        """Count all-gather nodes in the graph that use the given PG name."""
        count = 0
        for node in gm.graph.nodes:
            if is_all_gather(node) and node.args[2] == pg_name:
                count += 1
        return count

    def _count_all_ag_nodes(self, gm):
        """Count all all-gather nodes in the graph regardless of PG."""
        count = 0
        for node in gm.graph.nodes:
            if is_all_gather(node):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_reassign_rewrites_ag_nodes(self):
        """Apply reassign_to_pg_pass on the real backward graph and verify
        that all-gather nodes are rewritten to the target PG."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()
        target_pg_name = "test_target_pg"

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        # Before: all AG nodes should use the FSDP PG
        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0, "Expected AG nodes with FSDP PG name")

        # Apply the pass
        reassign_to_pg_pass(bw_gm, bw_example_inputs, fsdp_pg_name, target_pg_name)

        # After: AG nodes should use the target PG
        ag_with_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_with_new = self._count_ag_nodes_with_pg(bw_gm, target_pg_name)

        self.assertEqual(ag_with_old, 0, "No AG nodes should still use the old PG")
        self.assertEqual(
            ag_with_new, ag_before, "All AG nodes should now use the target PG"
        )

    def test_reassign_preserves_total_ag_count(self):
        """The pass should not add or remove AG nodes, only rewrite PG names."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        total_before = self._count_all_ag_nodes(bw_gm)
        reassign_to_pg_pass(bw_gm, bw_example_inputs, fsdp_pg_name, "new_pg")
        total_after = self._count_all_ag_nodes(bw_gm)

        self.assertEqual(total_before, total_after)

    def test_reassign_with_non_matching_pg_is_noop(self):
        """If the source PG name doesn't match any AG node, nothing changes."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)

        # Use a non-matching source PG name
        reassign_to_pg_pass(bw_gm, bw_example_inputs, "nonexistent_pg", "target_pg")

        # FSDP AG nodes should be unchanged
        ag_after = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertEqual(ag_before, ag_after)

    def test_reassign_with_extra_pg(self):
        """Test the production-like flow: create an extra FSDP PG and
        reassign AG nodes to it."""
        self._setup()
        model = self._make_fsdp_model()
        inputs = torch.randn(4, 16).cuda()
        fsdp_pg_name = self._get_fsdp_pg_name()

        # Create an extra PG mirroring the FSDP topology
        from torchtitan.experiments.compiler_toolkit.common_utils import (
            create_extra_fsdp_pg,
            get_extra_fsdp_pg_name,
        )

        create_extra_fsdp_pg(self.parallel_dims)
        extra_pg_name = get_extra_fsdp_pg_name(fsdp_pg_name)

        bw_gm, bw_example_inputs = self._export_and_get_bw_graph(model, inputs)

        ag_before = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        self.assertGreater(ag_before, 0)

        # Reassign to the real extra PG
        reassign_to_pg_pass(bw_gm, bw_example_inputs, fsdp_pg_name, extra_pg_name)

        ag_old = self._count_ag_nodes_with_pg(bw_gm, fsdp_pg_name)
        ag_new = self._count_ag_nodes_with_pg(bw_gm, extra_pg_name)

        self.assertEqual(ag_old, 0)
        self.assertEqual(ag_new, ag_before)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
