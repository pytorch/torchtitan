# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import CheckpointPolicy


class TestCpuOffloadPass(TestCase):
    """Unit tests for the CPU offload pass on synthetic FX graphs.

    Tests use hand-built graphs that simulate the joint fwd+bwd structure
    produced by make_fx, with ``autograd_backward`` for forward/backward
    classification and ``module_fqn`` for layer boundaries.
    """

    def _make_fake_val(self, shape=(64, 64), dtype=torch.float32, device="cuda:0"):
        """Create a real tensor for use as node.meta["val"].

        The offload pass reads .nelement(), .element_size(), .is_contiguous(),
        .device, and .to() from node metadata values.
        """
        return torch.empty(shape, dtype=dtype, device=device)

    def _build_joint_graph(self, num_layers=3, ops_per_layer=2):
        """Build a synthetic joint fwd+bwd graph with layer annotations.

        Structure per layer:
          Forward: mm -> relu
          Backward: mm_bwd -> relu_bwd

        All nodes get ``module_fqn = "layers.<layer_id>.block"`` in their custom
        metadata. Backward nodes have ``autograd_backward = True``.

        Forward mm nodes have backward consumers (relu_bwd uses fwd mm output),
        making them eligible for offloading.

        Returns:
            (gm, fwd_nodes, bwd_nodes) where fwd/bwd_nodes are lists of nodes
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        fwd_nodes = []
        bwd_nodes = []
        last_fwd = x

        # Forward pass: layer 0 -> layer N-1
        for layer_id in range(num_layers):
            for op_idx in range(ops_per_layer):
                if op_idx == 0:
                    node = graph.call_function(
                        torch.ops.aten.mm.default, args=(last_fwd, last_fwd)
                    )
                else:
                    node = graph.call_function(
                        torch.ops.aten.relu.default, args=(last_fwd,)
                    )
                node.meta["autograd_backward"] = False
                node.meta["custom"] = {"module_fqn": f"layers.{layer_id}.block"}
                node.meta["val"] = self._make_fake_val()
                fwd_nodes.append(node)
                last_fwd = node

        # Backward pass: layer N-1 -> layer 0
        last_bwd = last_fwd
        for layer_id in reversed(range(num_layers)):
            for op_idx in reversed(range(ops_per_layer)):
                # Backward node: uses the corresponding forward node's output
                fwd_idx = layer_id * ops_per_layer + op_idx
                fwd_node = fwd_nodes[fwd_idx]
                node = graph.call_function(
                    torch.ops.aten.mm.default, args=(last_bwd, fwd_node)
                )
                node.meta["autograd_backward"] = True
                node.meta["custom"] = {"module_fqn": f"layers.{layer_id}.block"}
                node.meta["val"] = self._make_fake_val()
                bwd_nodes.append(node)
                last_bwd = node

        graph.output(last_bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        return gm, fwd_nodes, bwd_nodes

    def test_tag_all_offloadable_activations_basic(self):
        """Forward nodes with backward consumers should be tagged MUST_CPU_OFFLOAD."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=3)
        tag_all_offloadable_activations(gm)

        # Last layer (layer 2) should NOT be tagged (skipped)
        # Layers 0 and 1 should have some tagged nodes
        tagged_nodes = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertGreater(len(tagged_nodes), 0, "Expected some nodes to be tagged")

        # Verify no last-layer nodes are tagged
        for node in tagged_nodes:
            fqn = node.meta.get("custom", {}).get("module_fqn", "")
            self.assertNotIn(
                "layers.2", fqn, "Last layer nodes should not be tagged for offload"
            )

    def test_must_save_sym_int_not_offloaded(self):
        """A MUST_SAVE sym-int node must survive tag_all_offloadable_activations.

        tag_sac_policy force-saves sym-int shape reads (sym_size etc.). The
        offload pass only considers nodes whose meta["val"] is a real tensor, so
        a sym-int (whose val is a SymInt) must be left untouched -- otherwise its
        MUST_SAVE tag would be flipped to MUST_CPU_OFFLOAD.
        """
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        shape_env = ShapeEnv()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Forward tensor in a non-last layer: a genuine offload candidate.
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = self._make_fake_val()

        # Sym-int shape read of mm in the same non-last layer, force-saved as
        # tag_sac_policy would. Its val is a SymInt, not a tensor.
        sym = graph.call_function(torch.ops.aten.sym_size.int, args=(mm, 0))
        sym.meta["autograd_backward"] = False
        sym.meta["custom"] = {"module_fqn": "layers.0.block"}
        sym.meta["val"] = shape_env.create_unbacked_symint()
        sym.meta["recompute"] = CheckpointPolicy.MUST_SAVE

        # Backward consumers (last layer): mm feeds a bwd matmul; sym feeds a
        # bwd view's size arg -- both forward values are live into backward.
        bwd_mm = graph.call_function(torch.ops.aten.mm.default, args=(mm, mm))
        bwd_mm.meta["autograd_backward"] = True
        bwd_mm.meta["custom"] = {"module_fqn": "layers.1.block"}
        bwd_mm.meta["val"] = self._make_fake_val()

        bwd_view = graph.call_function(
            torch.ops.aten.view.default, args=(bwd_mm, [sym, sym])
        )
        bwd_view.meta["autograd_backward"] = True
        bwd_view.meta["custom"] = {"module_fqn": "layers.1.block"}
        bwd_view.meta["val"] = self._make_fake_val()

        graph.output(bwd_view)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        # The sym-int keeps its MUST_SAVE tag (never flipped to CPU offload)...
        self.assertIs(sym.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        # ...while the real tensor in the same non-last layer WAS offloaded,
        # proving the pass ran and would have touched the sym node if eligible.
        self.assertIs(mm.meta["recompute"], CheckpointPolicy.MUST_CPU_OFFLOAD)

    def test_tag_no_backward_consumers(self):
        """Forward nodes without backward consumers should NOT be tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        # Build a graph with only forward nodes (no backward)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        node1 = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        node1.meta["autograd_backward"] = False
        node1.meta["custom"] = {"module_fqn": "layers.0.block"}
        node1.meta["val"] = self._make_fake_val()

        node2 = graph.call_function(torch.ops.aten.relu.default, args=(node1,))
        node2.meta["autograd_backward"] = False
        node2.meta["custom"] = {"module_fqn": "layers.0.block"}
        node2.meta["val"] = self._make_fake_val()

        graph.output(node2)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertEqual(
            len(tagged), 0, "No nodes should be tagged without bwd consumers"
        )

    def test_offload_pass_noop_when_no_tags(self):
        """apply_cpu_offload_pass should be a no-op when no nodes are tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()
        node = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        node.meta["val"] = self._make_fake_val()
        graph.output(node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        node_count_before = len(list(gm.graph.nodes))
        gm = apply_cpu_offload_pass(gm)
        node_count_after = len(list(gm.graph.nodes))

        self.assertEqual(
            node_count_before, node_count_after, "No-op pass should not add nodes"
        )

    def test_offload_reload_ops_inserted(self):
        """Verify offload/reload/wait_tensor ops are inserted for tagged nodes."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=3)
        tag_all_offloadable_activations(gm)

        # Count tagged nodes before applying offload pass
        tagged_count = sum(
            1
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        )
        self.assertGreater(tagged_count, 0)

        gm = apply_cpu_offload_pass(gm)

        # Verify offload, reload, and wait_tensor ops were inserted
        offload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.offload.default
        ]
        reload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.reload.default
        ]
        wait_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.wait_tensor.default
        ]

        self.assertGreater(len(offload_ops), 0, "Expected offload ops")
        self.assertGreater(len(reload_ops), 0, "Expected reload ops")
        # Each offloaded tensor gets 2 waits: one for offload, one for reload
        self.assertEqual(
            len(wait_ops),
            len(offload_ops) + len(reload_ops),
            "Each offload/reload should have a matching wait_tensor",
        )

    def test_view_ops_not_tagged(self):
        """View ops should never be tagged for offload."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Forward: mm -> view (layer 0)
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = self._make_fake_val()

        view = graph.call_function(torch.ops.aten.view.default, args=(mm, [64, 64]))
        view.meta["autograd_backward"] = False
        view.meta["custom"] = {"module_fqn": "layers.0.block"}
        view.meta["val"] = self._make_fake_val()

        # Forward layer 1 to avoid single-layer skip
        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(view, view))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.1.block"}
        mm2.meta["val"] = self._make_fake_val()

        # Backward: uses view output
        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, view))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        # The view node should not be tagged
        view_node = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        ]
        self.assertEqual(len(view_node), 1)
        self.assertIsNot(
            view_node[0].meta.get("recompute"),
            CheckpointPolicy.MUST_CPU_OFFLOAD,
            "View ops should not be tagged for offload",
        )

    def test_small_tensors_not_tagged(self):
        """Tensors smaller than _MIN_OFFLOAD_BYTES should not be tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        # Small tensor: 2x2 float32 = 16 bytes < 4096
        small_val = self._make_fake_val(shape=(2, 2))
        x.meta["val"] = small_val

        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = small_val

        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(mm, mm))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.1.block"}
        mm2.meta["val"] = small_val

        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, mm))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = small_val

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertEqual(len(tagged), 0, "Small tensors should not be tagged")

    def test_prefetch_moves_reloads_earlier(self):
        """Prefetch should move ao.reload N layers earlier in backward."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            prefetch_reloads,
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=4)
        tag_all_offloadable_activations(gm)
        apply_cpu_offload_pass(gm, prefetch_lookahead=0)

        # Record pre-prefetch reload positions
        nodes_before = list(gm.graph.nodes)
        reload_positions_before = {
            n: nodes_before.index(n)
            for n in nodes_before
            if n.op == "call_function" and n.target is torch.ops.ao.reload.default
        }
        self.assertGreater(len(reload_positions_before), 0)

        prefetch_reloads(gm, n_layers=1)

        nodes = list(gm.graph.nodes)
        moved_count = 0
        for node in nodes:
            if not (
                node.op == "call_function"
                and node.target is torch.ops.ao.reload.default
            ):
                continue
            wait_node = next(
                u for u in node.users if u.target is torch.ops.ao.wait_tensor.default
            )
            # Reload must always precede its wait
            self.assertLess(nodes.index(node), nodes.index(wait_node))

            # Check that moved reloads are now earlier than before
            if node in reload_positions_before:
                new_pos = nodes.index(node)
                old_pos = reload_positions_before[node]
                if new_pos < old_pos:
                    moved_count += 1

        self.assertGreater(moved_count, 0, "Expected at least one reload to be moved")

    def test_prefetch_noop_without_offloads(self):
        """Prefetch should be a no-op when no offload ops exist."""
        from torchtitan.experiments.graph_trainer.cpu_offload import prefetch_reloads

        gm, _, _ = self._build_joint_graph(num_layers=3)
        nodes_before = len(list(gm.graph.nodes))
        prefetch_reloads(gm, n_layers=1)
        nodes_after = len(list(gm.graph.nodes))
        self.assertEqual(nodes_before, nodes_after)

    def test_prefetch_via_apply_pass(self):
        """apply_cpu_offload_pass with prefetch_lookahead should insert and move reloads."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        def _reload_positions(graph_module):
            nodes = list(graph_module.graph.nodes)
            return [
                nodes.index(n)
                for n in nodes
                if n.op == "call_function" and n.target is torch.ops.ao.reload.default
            ]

        gm_no_prefetch, _, _ = self._build_joint_graph(num_layers=4)
        tag_all_offloadable_activations(gm_no_prefetch)
        gm_no_prefetch = apply_cpu_offload_pass(gm_no_prefetch, prefetch_lookahead=0)
        pos_no_prefetch = _reload_positions(gm_no_prefetch)

        gm_prefetch, _, _ = self._build_joint_graph(num_layers=4)
        tag_all_offloadable_activations(gm_prefetch)
        gm_prefetch = apply_cpu_offload_pass(gm_prefetch, prefetch_lookahead=1)
        pos_prefetch = _reload_positions(gm_prefetch)

        self.assertGreater(len(pos_prefetch), 0)
        self.assertEqual(len(pos_no_prefetch), len(pos_prefetch))
        # With prefetch, reloads should be earlier in the graph
        earlier_count = sum(1 for a, b in zip(pos_prefetch, pos_no_prefetch) if a < b)
        self.assertGreater(
            earlier_count,
            0,
            "prefetch_lookahead=1 should move reloads earlier than prefetch_lookahead=0",
        )

    def test_prefetch_n_layers_2(self):
        """n_layers=2 should move reloads further than n_layers=1."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        def _reload_positions(graph_module):
            nodes = list(graph_module.graph.nodes)
            return [
                nodes.index(n)
                for n in nodes
                if n.op == "call_function" and n.target is torch.ops.ao.reload.default
            ]

        gm1, _, _ = self._build_joint_graph(num_layers=5)
        tag_all_offloadable_activations(gm1)
        gm1 = apply_cpu_offload_pass(gm1, prefetch_lookahead=1)
        pos_1 = _reload_positions(gm1)

        gm2, _, _ = self._build_joint_graph(num_layers=5)
        tag_all_offloadable_activations(gm2)
        gm2 = apply_cpu_offload_pass(gm2, prefetch_lookahead=2)
        pos_2 = _reload_positions(gm2)

        self.assertEqual(len(pos_1), len(pos_2))
        # n_layers=2 should move at least some reloads further than n_layers=1
        further_count = sum(1 for a, b in zip(pos_2, pos_1) if a < b)
        self.assertGreater(
            further_count,
            0,
            "prefetch_lookahead=2 should move reloads further than prefetch_lookahead=1",
        )

    def test_wait_after_last_forward_consumer(self):
        """Forward waits should be placed after the last forward consumer."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            _is_backward_node,
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        gm, _, _ = self._build_joint_graph(num_layers=4)
        tag_all_offloadable_activations(gm)
        apply_cpu_offload_pass(gm, prefetch_lookahead=0)

        nodes = list(gm.graph.nodes)
        node_pos = {n: i for i, n in enumerate(nodes)}

        for node in nodes:
            if not (
                node.op == "call_function"
                and node.target is torch.ops.ao.wait_tensor.default
                and not node.meta.get("autograd_backward")
            ):
                continue
            gpu_tensor = node.args[1] if len(node.args) > 1 else None
            if gpu_tensor is None:
                continue
            wait_pos = node_pos[node]
            ao_ops = {
                torch.ops.ao.offload.default,
                torch.ops.ao.reload.default,
                torch.ops.ao.wait_tensor.default,
            }
            for user in gpu_tensor.users:
                if user.op != "call_function":
                    continue
                if _is_backward_node(user) or user.target in ao_ops:
                    continue
                self.assertLess(
                    node_pos[user],
                    wait_pos,
                    f"Forward consumer {user.name} at pos {node_pos[user]} "
                    f"should precede wait at pos {wait_pos}",
                )

    def _build_view_chain_graph(self):
        """Build a graph where backward consumers reach the base through a view chain.

        Forward:  mm (layer 0) -> view (layer 0) -> relu (layer 0) -> mm2 (layer 1)
        Backward: bwd_mm uses view output (NOT mm directly)
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = self._make_fake_val()

        view = graph.call_function(torch.ops.aten.view.default, args=(mm, [64, 64]))
        view.meta["autograd_backward"] = False
        view.meta["custom"] = {"module_fqn": "layers.0.block"}
        view.meta["val"] = self._make_fake_val()

        relu = graph.call_function(torch.ops.aten.relu.default, args=(view,))
        relu.meta["autograd_backward"] = False
        relu.meta["custom"] = {"module_fqn": "layers.0.block"}
        relu.meta["val"] = self._make_fake_val()

        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(relu, relu))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.1.block"}
        mm2.meta["val"] = self._make_fake_val()

        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, view))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def test_view_chain_offloaded_with_replay(self):
        """View replay offloads the base tensor and replays views in backward."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        gm = self._build_view_chain_graph()
        tag_all_offloadable_activations(gm)
        gm = apply_cpu_offload_pass(gm)

        offload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.offload.default
        ]
        reload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.reload.default
        ]
        view_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        ]

        self.assertGreater(len(offload_ops), 0, "Base tensor should be offloaded")
        self.assertGreater(len(reload_ops), 0, "Base tensor should be reloaded")
        # 2 views: original forward + replayed backward
        self.assertEqual(len(view_ops), 2, "View should be replayed in backward")

        # The replayed view should be marked as backward
        replayed_views = [v for v in view_ops if v.meta.get("autograd_backward")]
        self.assertEqual(len(replayed_views), 1, "Replayed view should be backward")

        # The backward consumer should use the replayed view, not the forward one
        bwd_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.mm.default
            and n.meta.get("autograd_backward")
        ]
        self.assertEqual(len(bwd_mm), 1)
        # bwd_mm should reference the replayed view (backward), not the original
        self.assertIn(replayed_views[0], bwd_mm[0].args)

    def test_wait_dep_points_to_last_forward_consumer(self):
        """Forward wait_tensor dep arg should reference the last forward consumer."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            _is_backward_node,
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        gm, _, _ = self._build_joint_graph(num_layers=4)
        tag_all_offloadable_activations(gm)
        apply_cpu_offload_pass(gm, defer_n_layers=1, prefetch_lookahead=0)

        nodes = list(gm.graph.nodes)
        node_pos = {n: i for i, n in enumerate(nodes)}
        ao_ops = {
            torch.ops.ao.offload.default,
            torch.ops.ao.reload.default,
            torch.ops.ao.wait_tensor.default,
        }

        for node in nodes:
            if not (
                node.op == "call_function"
                and node.target is torch.ops.ao.wait_tensor.default
                and not node.meta.get("autograd_backward")
            ):
                continue
            # dep is the third arg
            self.assertGreater(len(node.args), 2, "Forward wait should have dep arg")
            dep = node.args[2]
            keepalive = node.args[1]
            self.assertIsNotNone(dep)

            # dep must be a forward non-AO consumer of keepalive's storage
            self.assertFalse(_is_backward_node(dep))
            self.assertNotIn(dep.target, ao_ops)

            # dep must be >= all other forward non-AO consumers by position
            for user in keepalive.users:
                if user.op != "call_function":
                    continue
                if _is_backward_node(user) or user.target in ao_ops:
                    continue
                self.assertGreaterEqual(
                    node_pos[dep],
                    node_pos[user],
                    f"dep {dep.name} should be at or after consumer {user.name}",
                )

    def test_wait_dep_follows_view_chain(self):
        """dep should track through views to find the true last consumer."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Layer 0: mm -> view -> relu (view shares storage with mm)
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = self._make_fake_val()
        mm.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD

        view = graph.call_function(torch.ops.aten.view.default, args=(mm, [64, 64]))
        view.meta["autograd_backward"] = False
        view.meta["custom"] = {"module_fqn": "layers.0.block"}
        view.meta["val"] = self._make_fake_val()

        # Layer 1: consumes the view (which aliases mm's storage)
        relu = graph.call_function(torch.ops.aten.relu.default, args=(view,))
        relu.meta["autograd_backward"] = False
        relu.meta["custom"] = {"module_fqn": "layers.1.block"}
        relu.meta["val"] = self._make_fake_val()

        # Layer 2
        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(relu, relu))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.2.block"}
        mm2.meta["val"] = self._make_fake_val()

        # Backward: uses mm directly
        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, mm))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        apply_cpu_offload_pass(gm, defer_n_layers=1, prefetch_lookahead=0)

        # Find the forward wait for mm's offload
        fwd_waits = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.ao.wait_tensor.default
            and not n.meta.get("autograd_backward")
        ]
        self.assertEqual(len(fwd_waits), 1)
        wait = fwd_waits[0]

        # dep should be relu (layer 1), not mm itself (layer 0), because
        # the view chain extends mm's storage lifetime to relu's consumption.
        dep = wait.args[2]
        self.assertEqual(
            dep.target,
            torch.ops.aten.relu.default,
            "dep should follow view chain to the last consumer of the storage",
        )

    def test_wait_dep_non_tensor_consumer(self):
        """When the last consumer produces a non-Tensor (e.g. sort -> tuple),
        dep should use a Tensor-producing child (getitem) to preserve the
        topo edge in wait_tensor."""
        import operator

        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Layer 0: mm (tagged for offload)
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = self._make_fake_val()
        mm.meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD

        # Layer 1: sort consumes mm's output, returns tuple (non-Tensor val)
        sort = graph.call_function(torch.ops.aten.sort.default, args=(mm,))
        sort.meta["autograd_backward"] = False
        sort.meta["custom"] = {"module_fqn": "layers.1.block"}
        sort.meta["val"] = (self._make_fake_val(), self._make_fake_val())

        # getitem extracts a Tensor from sort's tuple output
        getitem = graph.call_function(operator.getitem, args=(sort, 0))
        getitem.meta["autograd_backward"] = False
        getitem.meta["custom"] = {"module_fqn": "layers.1.block"}
        getitem.meta["val"] = self._make_fake_val()

        # Layer 2
        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(getitem, getitem))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.2.block"}
        mm2.meta["val"] = self._make_fake_val()

        # Backward: uses mm directly
        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, mm))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        apply_cpu_offload_pass(gm, defer_n_layers=1, prefetch_lookahead=0)

        # Find the forward wait for mm's offload
        fwd_waits = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.ao.wait_tensor.default
            and not n.meta.get("autograd_backward")
        ]
        self.assertEqual(len(fwd_waits), 1)
        wait = fwd_waits[0]

        # dep (args[2]) should be the getitem (Tensor child of sort),
        # not None, preserving the topo edge through sort.
        dep = wait.args[2]
        self.assertIsNotNone(dep, "dep should not be None for non-Tensor consumer")
        self.assertIs(
            dep.target,
            operator.getitem,
            "dep should be getitem (Tensor child of the non-Tensor sort node)",
        )
        self.assertIsInstance(
            dep.meta.get("val"),
            torch.Tensor,
            "dep must produce a Tensor to satisfy wait_tensor schema",
        )

    def test_view_replay_multi_view_same_consumer(self):
        """A backward node consuming two different views of the same base must have both redirected."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Layer 0: mm -> view, mm -> reshape (two views of same base)
        mm = graph.call_function(torch.ops.aten.mm.default, args=(x, x))
        mm.meta["autograd_backward"] = False
        mm.meta["custom"] = {"module_fqn": "layers.0.block"}
        mm.meta["val"] = self._make_fake_val()

        view = graph.call_function(torch.ops.aten.view.default, args=(mm, [64, 64]))
        view.meta["autograd_backward"] = False
        view.meta["custom"] = {"module_fqn": "layers.0.block"}
        view.meta["val"] = self._make_fake_val()

        reshape = graph.call_function(
            torch.ops.aten.reshape.default, args=(mm, [64, 64])
        )
        reshape.meta["autograd_backward"] = False
        reshape.meta["custom"] = {"module_fqn": "layers.0.block"}
        reshape.meta["val"] = self._make_fake_val()

        # Layer 1
        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(view, reshape))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.1.block"}
        mm2.meta["val"] = self._make_fake_val()

        # Backward: uses BOTH view and reshape
        bwd = graph.call_function(torch.ops.aten.mm.default, args=(view, reshape))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)
        gm = apply_cpu_offload_pass(gm)

        # Both view and reshape should be replayed in backward
        bwd_views = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target
            in (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
            and n.meta.get("autograd_backward")
        ]
        self.assertEqual(len(bwd_views), 2, "Both view and reshape should be replayed")

        # The backward mm should use BOTH replayed views, not the originals
        bwd_mm = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.aten.mm.default
            and n.meta.get("autograd_backward")
        ]
        self.assertEqual(len(bwd_mm), 1)
        for arg in bwd_mm[0].args:
            if isinstance(arg, torch.fx.Node) and arg.target in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
            ):
                self.assertTrue(
                    arg.meta.get("autograd_backward"),
                    f"Backward mm should use replayed view, not original: {arg.name}",
                )

    def test_single_layer_tagged(self):
        """With only one layer, nodes are still tagged (last-layer skip only applies with multiple layers)."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            tag_all_offloadable_activations,
        )

        gm, _, _ = self._build_joint_graph(num_layers=1)
        tag_all_offloadable_activations(gm)

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertGreater(
            len(tagged), 0, "Single layer nodes can be tagged (last_layer_id=None)"
        )
