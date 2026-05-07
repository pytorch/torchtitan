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
            _get_reload_layer,
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
            layer_id = _get_reload_layer(node)
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

    def test_view_chain_only_backward_users_not_offloaded(self):
        """Nodes whose only backward users come through a view chain are not offloaded.

        Views are excluded from offloading, and the base tensor (mm) has no
        direct backward users — only indirect ones through the view. Since
        apply_cpu_offload_pass only collects direct backward users, the base
        tensor won't get offload ops inserted even if tagged.
        """
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Forward layer 0: mm -> view -> relu
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

        # Forward layer 1
        mm2 = graph.call_function(torch.ops.aten.mm.default, args=(relu, relu))
        mm2.meta["autograd_backward"] = False
        mm2.meta["custom"] = {"module_fqn": "layers.1.block"}
        mm2.meta["val"] = self._make_fake_val()

        # Backward: uses view output (NOT mm directly)
        bwd = graph.call_function(torch.ops.aten.mm.default, args=(mm2, view))
        bwd.meta["autograd_backward"] = True
        bwd.meta["custom"] = {"module_fqn": "layers.0.block"}
        bwd.meta["val"] = self._make_fake_val()

        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        tag_all_offloadable_activations(gm)
        gm = apply_cpu_offload_pass(gm)

        # No view replay — only one view op (the original forward one)
        view_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        ]
        self.assertEqual(len(view_ops), 1, "Expected only the original view op")

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
