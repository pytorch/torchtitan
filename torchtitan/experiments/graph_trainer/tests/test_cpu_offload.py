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
        """cpu_offload_pass should be a no-op when no nodes are tagged."""
        from torchtitan.experiments.graph_trainer.cpu_offload import cpu_offload_pass

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()
        node = graph.call_function(torch.ops.aten.relu.default, args=(x,))
        node.meta["val"] = self._make_fake_val()
        graph.output(node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        node_count_before = len(list(gm.graph.nodes))
        gm = cpu_offload_pass(gm)
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

        # Each offloaded tensor gets a dealloc to free GPU storage in forward
        dealloc_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.dealloc.default
        ]
        self.assertEqual(
            len(dealloc_ops),
            len(offload_ops),
            "Each offloaded tensor should have a matching dealloc",
        )

    def test_dealloc_after_last_forward_user(self):
        """Dealloc nodes must appear after all forward users of the offloaded tensor."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=3)
        tag_all_offloadable_activations(gm)
        gm = apply_cpu_offload_pass(gm)

        nodes = list(gm.graph.nodes)
        pos = {n: i for i, n in enumerate(nodes)}

        for node in nodes:
            if not (
                node.op == "call_function"
                and node.target is torch.ops.ao.dealloc.default
            ):
                continue
            gpu_node = node.args[0]
            # dealloc must come after every user of the gpu tensor
            for user in gpu_node.users:
                if user is node:
                    continue
                self.assertGreater(
                    pos[node],
                    pos[user],
                    f"dealloc of {gpu_node.name} at {pos[node]} "
                    f"should be after user {user.name} at {pos[user]}",
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
            prefetch_offloads,
            tag_all_offloadable_activations,
        )

        gm, fwd_nodes, bwd_nodes = self._build_joint_graph(num_layers=4)
        tag_all_offloadable_activations(gm)
        apply_cpu_offload_pass(gm)

        # Record pre-prefetch reload positions
        nodes_before = list(gm.graph.nodes)
        reload_positions_before = {
            n: nodes_before.index(n)
            for n in nodes_before
            if n.op == "call_function" and n.target is torch.ops.ao.reload.default
        }
        self.assertGreater(len(reload_positions_before), 0)

        prefetch_offloads(gm, n_layers=1)

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
        from torchtitan.experiments.graph_trainer.cpu_offload import prefetch_offloads

        gm, _, _ = self._build_joint_graph(num_layers=3)
        nodes_before = len(list(gm.graph.nodes))
        prefetch_offloads(gm, n_layers=1)
        nodes_after = len(list(gm.graph.nodes))
        self.assertEqual(nodes_before, nodes_after)

    def test_prefetch_via_cpu_offload_pass(self):
        """cpu_offload_pass with prefetch_n_layers should insert and move reloads."""
        from torchtitan.experiments.graph_trainer.cpu_offload import cpu_offload_pass

        gm, _, _ = self._build_joint_graph(num_layers=4)
        gm = cpu_offload_pass(gm, prefetch_n_layers=1)

        reload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.reload.default
        ]
        wait_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.ao.wait_tensor.default
            and isinstance(n.args[0], torch.fx.Node)
            and n.args[0].target is torch.ops.ao.reload.default
        ]
        self.assertGreater(len(reload_ops), 0)
        # Each reload should have a corresponding wait after it (not adjacent)
        nodes = list(gm.graph.nodes)
        for wait_node in wait_ops:
            reload_node = wait_node.args[0]
            self.assertLess(nodes.index(reload_node), nodes.index(wait_node))

    def test_view_replay_in_backward(self):
        """View replay: base tensor with view-chain backward users gets offloaded + deallocated."""
        from torchtitan.experiments.graph_trainer.cpu_offload import (
            apply_cpu_offload_pass,
            tag_all_offloadable_activations,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = self._make_fake_val()

        # Forward layer 0: mm -> view -> relu
        # bwd uses view (not mm directly), so mm only has view-chain backward users
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

        tagged = [
            n
            for n in gm.graph.nodes
            if n.meta.get("recompute") is CheckpointPolicy.MUST_CPU_OFFLOAD
        ]
        self.assertGreater(
            len(tagged), 0, "mm should be tagged via view chain backward users"
        )

        gm = apply_cpu_offload_pass(gm)

        offload_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.offload.default
        ]
        dealloc_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.ao.dealloc.default
        ]
        self.assertGreater(len(offload_ops), 0, "Expected offload ops")
        self.assertEqual(
            len(dealloc_ops),
            len(offload_ops),
            "View replay should allow dealloc for all offloaded tensors",
        )

        # Verify replayed view exists in backward
        view_ops = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        ]
        self.assertEqual(len(view_ops), 2, "Expected original + replayed view op")
        replayed = [v for v in view_ops if v.meta.get("autograd_backward")]
        self.assertEqual(len(replayed), 1, "Expected one replayed view in backward")

        # Backward consumer should reference the replayed view, not the original
        bwd_nodes = [
            n
            for n in gm.graph.nodes
            if n.meta.get("autograd_backward") and n.target is torch.ops.aten.mm.default
        ]
        for b in bwd_nodes:
            for arg in b.args:
                if (
                    isinstance(arg, torch.fx.Node)
                    and arg.target is torch.ops.aten.view.default
                ):
                    self.assertTrue(
                        arg.meta.get("autograd_backward"),
                        "Backward consumer should use replayed view",
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
