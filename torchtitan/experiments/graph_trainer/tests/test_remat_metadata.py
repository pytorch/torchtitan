# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
    annotate_module_fqns,
    compute_annotated_loss,
    compute_parameter_gradients,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_full_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.linear(x))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(4), _Block(4)])
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2, bias=False)
        self.tied_weight = self.layers[0].linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class TestRematMetadata(TestCase):
    def test_traced_forward_backward_metadata_contracts(self) -> None:
        model = _ToyModel()
        annotate_module_fqns(model)

        def loss_fn(
            prediction: torch.Tensor, target: torch.Tensor, **_kwargs
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(prediction, target)

        def full_step(x: torch.Tensor, target: torch.Tensor) -> list[torch.Tensor]:
            loss = compute_annotated_loss(loss_fn, model(x), target)
            named_params = [
                (name, parameter)
                for name, parameter in model.named_parameters(remove_duplicate=False)
                if parameter.requires_grad
            ]
            grads = compute_parameter_gradients(loss, named_params)
            updated_params = [
                parameter - 0.01 * grad
                for (_, parameter), grad in zip(named_params, grads, strict=True)
            ]
            return [loss, *updated_params, loss.detach()]

        x = torch.randn(2, 4)
        target = torch.randn(2, 2)
        traced = minimal_fx_tracer(full_step, module=model)(x, target)
        expected_outputs = run_traced(traced, module=model)(x, target)

        graph_outputs = set(traced.gm.graph.find_nodes(op="output")[0].all_input_nodes)
        marker_nodes = [
            node
            for node in traced.gm.graph.nodes
            if node.target is torch.ops.aten.alias.default
            and PARAMETER_GRADIENT_FQNS_META in node.meta.get("custom", {})
        ]
        marker_fqn_sets = tuple(
            node.meta["custom"][PARAMETER_GRADIENT_FQNS_META] for node in marker_nodes
        )
        markers_are_internal = all(node not in graph_outputs for node in marker_nodes)
        shared_gradient = marker_nodes[0].args[0]
        assert isinstance(shared_gradient, torch.fx.Node)
        shared_gradient.meta.setdefault("custom", {})["existing_metadata"] = "kept"

        remove_parameter_gradient_markers_pass(traced.gm, traced.example_inputs)
        gradient_fqn_sets = {
            frozenset(node.meta.get("custom", {}).get(PARAMETER_GRADIENT_FQNS_META, ()))
            for node in traced.gm.graph.nodes
            if PARAMETER_GRADIENT_FQNS_META in node.meta.get("custom", {})
        }
        marker_aliases_remain = any(
            node.target is torch.ops.aten.alias.default
            and PARAMETER_GRADIENT_FQNS_META in node.meta.get("custom", {})
            for node in traced.gm.graph.nodes
        )

        tag_sac_policy(traced.gm, policy_fn=_make_full_memory_policy())

        boundary_edges = set()
        for node in traced.gm.graph.nodes:
            if node.meta.get("recompute") != CheckpointPolicy.MUST_SAVE:
                continue
            for user in node.users:
                if user.meta.get("recompute") in (
                    CheckpointPolicy.PREFER_RECOMPUTE,
                    CheckpointPolicy.MUST_RECOMPUTE,
                ):
                    boundary_edges.add((_get_module_fqn(node), _get_module_fqn(user)))

        selective_activation_remat_pass(traced.gm)

        remat_regions: dict[str, set[int]] = defaultdict(set)
        for node in traced.gm.graph.nodes:
            layer_id = _get_layer_id(node)
            if layer_id < 0:
                continue
            if node.name.endswith("_recomputed"):
                remat_regions["recompute"].add(layer_id)
            elif _is_backward_node(node):
                remat_regions["backward"].add(layer_id)

        # This one structural comparison proves that SAC saves both ordinary
        # and final-layer boundaries, remat exposes matching per-layer regions,
        # and parameter identities survive as internal optimizer inputs rather
        # than relying on the graph-output layout. Marker cleanup merges the two
        # tied-parameter FQNs onto their shared gradient node.
        self.assertEqual(
            {
                "boundary_edges": boundary_edges,
                "remat_regions": dict(remat_regions),
                "marker_fqn_sets": marker_fqn_sets,
                "markers_are_internal": markers_are_internal,
                "gradient_fqn_sets": gradient_fqn_sets,
                "marker_aliases_remain": marker_aliases_remain,
                "existing_metadata": shared_gradient.meta["custom"].get(
                    "existing_metadata"
                ),
            },
            {
                "boundary_edges": {
                    ("layers.0", "layers.1.linear"),
                    ("layers.1", "norm"),
                },
                "remat_regions": {
                    "backward": {0, 1},
                    "recompute": {0, 1},
                },
                "marker_fqn_sets": (
                    ("tied_weight",),
                    ("layers.0.linear.weight",),
                    ("layers.1.linear.weight",),
                    ("norm.weight",),
                    ("norm.bias",),
                    ("head.weight",),
                ),
                "markers_are_internal": True,
                "gradient_fqn_sets": {
                    frozenset({"tied_weight", "layers.0.linear.weight"}),
                    frozenset({"layers.1.linear.weight"}),
                    frozenset({"norm.weight"}),
                    frozenset({"norm.bias"}),
                    frozenset({"head.weight"}),
                },
                "marker_aliases_remain": False,
                "existing_metadata": "kept",
            },
        )
        actual_outputs = run_traced(traced, module=model)(x, target)
        for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
