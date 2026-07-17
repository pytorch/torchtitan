"""Runnable two-layer proof for make_fx-native nested region capture."""

import contextlib
import copy
import importlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.fx.experimental.symbolic_shapes
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph
from torch.nn.utils import stateless

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)


@dataclass
class RegionEntry:
    identifier: str
    body: Any
    output_spec: list[pytree.TreeSpec | None]
    state_names: tuple[str, ...]


class MakeFxRegionRegistry:
    def __init__(self) -> None:
        self.entries: dict[tuple[Any, ...], RegionEntry] = {}

    def __call__(self, fn, *args, **kwargs):
        if kwargs:
            raise AssertionError("This focused proof only supports positional arguments")
        layer, *user_args = args
        if not isinstance(layer, nn.Module):
            raise AssertionError("Expected the marked region's first argument to be a module")

        state = {
            **dict(layer.named_parameters(remove_duplicate=False)),
            **dict(layer.named_buffers(remove_duplicate=False)),
        }
        state_names = tuple(state)
        reuse_hash_fn = fn.__marked_compile_region_reuse_hash_fn__
        reuse_key = reuse_hash_fn(*args, **kwargs)
        cache_key = (fn.__code__, reuse_key, state_names)

        entry = self.entries.get(cache_key)
        if entry is None:
            representative = layer
            output_spec: list[pytree.TreeSpec | None] = [None]
            identifier = f"nested_region_{len(self.entries)}"

            def body(*flat_operands):
                state_values = flat_operands[: len(state_names)]
                call_args = flat_operands[len(state_names) :]
                current_state = dict(zip(state_names, state_values, strict=True))
                with stateless._reparametrize_module(representative, current_state):
                    result = fn(representative, *call_args)
                flat_output, spec = pytree.tree_flatten(result)
                if output_spec[0] is None:
                    output_spec[0] = spec
                else:
                    assert output_spec[0] == spec
                return tuple(flat_output)

            entry = RegionEntry(identifier, body, output_spec, state_names)
            self.entries[cache_key] = entry
        else:
            assert entry.state_names == state_names

        flat_output = invoke_subgraph(
            entry.body,
            entry.identifier,
            *state.values(),
            *user_args,
        )
        if entry.output_spec[0] is None:
            raise AssertionError("Nested body did not record its output spec")
        return pytree.tree_unflatten(list(flat_output), entry.output_spec[0])


@contextlib.contextmanager
def capture_marked_regions(registry: MakeFxRegionRegistry):
    invoke_module = importlib.import_module(
        "torch._higher_order_ops.invoke_subgraph"
    )
    original = invoke_module.invoke_subgraph_placeholder
    invoke_module.invoke_subgraph_placeholder = registry
    try:
        yield
    finally:
        invoke_module.invoke_subgraph_placeholder = original


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.bias = nn.Parameter(torch.randn(4))

    @torch.compiler.nested_compile_region(
        reuse_hash_fn=lambda _layer, _x: 0,
    )
    def forward(self, x):
        return torch.sin(torch.nn.functional.linear(x, self.weight, self.bias))


class TwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_train_step(model):
    def train_step(x):
        output = model(x)
        loss = output.square().sum()
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        return output, loss, grads

    return train_step


def main() -> None:
    torch.manual_seed(0)
    model = TwoLayerModel()
    reference_model = copy.deepcopy(model)
    x = torch.randn(3, 4)

    reference = make_train_step(reference_model)(x)
    registry = MakeFxRegionRegistry()
    with capture_marked_regions(registry):
        traced = minimal_fx_tracer(make_train_step(model), module=model)(x)
    actual = run_traced(traced, module=model)(x)

    reference_flat = pytree.tree_leaves(reference)
    actual_flat = pytree.tree_leaves(actual)
    assert len(reference_flat) == len(actual_flat)
    for expected, result in zip(reference_flat, actual_flat, strict=True):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    hop_nodes = [
        node
        for node in traced.gm.graph.nodes
        if node.op == "call_function" and node.target is invoke_subgraph
    ]
    identifiers = Counter(node.args[1] for node in hop_nodes)
    subgraph_targets = {
        node.args[0].target
        for node in hop_nodes
        if isinstance(node.args[0], torch.fx.Node) and node.args[0].op == "get_attr"
    }

    assert len(registry.entries) == 1
    assert identifiers["fw_nested_region_0"] == 2
    assert identifiers["bw_nested_region_0_0"] == 2
    assert len(subgraph_targets) == 2  # One shared forward body and one shared backward body.

    print(f"region_entries={len(registry.entries)}")
    print(f"invoke_subgraph_identifiers={dict(identifiers)}")
    print(f"unique_forward_backward_bodies={sorted(subgraph_targets)}")
    for node in hop_nodes:
        print(f"hop_call={node.format_node()}")
    print("forward_and_parameter_gradients_match=True")


if __name__ == "__main__":
    main()
