"""Trace all six Llama debug-model blocks through one shared nested body."""

import copy
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
import torch.fx.experimental.symbolic_shapes
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

from prototype import (
    MakeFxRegionRegistry,
    capture_marked_regions,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)


def stub_unused_cuda_ep() -> None:
    if importlib.util.find_spec("triton") is not None:
        return

    api = types.ModuleType("torchtitan.distributed.minimal_async_ep.api")

    def unavailable(*args, **kwargs):
        raise AssertionError("The dense CPU proof reached minimal async EP")

    api.MinimalAsyncEPDispatchMetadata = object
    api.maybe_update_minimal_async_ep_config = lambda *args, **kwargs: None
    api.combine_op = unavailable
    api.dispatch_op = unavailable
    api.init_buffer = unavailable
    sys.modules[api.__name__] = api


def make_train_step(model):
    def train_step(tokens, labels):
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum"
        )
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        return loss, grads

    return train_step


def main() -> None:
    stub_unused_cuda_ep()

    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.models.llama3 import Llama3Model, llama3_configs
    from torchtitan.models.llama3.model import Llama3TransformerBlock

    torch.manual_seed(0)
    config = llama3_configs["debugmodel"](attn_backend="flex")
    for layer_config in config.layers:
        layer_config.attention.inner_attention = ScaledDotProductAttention.Config()

    model = Llama3Model(config).to("cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    reference_model = copy.deepcopy(model)
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    labels = torch.randint(0, config.vocab_size, (1, 4))
    reference = make_train_step(reference_model)(tokens, labels)

    original_forward = Llama3TransformerBlock.forward
    Llama3TransformerBlock.forward = torch.compiler.nested_compile_region(
        reuse_hash_fn=lambda _block, _x, _attention_masks, _positions=None: 0
    )(original_forward)

    registry = MakeFxRegionRegistry()
    try:
        with capture_marked_regions(registry):
            traced = minimal_fx_tracer(
                make_train_step(model), module=model
            )(tokens, labels)
    finally:
        Llama3TransformerBlock.forward = original_forward

    actual = run_traced(traced, module=model)(tokens, labels)
    for expected, result in zip(
        pytree.tree_leaves(reference), pytree.tree_leaves(actual), strict=True
    ):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    hop_nodes = [
        node
        for node in traced.gm.graph.nodes
        if node.op == "call_function" and node.target is invoke_subgraph
    ]
    identifiers = Counter(node.args[1] for node in hop_nodes)
    subgraph_targets = Counter(
        node.args[0].target
        for node in hop_nodes
        if isinstance(node.args[0], torch.fx.Node) and node.args[0].op == "get_attr"
    )
    assert len(registry.entries) == 1
    assert identifiers["fw_nested_region_0"] == len(model.layers)
    assert identifiers["bw_nested_region_0_0"] == len(model.layers)
    assert subgraph_targets == {
        "repeated_subgraph0": len(model.layers),
        "repeated_subgraph1": len(model.layers),
    }

    graph_text = traced.gm.print_readable(
        print_output=False,
        include_stride=True,
        include_device=True,
        expanded_def=True,
    )
    graph_path = Path(__file__).with_name("llama_six_block_fx_graph.txt")
    graph_path.write_text(graph_text)

    print(f"model={type(model).__name__}")
    print(f"layers={len(model.layers)}")
    print(f"parameters={sum(p.numel() for p in model.parameters())}")
    print("region_scope=Llama3TransformerBlock.forward")
    print(f"region_entries={len(registry.entries)}")
    print(f"invoke_subgraph_identifiers={dict(identifiers)}")
    print(f"subgraph_targets={dict(subgraph_targets)}")
    print("loss_and_parameter_gradients_match=True")
    print(f"graph_path={graph_path}")
    print(graph_text)


if __name__ == "__main__":
    main()
