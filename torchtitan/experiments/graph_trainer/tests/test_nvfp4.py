# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import pytest
import torch
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.inductor_passes import (
    full_inductor_compilation_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.passes import (
    functionalize_scaled_mm_out_pass,
    selective_activation_remat_pass,
)
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.quantization.mxfp8 import MXFP8Linear
from torchtitan.quantization.nvfp4 import NVFP4Linear
from torchtitan.quantization.nvfp4.experts import _get_nvfp4_grouped_linear_cls


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="NVFP4 requires SM100 or later",
)


@pytest.mark.parametrize("projection", ["dense", "w13", "w2", "attention"])
def test_quantized_projection_full_compile(projection):
    torch.manual_seed(42)
    if projection in ("w13", "w2"):
        config = _get_nvfp4_grouped_linear_cls(GroupedLinear).Config(
            group_size=2,
            in_features=128,
            out_features=128,
            num_linears=2 if projection == "w13" else 1,
        )
    else:
        cls = NVFP4Linear if projection == "dense" else MXFP8Linear
        config = cls.Config(
            in_features=128,
            out_features=128,
            bias=False,
            num_linears=2 if projection == "dense" else 1,
        )
    model = config.build().cuda()
    # Quantization consumes BF16 compute weights; optimizer masters remain FP32.
    weight = model.weight
    if hasattr(weight, "_tensor"):
        weight = weight._tensor
    model.weight = torch.nn.Parameter(
        (torch.randn_like(weight.float()) * 0.02).bfloat16()
    )
    model._init_self_buffers(buffer_device=torch.device("cuda"))
    input = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    input.requires_grad_()
    offsets = torch.tensor([128, 256], device="cuda", dtype=torch.int32)

    def train_step(input):
        output = model(input, offsets) if projection in ("w13", "w2") else model(input)
        input_grad, weight_grad = torch.autograd.grad(
            output.float().square().mean(), (input, model.weight)
        )
        return output, input_grad, weight_grad

    rng = torch.cuda.get_rng_state()
    expected = train_step(input)
    torch.cuda.set_rng_state(rng)
    traced = minimal_fx_tracer(train_step, module=model)(input)
    ops = [
        str(node.target) for node in traced.gm.graph.nodes if node.op == "call_function"
    ]
    print(f"{projection} forward/backward traced operations: {ops}")
    traced.gm = functionalize_scaled_mm_out_pass(traced.gm)
    for node in traced.gm.graph.nodes:
        if node.op == "call_function" and not _is_backward_node(node):
            node.meta["recompute"] = (
                CheckpointPolicy.MUST_SAVE
                if torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", ())
                else CheckpointPolicy.MUST_RECOMPUTE
            )
    traced.gm = selective_activation_remat_pass(traced.gm)
    assert any(node.name.endswith("_recomputed") for node in traced.gm.graph.nodes)
    # Eager and Inductor otherwise consume different stochastic-rounding RNG streams.
    traced.gm = full_inductor_compilation_pass(
        traced.gm, traced.example_inputs, inductor_configs={"fallback_random": True}
    )
    calls = [
        node.target for node in traced.gm.graph.nodes if node.op == "call_function"
    ]
    compiled = [
        target
        for target in calls
        if isinstance(getattr(target, "__wrapped__", None), AOTCompiledArtifact)
    ]
    assert (
        len(compiled) == 1
    ), "The full forward/backward graph must compile as one region"
    assert all(target in compiled or target is operator.getitem for target in calls)
    print(f"{projection} compiled artifact: {type(compiled[0].__wrapped__).__name__}")
    torch.cuda.set_rng_state(rng)
    actual = run_traced(traced, module=model)(input)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-4)
    reference_weight = model.weight.detach().float().clone()
    reference_optimizer = torch.optim.SGD([reference_weight], lr=0.1)
    reference_weight.grad = expected[2].float()
    reference_optimizer.step()
    updated_weight = model.weight.detach().float().clone()
    optimizer = torch.optim.SGD([updated_weight], lr=0.1)
    updated_weight.grad = actual[2].float()
    optimizer.step()
    torch.testing.assert_close(updated_weight, reference_weight, rtol=1e-3, atol=1e-5)
