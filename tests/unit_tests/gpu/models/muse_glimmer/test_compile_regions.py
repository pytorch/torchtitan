# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import functools

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.muse_glimmer import build_model_config
from torchtitan.models.muse_glimmer.model import _flatten_attention_output


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _residual_sites(block, x, branch):
    """Both residual sites on one input: the attention site's (h, ffn_input), then the FFN site."""
    h, ffn_input = block._attention_residual(x, branch)
    return h, ffn_input, block._ffn_residual(h, branch)


def _no_grad_and_grad_outputs(fn, params, inputs, grads):
    """no_grad outputs (the generator), then grad-mode outputs (the trainer) and the grads
    of every input and param."""
    with torch.no_grad():
        no_grad_outputs = fn(*inputs)
    leaves = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    outputs = fn(*leaves)
    input_grads = torch.autograd.grad(outputs, [*leaves, *params], grads)
    return no_grad_outputs, [*outputs, *input_grads]


def _compare_region_to_eager(region, fn, module, inputs, grads, token_counts):
    """Run `fn` on the first T rows of `inputs` for each T in `token_counts`: fp64 eager, bf16
    eager, then bf16 with `region` on, all T in one binding (as a generator compiles them).

    Checks per T:
    - forward outputs: under 1e-4 of the elements differ from eager, and between the no_grad
      (generator) and grad-mode (trainer) forwards.
    - outputs and grads: as close to fp64 as eager's.
    Returns the region's no_grad outputs per T.
    """
    reference_module = copy.deepcopy(module)

    def run(module, dtype):
        params = [param for param in module.parameters() if param.requires_grad]
        for param in params:
            param.data = param.data.to(dtype)
        return [
            _no_grad_and_grad_outputs(
                functools.partial(fn, module),
                params,
                [tensor[:num_tokens].to(dtype) for tensor in inputs],
                [grad[:num_tokens].to(dtype) for grad in grads],
            )
            for num_tokens in token_counts
        ]

    try:
        LocalCompileConfig().apply()
        reference = run(reference_module, torch.float64)
        expected = run(module, torch.bfloat16)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        LocalCompileConfig(regions=[region]).apply()
        actual = run(module, torch.bfloat16)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs
    finally:
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    assert graphs > 0
    for num_tokens, actual_case, expected_case, reference_case in zip(
        token_counts, actual, expected, reference, strict=True
    ):
        no_grad_outputs, actual_tensors = actual_case
        expected_tensors, reference_tensors = expected_case[1], reference_case[1]
        num_outputs = len(no_grad_outputs)
        for no_grad_output, grad_output, expected_output in zip(
            no_grad_outputs,
            actual_tensors[:num_outputs],
            expected_tensors[:num_outputs],
            strict=True,
        ):
            # A norm over a wide row is a looped reduction whose kernel config Inductor picks
            # per kernel, so the summation order can differ in a few rows.
            modes_differ = (no_grad_output != grad_output).float().mean()
            eager_differs = (no_grad_output != expected_output).float().mean()
            assert modes_differ < 1e-4, (num_tokens, modes_differ)
            assert eager_differs < 1e-4, (num_tokens, eager_differs)
        for actual_tensor, expected_tensor, reference_tensor in zip(
            actual_tensors, expected_tensors, reference_tensors, strict=True
        ):
            actual_error = (actual_tensor.double() - reference_tensor).norm()
            eager_error = (expected_tensor.double() - reference_tensor).norm()
            assert actual_error <= 1.05 * eager_error, (
                num_tokens,
                actual_error,
                eager_error,
            )
    return [no_grad_outputs for no_grad_outputs, _ in actual]


# The debug model's width (persistent reduction) and Muse Glimmer 30B's (looped reduction).
# Token counts: 2048 compiles a static graph, as a generator's first call does; 37 a symbolic
# one; 1 its own. Without emulate_precision_casts, 34-47% of the outputs at width 6656 differ
# from eager and up to 21% between no_grad and grad mode.
@pytest.mark.parametrize("width", [256, 6656], ids=["debug_width", "glimmer_30b_width"])
def test_residual_norm_region_matches_eager(width):
    torch.manual_seed(0)
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    layer_config = config.layers[0]
    block = layer_config.build().cuda()
    norm_names = ("post_attention_norm", "ffn_norm", "post_ffn_norm")
    for name in norm_names:
        norm_config = dataclasses.replace(
            getattr(layer_config, name), normalized_shape=width
        )
        norm = norm_config.build().cuda()
        torch.nn.init.normal_(norm.weight, std=0.1)  # gain-center weights start at 0
        setattr(block, name, norm)
    norm_weights = {id(getattr(block, name).weight) for name in norm_names}
    for param in block.parameters():
        param.requires_grad_(id(param) in norm_weights)
    inputs = [torch.randn(2048, width, device="cuda") for _ in range(2)]
    grads = [torch.randn(2048, width, device="cuda") for _ in range(3)]
    _compare_region_to_eager(
        "muse_glimmer_residual_norm",
        _residual_sites,
        block,
        inputs,
        grads,
        token_counts=(2048, 37, 1),
    )


def test_attention_output_region_matches_eager():
    torch.manual_seed(0)
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    attention = config.layers[0].attention.build()
    heads, head_dim = attention.n_heads, attention.head_dim
    inputs = [
        torch.randn(64, heads, head_dim, device="cuda"),
        torch.randn(64, heads * head_dim, device="cuda"),
    ]
    grads = [torch.randn(64, heads * head_dim, device="cuda")]
    token_counts = (64, 37, 1)
    no_grad_outputs = _compare_region_to_eager(
        "muse_glimmer_attention_output",
        lambda _, output, gate: (_flatten_attention_output(output, gate),),
        torch.nn.Module(),  # no parameters
        inputs,
        grads,
        token_counts,
    )
    # Batch invariant: each row's result does not depend on how many rows run with it.
    (full_output,) = no_grad_outputs[0]
    for num_tokens, (output,) in zip(token_counts, no_grad_outputs, strict=True):
        assert torch.equal(output, full_output[:num_tokens]), num_tokens
