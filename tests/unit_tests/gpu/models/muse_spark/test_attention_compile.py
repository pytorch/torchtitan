# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.muse_spark import build_model_config


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _run(fn, inputs_):
    leaves = [t.detach().clone().requires_grad_() for t in inputs_]
    outputs = fn(*leaves)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    torch.autograd.backward(outputs, [torch.ones_like(t) for t in outputs])
    return [*outputs, *(t.grad for t in leaves)]


# Debug model layer 0: sliding window with RoPE. Layer 3: global, position temperature, no RoPE.
@pytest.mark.parametrize("layer_id", [0, 3])
@pytest.mark.parametrize(
    "region", ["muse_spark_attention_qk", "muse_spark_attention_output"]
)
def test_attention_region_matches_eager(region, layer_id):
    torch.manual_seed(0)
    layer = build_model_config("debugmodel").layers[layer_id]
    attention = layer.attention.build().cuda()
    num_tokens, num_heads, num_kv_heads = 64, attention.n_heads, attention.n_kv_heads
    head_dim = attention.head_dim
    query_TNH = torch.randn(
        num_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    if region == "muse_spark_attention_qk":
        positions_T = torch.arange(num_tokens, device="cuda")

        def fn(query_TNH, key_TKH):
            return attention._prepare_query_and_key(query_TNH, key_TKH, positions_T)

        key_TKH = torch.randn(
            num_tokens, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        inputs_ = (query_TNH, key_TKH)
    else:
        fn = attention._flatten_attention_output
        gate_TO = torch.randn(
            num_tokens, num_heads * head_dim, device="cuda", dtype=torch.bfloat16
        )
        inputs_ = (query_TNH, gate_TO)

    try:
        LocalCompileConfig().apply()
        reference = _run(fn, [t.float() for t in inputs_])
        expected = _run(fn, inputs_)
        LocalCompileConfig(regions=[region]).apply()
        actual = _run(fn, inputs_)
    finally:
        LocalCompileConfig().apply()

    # The compiled region rounds to bf16 once at the end, eager rounds between ops,
    # so the two differ by a few bf16 ulps. Check the compiled path is at least as
    # close as eager to an fp32 run of the same math, for outputs and grads.
    for actual_tensor, expected_tensor, reference_tensor in zip(
        actual, expected, reference, strict=True
    ):
        actual_error = (actual_tensor.float() - reference_tensor).norm()
        eager_error = (expected_tensor.float() - reference_tensor).norm()
        assert actual_error <= 1.05 * eager_error, (actual_error, eager_error)


def test_rmsnorm_region_matches_eager():
    # The residual stream is fp32 and the norm weight bf16, as under FSDP mixed precision.
    norm = build_model_config("debugmodel").layers[0].attention_norm.build().cuda()
    norm = norm.to(torch.bfloat16)
    stream_TD = torch.randn(64, norm.normalized_shape[0], device="cuda")

    def norm_and_weight_grad(regions):
        LocalCompileConfig(regions=regions).apply()
        norm.weight.grad = None
        return [*_run(norm, (stream_TD,)), norm.weight.grad]

    try:
        expected = norm_and_weight_grad([])
        actual = norm_and_weight_grad(["muse_spark_rmsnorm"])
    finally:
        LocalCompileConfig().apply()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2)


# The input-scaled norm also runs on routed tokens, a different count per layer.
@pytest.mark.parametrize(
    "region, norm_name, layer1_tokens",
    [
        ("muse_spark_rmsnorm", "post_attention_norm", 64),
        ("muse_spark_input_scaled_rmsnorm", "post_ffn_norm", 48),
    ],
)
def test_norm_region_under_activation_checkpoint(region, norm_name, layer1_tokens):
    # Two layers' post norms share one region and differ in gain_center.
    torch.manual_seed(0)
    layers = build_model_config("debugmodel").layers[:2]
    norms = [getattr(layer, norm_name).build().cuda() for layer in layers]
    assert norms[0].gain_center != norms[1].gain_center
    for norm in norms:
        torch.nn.init.normal_(norm.weight, std=0.1)
    input_TD = torch.randn(64, norms[0].normalized_shape[0], device="cuda")

    def two_norms(x):
        return norms[1](norms[0](x)[:layer1_tokens])

    def two_norms_checkpointed(x):
        return torch.utils.checkpoint.checkpoint(two_norms, x, use_reentrant=False)

    def output_and_grads(fn):
        # Empty cache, so layer 0 compiles the region first, as in training.
        torch._dynamo.reset()
        for norm in norms:
            norm.weight.grad = None
        return [*_run(fn, (input_TD,)), *(norm.weight.grad for norm in norms)]

    try:
        LocalCompileConfig(regions=[region]).apply()
        expected = output_and_grads(two_norms)
        actual = output_and_grads(two_norms_checkpointed)
    finally:
        LocalCompileConfig().apply()

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
