# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
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
        apply_local_compile([])
        reference = _run(fn, [t.float() for t in inputs_])
        expected = _run(fn, inputs_)
        apply_local_compile([region])
        actual = _run(fn, inputs_)
    finally:
        apply_local_compile([])

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
        apply_local_compile(regions)
        norm.weight.grad = None
        return [*_run(norm, (stream_TD,)), norm.weight.grad]

    try:
        expected = norm_and_weight_grad([])
        actual = norm_and_weight_grad(["muse_spark_rmsnorm"])
    finally:
        apply_local_compile([])

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2)
