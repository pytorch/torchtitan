# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.deepseek_v3 import build_model_config


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    torch._dynamo.reset()
    yield
    apply_local_compile([])


def test_compiled_mla_qk_matches_eager() -> None:
    torch.manual_seed(0)
    attention = build_model_config("debugmodel").layers[0].attention.build()
    attention.init_states()
    num_tokens = 32
    num_heads = attention.n_heads
    q = torch.randn(num_tokens, num_heads, attention.qk_head_dim)
    k_pe = torch.randn(num_tokens, attention.qk_rope_head_dim)
    kv = torch.randn(
        num_tokens, num_heads, attention.qk_nope_head_dim + attention.v_head_dim
    )
    positions = torch.arange(num_tokens) % 10
    grad_q, grad_k, grad_v = (
        torch.randn_like(q),
        torch.randn(num_tokens, num_heads, attention.qk_head_dim),
        torch.randn(num_tokens, num_heads, attention.v_head_dim),
    )

    results = []
    for regions in ([], ["mla_qk"]):
        apply_local_compile(regions)
        inputs = [t.clone().requires_grad_(True) for t in (q, k_pe, kv)]
        rotated_q, rotated_k_pe = attention._apply_rope_to_q_and_k_pe(
            inputs[0], inputs[1], positions
        )
        k, v = attention._split_kv_and_build_k(inputs[2], rotated_k_pe)
        torch.autograd.backward((rotated_q, k, v), (grad_q, grad_k, grad_v))
        results.append((rotated_q, k, v, *(t.grad for t in inputs)))

    for eager, compiled in zip(*results, strict=True):
        torch.testing.assert_close(compiled, eager)
