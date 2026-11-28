# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.deepseek_v3 import build_model_config


@pytest.fixture(autouse=True)
def reset_local_compile():
    LocalCompileConfig().apply()
    torch._dynamo.reset()
    yield
    LocalCompileConfig().apply()


@pytest.mark.parametrize("use_positions", [True, False])
def test_compiled_mla_qk_matches_eager(use_positions: bool) -> None:
    torch.manual_seed(0)
    attention = (
        build_model_config("debugmodel", parallelism=ParallelismConfig())
        .layers[0]
        .attention.build()
    )
    attention.init_states()
    num_tokens = 32
    num_heads = attention.n_heads
    q = torch.randn(num_tokens, num_heads, attention.qk_head_dim)
    k_pe = torch.randn(num_tokens, attention.qk_rope_head_dim)
    positions = torch.arange(num_tokens) % 10 if use_positions else None
    grad_q = torch.randn_like(q)
    grad_k_pe = torch.randn(num_tokens, 1, attention.qk_rope_head_dim)

    results = []
    for regions in ([], ["mla_qk"]):
        LocalCompileConfig(regions=regions).apply()
        inputs = [t.clone().requires_grad_(True) for t in (q, k_pe)]
        rotated_q, rotated_k_pe = attention._apply_rope_to_q_and_k_pe(
            inputs[0], inputs[1], positions
        )
        torch.autograd.backward((rotated_q, rotated_k_pe), (grad_q, grad_k_pe))
        results.append((rotated_q, rotated_k_pe, *(t.grad for t in inputs)))

    for eager, compiled in zip(*results, strict=True):
        torch.testing.assert_close(compiled, eager)
