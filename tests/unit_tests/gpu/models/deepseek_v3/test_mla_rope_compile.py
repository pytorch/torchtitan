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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def reset_local_compile():
    LocalCompileConfig().apply()
    torch._dynamo.reset()
    yield
    LocalCompileConfig().apply()


@pytest.mark.parametrize("use_positions", [True, False])
def test_compiled_bf16_rope_matches_eager(use_positions: bool) -> None:
    torch.manual_seed(0)
    # DeepSeek-V3's 128 query heads; an odd token count leaves a partial block.
    num_heads, num_tokens = 128, 1027
    attention = (
        build_model_config("debugmodel", parallelism=ParallelismConfig(), seq_len=2048)
        .layers[0]
        .attention.build()
        .cuda()
    )
    attention.init_states(buffer_device=torch.device("cuda"))
    q = torch.randn(
        num_tokens,
        num_heads,
        attention.qk_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    # k_pe is a strided view of the wkv_a output, as in the forward.
    kv = torch.randn(
        num_tokens,
        attention.kv_lora_rank + attention.qk_rope_head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = (
        torch.randint(0, 2048, (num_tokens,), device="cuda") if use_positions else None
    )
    grad_q = torch.randn_like(q)
    grad_k_pe = torch.randn(
        num_tokens, 1, attention.qk_rope_head_dim, device="cuda", dtype=torch.bfloat16
    )

    results = []
    # Eager, the MLA region (int32 path), and the rope region alone (split path).
    for regions in ([], ["mla_qk"], ["rope"]):
        LocalCompileConfig(regions=regions).apply()
        q_in, kv_in = (t.clone().requires_grad_(True) for t in (q, kv))
        k_pe = kv_in.split(
            [attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1
        )[1]
        rotated_q, rotated_k_pe = attention._apply_rope_to_q_and_k_pe(
            q_in, k_pe, positions
        )
        torch.autograd.backward((rotated_q, rotated_k_pe), (grad_q, grad_k_pe))
        results.append((rotated_q, rotated_k_pe, q_in.grad, kv_in.grad))

    eager, *compiled_runs = results
    # Bitwise on GB300 but not on A10G: compiled real arithmetic can round
    # differently from eager's complex multiply.
    for compiled in compiled_runs:
        for got, want in zip(compiled, eager, strict=True):
            torch.testing.assert_close(got, want)
