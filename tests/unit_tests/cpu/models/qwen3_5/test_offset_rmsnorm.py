# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel
from torchtitan_recipes.tests.models.qwen3_6 import qwen36_debugmodel
from torchtitan_recipes.tests.models.qwen3_8 import qwen38_debugmodel


def test_qwen35_architecture_enables_offset_rmsnorm_local_compile() -> None:
    for config_factory in (qwen35_debugmodel, qwen36_debugmodel, qwen38_debugmodel):
        config = config_factory()
        assert config.model.local_compile.regions == [
            "loss",
            "fused_binary_activation",
            "gated_rmsnorm",
            "gdn_split_in_proj",
            "offset_rmsnorm",
            "partial_rope",
            "shared_expert_gate",
            "fp32_to_bf16_split",
            "attn_output_gate",
            "gdn_gate",
        ]


def test_qwen35_fused_output_gate_and_gdn_gate_match_eager() -> None:
    import torch
    import torch.nn.functional as F
    from torchtitan.distributed.local_compile import LocalCompileConfig
    from torchtitan.models.common.attention.gdn import _compute_gdn_gates
    from torchtitan.models.qwen3_5.model import _apply_output_gate

    torch.manual_seed(0)
    LocalCompileConfig(regions=["attn_output_gate", "gdn_gate"]).apply()
    try:
        out_THV = torch.randn(16, 4, 8, dtype=torch.bfloat16, requires_grad=True)
        gate_THV = torch.randn(16, 4, 8, dtype=torch.bfloat16, requires_grad=True)
        out_ref = out_THV.detach().clone().requires_grad_()
        gate_ref = gate_THV.detach().clone().requires_grad_()

        non_contig = out_THV.transpose(0, 1).contiguous().transpose(0, 1)
        assert not non_contig.is_contiguous()
        y = _apply_output_gate(non_contig, gate_THV)
        y_ref = (out_ref.contiguous() * torch.sigmoid(gate_ref)).view(16, -1)
        torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

        grad_y = torch.randn_like(y)
        y.backward(grad_y)
        y_ref.backward(grad_y)
        torch.testing.assert_close(out_THV.grad, out_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(gate_THV.grad, gate_ref.grad, rtol=1e-2, atol=1e-2)

        a_TH = torch.randn(16, 6, dtype=torch.bfloat16, requires_grad=True)
        b_TH = torch.randn(16, 6, dtype=torch.bfloat16, requires_grad=True)
        A_log_H = torch.randn(6, dtype=torch.float32, requires_grad=True)
        dt_bias_H = torch.randn(6, dtype=torch.float32, requires_grad=True)
        a_ref = a_TH.detach().clone().requires_grad_()
        b_ref = b_TH.detach().clone().requires_grad_()
        A_ref = A_log_H.detach().clone().requires_grad_()
        dt_ref = dt_bias_H.detach().clone().requires_grad_()

        g_TH, beta_TH = _compute_gdn_gates(a_TH, b_TH, A_log_H, dt_bias_H)
        g_ref = -torch.exp(A_ref.float()) * F.softplus(a_ref.float() + dt_ref)
        beta_ref = torch.sigmoid(b_ref)
        torch.testing.assert_close(g_TH, g_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(beta_TH, beta_ref, rtol=1e-2, atol=1e-2)

        grad_g = torch.randn_like(g_TH)
        grad_beta = torch.randn_like(beta_TH)
        (g_TH * grad_g).sum().add((beta_TH * grad_beta).sum()).backward()
        (g_ref * grad_g).sum().add((beta_ref * grad_beta).sum()).backward()
        torch.testing.assert_close(a_TH.grad, a_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(b_TH.grad, b_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(A_log_H.grad, A_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dt_bias_H.grad, dt_ref.grad, rtol=1e-2, atol=1e-2)
    finally:
        LocalCompileConfig(regions=[]).apply()
