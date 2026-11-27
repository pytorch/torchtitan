# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn import functional as F

from torchtitan.config import PrecisionConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.distributed.fsdp import maybe_store_frozen_params_in_param_dtype_
from torchtitan.experiments.inkling import build_model_config
from torchtitan.experiments.inkling.experts import InklingMoE
from torchtitan.tools.utils import set_default_dtype


def test_mixed_precision_keeps_frozen_checkpoint_dtypes():
    cfg = LoRATransform(rank=2, alpha=2, target_modules=["attn.wq_du"]).transform(
        build_model_config("debugmodel")
    )
    # In mixed precision only trainable masters differ from the BF16 recipe.
    with torch.device("meta"), set_default_dtype(torch.bfloat16):
        expected = {
            name: torch.float32 if param.requires_grad else param.dtype
            for name, param in cfg.build().named_parameters()
        }
    with torch.device("meta"), set_default_dtype(torch.float32):
        model = cfg.build()
    maybe_store_frozen_params_in_param_dtype_(
        model, PrecisionConfig(dtype="float32", param_dtype="bfloat16")
    )
    assert {name: param.dtype for name, param in model.named_parameters()} == expected


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_routed_packed_experts_propagate_input_gradient(fmt):
    # Diagonal projections keep each input row constant across columns. In that
    # case NVFP4 exactly represents the row max, and the MXFP8 reference only
    # needs a scalar power-of-two scale. No production decoder is the oracle.
    dim = 32
    model = InklingMoE.Config(
        hidden_size=dim,
        intermediate_size=dim,
        num_experts=3,
        experts_per_token=2,
        route_scale=1.5,
        gate_softcap=4,
        up_softcap=6,
        expert_format=fmt,
    ).build()
    model.init_states()
    # BASE-role checkpoint filtering operates on named_parameters, not buffers.
    assert model.state_dict().keys() == dict(model.named_parameters()).keys()
    assert not any(p.requires_grad for p in model.parameters())
    idx = torch.arange(dim)
    with torch.no_grad():
        model.gate.weight.copy_(torch.tensor([0.5, -0.5, 0.75])[:, None] / dim)
        model.gate.bias.copy_(torch.tensor([0.5, 0.5, -2]))
        for e in range(3):
            gate, up, down = (1.0, -2.0, 0.5) if e == 0 else (0.5, 1.0, 2.0)
            for name in ("w13_weight", "w2_weight"):
                matrix = torch.zeros((2 if name == "w13_weight" else 1) * dim, dim)
                if name == "w13_weight":
                    matrix[2 * idx, idx] = gate
                    matrix[2 * idx + 1, idx] = up
                else:
                    matrix[idx, idx] = down
                if fmt == "nvfp4":
                    # Give non-unit per-expert factors to only the down projection.
                    factor = 2 if name == "w2_weight" else 1
                    codes = torch.zeros_like(matrix, dtype=torch.uint8)
                    for value, code in (
                        (0.5, 2),
                        (1.0, 4),
                        (-2.0, 14),
                        (2.0, 6),
                    ):
                        codes[matrix == value] = code
                    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
                    model.experts.get_parameter(name + "_amax")[e] = factor
                    model.experts.get_parameter(
                        name + "_outer_scale_denominator"
                    ).fill_(2)
                    # scale=1, codes represent 2*matrix, so W=matrix*factor.
                    model.experts.get_parameter(name + "_scale")[e].fill_(1)
                else:
                    # Different E8M0 scales per row and expert still decode to matrix.
                    exponents = (torch.arange(matrix.size(0))[:, None] + e) % 2 * 2 - 1
                    data = (matrix / 2.0**exponents).to(torch.float8_e4m3fn)
                    model.experts.get_parameter(name + "_scale")[e].copy_(
                        exponents + 127
                    )
                model.experts.get_parameter(name)[e].copy_(data)

    x = (
        torch.tensor([0.75, -1.25])
        .bfloat16()[:, None]
        .expand(-1, dim)
        .clone()
        .requires_grad_()
    )
    x_ref = x.detach().clone().requires_grad_()
    scores = F.linear(x_ref.float(), model.gate.weight.float()).sigmoid()[:, :2]
    weights = scores / scores.sum(-1, keepdim=True) * 1.5

    def quantize(rows):
        if fmt == "nvfp4":
            return rows
        scale = 2 ** (rows.float().abs().amax(-1, keepdim=True) / 448).log2().ceil()
        values = (
            (rows.float() / scale).to(torch.float8_e4m3fn).float() * scale
        ).bfloat16()
        return values.detach() + (rows - rows.detach())

    result = torch.zeros_like(x_ref, dtype=torch.float32)
    for e, (gate, up, down) in enumerate(((1.0, -2.0, 0.5), (0.5, 1.0, 2.0))):
        eye = torch.eye(dim)
        w13 = torch.stack((eye * gate, eye * up), 1).flatten(0, 1)
        gu = F.linear(quantize(x_ref).float(), w13).bfloat16().float()
        g, u = gu[:, 0::2], gu[:, 1::2]
        act = (g / 4).tanh() * 4 * g.sigmoid() * ((u / 6).tanh() * 6)
        routed = (act * weights[:, e : e + 1]).bfloat16()
        result = result + quantize(routed) * down * (2 if fmt == "nvfp4" else 1)

    y = model(x)
    torch.testing.assert_close(y, result.bfloat16(), rtol=0, atol=0)
    # Nonconstant upstream gradients also check the transpose of the interleaved W13.
    dy = torch.linspace(-1, 1, x.numel()).reshape_as(x).bfloat16()
    y.backward(dy)
    result.bfloat16().backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


def test_nvfp4_projection_rounds_each_block_with_its_own_scale():
    experts = (
        InklingMoE.Config(
            hidden_size=32,
            intermediate_size=32,
            num_experts=1,
            experts_per_token=1,
            route_scale=1,
            gate_softcap=4,
            up_softcap=6,
            expert_format="nvfp4",
        )
        .build()
        .experts
    )
    # Code 2 is +1 in E2M1. An identity weight exposes activation rounding.
    codes = 2 * torch.eye(32, dtype=torch.uint8)
    experts.w2_weight[0].copy_(codes[:, ::2] | (codes[:, 1::2] << 4))
    experts.w2_weight_scale.fill_(1)
    experts.w2_weight_amax.fill_(1)
    experts.w2_weight_outer_scale_denominator.fill_(1)

    # With a row max of six, the candidate scales are 1 and 1.5. The
    # first block favors 1 and includes ties; the second is exact at 1.5.
    six = [6, 1.25, -2.5, 5] + [1] * 12
    four = [6, -0.75, 2.25, 4.5] * 4
    x = torch.tensor([six + four, [0] * 32], dtype=torch.bfloat16)
    expected = torch.tensor(
        [[6, 1, -2, 4] + [1] * 12 + four, [0] * 32], dtype=torch.bfloat16
    )
    torch.testing.assert_close(
        experts.project("w2_weight", 0, x), expected, atol=0, rtol=0
    )
