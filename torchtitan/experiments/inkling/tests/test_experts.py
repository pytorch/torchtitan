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


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA grouped GEMM requires a GPU")
    return request.param


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_routed_packed_experts_propagate_input_gradient(fmt, device):
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
    router_ref = model.gate.weight.detach().clone().requires_grad_()
    scores = F.linear(x_ref.float(), router_ref.float()).sigmoid()[:, :2]
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

    model.to(device)
    model.gate.weight.requires_grad_()
    x = x.detach().to(device).requires_grad_()
    y = model(x)
    torch.testing.assert_close(y.cpu(), result.bfloat16(), rtol=0, atol=0)
    # Nonconstant upstream gradients also check the transpose of the interleaved W13.
    dy = torch.linspace(-1, 1, x.numel()).reshape_as(x).bfloat16()
    y.backward(dy.to(device))
    result.bfloat16().backward(dy)
    assert x.grad is not None
    torch.testing.assert_close(x.grad.cpu(), x_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        model.gate.weight.grad.cpu(), router_ref.grad, rtol=0, atol=0
    )


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_ragged_packed_projection_and_transpose(fmt, device):
    # Unequal axes, partial tiles, and empty experts at either end of the grid.
    torch.manual_seed(312)
    nvfp4 = fmt == "nvfp4"
    out_dim, in_dim = (48, 80) if nvfp4 else (96, 64)
    experts = (
        InklingMoE.Config(
            hidden_size=out_dim,
            intermediate_size=in_dim,
            num_experts=6,
            experts_per_token=2,
            route_scale=1,
            gate_softcap=4,
            up_softcap=6,
            expert_format=fmt,
        )
        .build()
        .experts
    )
    shape = (6, out_dim, in_dim)
    scale = torch.randint(1, 4, (6, out_dim, in_dim // (16 if nvfp4 else 32)))
    if nvfp4:
        codes = torch.randint(16, shape, dtype=torch.uint8)
        lookup = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        dense = lookup[codes.long()] * scale.repeat_interleave(16, -1)
        experts.w2_weight.copy_(codes[..., ::2] | (codes[..., 1::2] << 4))
        experts.w2_weight_scale.copy_(scale)
        experts.w2_weight_amax.copy_(torch.arange(1, 7))
        experts.w2_weight_outer_scale_denominator.fill_(4)
        dense *= torch.arange(1, 7)[:, None, None] / 4
    else:
        codes = torch.randint(-16, 17, shape).float()
        dense = codes * (2.0 ** (scale - 2)).repeat_interleave(32, -1)
        experts.w2_weight.copy_(codes)
        experts.w2_weight_scale.copy_(scale + 125)
    # Skew one expert across multiple parallel row workers and a partial tail.
    counts = torch.tensor([0, 305, 0, 3, 1, 0])
    edges = F.pad(counts.cumsum(0), (1, 0)).int()
    ids = torch.arange(len(counts)).repeat_interleave(counts)
    # The peaks and remaining values are exactly representable by both encoders.
    x = torch.randint(-2, 3, (len(ids), in_dim)).bfloat16()
    x[:, ::16] = 6
    x *= torch.where(torch.arange(len(ids))[:, None] % 2 == 0, 2, 1)
    dy = torch.randint(-2, 3, (len(ids), out_dim)).bfloat16()
    w = dense[ids].double()
    expected_y = (x.double()[:, None] @ w.mT).squeeze(1).bfloat16()
    expected_dx = (dy.double()[:, None] @ w).squeeze(1).bfloat16()
    x = x.to(device).requires_grad_()
    y = experts.to(device).project("w2_weight", x, edges.to(device), ids.to(device))
    y.backward(dy.to(device))
    torch.testing.assert_close(y.cpu(), expected_y, atol=0, rtol=0)
    assert x.grad is not None
    torch.testing.assert_close(x.grad.cpu(), expected_dx, atol=0, rtol=0)


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
        experts.project("w2_weight", x, torch.tensor([0, 2]), torch.tensor([0, 0])),
        expected,
        atol=0,
        rtol=0,
    )
