# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed experts stored as MXFP8, fed pre-quantized weights by the trainer.

The claim under test is that a generator built this way needs no high-precision
weights at all: loading the trainer's published state dict is enough to produce
numerics identical to today's quantize-on-the-generator path.
"""

import pytest
import torch
from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_per_group_3d as _swizzle,
)

from torchtitan.components.quantization.mxfp8_utils import (
    quantize_expert_state_dict_to_mxfp8,
)
from torchtitan.overrides.mxfp8_inference_grouped_experts import (
    MXFP8InferenceGroupedExperts,
)
from torchtitan.tools.utils import has_cuda_capability

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_cuda_capability(10, 0)),
    reason="MXFP8 requires SM100 or later",
)

E, F, D = 4, 768, 512
TOKENS_PER_EXPERT = 128
BLOCK = 32


def _single_rank_mesh():
    """A 1-rank device mesh: the publish path takes DTensors."""
    import os

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29592")
        dist.init_process_group("nccl", rank=0, world_size=1)
    return init_device_mesh("cuda", (1,), mesh_dim_names=("ep",))


def _published(seed: int = 0):
    """The weights, and what the trainer would publish for them."""
    from torch.distributed.tensor import distribute_tensor, DTensor, Shard

    torch.manual_seed(seed)
    weights = {
        "w1_EFD": torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16),
        "w3_EFD": torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16),
        "w2_EDF": torch.randn(E, D, F, device="cuda", dtype=torch.bfloat16),
    }
    mesh = _single_rank_mesh()
    dist_sd = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in weights.items()}
    published = {
        k: v.to_local() if isinstance(v, DTensor) else v
        for k, v in quantize_expert_state_dict_to_mxfp8(dist_sd).items()
    }
    return weights, published


def _build(dim: int = D, hidden_dim: int = F):
    """Build the module the way the override does, then materialize and init it."""
    from torchtitan.models.common.moe import GroupedExperts
    from torchtitan.overrides.mxfp8_inference_grouped_experts import (
        mxfp8_inference_grouped_experts,
    )

    cfg = mxfp8_inference_grouped_experts(
        GroupedExperts.Config(dim=dim, hidden_dim=hidden_dim, num_experts=E)
    )
    mod = cfg.build().to_empty(device="cuda")
    mod.init_states()
    return mod


def _inputs():
    torch.manual_seed(7)
    x = torch.randn(E * TOKENS_PER_EXPERT, D, device="cuda", dtype=torch.bfloat16)
    counts = torch.full((E,), TOKENS_PER_EXPERT, device="cuda", dtype=torch.int32)
    return x, counts


def test_state_dict_is_exactly_the_wire_format():
    """No hooks, no derived entries: what the trainer publishes is what loads."""
    _, published = _published()
    mod = _build()
    assert set(mod.state_dict()) == set(published)
    missing, unexpected = mod.load_state_dict(published, strict=True)
    assert not missing and not unexpected


def test_no_high_precision_parameters():
    mod = _build()
    for name, tensor in mod.state_dict().items():
        assert tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu), (
            f"{name} is {tensor.dtype}; the generator should hold no "
            f"high-precision expert weights"
        )


def test_matches_a_reference_forward_that_quantizes_locally():
    """Loading pre-quantized weights gives the same output as quantizing here.

    The reference composes torchao's upstream quantization kernels and
    ``torch._scaled_grouped_mm`` directly on the high-precision weights, which is
    what the generator used to do for itself. Matching it bitwise is the
    guarantee that moving quantization to the trainer changed no numerics.
    """
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        mx_block_rearrange_2d_M_groups_cuda,
        triton_mx_block_rearrange_per_group_3d,
    )
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
    from torchtitan.overrides.fused_swiglu import silu_and_mul_op

    weights, published = _published()
    x, counts = _inputs()

    new = _build()
    new.load_state_dict(published, strict=True)
    with torch.no_grad():
        got = new(x, counts)

    fused_ENK = (
        torch.stack([weights["w1_EFD"], weights["w3_EFD"]], dim=2)
        .reshape(E, F * 2, D)
        .contiguous()
    )

    def quantize(weight_ENK):
        qdata, scales = triton_to_mxfp8_dim0(weight_ENK, BLOCK, "rceil")
        return qdata, triton_mx_block_rearrange_per_group_3d(scales)

    w13_q, w13_s = quantize(fused_ENK)
    w2_q, w2_s = quantize(weights["w2_EDF"])
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

    def gemm(act, qdata_ENK, scales_blocked):
        act_q, act_s = triton_to_mxfp8_dim0(act.contiguous(), BLOCK, "rceil")
        return torch._scaled_grouped_mm(
            act_q,
            qdata_ENK.transpose(-2, -1),
            mx_block_rearrange_2d_M_groups_cuda(act_s, offsets),
            scales_blocked,
            offs=offsets,
            out_dtype=torch.bfloat16,
        )

    with torch.no_grad():
        gate_up = gemm(x.bfloat16(), w13_q, w13_s)
        half = gate_up.shape[-1] // 2
        g, u = gate_up.reshape(-1, half, 2).unbind(-1)
        want = gemm(silu_and_mul_op(g, u, offsets), w2_q, w2_s).type_as(x)

    assert torch.equal(got, want)


def test_reload_updates_output_in_place():
    """A second load changes the output and keeps the buffer addresses fixed.

    Stable addresses are what let a captured CUDA graph see a weight sync; a
    changed output is what proves the sync was not silently dropped.
    """
    _, first = _published(seed=0)
    _, second = _published(seed=1)
    x, counts = _inputs()

    mod = _build()
    mod.load_state_dict(first, strict=True)
    blocked = ("w13_scales_blocked", "w2_EDF_scales_blocked")
    ptrs = {n: getattr(mod, n).data_ptr() for n in blocked}
    with torch.no_grad():
        before = mod(x, counts).clone()

    mod.load_state_dict(second, strict=True)
    with torch.no_grad():
        after = mod(x, counts)

    assert not torch.equal(before, after), "reload did not change the output"
    for name in blocked:
        assert getattr(mod, name).data_ptr() == ptrs[name], name


@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(D, F), (96, 100)],  # the second pads on both axes: 2F=200 rows, D/32=3 cols
)
def test_allocated_blocked_scales_match_the_kernel(dim, hidden_dim):
    """The buffers allocated at init must be exactly what the swizzle produces.

    ``_init_self_buffers`` predicts the shape instead of running the kernel, so
    a drift in torchao's padding would mis-size them.
    """

    mod = _build(dim=dim, hidden_dim=hidden_dim)
    for name in ("w13_scales", "w2_EDF_scales"):
        want = _swizzle(getattr(mod, name)).shape
        assert getattr(mod, f"{name}_blocked").shape == want, name


def test_override_claims_the_qat_converted_config():
    """The override must claim the config the 30B path actually produces.

    The mxfp8 converter runs first, so the config arriving here is
    ``MXFP8QATGroupedExperts.Config``. That class comes from a factory, and the
    generator actor builds its own copy, so the override has to match it by
    ``isinstance``. An identity check declines there in silence and leaves the
    generator on the training module while the trainer publishes quantized names.
    """
    from torchtitan.components.quantization.mx import _get_mxfp8_qat_grouped_experts_cls
    from torchtitan.models.common.moe import GroupedExperts
    from torchtitan.overrides.mxfp8_inference_grouped_experts import (
        mxfp8_inference_grouped_experts,
    )

    qat_cls = _get_mxfp8_qat_grouped_experts_cls(GroupedExperts)
    qat_cfg = qat_cls.Config(dim=D, hidden_dim=F, num_experts=E)
    assert type(qat_cfg) is not GroupedExperts.Config  # the trap this guards

    out = mxfp8_inference_grouped_experts(qat_cfg)
    assert isinstance(
        out, MXFP8InferenceGroupedExperts.Config
    ), "override declined the QAT-converted config"
