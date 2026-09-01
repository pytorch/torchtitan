# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantizing MoE expert weights on the trainer, for transfer to a generator.

The point of quantizing before publishing is that the generator's parameters are
the MXFP8 operands themselves, with no high-precision copy anywhere. That is only
sound if the published tensors survive the trip: they are resharded from the
trainer's expert layout to the generator's, and the generator fuses gate and up.
These tests pin the two commutation properties that make it work, and the
alignment condition that bounds it.
"""

import pytest
import torch
from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_per_group_3d as _swizzle,
)

from torchtitan.components.quantization.mxfp8_utils import (
    quantize_expert_state_dict_to_mxfp8,
)
from torchtitan.tools.utils import has_cuda_capability

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_cuda_capability(10, 0)),
    reason="MXFP8 requires SM100 or later",
)

E, F, D = 4, 768, 256
BLOCK = 32


def _publish(sd: dict) -> dict:
    """Run the trainer's publish step, returning plain local tensors."""
    from torch.distributed.tensor import distribute_tensor, DTensor, Shard

    mesh = _single_rank_mesh()
    dist_sd = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in sd.items()}
    return {
        k: v.to_local() if isinstance(v, DTensor) else v
        for k, v in quantize_expert_state_dict_to_mxfp8(dist_sd).items()
    }


def _experts(prefix: str = "layers.0.moe.routed_experts.inner_experts."):
    torch.manual_seed(0)
    return prefix, {
        f"{prefix}w1_EFD": torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16),
        f"{prefix}w3_EFD": torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16),
        f"{prefix}w2_EDF": torch.randn(E, D, F, device="cuda", dtype=torch.bfloat16),
    }


def _reference_quantize(weight_ENK):
    """qdata and blocked scales for an (E, N, K) operand, via upstream kernels."""
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_per_group_3d,
    )
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    qdata, scales = triton_to_mxfp8_dim0(weight_ENK, BLOCK, "rceil")
    return qdata, triton_mx_block_rearrange_per_group_3d(scales)


def test_matches_torchaos_reference_quantization():
    """Published operands are bitwise what torchao's own kernels produce.

    The reference composes the upstream primitives directly, so this pins the
    published bytes against torchao rather than against our own helper.
    """
    prefix, sd = _experts()
    published = _publish(sd)

    gate, up = sd[f"{prefix}w1_EFD"], sd[f"{prefix}w3_EFD"]
    fused_ENK = torch.stack([gate, up], dim=2).reshape(E, F * 2, D).contiguous()
    reference = {
        "w13": _reference_quantize(fused_ENK),
        "w2_EDF": _reference_quantize(sd[f"{prefix}w2_EDF"]),
    }
    for name, (want_q, want_s) in reference.items():
        assert torch.equal(published[f"{prefix}{name}_qdata"], want_q), name
        # torchao's helper swizzles; we publish the natural layout, so swizzle
        # ours to compare.
        got_s = _swizzle(published[f"{prefix}{name}_scales"])
        assert torch.equal(got_s, want_s), name


def test_expert_sharding_commutes_with_quantization():
    """A per-expert shard quantizes to a slice of the global quantization.

    This is what lets TorchStore reshard the published tensors between a trainer
    and a generator with different expert-parallel degrees.
    """
    prefix, sd = _experts()
    full = _publish(sd)
    shard = _publish({k: v[2:4].contiguous() for k, v in sd.items()})
    for key in (
        f"{prefix}w13_qdata",
        f"{prefix}w13_scales",
        f"{prefix}w2_EDF_qdata",
        f"{prefix}w2_EDF_scales",
    ):
        assert torch.equal(shard[key], full[key][2:4]), key


def test_gate_up_fusion_commutes_with_quantization():
    """Fusing then quantizing equals quantizing then interleaving rows.

    Fusion only permutes rows and each row is scaled independently, so the
    trainer may fuse before publishing without changing any quantized value.
    """
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    _, sd = _experts(prefix="")
    gate, up = sd["w1_EFD"], sd["w3_EFD"]
    fused_q = _publish(sd)["w13_qdata"]

    q_gate, _ = triton_to_mxfp8_dim0(gate.contiguous(), BLOCK, "rceil")
    q_up, _ = triton_to_mxfp8_dim0(up.contiguous(), BLOCK, "rceil")
    interleaved = torch.stack([q_gate, q_up], dim=2).reshape(E, F * 2, D)
    assert torch.equal(fused_q, interleaved)


def test_non_expert_entries_pass_through():
    prefix, sd = _experts()
    sd["layers.0.attention.wq.weight"] = torch.randn(8, 8, device="cuda")
    published = _publish(sd)
    assert torch.equal(
        published["layers.0.attention.wq.weight"], sd["layers.0.attention.wq.weight"]
    )


def test_incomplete_expert_group_is_rejected():
    prefix, sd = _experts()
    del sd[f"{prefix}w3_EFD"]
    with pytest.raises(ValueError, match="incomplete"):
        _publish(sd)


@pytest.mark.parametrize("k_per_shard", [F // 2, F // 4])
def test_aligned_contracting_split_is_exact(k_per_shard):
    """A tensor-parallel split of the contracting dim is exact when 32-aligned.

    ``w2``'s contracting dim is the one TP shards, so this is the condition that
    decides whether TP composes with quantized transfer.
    """
    assert k_per_shard % BLOCK == 0
    _, sd = _experts(prefix="")
    full = _publish(sd)
    full_q, full_s = full["w2_EDF_qdata"], full["w2_EDF_scales"]
    sliced = dict(sd)
    sliced["w2_EDF"] = sd["w2_EDF"][:, :, :k_per_shard].contiguous()
    shard = _publish(sliced)
    shard_q, shard_s = shard["w2_EDF_qdata"], shard["w2_EDF_scales"]
    # (E, D, F) quantizes to qdata (E, D, F); the split is along the last dim.
    assert torch.equal(shard_q, full_q[:, :, :k_per_shard])
    # Unswizzled scales are (E, N, K/32), so the K split is a plain slice. The
    # swizzled form is not sliceable here, which is why it is not what we
    # publish.
    assert torch.equal(shard_s, full_s[..., : k_per_shard // BLOCK])


def _single_rank_mesh():
    """A 1-rank device mesh, enough to exercise the DTensor branch."""
    import os

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group("nccl", rank=0, world_size=1)
    return init_device_mesh("cuda", (1,), mesh_dim_names=("ep",))


def test_dtensor_input_stays_a_dtensor():
    """A DTensor weight publishes a DTensor, with the same placements.

    TorchStore reshards the published tensors from the trainer's layout to the
    generator's, which it can only do if they arrive as DTensors. Dropping to a
    plain local shard silently publishes one rank's slice as the whole tensor.
    """
    from torch.distributed.tensor import distribute_tensor, DTensor, Shard

    mesh = _single_rank_mesh()
    _, sd = _experts(prefix="")
    local = _publish(sd)
    dist_sd = {k: distribute_tensor(v, mesh, [Shard(0)]) for k, v in sd.items()}
    published = quantize_expert_state_dict_to_mxfp8(dist_sd)
    for key in ("w13_qdata", "w13_scales", "w2_EDF_qdata", "w2_EDF_scales"):
        got = published[key]
        assert isinstance(got, DTensor), f"{key} lost its DTensor wrapper"
        assert got.placements == (Shard(0),), key
        assert torch.equal(got.to_local(), local[key]), key
        # The global shape must describe the whole weight, not one shard.
        assert got.shape == local[key].shape, key


def test_unaligned_contracting_shard_is_rejected():
    """A contracting dim that is not a multiple of 32 is refused.

    A shard boundary inside a scaling group would make per-shard quantization
    differ from the unsharded weight, so this must not silently succeed.
    """
    from torch.distributed.tensor import distribute_tensor, Shard

    mesh = _single_rank_mesh()
    torch.manual_seed(0)
    bad_k = 100  # not a multiple of the 32-element scaling group
    sd = {
        "w1_EFD": torch.randn(E, F, bad_k, device="cuda", dtype=torch.bfloat16),
        "w3_EFD": torch.randn(E, F, bad_k, device="cuda", dtype=torch.bfloat16),
        "w2_EDF": torch.randn(E, bad_k, F, device="cuda", dtype=torch.bfloat16),
    }
    dist_sd = {k: distribute_tensor(v, mesh, [Shard(2)]) for k, v in sd.items()}
    with pytest.raises(AssertionError, match="divisible by inner block size"):
        quantize_expert_state_dict_to_mxfp8(dist_sd)


def test_quantizes_through_the_qat_weight_wrapper():
    """Publishing works when the trainer runs MXFP8 QAT.

    QAT installs torchao's ``MXFP8TrainingWeightWrapperTensor`` on every expert
    weight, so the trainer's state dict holds subclass tensors rather than plain
    ones. Quantizing must see through that and produce the same bytes.
    """
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.quantization.quant_api import quantize_

    from torchtitan.models.common.moe import GroupedExperts

    _, sd = _experts(prefix="")
    plain = _publish(dict(sd))

    mod = GroupedExperts.Config(dim=D, hidden_dim=F, num_experts=E).build()
    with torch.no_grad():
        for name, key in (
            ("w1_EFD", "w1_EFD"),
            ("w2_EDF", "w2_EDF"),
            ("w3_EFD", "w3_EFD"),
        ):
            setattr(mod, name, torch.nn.Parameter(sd[key], requires_grad=False))
    quantize_(
        mod,
        config=MXFP8TrainingOpConfig(
            scale_calculation_mode=ScaleCalculationMode.RCEIL, bf16_bwd=True
        ),
        filter_fn=lambda m, _fqn: isinstance(m, GroupedExperts),
    )
    wrapped = {k: v for k, v in mod.state_dict().items()}
    assert any(
        hasattr(v, "_data") for v in wrapped.values()
    ), "QAT wrapper not installed"

    published = _publish(wrapped)
    for key in ("w13_qdata", "w13_scales", "w2_EDF_qdata", "w2_EDF_scales"):
        assert torch.equal(published[key], plain[key]), key
