# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Publish MoE expert weights to the generator already quantized to MXFP8.

One weight sync::

    TRAINER            w1, w2, w3 bf16, FSDP-sharded
                            |
                            |  fuse w1+w3, quantize     <- this module
                            v
                       w13_qdata / w13_scales
                       w2_qdata  / w2_scales            unswizzled
                            |
    TORCHSTORE              |  reshard to the generator's EP/TP layout
                            v
    GENERATOR          loaded straight into its parameters
                            |
                            |  swizzle scales into the GEMM layout
                            v
                       ready for _scaled_grouped_mm

The generator never holds a high-precision copy: what it receives is what its
GEMMs read. That also halves the bytes on the wire.

Quantizing before the reshard is safe because a scaling group is 32 contiguous
elements of ``K`` within one row of one expert, so no split of ``E`` or ``N`` can
intersect one and a ``K`` split cannot either while each shard's ``K`` stays a
multiple of 32. Fusing first is safe because it only interleaves rows. Both are

Scales go out unswizzled because the swizzle does *not* commute with a ``K``
split: the blocked layout interleaves across rows and groups, so slicing it
would mix in another shard's scales. Swizzling therefore has to happen after
resharding, which is why the generator does it and the trainer needs to know
nothing about the inference kernel's layout.
"""

import re

import torch
from torch.distributed.tensor import DTensor
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

# Expert weights as the trainer names them, mapped to the generator's storage.
# ``w1_EFD``/``w3_EFD`` fuse into a single ``w13``; ``w2_EDF`` stands alone.
_GATE = "w1_EFD"
_UP = "w3_EFD"
_DOWN = "w2_EDF"

_EXPERT_SUFFIXES = (_GATE, _UP, _DOWN)
_EXPERT_KEY = re.compile(r"^(?P<prefix>.*\.)?(?P<name>w1_EFD|w2_EDF|w3_EFD)$")

# MXFP8 scaling-group size, along the contracting dim.
_BLOCK_SIZE = 32
# Must match what the inference path quantizes with, so both GEMM operands agree.
_SCALE_MODE = "rceil"


# TODO: remove this in favor of FSDP-managed version
# similar to https://github.com/pytorch/torchtitan/pull/4203
class _MXFP8GroupedMMFwdBF16Bwd(torch.autograd.Function):
    """Grouped GEMM with an MXFP8 forward and a bf16 backward.

    QAT for the routed experts: the forward really is mxfp8, so training sees
    quantization error, while the backward is a plain bf16 ``torch._grouped_mm``
    on the saved high-precision tensors (straight-through estimator).

    Adapted from torchao's ``_MXFP8GroupedMM`` with the paths we do not use
    removed: no emulation, no kernel-preference switch, and no internal token
    padding. The MXFP8 converter pads token groups in the dispatcher instead (see
    ``swap_token_dispatcher``), so groups are already aligned on arrival.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        A_MK: torch.Tensor,
        B_t_EKN: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            mxfp8_quantize_2d_1x32_cutedsl,
            triton_mx_block_rearrange_per_group_3d,
        )

        if A_MK.ndim != 2 or B_t_EKN.ndim != 3:
            raise ValueError(
                "MXFP8 grouped mm expects a 2D activation and 3D weight, got "
                f"{A_MK.ndim}D and {B_t_EKN.ndim}D."
            )

        # Activation: CuteDSL emits the blocked scale layout the GEMM wants, so
        # there is no separate swizzle pass.
        act_qdata, act_scales_blocked = mxfp8_quantize_2d_1x32_cutedsl(
            A_MK, scaling_mode=_SCALE_MODE, offs=offsets_E
        )
        # Weight: quantize along K in the (E, N, K) view, then swizzle.
        w_qdata, w_scales = triton_to_mxfp8_dim0(
            B_t_EKN.transpose(-2, -1).contiguous(), _BLOCK_SIZE, _SCALE_MODE
        )
        w_scales_blocked = triton_mx_block_rearrange_per_group_3d(w_scales)

        out = torch._scaled_grouped_mm(
            act_qdata,
            w_qdata.transpose(-2, -1),
            act_scales_blocked,
            w_scales_blocked,
            offs=offsets_E,
            out_dtype=torch.bfloat16,
        )
        ctx.save_for_backward(A_MK, B_t_EKN, offsets_E)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_out_MN: torch.Tensor):
        A_MK, B_t_EKN, offsets_E = ctx.saved_tensors
        # grad_out may be non-contiguous (e.g. expanded with stride 0);
        # torch._grouped_mm needs real strides.
        grad_out_MN = grad_out_MN.contiguous()
        grad_A_MK = torch._grouped_mm(
            grad_out_MN,
            B_t_EKN.transpose(-2, -1),
            offs=offsets_E,
            out_dtype=torch.bfloat16,
        )
        grad_B_ENK = torch._grouped_mm(
            grad_out_MN.transpose(-2, -1),
            A_MK,
            offs=offsets_E,
            out_dtype=torch.bfloat16,
        )
        return grad_A_MK, grad_B_ENK.transpose(-2, -1), None


def quantize_expert_state_dict_to_mxfp8(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Replace the bf16 expert weights with the MXFP8 operands the generator stores.

    Gate and up are fused into a single ``w13`` to match
    ``mxfp8_inference_grouped_experts``, which is the only consumer of these.

    Args:
        state_dict: the trainer's state dict; non-expert entries pass through.

    Returns:
        A new state dict, each expert weight replaced by ``_qdata`` and
        unswizzled ``_scales`` entries named for the generator's parameters.
    """
    groups: dict[str, dict[str, str]] = {}
    for key in state_dict:
        match = _EXPERT_KEY.match(key)
        if match:
            prefix = match.group("prefix") or ""
            groups.setdefault(prefix, {})[match.group("name")] = key

    out = {k: v for k, v in state_dict.items()}
    for prefix, keys in groups.items():
        missing = [s for s in _EXPERT_SUFFIXES if s not in keys]
        if missing:
            raise ValueError(
                f"Expert weights under {prefix!r} are incomplete: missing "
                f"{missing}. All of {list(_EXPERT_SUFFIXES)} are needed to build "
                f"the generator's MXFP8 operands."
            )
        gate = out.pop(keys[_GATE])
        up = out.pop(keys[_UP])
        down = out.pop(keys[_DOWN])
        gate_local = gate.to_local()
        up_local = up.to_local()
        down_local = down.to_local()

        with torch.no_grad():
            # Interleave gate and up into the (E, N=2F, K=D) view the GEMM reads.
            E, F, D = gate_local.shape
            fused = torch.stack([gate_local.bfloat16(), up_local.bfloat16()], dim=2)
            w13 = fused.reshape(E, 2 * F, D).contiguous()
            w13_qdata, w13_scales = triton_to_mxfp8_dim0(w13, _BLOCK_SIZE, _SCALE_MODE)

            # w2 is already (E, N=D, K=F).
            w2 = down_local.bfloat16().contiguous()
            w2_qdata, w2_scales = triton_to_mxfp8_dim0(w2, _BLOCK_SIZE, _SCALE_MODE)

            # Rewrap as DTensors so TorchStore can reshard them. from_local
            # infers the global shape from the local shape and the placements,
            # so the fused w13 comes out twice as wide on its own.
            out[f"{prefix}w13_qdata"] = DTensor.from_local(
                w13_qdata, gate.device_mesh, gate.placements, run_check=False
            )
            out[f"{prefix}w13_scales"] = DTensor.from_local(
                w13_scales, gate.device_mesh, gate.placements, run_check=False
            )
            out[f"{prefix}{_DOWN}_qdata"] = DTensor.from_local(
                w2_qdata, down.device_mesh, down.placements, run_check=False
            )
            out[f"{prefix}{_DOWN}_scales"] = DTensor.from_local(
                w2_scales, down.device_mesh, down.placements, run_check=False
            )
    return out
