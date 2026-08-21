# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Composite MXFP8 SwiGLU MLP for a fused w13 projection.

One autograd function covers the full dense MLP

    x -> MXFP8 w13 GEMM -> [gate | up] -> silu(gate) * up -> MXFP8 w2 GEMM

with both directions of every quantization done by the CuTeDSL kernels. The
``fuse_activation`` flag selects how the activation boundary is quantized:

* ``True``: the unified SwiGLU+MXFP8 kernel produces the rowwise and colwise
  MXFP8 copies directly; the BF16 activation ``h`` is never written to global
  memory.
* ``False``: ``h`` (forward) and ``[dGate | dUp]`` (backward) are materialized
  in BF16 and quantized by the standalone 1x32 / 32x1 CuTeDSL kernels.

Everything outside that boundary -- the w13/w2 GEMMs, their input, weight and
gradient casts -- is byte-for-byte identical between the two modes, so an A/B
comparison isolates the activation+quantization implementation.

There is no silent fallback: configurations the kernels cannot execute
(missing CuTeDSL runtime, DTensor operands, non-BF16 dtypes, or dimensions
violating the kernels' 128-alignment contract) raise an actionable error so
the caller can change the config -- e.g. narrow the override's ``fqns`` or
drop it for the offending module -- rather than train silently on a
different numerical path.

Two self-contained overrides wire the composites into a model:

* ``mxfp8_fused_swiglu`` (dense ``FeedForward``) builds
  :class:`MXFP8FusedSwiGLU`, a :class:`FusedSwiGLU` whose forward runs the
  dense composite.
* ``mxfp8_fused_grouped_experts`` (``RoutedExperts``) builds
  :class:`MXFP8FusedGroupedExperts` and swaps the token dispatcher for the
  padded variant the grouped composite requires (``pad_multiple=128``).

Activate by naming the factories, e.g. ``--override.imports
torchtitan.overrides.mxfp8_fused_swiglu.mxfp8_fused_swiglu``; both accept a
``fuse_activation`` kwarg via ``(target, kwargs)`` imports entries. The
composites quantize every GEMM themselves, so these overrides must not be
combined with the MXFP8 linear / grouped-experts converters on the same
modules (the factories raise if they are).
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _compute_dgrad_sm100,
    _compute_fwd_sm100,
)
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from torchtitan.components.quantization.utils import swap_token_dispatcher
from torchtitan.config import derive, override
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.overrides.fused_swiglu import (
    _fuse_w13_grouped_experts_param_init,
    _fuse_w13_grouped_experts_sharding,
    _make_fused_gate_up_init,
    FusedGroupedExperts,
    FusedSwiGLU,
)
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.utils import has_cuda_capability

__all__ = [
    "MXFP8FusedGroupedExperts",
    "MXFP8FusedSwiGLU",
    "mxfp8_fused_grouped_experts",
    "mxfp8_fused_swiglu",
    "mxfp8_swiglu_mlp_w13",
]

_BLOCK_SIZE = 32
_ELEM_DTYPE = torch.float8_e4m3fn
_KERNEL_PREFERENCE = KernelPreference.AUTO
_SCALE_MODE = ScaleCalculationMode.RCEIL
_INT32_MAX = 2**31 - 1


def _wrap_rowwise(qdata, scales, orig_dtype):
    return MXTensor.from_qdata_and_scales(
        qdata,
        scales,
        orig_dtype,
        block_size=_BLOCK_SIZE,
        kernel_preference=_KERNEL_PREFERENCE,
        is_swizzled_scales=True,
    )


def _wrap_colwise(qdata, scales, orig_dtype):
    # Colwise kernel outputs are (M, N) with strides (1, M); wrapping the
    # transpose keeps qdata row-major, which torch.mm's MXFP8 dispatch
    # requires. The flat 1D blocked scales are unaffected by the transpose.
    return _wrap_rowwise(qdata.t(), scales, orig_dtype)


def _mx_rowwise(t):
    qdata, scales = mxfp8_quantize_2d_1x32_cutedsl(t, scaling_mode=_SCALE_MODE.value)
    return _wrap_rowwise(qdata, scales, t.dtype)


def _mx_colwise(t):
    # Returns the MXTensor for t.t() quantized along t's rows (32x1 blocks).
    return _to_mxfp8_dim1_kernel_wrapper(
        t,
        _BLOCK_SIZE,
        _ELEM_DTYPE,
        t.dtype,
        _KERNEL_PREFERENCE,
        MXFP8Dim1CastKernelChoice.CUTEDSL,
        _SCALE_MODE,
    )


def _pack_w13(w13):
    # (H, 2, D) with w13[:, 0] = gate and w13[:, 1] = up, packed to (2H, D)
    # with all gate rows first -- the layout whose GEMM output feeds the
    # SwiGLU kernel's [gate | up] contract.
    hidden, _, dim = w13.shape
    return w13.transpose(0, 1).reshape(2 * hidden, dim).contiguous()


def _swiglu_forward_hp(gated):
    k = gated.shape[1] // 2
    return (F.silu(gated[:, :k].float()) * gated[:, k:].float()).to(gated.dtype)


def _swiglu_forward_casts(gated, fuse_activation):
    if fuse_activation:
        # Lazy: the kernel module imports the CuTe DSL runtime at module
        # scope; the standalone-cast arm must work without it.
        from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
            gated_act_mxfp8_cutedsl_forward,
        )

        return gated_act_mxfp8_cutedsl_forward(gated, rowwise=True, colwise=True)
    h = _swiglu_forward_hp(gated)
    h_rw, hs_rw = mxfp8_quantize_2d_1x32_cutedsl(h, scaling_mode=_SCALE_MODE.value)
    h_cw, hs_cw = mxfp8_quantize_2d_32x1_cutedsl(h, scaling_mode=_SCALE_MODE.value)
    return h_rw, h_cw, hs_rw, hs_cw


def _swiglu_backward_hp(grad_h, gated):
    k = gated.shape[1] // 2
    gate = gated[:, :k].float()
    up = gated[:, k:].float()
    grad_h_f = grad_h.float()
    # Same evaluation order as the unified kernel (which contracts `deriv`
    # into one FMA), so the two modes differ only in sigmoid lowering and
    # that contraction, not in association.
    sigmoid_gate = torch.sigmoid(gate)
    silu = gate * sigmoid_gate
    deriv = gate * (1.0 - sigmoid_gate) + 1.0
    return torch.cat(
        [
            ((grad_h_f * up) * (sigmoid_gate * deriv)).to(gated.dtype),
            (grad_h_f * silu).to(gated.dtype),
        ],
        dim=1,
    )


def _swiglu_backward_casts(grad_h, gated, fuse_activation):
    if fuse_activation:
        from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
            gated_act_mxfp8_cutedsl_backward,
        )

        return gated_act_mxfp8_cutedsl_backward(
            grad_h, gated, rowwise=True, colwise=True
        )
    d = _swiglu_backward_hp(grad_h, gated)
    d_rw, ds_rw = mxfp8_quantize_2d_1x32_cutedsl(d, scaling_mode=_SCALE_MODE.value)
    d_cw, ds_cw = mxfp8_quantize_2d_32x1_cutedsl(d, scaling_mode=_SCALE_MODE.value)
    return d_rw, d_cw, ds_rw, ds_cw


@torch._dynamo.allow_in_graph
class _MXFP8SwiGLUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, fuse_activation):
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        w13_packed = _pack_w13(w13)
        gated = torch.mm(_mx_rowwise(x2d), _mx_rowwise(w13_packed).t())
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated, fuse_activation)
        out = torch.mm(_wrap_rowwise(h_rw, hs_rw, x2d.dtype), _mx_rowwise(w2).t())
        ctx.save_for_backward(x2d, w13_packed, w2, gated, h_cw, hs_cw)
        ctx.fuse_activation = fuse_activation
        ctx.x_shape = x.shape
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w13_packed, w2, gated, h_cw, hs_cw = ctx.saved_tensors
        hidden = w13_packed.shape[0] // 2
        go = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_h = torch.mm(_mx_rowwise(go), _mx_colwise(w2).t())
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(
            grad_h, gated, ctx.fuse_activation
        )
        grad_x = torch.mm(
            _wrap_rowwise(d_rw, ds_rw, go.dtype), _mx_colwise(w13_packed).t()
        )
        grad_w13_packed = torch.mm(
            _wrap_colwise(d_cw, ds_cw, go.dtype), _mx_colwise(x2d).t()
        )
        grad_w2 = torch.mm(_mx_colwise(go), _wrap_colwise(h_cw, hs_cw, go.dtype).t())
        grad_w13 = grad_w13_packed.view(2, hidden, -1).transpose(0, 1).contiguous()
        return grad_x.reshape(ctx.x_shape), grad_w13, grad_w2, None


def _require_kernels(op_name):
    if not _mxfp8_cutedsl_kernels_available:
        raise NotImplementedError(
            f"{op_name} requires the MXFP8 CuTeDSL kernels (nvidia-cutlass-dsl "
            "on an SM100-class GPU); install the runtime or exclude this module "
            "from the MXFP8 converter."
        )


def _validate_dense_inputs(x, w13, w2):
    _require_kernels("mxfp8_swiglu_mlp_w13")
    if isinstance(x, DTensor) or isinstance(w13, DTensor) or isinstance(w2, DTensor):
        raise ValueError(
            "mxfp8_swiglu_mlp_w13 takes plain local tensors, not DTensor; pass "
            "local shards or exclude this module from the fused MXFP8 path."
        )
    if not x.is_cuda:
        raise ValueError(
            f"mxfp8_swiglu_mlp_w13 requires CUDA tensors, got device {x.device}"
        )
    if (
        x.dtype != torch.bfloat16
        or w13.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        raise ValueError(
            "mxfp8_swiglu_mlp_w13 requires BF16 inputs and weights, got "
            f"x={x.dtype}, w13={w13.dtype}, down_weight={w2.dtype}"
        )
    if w13.ndim != 3 or w13.shape[1] != 2 or w2.ndim != 2:
        raise ValueError(
            "expected w13 of shape (H, 2, D) and down_weight of shape (D_out, H), "
            f"got w13={tuple(w13.shape)}, down_weight={tuple(w2.shape)}"
        )
    hidden, _, dim = w13.shape
    n = w2.shape[0]
    m = x.numel() // x.shape[-1]
    if x.shape[-1] != dim or w2.shape[1] != hidden:
        raise ValueError(
            f"shape mismatch: x={tuple(x.shape)}, w13={tuple(w13.shape)}, "
            f"down_weight={tuple(w2.shape)}"
        )
    if m % 128 != 0 or hidden % 128 != 0 or dim % 128 != 0 or n % 128 != 0:
        raise ValueError(
            "the MXFP8 CuTeDSL kernels require every dimension to be a multiple "
            f"of 128, got tokens={m}, hidden={hidden}, dim={dim}, d_out={n}; "
            "exclude this module from the fused MXFP8 path if its shapes cannot "
            "satisfy this."
        )
    # 32-bit index-math limit over BOTH A/B arms: the unified kernel's input
    # layout reaches element 2*hidden*m - hidden - 1, but the unfused arm's
    # standalone casts of the (m, 2*hidden) backward tensor reach
    # 2*hidden*m - 1, and those kernels do not validate. Gate on the max so
    # the two arms accept identical shapes.
    if 2 * hidden * m - 1 > _INT32_MAX:
        raise ValueError(
            "tokens*hidden exceeds the kernels' 32-bit index math: "
            f"2*{hidden}*{m} - 1 > {_INT32_MAX}"
        )


def mxfp8_swiglu_mlp_w13(x, w13, down_weight, *, fuse_activation=True):
    """Dense MXFP8 SwiGLU MLP with a fused (H, 2, D) w13 weight.

    Args:
        x: BF16 input of shape (..., D).
        w13: BF16 fused gate/up weight of shape (H, 2, D); w13[:, 0] is the
            gate (w1) and w13[:, 1] the up (w3) projection.
        down_weight: BF16 down-projection weight of shape (D_out, H).
        fuse_activation: quantize the SwiGLU boundary with the unified
            SwiGLU+MXFP8 kernel instead of standalone BF16 + cast kernels.

    Returns:
        BF16 tensor of shape (..., D_out).

    Raises:
        NotImplementedError: the MXFP8 CuTeDSL kernels are unavailable.
        ValueError: DTensor operands, non-BF16 dtypes, or dimensions the
            kernels cannot execute (every dim must be a multiple of 128).
            There is no silent fallback; change the config instead.
    """
    _validate_dense_inputs(x, w13, down_weight)
    return _MXFP8SwiGLUMLP.apply(x, w13, down_weight, fuse_activation)


def _pack_w13_grouped(w13):
    # (E, F, 2, D) with w13[:, :, 0] = gate and w13[:, :, 1] = up, packed to
    # (E, 2F, D) with all gate rows first per expert (the [gate | up] layout
    # the SwiGLU kernel consumes). The reshape of the transposed view copies.
    e, f, _, d = w13.shape
    return w13.transpose(1, 2).reshape(e, 2 * f, d)


def _reblock_scales_k_groups(scales, n_rows, m_total, offs):
    # The CuTeDSL kernels emit blocked scales in full-tensor row-block-major
    # tile order; a 2d-2d grouped GEMM contracting over tokens needs the tiles
    # regrouped per token group (row-block-major within each group). With every
    # group a multiple of 128 rows the two layouts hold identical (128, 4)
    # tiles, so this is a pure tile gather.
    rb = n_rows // 128
    cb = m_total // 128
    ends = (offs // 128).long()
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    sizes = (ends - starts).clamp(min=1)
    t = torch.arange(rb * cb, device=scales.device)
    # The dispatcher may pad the token buffer globally past offs[-1]; tiles in
    # that tail belong to no group and are never read by the grouped GEMM, so
    # clamping them anywhere in bounds is enough to keep the gather valid.
    g = torch.searchsorted(ends * rb, t, right=True).clamp(max=ends.numel() - 1)
    local = t - starts[g] * rb
    src = ((local // sizes[g]) * cb + starts[g] + local % sizes[g]).clamp(
        max=rb * cb - 1
    )
    return scales.view(rb * cb, 512)[src].view(n_rows, -1)


def _wgrad_k_groups(a_qdata, a_scales, b, offs, out_dtype):
    # grad[e] = a[start:end].T @ b[start:end] for each token group. `a` arrives
    # colwise-quantized from the SwiGLU boundary ((M, Ka) with strides (1, M)
    # plus flat blocked scales); `b` gets the same GEMM-operand colwise cast the
    # existing grouped wgrad path uses.
    m, ka = a_qdata.shape
    b_t_mx = _to_mxfp8_dim1_kernel_wrapper(
        b,
        _BLOCK_SIZE,
        _ELEM_DTYPE,
        b.dtype,
        _KERNEL_PREFERENCE,
        MXFP8Dim1CastKernelChoice.CUDA,
        _SCALE_MODE,
    )
    b_scales = triton_mx_block_rearrange_2d_K_groups(b_t_mx.scale, offs // _BLOCK_SIZE)
    a_scales_2d = _reblock_scales_k_groups(a_scales, ka, m, offs)
    return torch._scaled_grouped_mm(
        a_qdata.t(),
        b_t_mx.qdata.transpose(-2, -1),
        a_scales_2d,
        b_scales,
        offs=offs,
        out_dtype=out_dtype,
    )


@torch._dynamo.allow_in_graph
class _MXFP8SwiGLUGroupedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2_t, offs, fuse_activation):
        x = x.contiguous()
        w13_packed = _pack_w13_grouped(w13)
        gated = _compute_fwd_sm100(
            x, w13_packed.transpose(-2, -1), offs, _BLOCK_SIZE, x.dtype, _SCALE_MODE
        )
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated, fuse_activation)
        out = _compute_fwd_sm100(
            _wrap_rowwise(h_rw, hs_rw, x.dtype),
            w2_t,
            offs,
            _BLOCK_SIZE,
            x.dtype,
            _SCALE_MODE,
        )
        ctx.save_for_backward(x, w13_packed, w2_t, offs, gated, h_cw, hs_cw)
        ctx.fuse_activation = fuse_activation
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w13_packed, w2_t, offs, gated, h_cw, hs_cw = ctx.saved_tensors
        e, two_f, d = w13_packed.shape
        go = grad_out.contiguous()
        grad_h = _compute_dgrad_sm100(
            go, w2_t, offs, _BLOCK_SIZE, go.dtype, _SCALE_MODE
        )
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(
            grad_h, gated, ctx.fuse_activation
        )
        grad_x = _compute_dgrad_sm100(
            _wrap_rowwise(d_rw, ds_rw, go.dtype),
            w13_packed.transpose(-2, -1),
            offs,
            _BLOCK_SIZE,
            go.dtype,
            _SCALE_MODE,
        )
        grad_w13_packed = _wgrad_k_groups(d_cw, ds_cw, x, offs, go.dtype)
        grad_w2_t = _wgrad_k_groups(h_cw, hs_cw, go, offs, go.dtype)
        grad_w13 = grad_w13_packed.view(e, 2, two_f // 2, d).transpose(1, 2)
        return grad_x, grad_w13, grad_w2_t, None, None


def _validate_grouped_inputs(x, w13, w2_t, offs):
    # The only caller is MXFP8FusedGroupedExperts.forward, which guarantees
    # plain local BF16 tensors in the module's own (M, D) / (E, F, 2, D) /
    # (E, F, D_out) shapes; only environment, config dims, and the
    # routing-dependent token count need checking.
    _require_kernels("MXFP8FusedGroupedExperts")
    _, f, _, d = w13.shape
    m = x.shape[0]
    d_out = w2_t.shape[2]
    if f % 128 != 0 or d % 128 != 0 or d_out % 128 != 0:
        raise ValueError(
            "the MXFP8 CuTeDSL kernels require expert dimensions to be "
            f"multiples of 128, got hidden={f}, dim={d}, d_out={d_out}; exclude "
            "this module from the fused MXFP8 path if its shapes cannot "
            "satisfy this."
        )
    # Group boundaries must additionally be 128-row aligned (the token
    # dispatcher's pad_multiple guarantees it); checking offs here would sync.
    # M is routing-dependent under compile (an unbacked SymInt, which type
    # tests cannot tell apart from int inside traced code), so the M
    # conditions use identity tests: literal bools raise immediately,
    # symbolic ones become deferred runtime asserts. The m >= 128 and m % 32
    # forms are redundant with m % 128 (plus non-emptiness) but must be
    # recorded separately: downstream cast-kernel wrappers and GEMM metas
    # check exactly those forms, and the symbolic engine resolves them by
    # expression match / value range, not by deriving them from mod-128.
    # The last condition is the 32-bit index-math limit over BOTH A/B arms
    # (the unfused arm's standalone casts of the (m, 2f) backward tensor
    # reach element 2*f*m - 1, slightly past the unified kernel's own input
    # bound, and those kernels do not validate).
    for cond, requirement in (
        (m >= 128, "at least 128"),
        (
            m % 128 == 0,
            "a multiple of 128 (configure the token dispatcher with "
            "pad_multiple=128)",
        ),
        (m % 32 == 0, "a multiple of 32"),
        (
            2 * f * m - 1 <= _INT32_MAX,
            "small enough for the kernels' 32-bit index math "
            "(2*hidden*tokens - 1 <= 2**31 - 1)",
        ),
    ):
        if cond is False:
            raise ValueError(
                f"MXFP8FusedGroupedExperts: token count {m} (hidden={f}) "
                f"must be {requirement}; there is no silent fallback."
            )
        if cond is not True:
            torch._check(cond)


class MXFP8FusedSwiGLU(FusedSwiGLU):
    """:class:`FusedSwiGLU` whose forward runs the composite MXFP8 SwiGLU MLP.

    Inherits the fused ``w13`` parameter, the stock-layout checkpoint hooks,
    and the FSDP/TP sharding story from :class:`FusedSwiGLU`; only ``forward``
    changes.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FusedSwiGLU.Config):
        fuse_activation: bool = True
        """Quantize the SwiGLU boundary with the unified SwiGLU+MXFP8 kernel
        (False: standalone BF16 + cast kernels; identical GEMMs either way)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.fuse_activation = config.fuse_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, DTensor):
            raise ValueError(
                "MXFP8FusedSwiGLU does not support DTensor activations (dense "
                "tensor parallelism); narrow the mxfp8_fused_swiglu override's "
                "fqns or drop it for this module."
            )
        output = mxfp8_swiglu_mlp_w13(
            x,
            self.w13,
            self.w2.weight,
            fuse_activation=self.fuse_activation,
        )
        if self.w2.bias is not None:
            output = output + self.w2.bias.to(output.dtype)
        return output


class MXFP8FusedGroupedExperts(FusedGroupedExperts):
    """:class:`FusedGroupedExperts` whose forward runs the composite MXFP8
    SwiGLU grouped MLP.

    Requires token groups padded to multiples of 128 rows (zero-filled) --
    the ``mxfp8_fused_grouped_experts`` factory swaps the token dispatcher
    accordingly. Inherits ``w13``, checkpoint hooks, and sharding from
    :class:`FusedGroupedExperts`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FusedGroupedExperts.Config):
        fuse_activation: bool = True
        """Quantize the SwiGLU boundary with the unified SwiGLU+MXFP8 kernel
        (False: standalone BF16 + cast kernels; identical GEMMs either way)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.fuse_activation = config.fuse_activation

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w13, DTensor):
            w13 = self.w13.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
        else:
            w13 = self.w13
            w2_EDF = self.w2_EDF

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        x = x_RD.bfloat16()
        w13 = w13.bfloat16()
        w2_t = w2_EDF.bfloat16().transpose(-2, -1)
        _validate_grouped_inputs(x, w13, w2_t, offsets_E)
        return _MXFP8SwiGLUGroupedMLP.apply(
            x,
            w13,
            w2_t,
            offsets_E,
            self.fuse_activation,
        ).type_as(x_RD)


@override(
    target=FeedForward.Config,
    description="Dense SwiGLU FFN via the composite MXFP8 SwiGLU MLP (fused w13).",
)
def mxfp8_fused_swiglu(
    cfg: FeedForward.Config,
    *,
    fuse_activation: bool = True,
) -> "MXFP8FusedSwiGLU.Config":
    # Config-application-time gate, matching the MXFP8 converters' UX; the
    # composite re-validates at runtime.
    if not has_cuda_capability(10, 0):
        raise ValueError(
            "mxfp8_fused_swiglu requires SM100 or later; remove the override "
            "or run on supported hardware."
        )
    # Fail loud on anything but the stock config: this override owns the whole
    # MLP's quantization, so composing it with another FFN variant or a linear
    # quantization converter is a config error, not a silent no-op.
    if type(cfg) is not FeedForward.Config:
        raise ValueError(
            "mxfp8_fused_swiglu targets the stock FeedForward.Config, got "
            f"{type(cfg).__qualname__}; narrow this override's fqns or remove "
            "the conflicting override/converter."
        )
    for name in ("w1", "w2", "w3"):
        sub = getattr(cfg, name)
        if type(sub) is not Linear.Config:
            raise ValueError(
                "mxfp8_fused_swiglu requires stock Linear.Config projections, "
                f"but {name} is {type(sub).__qualname__}. The composite "
                "quantizes every GEMM itself -- do not combine it with a "
                "linear quantization converter on the same module."
            )

    # Same param-init and sharding remaps as the fused_swiglu factory.
    w1_init = (cfg.w1.param_init or {}).get("weight")
    w3_init = (cfg.w3.param_init or {}).get("weight")
    param_init = None
    if w1_init is not None and w3_init is not None:
        param_init = {"w13": _make_fused_gate_up_init(w1_init, w3_init, gate_up_axis=1)}

    fused = derive(
        cfg,
        MXFP8FusedSwiGLU.Config,
        param_init=param_init,
        fuse_activation=fuse_activation,
    )
    base = cfg.sharding_config
    fused.sharding_config = ShardingConfig(
        state_shardings={"w13": dense_param_placement(tp=spmd.S(0))},
        in_src_shardings=base.in_src_shardings if base is not None else None,
        in_dst_shardings=base.in_dst_shardings if base is not None else None,
    )
    return fused


@override(
    target=RoutedExperts.Config,
    description="Routed experts via the composite MXFP8 SwiGLU grouped MLP "
    "(fused w13, 128-row-padded token groups).",
)
def mxfp8_fused_grouped_experts(
    cfg: RoutedExperts.Config,
    *,
    fuse_activation: bool = True,
) -> RoutedExperts.Config:
    # Config-application-time gate, matching the MXFP8 converters' UX; the
    # composite re-validates at runtime.
    if not has_cuda_capability(10, 0):
        raise ValueError(
            "mxfp8_fused_grouped_experts requires SM100 or later; remove the "
            "override or run on supported hardware."
        )
    # Targets RoutedExperts.Config (not GroupedExperts.Config) because the
    # grouped composite constrains BOTH the experts and the token dispatcher:
    # its kernels require every per-expert token group padded to a multiple of
    # 128 rows (zero-filled), which only the padded dispatch path produces.
    if type(cfg) is not RoutedExperts.Config:
        raise ValueError(
            "mxfp8_fused_grouped_experts targets the stock "
            f"RoutedExperts.Config, got {type(cfg).__qualname__}; narrow this "
            "override's fqns or remove the conflicting override."
        )
    inner = cfg.inner_experts
    if type(inner) is not GroupedExperts.Config:
        raise ValueError(
            "mxfp8_fused_grouped_experts requires the stock "
            f"GroupedExperts.Config, but inner_experts is "
            f"{type(inner).__qualname__}. The composite quantizes every "
            "grouped GEMM itself -- do not combine it with the MXFP8 "
            "grouped-experts converter."
        )

    swap_token_dispatcher(cfg, pad_multiple=128)

    # Same param-init and sharding remaps as the fused_grouped_experts factory.
    param_init = _fuse_w13_grouped_experts_param_init(inner.param_init)
    fused = derive(
        inner,
        MXFP8FusedGroupedExperts.Config,
        param_init=param_init,
        fuse_activation=fuse_activation,
    )
    base = inner.sharding_config
    if base is not None:
        fused.sharding_config = _fuse_w13_grouped_experts_sharding(base)
    cfg.inner_experts = fused
    return cfg
