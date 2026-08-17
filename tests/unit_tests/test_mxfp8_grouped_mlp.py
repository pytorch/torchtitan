# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the MXFP8 fused grouped-MLP override (mxfp8_grouped_mlp.py).

The five groups of INTEGRATION_DESIGN.md §2.4:

1. Composite numerics: forward+backward vs an independent quantized-unfused
   reference (CONTRACT §10 ref 2) built here from standalone RCEIL casts
   (``to_mx``), raw ``torch._scaled_grouped_mm``, and first-principles eager
   SwiGLU forward/backward -- deliberately NOT the override module's own cast
   helpers. Shapes include a D != F case, zero-token experts, and a strict
   inactive tail (A < R) filled with deliberate garbage.
2. Composition: the named fused config through the real converter +
   ``apply_overrides`` pipeline, plus the three decline paths (pad_multiple,
   non-converter config, recipe gate). Activation evidence is module/config
   TYPE only: a declined run still emits identity "[Override] ... -> ..."
   replacement lines, so log text is never taken as evidence (V2-5).
3. Autograd/AC: under the real SelectiveAC policy the composite forward runs
   exactly twice (save-from-recompute), gradients match no-AC bitwise, and
   the by-reference parameter save survives optimizer.step into the next step.
4. State dict: the inherited FusedGroupedExperts hooks round-trip the stock
   w1_EFD/w3_EFD checkpoint layout through the subclass.
5. Trace shape: one fwd+bwd launches exactly 1x kernel A, 1x kernel B, and
   2x kernel C, counted via ``torchao::`` op names ONLY (aten CPU events
   double-count under SAC recompute, V3-6).

All tests run on pristine cute_utils: nothing here (and nothing in the module
under test) calls into it; the non-CuTe casts and kernels A/B/C are safe.
"""

import importlib.util
import logging

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)):
    pytest.skip("Requires CUDA SM 10.0 (Blackwell)", allow_module_level=True)

try:
    import torchao.prototype.moe_training.mxfp8_grouped_mlp as _kernels
except ImportError:
    pytest.skip(
        "torchao MXFP8 grouped-MLP kernel module not importable",
        allow_module_level=True,
    )

if not _kernels._mxfp8_grouped_mlp_kernels_available:
    pytest.skip(
        "torchao MXFP8 grouped-MLP kernels unavailable", allow_module_level=True
    )

if importlib.util.find_spec("torchtitan.overrides.mxfp8_grouped_mlp") is None:
    pytest.skip(
        "torchtitan.overrides.mxfp8_grouped_mlp module under construction",
        allow_module_level=True,
    )

from torch.profiler import profile, ProfilerActivity

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization.utils import compute_error

from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.config import apply_overrides
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.common.moe import GroupedExperts, RoutedExperts
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_mxfp8,
)
from torchtitan.overrides.mxfp8_grouped_mlp import (
    MXFP8FusedGroupedExperts,
    mxfp8_fused_grouped_mlp,
)

# The frozen activation string (design §2.1, V2-4).
_OVERRIDE_TARGET = "torchtitan.overrides.mxfp8_grouped_mlp.mxfp8_grouped_experts"

_OP_A = "torchao::mxfp8_grouped_gemm_swiglu_fwd"
_OP_B = "torchao::mxfp8_grouped_gemm_dswiglu_bwd"
_OP_C = "torchao::mxfp8_grouped_gemm_wgrad"

_BLOCK = 32
_E4M3 = torch.float8_e4m3fn
_RCEIL = ScaleCalculationMode.RCEIL

# ---------------------------------------------------------------------------
# Test-1 tolerances (CONTRACT §10: establish tolerance from the measured
# unfused variability; no blanket epsilon shared across qdata/scales/BF16).
# Measured on GB200 (torch 2.14 nightly, ao b822dbe6, clocks capped 1200 MHz)
# with this file's reference against (a) a second reduction order of the SAME
# unfused math (per-expert fp32 dequant-loop GEMMs, split-ragged-axis wgrad
# accumulation) and (b) the design §2.1 composite dataflow (real kernels
# A/B/C + the design's triton casts), at both parametrized shapes
# (agent_scratch/mxfp8_grouped_mlp/integration/impl_T/calibrate_reference.py):
#
#   output  ref-vs-fp32 (dB)  ref-vs-alt-order (dB)  ref-vs-composite (dB)
#   y       23.62 .. 23.68      155.8 / 173.3           inf / inf
#   dx      23.72 .. 23.75      inf   / inf             inf / inf
#   dw13    23.67 .. 23.71      116.8 / 104.1           116.8 / 104.1
#   dw2     23.65 .. 23.71       95.7 / inf             149.8 / inf
#
# Identical RCEIL quantization boundaries make the lanes near-bitwise; the
# residual difference is a handful of adjacent-code flips where a value sits
# exactly on a quantization boundary, so the unfused-vs-unfused variability
# floor is 95.7 dB. The gate sits 45 dB below that floor (>10^4x error-energy
# headroom for flip-COUNT churn across cuBLAS/driver updates -- flips are
# rare boundary events, so their count, not their size, moves the SQNR) and
# 26 dB (>400x error energy) above the 23.6-23.8 dB compound-quantization
# floor that a real dataflow/layout/offsets bug collapses toward (a
# wrong-axis weight cast lands near 0 dB, not a few dB lower).
_SQNR_VS_REFERENCE_DB = 50.0
# Secondary tracking gate vs the FP32 eager MLP: measured 23.58-23.75 dB for
# every output at every shape (the I3 probe's known-good 23.6-23.7 regime);
# 2.5 dB of seed headroom. Catches a blind spot shared by both MXFP8 lanes.
_SQNR_VS_FP32_DB = 21.0
# ---------------------------------------------------------------------------

# Fixture shapes: per-expert row counts are 128-multiples (the dispatcher's
# pad_multiple=128 ABI guarantee), zero-token experts are legal anywhere, and
# `tail` allocates inactive rows past offsets[-1] (A < R). D != F in the
# second case so a wrong-axis weight cast cannot cancel (V1-5).
_CASES = {
    "debugmodel_tail_zero_expert": dict(
        d=256, f=256, sizes=[128, 128, 0, 128, 128, 128, 0, 256], tail=128, seed=0
    ),
    "asym_d_ne_f_tail": dict(d=256, f=512, sizes=[256, 0, 128], tail=128, seed=1),
}


def _make_case(*, d, f, sizes, tail, seed=0):
    """Dispatcher-shaped fixture: expert-major x [R, D] with per-expert row
    counts ``sizes``, offsets = inclusive cumsum, and a strict inactive tail
    of ``tail`` rows. Tail rows carry large deliberate garbage (the D11
    attack): producers do not define them, kernels must never read them, and
    y/dx comparisons mask them because ``_scaled_grouped_mm`` leaves output
    tail rows unwritten (V1-6)."""
    g = len(sizes)
    a = sum(sizes)
    r = a + tail
    torch.manual_seed(seed)
    offsets = torch.tensor(
        [sum(sizes[: i + 1]) for i in range(g)], device="cuda", dtype=torch.int32
    )
    x = torch.randn(r, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    dy = torch.randn(r, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    w13 = torch.randn(g, f, 2, d, device="cuda", dtype=torch.bfloat16) / d**0.5
    w2 = torch.randn(g, d, f, device="cuda", dtype=torch.bfloat16) / d**0.5
    if tail:
        x[a:] = 12345.0
        dy[a:] = -6789.0
    return dict(x=x, dy=dy, w13=w13, w2=w2, offsets=offsets, sizes=sizes, a=a, r=r)


# ---------------------------------------------------------------------------
# Independent quantized-unfused reference (CONTRACT §10 ref 2). Standalone
# torchao RCEIL casts (``to_mx``) + raw ``torch._scaled_grouped_mm`` + eager
# SwiGLU forward/backward at the CONTRACT §5.3/§6.3 boundary order (the BF16
# round of z precedes SwiGLU; the BF16 rounds of h/dz precede their
# quantizers); wgrads follow the §7.3 normative math (colwise quant-dequant +
# fp32 matmul per expert). Built from first principles, sharing no code with
# the module under test. GPU-proven to reproduce the probe-known 23.6-23.7 dB
# SQNR vs FP32 at the I3 probe shape (impl_T/calibrate_reference.py).
# ---------------------------------------------------------------------------


def _rceil_rowwise(t):
    """[M, K] bf16 -> (E4M3 [M, K] row-major qdata, whole-matrix blocked E8M0
    scales)."""
    scale, q = to_mx(t, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, to_blocked(scale)


def _rceil_rowwise_3d(w):
    """[G, N, K] -> per-group qdata [G, N, K] quantized along K + per-group
    blocked scales [G, numel]."""
    qs, sfs = zip(*(_rceil_rowwise(w[g]) for g in range(w.shape[0])))
    return torch.stack(list(qs)), torch.stack(list(sfs))


def _rceil_colwise_3d(w):
    """[G, N, K] -> qdata [G, N, K] stride (N*K, 1, N) quantized along N +
    per-group blocked scales (the ``mat2`` of a dgrad ``_scaled_grouped_mm``)."""
    qs, sfs = zip(
        *(_rceil_rowwise(w[g].t().contiguous()) for g in range(w.shape[0]))
    )
    return torch.stack(list(qs)).transpose(-2, -1), torch.stack(list(sfs))


def _dequant(q, scale):
    """Exact fp32 dequantization of a row-major RCEIL cast."""
    m, k = q.shape
    return (
        q.float().view(m, k // _BLOCK, _BLOCK)
        * scale.to(torch.float32).view(m, k // _BLOCK, 1)
    ).view(m, k)


def _quant_dequant_colwise(t):
    """[m, N] bf16 -> fp32 [N, m]: RCEIL-quantize along the row axis (32x1,
    what kernel C's colwise operands hold) and dequantize. Per-expert slices
    quantize identically to the whole matrix because 128-multiple group sizes
    keep every 32-value block inside one group."""
    scale, q = to_mx(t.t().contiguous(), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return _dequant(q, scale)


def _wgrad_expert(a, b):
    """CONTRACT §7.3 normative wgrad for one expert: dequant(a_col).T @
    dequant(b_col), fp32 accumulation, one BF16 round. a [m, N], b [m, K] ->
    [N, K]."""
    return (_quant_dequant_colwise(a) @ _quant_dequant_colwise(b).t()).to(
        torch.bfloat16
    )


def _reference_forward_backward(x, w13, w2, dy, offsets, sizes, a):
    """Returns (y, dx, dw13 [G, F, 2, D], dw2 [G, D, F]). y/dx tail rows [a:]
    are defined as zero here (the real GEMMs leave them unwritten; callers
    mask them out of every comparison)."""
    r, d = x.shape
    g, f = w13.shape[0], w13.shape[1]
    w13i = w13.reshape(g, 2 * f, d)  # element-interleaved [gate_0, up_0, ...]

    # FC1 forward, then eager SwiGLU on the BF16-rounded z (§5.3 order).
    x_q, x_sf = _rceil_rowwise(x)
    w13_row_q, w13_row_sf = _rceil_rowwise_3d(w13i)
    z = torch._scaled_grouped_mm(
        x_q,
        w13_row_q.transpose(-2, -1),
        x_sf.reshape(r, -1),
        w13_row_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    z[a:] = 0
    gate = z[:, 0::2].float()
    up = z[:, 1::2].float()
    h = (F.silu(gate) * up).to(torch.bfloat16)

    # FC2 forward.
    h_q, h_sf = _rceil_rowwise(h)
    w2_row_q, w2_row_sf = _rceil_rowwise_3d(w2)
    y = torch._scaled_grouped_mm(
        h_q,
        w2_row_q.transpose(-2, -1),
        h_sf.reshape(r, -1),
        w2_row_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    y[a:] = 0

    # FC2 dgrad, then eager dSwiGLU on the BF16-rounded dh (§6.3 order).
    dy_q, dy_sf = _rceil_rowwise(dy)
    w2_col_q, w2_col_sf = _rceil_colwise_3d(w2)
    dh = torch._scaled_grouped_mm(
        dy_q,
        w2_col_q,
        dy_sf.reshape(r, -1),
        w2_col_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    dh[a:] = 0
    sig = torch.sigmoid(gate)
    silu_g = gate * sig
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    dhf = dh.float()
    dgate = (dhf * up * dsilu).to(torch.bfloat16)
    dup = (dhf * silu_g).to(torch.bfloat16)
    dz = torch.stack((dgate, dup), dim=-1).reshape(r, 2 * f)

    # FC1 dgrad.
    dz_q, dz_sf = _rceil_rowwise(dz)
    w13_col_q, w13_col_sf = _rceil_colwise_3d(w13i)
    dx = torch._scaled_grouped_mm(
        dz_q,
        w13_col_q,
        dz_sf.reshape(r, -1),
        w13_col_sf.reshape(g, -1),
        offs=offsets,
        out_dtype=torch.bfloat16,
    )
    dx[a:] = 0

    # Wgrads over active rows only; zero-token experts stay all-zero.
    dw13 = torch.zeros(g, 2 * f, d, device=x.device, dtype=torch.bfloat16)
    dw2 = torch.zeros(g, d, f, device=x.device, dtype=torch.bfloat16)
    prev = 0
    for gi in range(g):
        end = int(offsets[gi])
        if end > prev:
            dw13[gi] = _wgrad_expert(dz[prev:end], x[prev:end])
            dw2[gi] = _wgrad_expert(dy[prev:end], h[prev:end])
        prev = end
    return y, dx, dw13.view(g, f, 2, d), dw2


def _fp32_reference(x, w13, w2, dy, offsets, a):
    """FP32 eager autograd MLP over the active rows (the probes' high-
    precision reference lane)."""
    g = w13.shape[0]
    x32 = x[:a].float().detach().requires_grad_(True)
    w13_32 = w13.float().detach().requires_grad_(True)
    w2_32 = w2.float().detach().requires_grad_(True)
    outs, prev = [], 0
    for gi in range(g):
        end = int(offsets[gi])
        gate = x32[prev:end] @ w13_32[gi, :, 0].t()
        up = x32[prev:end] @ w13_32[gi, :, 1].t()
        h = F.silu(gate) * up
        outs.append(h @ w2_32[gi].t())
        prev = end
    y_ref = torch.cat(outs, dim=0)
    y_ref.backward(dy[:a].float())
    return y_ref, x32.grad, w13_32.grad, w2_32.grad


# ---------------------------------------------------------------------------
# 1. Composite numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_CASES))
def test_composite_matches_quantized_unfused_reference(case):
    fx = _make_case(**_CASES[case])
    a, r, d = fx["a"], fx["r"], fx["x"].shape[1]

    x = fx["x"].clone().detach().requires_grad_(True)
    w13 = fx["w13"].clone().detach().requires_grad_(True)
    w2 = fx["w2"].clone().detach().requires_grad_(True)
    y = mxfp8_fused_grouped_mlp(x, w13, w2, fx["offsets"])
    assert y.shape == (r, d)
    assert y.dtype == torch.bfloat16
    y.backward(fx["dy"])

    ref_y, ref_dx, ref_dw13, ref_dw2 = _reference_forward_backward(
        fx["x"], fx["w13"], fx["w2"], fx["dy"], fx["offsets"], fx["sizes"], a
    )
    fp32 = _fp32_reference(fx["x"], fx["w13"], fx["w2"], fx["dy"], fx["offsets"], a)

    # Zero-token experts must produce exactly-zero weight gradients
    # (kernel C's write-defined contract), including through the [G, F, 2, D]
    # param view.
    for gi, m in enumerate(fx["sizes"]):
        if m == 0:
            assert w13.grad[gi].abs().max().item() == 0.0
            assert w2.grad[gi].abs().max().item() == 0.0

    # y/dx are compared over active rows only: the FC2-fwd/FC1-dgrad
    # _scaled_grouped_mm leaves the inactive tail [A, R) unwritten in BOTH
    # lanes (V1-6), and the garbage planted in the x/dy tails must not move
    # any active output (D11).
    for name, got, ref, hp in [
        ("y", y[:a], ref_y[:a], fp32[0]),
        ("dx", x.grad[:a], ref_dx[:a], fp32[1]),
        ("dw13", w13.grad, ref_dw13, fp32[2]),
        ("dw2", w2.grad, ref_dw2, fp32[3]),
    ]:
        assert torch.isfinite(got).all(), f"{name} contains non-finite values"
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= _SQNR_VS_REFERENCE_DB, (
            f"{name} SQNR vs quantized-unfused reference {sqnr} < "
            f"{_SQNR_VS_REFERENCE_DB}"
        )
        sqnr_hp = compute_error(hp.float(), got.float())
        assert sqnr_hp >= _SQNR_VS_FP32_DB, (
            f"{name} SQNR vs fp32 {sqnr_hp} < {_SQNR_VS_FP32_DB}"
        )


# ---------------------------------------------------------------------------
# 2. Composition (config-time factory gating through the real pipeline)
# ---------------------------------------------------------------------------


def _prepare(config):
    """Mimic the trainer's pre-override step (sharding fill)."""
    config.model_spec.model.update_from_config(config=config)
    return config


def _routed_experts_nodes(config):
    return list(config.traverse(RoutedExperts.Config))


def test_composition_fired_on_named_config():
    try:
        from torchtitan.models.deepseek_v3.config_registry import (
            deepseek_v3_debugmodel_mxfp8_grouped_mlp,
        )
    except ImportError:
        pytest.skip("deepseek_v3_debugmodel_mxfp8_grouped_mlp under construction")

    config = deepseek_v3_debugmodel_mxfp8_grouped_mlp()
    # The activation string is part of the frozen interface (V2-4).
    assert _OVERRIDE_TARGET in config.override.imports
    _prepare(config)
    replacements = apply_overrides(config.override, config)
    assert replacements

    nodes = _routed_experts_nodes(config)
    assert nodes
    for _fqn, cfg, _parent, _attr in nodes:
        assert type(cfg.inner_experts) is MXFP8FusedGroupedExperts.Config
        assert isinstance(cfg.token_dispatcher, TorchAOTokenDispatcher.Config)
        assert cfg.token_dispatcher.pad_multiple == 128

    # Module TYPE is the accepted activation evidence (never log text, V2-5).
    with torch.device("meta"):
        experts = nodes[0][1].inner_experts.build()
    assert type(experts) is MXFP8FusedGroupedExperts


def test_composition_declines_pad_multiple_32_dispatcher(caplog):
    config = _prepare(deepseek_v3_debugmodel_mxfp8())
    config.override.imports.append(_OVERRIDE_TARGET)
    for _fqn, cfg, _parent, _attr in _routed_experts_nodes(config):
        cfg.token_dispatcher.pad_multiple = 32
    with caplog.at_level(logging.WARNING):
        replacements = apply_overrides(config.override, config)

    converter_cfg_cls = _get_mxfp8_grouped_experts_cls(GroupedExperts).Config
    nodes = _routed_experts_nodes(config)
    assert nodes
    for _fqn, cfg, _parent, _attr in nodes:
        # Declined: the converter's unfused MXFP8 config stays in place.
        assert type(cfg.inner_experts) is converter_cfg_cls
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)
    # The misleading-identity-log caveat (V2-5): the declined apply still
    # reports "[Override] ... RoutedExperts.Config -> RoutedExperts.Config"
    # replacements, so their presence proves nothing about activation.
    assert replacements


def test_composition_declines_wrong_recipe(caplog):
    config = _prepare(deepseek_v3_debugmodel_mxfp8())
    config.override.imports.append(_OVERRIDE_TARGET)
    for _fqn, cfg, _parent, _attr in _routed_experts_nodes(config):
        cfg.inner_experts.recipe_name = "mxfp8_floor"
    with caplog.at_level(logging.WARNING):
        apply_overrides(config.override, config)

    converter_cfg_cls = _get_mxfp8_grouped_experts_cls(GroupedExperts).Config
    nodes = _routed_experts_nodes(config)
    assert nodes
    for _fqn, cfg, _parent, _attr in nodes:
        assert type(cfg.inner_experts) is converter_cfg_cls


def test_composition_leaves_stock_config_untouched():
    # No converter ran: the factory sees stock GroupedExperts.Config nodes and
    # must never claim them.
    config = _prepare(deepseek_v3_debugmodel())
    config.override.imports.append(_OVERRIDE_TARGET)
    apply_overrides(config.override, config)

    nodes = _routed_experts_nodes(config)
    assert nodes
    for _fqn, cfg, _parent, _attr in nodes:
        assert type(cfg.inner_experts) is GroupedExperts.Config


# ---------------------------------------------------------------------------
# 3/5. Module-level fixtures (SelectiveAC + trace shape)
# ---------------------------------------------------------------------------

_MOD_D, _MOD_F, _MOD_E = 256, 256, 4
_MOD_SIZES = [128, 256, 128, 128]


def _build_module(seed):
    torch.manual_seed(seed)
    module = MXFP8FusedGroupedExperts.Config(
        dim=_MOD_D, hidden_dim=_MOD_F, num_experts=_MOD_E
    ).build()
    module = module.to("cuda")
    with torch.no_grad():
        # fp32 master weights: the .bfloat16() casts stay outside the Function
        # (design §2.1 dtype contract), so autograd routes bf16 grads back to
        # fp32 params.
        module.w13.normal_(0.0, _MOD_D**-0.5)
        module.w2_EDF.normal_(0.0, _MOD_D**-0.5)
    return module


def _module_inputs(seed=0):
    torch.manual_seed(seed)
    r = sum(_MOD_SIZES)
    x = torch.randn(r, _MOD_D, device="cuda", dtype=torch.bfloat16) / _MOD_D**0.5
    dy = torch.randn(r, _MOD_D, device="cuda", dtype=torch.bfloat16) / _MOD_D**0.5
    num_tokens = torch.tensor(_MOD_SIZES, device="cuda")
    return x, dy, num_tokens


def _run_module(module, x, dy, num_tokens):
    x = x.clone().detach().requires_grad_(True)
    y = module(x, num_tokens)
    y.backward(dy)
    return y.detach(), x.grad


def _torchao_op_counts(prof):
    # Count ONLY the torchao:: custom-op events: aten CPU events double-count
    # under SAC recompute (V3-6), so they are never used for launch evidence.
    counts = {}
    for evt in prof.key_averages():
        if evt.key in (_OP_A, _OP_B, _OP_C):
            counts[evt.key] = counts.get(evt.key, 0) + evt.count
    return counts


def test_selective_ac_recompute_count_and_bitwise_grads():
    x, dy, num_tokens = _module_inputs()

    ref = _build_module(seed=1)
    y_ref, dx_ref = _run_module(ref, x, dy, num_tokens)

    acm = _build_module(seed=2)
    with torch.no_grad():
        acm.w13.copy_(ref.w13)
        acm.w2_EDF.copy_(ref.w2_EDF)
    wrapped = SelectiveAC(SelectiveAC.Config())._wrap_block(acm)

    # Warm up kernel JIT outside the profiled region.
    _run_module(wrapped, x, dy, num_tokens)
    acm.zero_grad(set_to_none=True)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        y_ac, dx_ac = _run_module(wrapped, x, dy, num_tokens)
    counts = _torchao_op_counts(prof)

    # D8: the composite forward runs exactly twice under SelectiveAC (original
    # + recompute; saves come from the recompute pass), backward once.
    assert counts.get(_OP_A, 0) == 2, counts
    assert counts.get(_OP_B, 0) == 1, counts
    assert counts.get(_OP_C, 0) == 2, counts

    # Deterministic kernels + save-from-recompute => bitwise-identical results.
    assert torch.equal(y_ac, y_ref)
    assert torch.equal(dx_ac, dx_ref)
    assert torch.equal(acm.w13.grad, ref.w13.grad)
    assert torch.equal(acm.w2_EDF.grad, ref.w2_EDF.grad)


def test_param_ref_save_survives_optimizer_step():
    # The Function saves w13/w2 by reference; the same-step backward precedes
    # the optimizer update, so step -> next fwd+bwd must work (and a step
    # BETWEEN forward and backward is the documented version-counter hazard,
    # not this pattern).
    x, dy, num_tokens = _module_inputs()
    module = _build_module(seed=3)
    wrapped = SelectiveAC(SelectiveAC.Config())._wrap_block(module)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

    _run_module(wrapped, x, dy, num_tokens)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    _, dx = _run_module(wrapped, x, dy, num_tokens)
    assert dx is not None
    assert torch.isfinite(dx).all()
    assert module.w13.grad is not None
    assert torch.isfinite(module.w13.grad).all()


# ---------------------------------------------------------------------------
# 4. State dict (inherited FusedGroupedExperts checkpoint hooks)
# ---------------------------------------------------------------------------


def test_state_dict_round_trips_stock_layout():
    src = _build_module(seed=4)
    sd = src.state_dict()

    # Saved in the stock GroupedExperts layout, not as the fused parameter.
    assert set(sd) == {"w1_EFD", "w3_EFD", "w2_EDF"}
    assert torch.equal(sd["w1_EFD"], src.w13[:, :, 0, :])
    assert torch.equal(sd["w3_EFD"], src.w13[:, :, 1, :])

    dst = _build_module(seed=5)
    dst.load_state_dict(sd)
    assert torch.equal(dst.w13, src.w13)
    assert torch.equal(dst.w2_EDF, src.w2_EDF)


def test_state_dict_loads_stock_grouped_experts_checkpoint():
    stock = GroupedExperts.Config(
        dim=_MOD_D, hidden_dim=_MOD_F, num_experts=_MOD_E
    ).build()
    with torch.no_grad():
        for param in stock.parameters():
            param.normal_()

    fused = _build_module(seed=6).cpu()
    fused.load_state_dict(stock.state_dict())
    assert torch.equal(fused.w13[:, :, 0, :], stock.w1_EFD)
    assert torch.equal(fused.w13[:, :, 1, :], stock.w3_EFD)
    assert torch.equal(fused.w2_EDF, stock.w2_EDF)


# ---------------------------------------------------------------------------
# 5. Trace shape
# ---------------------------------------------------------------------------


def test_trace_counts_exactly_one_a_one_b_two_c():
    x, dy, num_tokens = _module_inputs()
    module = _build_module(seed=7)

    # Warm up kernel JIT outside the profiled region.
    _run_module(module, x, dy, num_tokens)
    module.zero_grad(set_to_none=True)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run_module(module, x, dy, num_tokens)
    counts = _torchao_op_counts(prof)

    assert counts.get(_OP_A, 0) == 1, counts
    assert counts.get(_OP_B, 0) == 1, counts
    assert counts.get(_OP_C, 0) == 2, counts
