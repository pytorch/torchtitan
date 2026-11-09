# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DSv3 671B loop's verified kernels, called from eager autograd.Functions.

Each kernel is the standalone ``run(*args)`` file the loop promoted into its kernel cache; its
contract (argument and output order, shapes, strides, dtypes) is the lifted graph region's, which
``reference.py`` next to it spells out. A pattern can have several specializations (one per shape);
``kernel`` picks the one matching its arguments and returns None when that shape has no promoted
kernel, in which case the caller runs the stock ops.

As in the Qwen3 30B-A3B port (``qwen3_30b_eager_study/graph_kernels.py``, whose linear helpers this
module imports), a Function owns every forward op whose backward a backward region contains and
returns a tensor once per consumer where the graph region contains an autograd-engine add. One
region crosses a layer boundary (G-049: layer L's attention_norm backward plus the quantization of
that gradient for layer L-1's FFN down projection). Its kernel runs in layer L's backward; the
quantized gradient reaches layer L-1's backward through ``_GRAD_QUANT``, keyed by the gradient
tensor the autograd engine hands from one to the other.

Shape suffixes: T tokens, D model dim, E experts, K top-k, H heads, F FFN hidden dim, M rows of a
GEMM, N its output features.
"""

import collections
import glob
import importlib.util
import json
import os

import torch
from qwen3_30b_eager_study.graph_kernels import (
    linear_meta,
    mxfp8_operands,
    scaled_mm,
    wgrad,
    wgrad_from_col,
)
from torch.autograd.function import once_differentiable
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.models.common.linear.inplace_wgrad_accum import running_grad

from torchtitan_recipes.overrides.fused_mla import _fused_mla_k_rope_op
from torchtitan_recipes.overrides.fused_swiglu import (
    silu_and_mul_backward_op,
    silu_and_mul_op,
)

KERNEL_CACHE = os.environ.get(
    "EAGER_KERNEL_CACHE",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../dsv3_671b_distmoe_mxfp8/kernel_cache",
    ),
)
_SCALING_MODE = "rceil"
# prefix -> [(input signature, kernel path or None)], one entry per specialization.
_SPECS: dict[str, list] = {}
_RUNS: dict[str, object] = {}
_MODULES: dict[str, object] = {}
# Calls per kernel and fallbacks to stock ops, printed at exit by eager_recipe.
CALLS: collections.Counter = collections.Counter()


def _signature(tensors):
    return tuple((tuple(t.shape), str(t.dtype).removeprefix("torch.")) for t in tensors)


def kernel(prefix: str, *args):
    """``run`` of pattern ``pat_<prefix>*``'s promoted kernel for these arguments, or None."""
    if prefix not in _SPECS:
        (pat,) = glob.glob(os.path.join(KERNEL_CACHE, f"pat_{prefix}*"))
        entries = []
        for spec_path in sorted(glob.glob(os.path.join(pat, "specialization_*.json"))):
            spec = json.load(open(spec_path))
            path = os.path.join(pat, spec["kernel_name"])
            sig = tuple(
                (tuple(t["shape"]), t["dtype"].removeprefix("torch."))
                for t in spec["inputs"]
            )
            entries.append((sig, path if os.path.exists(path) else None))
        _SPECS[prefix] = entries
    sig = _signature(args)
    for spec_sig, path in _SPECS[prefix]:
        if spec_sig == sig:
            break
    else:
        path = None
    if path is None:
        return None
    if path not in _RUNS:
        spec = importlib.util.spec_from_file_location(
            f"graph_kernel_{os.path.basename(path)[:8]}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RUNS[path] = module.run
        _MODULES[path] = module
    CALLS[prefix] += 1
    return _RUNS[path]


def quantize_rows(x_MK):
    """Rowwise MXFP8 FPROP operands of an activation, as ``_MXFP8LinearFunction.forward``."""
    q, _, s, _ = mxfp8_quantize_cuda(
        x_MK, rowwise=True, colwise=False, scaling_mode=_SCALING_MODE
    )
    return q, triton_mx_block_rearrange(s)


def _quantize_rows_cols(x_MK):
    """Rowwise and columnwise operands, rearranged; the columnwise qdata is not transposed."""
    q_row, q_col, s_row, s_col = mxfp8_quantize_cuda(
        x_MK, rowwise=True, colwise=True, scaling_mode=_SCALING_MODE
    )
    return (
        q_row,
        triton_mx_block_rearrange(s_row),
        q_col,
        triton_mx_block_rearrange(s_col),
    )


def _grad_operands(g_MN):
    """(row qdata, row scale, column qdata transposed, column scale) of an output gradient."""
    q_row, s_row, q_col, s_col = _quantize_rows_cols(g_MN)
    return q_row, s_row, q_col.t(), s_col


def _dgrad_and_grad_col(g_MN, w_dgrad, w_sdgrad):
    """G-015 (lfu5 / bqhy): quantize an output gradient and run the DGRAD GEMM.

    Returns (DGRAD, column scale, column qdata transposed). Stock ops when the shape has no
    promoted kernel.
    """
    run = kernel("lfu5", g_MN, w_dgrad, w_sdgrad)
    if run is not None:
        return run(g_MN, w_dgrad, w_sdgrad)
    run = kernel("bqhy", g_MN, w_dgrad, w_sdgrad)
    if run is not None:
        q_col_t, dgrad, s_col = run(g_MN, w_dgrad, w_sdgrad)
        return dgrad, s_col, q_col_t
    CALLS["G-015 stock"] += 1
    q_row, s_row, q_col_t, s_col = _grad_operands(g_MN)
    return scaled_mm(q_row, w_dgrad, s_row, w_sdgrad), s_col, q_col_t


# The quantized input gradient of the latest layer whose attention-input backward produced one:
# id(gradient) -> (gradient, row qdata, row scale, column qdata transposed, column scale).
_GRAD_QUANT: dict[int, tuple] = {}
GRAD_QUANT_LOOKUPS: collections.Counter = collections.Counter()


def _stash_grad_quant(g, *operands):
    _GRAD_QUANT.clear()
    _GRAD_QUANT[id(g)] = (g, *operands)


def _take_grad_quant(g):
    entry = _GRAD_QUANT.pop(id(g), None)
    if entry is None or entry[0] is not g:
        GRAD_QUANT_LOOKUPS["miss"] += 1
        return None
    GRAD_QUANT_LOOKUPS["hit"] += 1
    return entry[1:]


class Linear(torch.autograd.Function):
    """An MXFP8 linear (stock FPROP) whose backward runs G-015 (gradient quantization and DGRAD
    in one kernel, lfu5) where a kernel exists for the gradient's shape."""

    @staticmethod
    def forward(ctx, x, weight, meta, save_mxfp8_input):
        x_MK = x.reshape(-1, x.shape[-1]).contiguous()
        ops = mxfp8_operands(weight)
        if save_mxfp8_input:
            x_row, s_row, x_col, s_col = _quantize_rows_cols(x_MK)
            ctx.save_for_backward(x_col, s_col, weight)
        else:
            x_row, s_row = quantize_rows(x_MK)
            ctx.save_for_backward(x_MK, weight)
        out_MN = scaled_mm(
            x_row, ops.weight_qdata_fprop_KN, s_row, ops.weight_scale_fprop_swizzled
        )
        ctx.meta = meta
        ctx.save_mxfp8_input = save_mxfp8_input
        ctx.input_shape = x.shape
        ctx.weight_shape = weight.shape
        return out_MN.view(*x.shape[:-1], out_MN.shape[-1])

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out):
        *saved_input, weight = ctx.saved_tensors
        wgrad_dtype, weight_param = ctx.meta
        ops = mxfp8_operands(weight)
        dgrad, s_col, q_col_t = _dgrad_and_grad_col(
            g_out.reshape(-1, g_out.shape[-1]).contiguous(),
            ops.weight_qdata_dgrad_NK,
            ops.weight_scale_dgrad_swizzled,
        )
        if ctx.save_mxfp8_input:
            g_w = wgrad_from_col(
                q_col_t,
                s_col,
                *saved_input,
                weight_param,
                wgrad_dtype,
                ctx.weight_shape,
            )
        else:
            g_w = wgrad(
                q_col_t,
                s_col,
                *saved_input,
                weight_param,
                wgrad_dtype,
                ctx.weight_shape,
            )
        return dgrad.view(ctx.input_shape), g_w, None, None


def linear(module, x):
    """``MXFP8Linear.forward`` through ``Linear``."""
    if module.bias is not None:
        raise ValueError("graph-kernel linear: bias is not supported")
    return Linear.apply(
        x,
        module.weight,
        linear_meta(module, module.weight),
        module.input_activation_format_for_backward == "mxfp8",
    )


class AttentionInput(torch.autograd.Function):
    """attention_norm -> wq_a and wkv_a (MXFP8, BF16 input saved for WGRAD).

    Forward: G-047 (rmsnorm and both rowwise quantizations, use7), the unchanged FPROP GEMMs.
    Backward: G-015 for wq_a and wkv_a (gradient quantization and DGRAD, lfu5), the WGRADs, the
    engine add of both DGRADs, G-049 (rmsnorm backward, the residual gradient's add and the row and
    column quantization of the result for layer L-1's FFN down projection, 3ybb). Returns ``x`` as
    a third output for the residual add, whose gradient G-049 adds.
    """

    @staticmethod
    def forward(ctx, x_TD, attn_norm_w, wq_a_w, wkv_a_w, wq_a_meta, wkv_a_meta):
        y_TD, q1, s1, rstd, q2, s2 = kernel("use7", x_TD, attn_norm_w)(
            x_TD, attn_norm_w
        )
        wq_a_ops = mxfp8_operands(wq_a_w)
        wkv_a_ops = mxfp8_operands(wkv_a_w)
        q_a = scaled_mm(
            q1, wq_a_ops.weight_qdata_fprop_KN, s1, wq_a_ops.weight_scale_fprop_swizzled
        )
        kv_a = scaled_mm(
            q2,
            wkv_a_ops.weight_qdata_fprop_KN,
            s2,
            wkv_a_ops.weight_scale_fprop_swizzled,
        )
        ctx.save_for_backward(x_TD, rstd, y_TD, attn_norm_w, wq_a_w, wkv_a_w)
        ctx.metas = (wq_a_meta, wkv_a_meta)
        ctx.weight_shapes = (wq_a_w.shape, wkv_a_w.shape)
        return q_a, kv_a, x_TD.view_as(x_TD)

    @staticmethod
    @once_differentiable
    def backward(ctx, g_q_a, g_kv_a, g_x_res):
        x_TD, rstd, y_TD, attn_norm_w, wq_a_w, wkv_a_w = ctx.saved_tensors
        g_y = None
        g_ws = []
        for g, w, (wgrad_dtype, weight_param), shape in zip(
            (g_q_a, g_kv_a), (wq_a_w, wkv_a_w), ctx.metas, ctx.weight_shapes
        ):
            ops = mxfp8_operands(w)
            dgrad, s_col, q_col_t = _dgrad_and_grad_col(
                g.contiguous(),
                ops.weight_qdata_dgrad_NK,
                ops.weight_scale_dgrad_swizzled,
            )
            g_ws.append(wgrad(q_col_t, s_col, y_TD, weight_param, wgrad_dtype, shape))
            g_y = dgrad if g_y is None else g_y + dgrad
        g_norm_w, g_x, q_row, s_row, s_col, q_col_t = kernel(
            "3ybb", rstd, g_y, x_TD, attn_norm_w, g_x_res
        )(rstd, g_y, x_TD, attn_norm_w, g_x_res.contiguous())
        _stash_grad_quant(g_x, q_row, s_row, q_col_t, s_col)
        return g_x, g_norm_w, g_ws[0], g_ws[1], None, None


class MLAKeyValue(torch.autograd.Function):
    """wkv_b (MXFP8, columnwise input saved) -> per-head K (RoPE on the shared key) and V.

    Forward: the stock wkv_b FPROP and ``torchtitan::fused_mla_k_rope``.
    Backward: G-035 (``torchtitan::fused_mla_kv_backward`` and the gradient's row and column
    quantization, k6pn), the DGRAD and WGRAD GEMMs.
    """

    @staticmethod
    def forward(
        ctx,
        kv_c_TL,
        k_pe_TR,
        wkv_b_w,
        cache_real,
        positions_1T,
        q_nope_dim,
        packed_head_dim,
        meta,
    ):
        ops = mxfp8_operands(wkv_b_w)
        x_row, s_row, x_col, s_col = _quantize_rows_cols(kv_c_TL)
        kv_TN = scaled_mm(
            x_row, ops.weight_qdata_fprop_KN, s_row, ops.weight_scale_fprop_swizzled
        )
        kv_1THP = kv_TN.view(1, kv_TN.shape[0], -1, packed_head_dim)
        k_1THK = _fused_mla_k_rope_op(
            kv_1THP, k_pe_TR.unsqueeze(0), cache_real, positions_1T, q_nope_dim
        )
        v_THV = kv_1THP[..., q_nope_dim:].squeeze(0)
        ctx.save_for_backward(x_col, s_col, wkv_b_w, cache_real, positions_1T)
        ctx.meta = meta
        ctx.weight_shape = wkv_b_w.shape
        return k_1THK.squeeze(0), v_THV

    @staticmethod
    @once_differentiable
    def backward(ctx, g_k_THK, g_v_THV):
        x_col, x_scale_col, wkv_b_w, cache_real, positions_1T = ctx.saved_tensors
        wgrad_dtype, weight_param = ctx.meta
        g_k_pe, q_row, s_row, s_col, q_col_t = kernel(
            "k6pn", g_v_THV, g_k_THK, cache_real, positions_1T
        )(g_v_THV.contiguous(), g_k_THK.contiguous(), cache_real, positions_1T)
        ops = mxfp8_operands(wkv_b_w)
        g_kv_c = scaled_mm(
            q_row, ops.weight_qdata_dgrad_NK, s_row, ops.weight_scale_dgrad_swizzled
        )
        g_w = wgrad_from_col(
            q_col_t,
            s_col,
            x_col,
            x_scale_col,
            weight_param,
            wgrad_dtype,
            ctx.weight_shape,
        )
        return g_kv_c, g_k_pe, g_w, None, None, None, None, None


class AttentionOutput(torch.autograd.Function):
    """wo (MXFP8, BF16 input saved) -> residual add -> ffn_norm, plus ffn_norm's output quantized
    for the FFN up projection.

    Forward: the stock wo FPROP, G-082 (add, rmsnorm, rowwise quantization, dre3).
    Backward: G-049's same-layer form (ffn_norm backward, the residual gradient's add and the row
    and column quantization of the result for wo, a6rv), the wo DGRAD and WGRAD GEMMs (the graph's
    G-003 also packs the WGRAD into the FSDP reduce-scatter bucket, which eager FSDP2 does itself).
    Returns the residual stream ``h`` for the block's last add and the normalized ``y`` with its
    rowwise MXFP8 operands (no gradient) for the FFN up projection.
    """

    @staticmethod
    def forward(ctx, attn_TD, x_TD, wo_w, ffn_norm_w, meta):
        ops = mxfp8_operands(wo_w)
        a_q, a_s = quantize_rows(attn_TD)
        out_TD = scaled_mm(
            a_q, ops.weight_qdata_fprop_KN, a_s, ops.weight_scale_fprop_swizzled
        )
        h_TD, y_TD, y_q, y_s, rstd = kernel("dre3", x_TD, out_TD, ffn_norm_w)(
            x_TD, out_TD, ffn_norm_w
        )
        ctx.save_for_backward(attn_TD, h_TD, rstd, ffn_norm_w, wo_w)
        ctx.meta = meta
        ctx.weight_shape = wo_w.shape
        ctx.mark_non_differentiable(y_q, y_s)
        return h_TD, y_TD, y_q, y_s

    @staticmethod
    @once_differentiable
    def backward(ctx, g_h_res, g_y, _g_y_q, _g_y_s):
        attn_TD, h_TD, rstd, ffn_norm_w, wo_w = ctx.saved_tensors
        wgrad_dtype, weight_param = ctx.meta
        g_h, q_row, q_col_t, s_row, s_col, g_norm_w = kernel(
            "a6rv", rstd, g_y, h_TD, ffn_norm_w, g_h_res
        )(rstd, g_y.contiguous(), h_TD, ffn_norm_w, g_h_res.contiguous())
        ops = mxfp8_operands(wo_w)
        g_attn = scaled_mm(
            q_row, ops.weight_qdata_dgrad_NK, s_row, ops.weight_scale_dgrad_swizzled
        )
        g_w = wgrad(
            q_col_t, s_col, attn_TD, weight_param, wgrad_dtype, ctx.weight_shape
        )
        return g_attn, g_h, g_w, g_norm_w, None


class Router(torch.autograd.Function):
    """Router gate (HiMidLoLinear) -> sigmoid -> group-limited top-k, normalized and scaled scores,
    the microbatch-wise aux loss and its gradient injection, and Dist-MoE's padding masks.

    Forward: the unchanged gate GEMM, G-001 (gjn6) for the routing outputs, the usage-counter and
    metric side effects with the stock router's in-place adds, and G-020 (h4xd), the graph's SAC
    recompute region, for the tensors backward reads (eager saves them instead of recomputing).
    Backward: G-006 (uu33: aux-loss, normalization, sigmoid backward and the split of the fp32 gate
    gradient into bf16 pieces), then the gate's ``_narrow_backward`` GEMMs on those pieces.
    """

    @staticmethod
    def forward(
        ctx, x_TD, gate_weight, padding_mask_T, denominator, router, expert_bias_E
    ):
        gate_w = gate_weight.flatten(0, -2)
        gate_TE = torch.mm(x_TD, gate_w.T, out_dtype=torch.float32)
        ids_TK, routing_map_TE, counts_E, metric, masked_ids_TK, scores_TK = kernel(
            "gjn6", gate_TE, expert_bias_E, padding_mask_T, denominator
        )(gate_TE, expert_bias_E, padding_mask_T, denominator)
        router.tokens_per_expert_E.add_(counts_E)
        router.aux_loss.instance_acc.add_(metric)
        pad_TK = padding_mask_T.unsqueeze(-1)
        (
            probs_TE,
            topk_raw_TK,
            denom_T1,
            _,
            l1_T1,
            l1_TE,
            notpad_T1,
            f_E,
            coeff_scale,
            injected,
            probs_alias_TE,
        ) = kernel(
            "h4xd", gate_TE, ids_TK, pad_TK, routing_map_TE, padding_mask_T, denominator
        )(
            gate_TE, ids_TK, pad_TK, routing_map_TE, padding_mask_T, denominator
        )
        ctx.save_for_backward(
            x_TD,
            gate_w,
            pad_TK,
            injected,
            coeff_scale,
            f_E,
            notpad_T1,
            probs_TE,
            l1_TE,
            l1_T1,
            topk_raw_TK,
            denom_T1,
            ids_TK,
            probs_alias_TE,
        )
        ctx.weight_shape = gate_weight.shape
        ctx.num_pieces = router.gate.num_pieces
        ctx.mark_non_differentiable(masked_ids_TK)
        return scores_TK, masked_ids_TK

    @staticmethod
    @once_differentiable
    def backward(ctx, g_scores_TK, _g_ids):
        x_TD, gate_w, *saved = ctx.saved_tensors
        stacked, p0, p1, p2 = kernel("uu33", g_scores_TK, *saved)(
            g_scores_TK.contiguous(), *saved
        )
        g_x = torch.mm(stacked, torch.cat([gate_w] * ctx.num_pieces))
        g_w = torch.mm(p0, x_TD, out_dtype=torch.float32)
        g_w += torch.mm(p1, x_TD, out_dtype=torch.float32)
        g_w += torch.mm(p2, x_TD, out_dtype=torch.float32)
        if g_w.shape != ctx.weight_shape:
            g_w = g_w.view(ctx.weight_shape)
        return g_x, g_w, None, None, None, None


def router_forward(router, x_TD, expert_bias_E, padding_mask_T, aux_loss_denominator):
    """``DeepSeekV3Router.forward`` through ``Router``, with Dist-MoE's padding masks applied.

    Returns (scores, expert ids) with padded rows already routed to expert -1 at score 0, so the
    routed experts are called without a padding mask.
    """
    from torchtitan.models.common.moe.moe import MicrobatchWiseLoadBalanceLoss
    from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

    if not (
        isinstance(router, DeepSeekV3Router)
        and router.training
        and isinstance(router.aux_loss, MicrobatchWiseLoadBalanceLoss)
        and router.route_norm
        and padding_mask_T is not None
        and router.gate.num_pieces == 3
        and router.gate.bias is None
        and router.gate.weight.requires_grad
    ):
        raise ValueError("graph-kernel router: unsupported router configuration")
    return Router.apply(
        x_TD,
        router.gate.weight,
        padding_mask_T,
        aux_loss_denominator,
        router,
        expert_bias_E,
    )


class SwiGLUFeedForward(torch.autograd.Function):
    """Shared-expert or dense FFN: w13 (MXFP8, BF16 input saved) -> SwiGLU -> w2 (MXFP8,
    columnwise input saved).

    Forward: w13's FPROP on ``y``'s rowwise operands from ``AttentionOutput`` (G-082's
    quantization), ``torchtitan::silu_and_mul``, the stock w2 FPROP.
    Backward: w2 takes its gradient's quantized operands from the next layer's G-049 kernel when
    that gradient is the one the kernel quantized (stock quantization otherwise), the w2 DGRAD and
    WGRAD, SwiGLU backward, G-015 for w13 (bqhy, where a kernel exists for the shape) and its WGRAD.
    With ``pre_permute_w13`` the w13 WGRAD is G-007 (5nj2: the GEMM plus the row-block permutation
    FSDP's reduce-scatter copy-in applies to a parameter sharded on dim 1), returned in the
    parameter's shape for ``reduce_scatter_input_fn`` to use as already permuted.
    """

    @staticmethod
    def forward(ctx, y_TD, y_q, y_s, w13_w, w2_w, w13_meta, w2_meta, pre_permute_w13):
        w13_ops = mxfp8_operands(w13_w)
        gate_up_TN = scaled_mm(
            y_q, w13_ops.weight_qdata_fprop_KN, y_s, w13_ops.weight_scale_fprop_swizzled
        )
        gate_up_T2F = gate_up_TN.view(gate_up_TN.shape[0], 2, -1)
        act_TF = silu_and_mul_op(gate_up_T2F)
        w2_ops = mxfp8_operands(w2_w)
        a_row, a_srow, a_col, a_scol = _quantize_rows_cols(act_TF)
        out_TD = scaled_mm(
            a_row,
            w2_ops.weight_qdata_fprop_KN,
            a_srow,
            w2_ops.weight_scale_fprop_swizzled,
        )
        ctx.save_for_backward(y_TD, gate_up_T2F, a_col, a_scol, w13_w, w2_w)
        ctx.metas = (w13_meta, w2_meta)
        ctx.weight_shapes = (w13_w.shape, w2_w.shape)
        ctx.pre_permute_w13 = pre_permute_w13
        return out_TD

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out_TD):
        y_TD, gate_up_T2F, a_col, a_scol, w13_w, w2_w = ctx.saved_tensors
        (w13_dtype, w13_param), (w2_dtype, w2_param) = ctx.metas
        operands = _take_grad_quant(g_out_TD)
        if operands is None:
            operands = _grad_operands(g_out_TD.contiguous())
        q_row, s_row, q_col_t, s_col = operands
        w2_ops = mxfp8_operands(w2_w)
        g_act = scaled_mm(
            q_row,
            w2_ops.weight_qdata_dgrad_NK,
            s_row,
            w2_ops.weight_scale_dgrad_swizzled,
        )
        g_w2 = wgrad_from_col(
            q_col_t, s_col, a_col, a_scol, w2_param, w2_dtype, ctx.weight_shapes[1]
        )
        g_gate_up = silu_and_mul_backward_op(g_act, gate_up_T2F)
        g_gate_up_TN = g_gate_up.view(g_gate_up.shape[0], -1)
        w13_ops = mxfp8_operands(w13_w)
        g_y, s_col13, q_col_t13 = _dgrad_and_grad_col(
            g_gate_up_TN,
            w13_ops.weight_qdata_dgrad_NK,
            w13_ops.weight_scale_dgrad_swizzled,
        )
        g_w13 = None
        if ctx.pre_permute_w13 and running_grad(w13_param) is None:
            _, y_col, _, y_scol = mxfp8_quantize_cuda(
                y_TD, rowwise=False, colwise=True, scaling_mode=_SCALING_MODE
            )
            y_scol = triton_mx_block_rearrange(y_scol)
            run = kernel("5nj2", q_col_t13, y_col, s_col13, y_scol)
            if run is not None and w13_dtype == torch.bfloat16:
                permuted = run(q_col_t13, y_col, s_col13, y_scol)
                _PREPERMUTED[permuted.data_ptr()] = permuted
                g_w13 = permuted.view(ctx.weight_shapes[0])
        if g_w13 is None:
            g_w13 = wgrad(
                q_col_t13, s_col13, y_TD, w13_param, w13_dtype, ctx.weight_shapes[0]
            )
        return g_y, None, None, g_w13, g_w2, None, None, None


# W13 gradients whose rows G-007 already permuted: data_ptr -> the [W * 2, F / W, D] tensor.
_PREPERMUTED: dict[int, torch.Tensor] = {}


def _pack_module(prefix):
    """The module of a pack kernel (single specialization), for its internal ``_fn``."""
    kernel(prefix)  # indexes the pattern's specializations
    ((_, path),) = _SPECS[prefix]
    if path not in _MODULES:
        spec = importlib.util.spec_from_file_location(
            f"graph_kernel_{os.path.basename(path)[:8]}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[path] = module
    return _MODULES[path]


def _pack_args_ok(module, tensors):
    """The checks the pack kernel's ``run`` applies to its sources."""
    return all(
        t.dtype == torch.bfloat16
        and tuple(t.shape) == shape
        and t.is_contiguous()
        and t.data_ptr() % align == 0
        for t, (_, shape), align in zip(tensors, module.SHAPES, module.ALIGN)
    )


def reduce_scatter_input_fn(unsharded_grads, *, shard_dims, world_size):
    """FSDP2 reduce-scatter input for a DSv3 block, via ``FSDPModule.set_reduce_scatter_input_fn``.

    The stock function reorders dim-1-sharded gradients (the stacked w13) by rank with a chunk
    and a 256-way cat, then copies every gradient into the flat buffer with
    ``torch._chunk_cat(grads, 0, world_size)``. Here a w13 gradient that G-007 already permuted
    skips the reorder, and the copy-in of the attention / shared-expert group is G-003's pack
    (ponk; its GEMM is the wo WGRAD eager has already computed, so only the pack runs). The
    routed-expert group's G-028 pack (czaf) writes w2 before w13, the reverse of FSDP2's order, so
    that group keeps the stock copy-in.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        foreach_reduce_scatter_copy_in,
    )

    for i, shard_dim in enumerate(shard_dims):
        if shard_dim == 0:
            continue
        g = unsharded_grads[i]
        permuted = _PREPERMUTED.pop(g.data_ptr(), None)
        if permuted is not None:
            CALLS["G-007 reorder skipped"] += 1
            unsharded_grads[i] = permuted
            continue
        if _PREPERMUTED:
            raise RuntimeError(
                "a G-007-permuted w13 gradient did not reach FSDP's copy-in"
            )
        chunks = torch.chunk(g, world_size, dim=shard_dim)
        unsharded_grads[i] = torch.cat(chunks, dim=0)
    module = _pack_module("ponk")
    if len(unsharded_grads) == len(module.SHAPES) and _pack_args_ok(
        module, unsharded_grads
    ):

        def copy_in(out):
            if out.dtype != torch.bfloat16 or out.numel() != module.NOUT:
                foreach_reduce_scatter_copy_in(unsharded_grads, out, world_size)
                return
            CALLS["G-003 pack"] += 1
            module._fn(*unsharded_grads, out)

        return copy_in
    CALLS[
        "stock copy-in " + " ".join(str(tuple(g.shape)) for g in unsharded_grads)
    ] += 1
    return lambda out: foreach_reduce_scatter_copy_in(unsharded_grads, out, world_size)


def install_expert_weight_prep():
    """G-021 (W13, x6t6) / G-061 (W2, 5uzo): ``dist_moe.prepare_mxfp8_weight`` of the
    all-gathered BF16 expert weight in one kernel, writing into FSDP's existing operand tensors.

    torchtitan exposes no config node for this FSDP extension, so this patches its two hooks.
    Shapes without a promoted kernel keep the stock preparation.
    """
    from torchtitan.models.common.dist_moe.mxfp8 import (
        _DistMoeMXFP8Operands,
        _DistMoeW13ShardedTensor,
        _DistMoeW2ShardedTensor,
    )

    for cls, prefix, flatten in (
        (_DistMoeW13ShardedTensor, "x6t6", True),
        (_DistMoeW2ShardedTensor, "5uzo", False),
    ):
        stock_build = cls._build_operands

        def _build_operands(
            self,
            logical_tensor,
            out=None,
            _prefix=prefix,
            _flatten=flatten,
            _stock=stock_build,
        ):
            weight = logical_tensor.flatten(1, 2) if _flatten else logical_tensor
            flat = weight.reshape(-1) if weight.is_contiguous() else None
            run = None
            if flat is not None and not self.mega_fused_inference:
                run = kernel(_prefix, flat)
            if run is None:
                CALLS[f"{_prefix} stock"] += 1
                return _stock(self, logical_tensor, out)
            if out is None:
                qdata, fprop_scale, dgrad_scale = run(flat)
                return _DistMoeMXFP8Operands(
                    qdata=qdata, fprop_scale=fprop_scale, dgrad_scale=dgrad_scale
                )
            module = _MODULES[next(p for p, r in _RUNS.items() if r is run)]
            fn = module._fn if flat.data_ptr() % module._ALIGN == 0 else module._fn16
            fn(
                flat,
                out.qdata.view(-1).view(torch.uint8),
                out.fprop_scale.view(-1).view(torch.uint8),
                out.dgrad_scale.view(-1).view(torch.uint8),
            )
            return out

        cls._build_operands = _build_operands
