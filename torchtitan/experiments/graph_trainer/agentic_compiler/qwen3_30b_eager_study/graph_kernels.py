# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Qwen3 30B-A3B loop's verified kernels, called from eager autograd.Functions.

Each kernel is the standalone ``run(*args)`` file the loop promoted into its kernel cache; its
contract (argument and output order, shapes, strides, dtypes) is the lifted graph region's, which
``reference.py`` next to it spells out. A Function owns every forward op whose backward a backward
region contains, saves exactly the tensors that region reads, and returns a tensor once per
consumer where the graph region contains an autograd-engine accumulation add, so that its backward
receives the gradients separately.

Shape suffixes: T tokens, D model dim, E experts, K top-k, N/H heads, V vocab.
"""

import glob
import importlib.util
import json
import os

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.models.common.linear.inplace_wgrad_accum import (
    running_grad,
    uses_inplace_wgrad_accum,
)
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.tensor import (
    _LinearShardedTensorWithMXFP8Compute,
    _quantize_mxfp8_weight,
)

KERNEL_CACHE = os.environ.get(
    "EAGER_KERNEL_CACHE",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../qwen3_30b_a3b_distmoe_mxfp8/kernel_cache",
    ),
)
_SCALING_MODE = "rceil"
_KERNELS = {}
_MODULES = {}


def kernel(prefix: str):
    """``run`` of the promoted kernel of pattern ``pat_<prefix>*``."""
    if prefix not in _KERNELS:
        (pat,) = glob.glob(os.path.join(KERNEL_CACHE, f"pat_{prefix}*"))
        (spec_path,) = glob.glob(os.path.join(pat, "specialization_*.json"))
        name = json.load(open(spec_path))["kernel_name"]
        spec = importlib.util.spec_from_file_location(
            f"graph_kernel_{prefix}", os.path.join(pat, name)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _KERNELS[prefix] = module.run
        _MODULES[prefix] = module
    return _KERNELS[prefix]


def mxfp8_operands(weight: torch.Tensor):
    """The MXFP8 linear weight operands, as ``MXFP8Linear._linear`` obtains them."""
    if isinstance(weight, _UnshardedFSDPTensor):
        return weight.operands
    hp = (
        weight._tensor
        if isinstance(weight, _LinearShardedTensorWithMXFP8Compute)
        else weight
    )
    with torch.no_grad():
        return _quantize_mxfp8_weight(hp.flatten(0, -2))


def _scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16):
    return F.scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=scale_b,
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=out_dtype,
    )


def _wgrad(grad_col_t, grad_scale_col, x_MK, weight_param, wgrad_dtype, weight_shape):
    """WGRAD of an MXFP8 linear whose BF16 input was saved, as ``_MXFP8LinearFunction``."""
    _, x_col, _, x_scale_col = mxfp8_quantize_cuda(
        x_MK, rowwise=False, colwise=True, scaling_mode=_SCALING_MODE
    )
    x_scale_col = triton_mx_block_rearrange(x_scale_col)
    grad = running_grad(weight_param)
    if weight_param is None or grad is None:
        return _scaled_mm(
            grad_col_t, x_col, grad_scale_col, x_scale_col, wgrad_dtype
        ).view(weight_shape)
    F.scaled_addmm_(
        grad.view(-1, grad.shape[-1]),
        grad_col_t,
        x_col,
        scale_a=grad_scale_col,
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=x_scale_col,
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
    )
    weight_param.grad = None
    return grad


def linear_meta(linear: torch.nn.Module, weight: torch.Tensor):
    """(wgrad dtype, parameter for in-place WGRAD accumulation or None), as MXFP8Linear."""
    inplace = uses_inplace_wgrad_accum(linear.inplace_wgrad_accum, weight, linear)
    return (weight.grad_dtype or weight.dtype, weight if inplace else None)


class AttentionInput(torch.autograd.Function):
    """attention_norm -> wqkv (MXFP8) -> split -> q/k RMSNorm -> RoPE.

    Forward: G-41 (norm + quant), the unchanged FPROP GEMM, G-02 (split, q/k norm, RoPE).
    Backward: G-01 (RoPE and q/k norm bwd, cat, quant), WGRAD, G-15 (DGRAD GEMM + norm bwd +
    residual add). Returns ``x`` as a fourth output for the residual add, whose gradient G-15 adds.
    """

    @staticmethod
    def forward(
        ctx,
        x_TD,
        attn_norm_w,
        weight,
        w_fprop,
        w_sfprop,
        w_dgrad,
        w_sdgrad,
        q_norm_w,
        k_norm_w,
        cache,
        positions_T,
        meta,
    ):
        normed_TD, rstd, x_q, x_s = kernel("kbxm")(x_TD, attn_norm_w)
        qkv = _scaled_mm(x_q, w_fprop, x_s, w_sfprop)
        xq, xk, xv, rq, rk, cos, sin, q, k = kernel("4dzo")(
            qkv, q_norm_w, k_norm_w, cache, positions_T
        )
        ctx.save_for_backward(
            x_TD,
            rstd,
            normed_TD,
            attn_norm_w,
            w_dgrad,
            w_sdgrad,
            xq,
            xk,
            rq,
            rk,
            cos,
            sin,
            q_norm_w,
            k_norm_w,
        )
        ctx.meta = meta
        ctx.weight_shape = weight.shape
        return q, k, xv, x_TD.view_as(x_TD)

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk, gv, gx_res):
        (
            x_TD,
            rstd,
            normed_TD,
            attn_norm_w,
            w_dgrad,
            w_sdgrad,
            xq,
            xk,
            rq,
            rk,
            cos,
            sin,
            q_norm_w,
            k_norm_w,
        ) = ctx.saved_tensors
        wgrad_dtype, weight_param = ctx.meta
        g_kw, g_qw, g_row, g_srow, g_scol, g_col_t = kernel("qnfe")(
            gk.contiguous(),
            gq.contiguous(),
            sin,
            cos,
            rk,
            xk,
            k_norm_w,
            rq,
            xq,
            q_norm_w,
            gv.contiguous(),
        )
        g_w = _wgrad(
            g_col_t, g_scol, normed_TD, weight_param, wgrad_dtype, ctx.weight_shape
        )
        g_attn_norm_w, g_x = kernel("tl5t")(
            g_row,
            w_dgrad,
            g_srow,
            w_sdgrad,
            rstd,
            x_TD,
            attn_norm_w,
            gx_res.contiguous(),
        )
        return (
            g_x,
            g_attn_norm_w,
            g_w,
            None,
            None,
            None,
            None,
            g_qw,
            g_kw,
            None,
            None,
            None,
        )


class AttentionOutput(torch.autograd.Function):
    """wo (MXFP8) -> residual add -> ffn_norm.

    Forward: G-28 (quant + FPROP GEMM), G-45 (add + norm). Backward: G-19 (sum of the two
    consumers' gradients, norm bwd, residual add, quant), then wo's DGRAD and WGRAD GEMMs.
    Returns ``h`` for the trailing residual add and the normalized ``h`` once each for the
    router and the routed experts.
    """

    @staticmethod
    def forward(
        ctx,
        attn_TD,
        x_TD,
        weight,
        w_fprop,
        w_sfprop,
        w_dgrad,
        w_sdgrad,
        ffn_norm_w,
        meta,
    ):
        wo_TD = kernel("vf5c")(attn_TD, w_fprop, w_sfprop)
        h_TD, normed_TD, rstd = kernel("na2k")(x_TD, wo_TD, ffn_norm_w)
        ctx.save_for_backward(attn_TD, h_TD, rstd, ffn_norm_w, w_dgrad, w_sdgrad)
        ctx.meta = meta
        ctx.weight_shape = weight.shape
        return h_TD, normed_TD, normed_TD.view_as(normed_TD)

    @staticmethod
    @once_differentiable
    def backward(ctx, g_h_res, g_router, g_experts):
        attn_TD, h_TD, rstd, ffn_norm_w, w_dgrad, w_sdgrad = ctx.saved_tensors
        wgrad_dtype, weight_param = ctx.meta
        g_ffn_w, g_h, g_row, g_srow, g_scol, g_col_t = kernel("zfby")(
            g_router.contiguous(),
            g_experts.contiguous(),
            rstd,
            h_TD,
            ffn_norm_w,
            g_h_res.contiguous(),
        )
        g_attn = _scaled_mm(g_row, w_dgrad, g_srow, w_sdgrad)
        g_w = _wgrad(
            g_col_t, g_scol, attn_TD, weight_param, wgrad_dtype, ctx.weight_shape
        )
        return (g_attn, g_h, g_w, None, None, None, None, g_ffn_w, None)


class Router(torch.autograd.Function):
    """Router gate (HiMidLoLinear) -> softmax -> top-k gather/normalize, with the batch-wise
    aux loss and its gradient injection.

    Forward: the unchanged gate GEMM, G-47 (softmax, gather, normalize), G-17 (expert counts),
    G-91 (aux-loss score sums); the router's side effects (usage counter, cumulative-count
    all-reduce, metric) run here with the same torchtitan calls as the stock router.
    Backward: G-06 (aux-loss and router bwd, softmax bwd, split of the fp32 gate gradient into
    bf16 pieces), then the gate's ``_narrow_backward`` GEMMs on those pieces.
    """

    @staticmethod
    def forward(
        ctx, x_TD, gate_weight, padding_mask_T, aux_denominator, router, expert_bias_E
    ):
        import spmd_types as spmd
        import torch_remat as remat
        from torchtitan.distributed.spmd_types import spmd_local_context, spmd_mesh_size

        aux = router.aux_loss
        gate_w = gate_weight.flatten(0, -2)
        gate_TE = torch.mm(x_TD, gate_w.T, out_dtype=torch.float32)
        ids_TK = router._select_experts(
            gate_TE, expert_bias_E, padding_mask_T=padding_mask_T
        )
        scores, scores_a, topk_raw, denom_T1, topk_TK = kernel("5ifh")(gate_TE, ids_TK)
        counts_i64, counts_f32 = kernel("bp6b")(scores, ids_TK, padding_mask_T)
        if not remat.is_recomputing():
            router.tokens_per_expert_E.add_(counts_i64)
        cumulative_E = aux._accumulate_expert_counts(counts_f32)
        f_E = F.normalize(cumulative_E, p=1, dim=0) * scores.size(-1)
        if spmd_mesh_size("dp") > 1:
            f_E = spmd.convert(f_E, "dp", src=spmd.I, dst=spmd.R)
        notpad_T1 = ~padding_mask_T.unsqueeze(-1)
        with spmd_local_context("dp"):
            l1_T1, denom_TE, prob_sums_E = kernel("bj3b")(scores, notpad_T1)
            prob_sums_E = aux._reduce_token_partials(prob_sums_E)
            raw = (f_E * prob_sums_E).sum()
        with spmd.no_typecheck():
            scale = 1.0 / aux_denominator.clamp_min(1)
            coeff_scale = aux.coeff * scale
            injected = raw * coeff_scale
            aux.instance_acc.add_(raw * scale)
        ctx.save_for_backward(
            x_TD,
            gate_w,
            padding_mask_T.unsqueeze(-1),
            injected,
            coeff_scale,
            f_E,
            notpad_T1,
            scores,
            denom_TE,
            l1_T1,
            topk_raw,
            denom_T1,
            ids_TK,
            scores_a,
        )
        ctx.weight_shape = gate_weight.shape
        ctx.num_pieces = router.gate.num_pieces
        ctx.mark_non_differentiable(ids_TK)
        return topk_TK, ids_TK

    @staticmethod
    @once_differentiable
    def backward(ctx, g_topk_TK, _g_ids):
        (
            x_TD,
            gate_w,
            pad_TK,
            injected,
            coeff_scale,
            f_E,
            notpad_T1,
            scores,
            denom_TE,
            l1_T1,
            topk_raw,
            denom_T1,
            ids_TK,
            scores_a,
        ) = ctx.saved_tensors
        stacked, p0, p1, p2 = kernel("s3of")(
            g_topk_TK.contiguous(),
            pad_TK,
            injected,
            coeff_scale,
            f_E,
            notpad_T1,
            scores,
            denom_TE,
            l1_T1,
            topk_raw,
            denom_T1,
            ids_TK,
            scores_a,
        )
        g_x = torch.mm(stacked, torch.cat([gate_w] * ctx.num_pieces))
        g_w = torch.mm(p0, x_TD, out_dtype=torch.float32)
        g_w += torch.mm(p1, x_TD, out_dtype=torch.float32)
        g_w += torch.mm(p2, x_TD, out_dtype=torch.float32)
        if g_w.shape != ctx.weight_shape:
            g_w = g_w.view(ctx.weight_shape)
        return g_x, g_w, None, None, None, None


def router_forward(router, x_TD, expert_bias_E, padding_mask_T, aux_loss_denominator):
    """``TokenChoiceTopKRouter.forward`` through ``Router``, for the layout the loop measured."""
    from torchtitan.models.common.moe.moe import (
        BatchWiseLoadBalanceLoss,
        RoundRobinTokenChoiceTopKRouter,
    )

    if not (
        isinstance(router, RoundRobinTokenChoiceTopKRouter)
        and router.training
        and isinstance(router.aux_loss, BatchWiseLoadBalanceLoss)
        and router.route_norm
        and router.route_scale == 1.0
        and padding_mask_T is not None
        and router.gate.num_pieces == 3
        and router.gate.bias is None
    ):
        raise ValueError("graph-kernel router: unsupported router configuration")
    topk_TK, ids_TK = Router.apply(
        x_TD,
        router.gate.weight,
        padding_mask_T,
        aux_loss_denominator,
        router,
        expert_bias_E,
    )
    # Dist-MoE dispatches from the expert ids and ignores the routing map.
    return topk_TK, ids_TK, None


class CrossEntropyChunk(torch.autograd.Function):
    """``cross_entropy_loss(pred, labels) / global_loss_token_counts`` for one loss chunk.

    Forward: G-90 (cast, log_softmax, nll, div). Backward: G-07.
    """

    @staticmethod
    def forward(ctx, pred_TV, labels_T, denominator):
        logsm_TV, total_weight, loss = kernel("bme7")(pred_TV, labels_T, denominator)
        ctx.save_for_backward(logsm_TV, labels_T, total_weight, denominator)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, g_loss):
        logsm_TV, labels_T, total_weight, denominator = ctx.saved_tensors
        g_pred = kernel("53aq")(
            g_loss, denominator, logsm_TV, labels_T, total_weight, logsm_TV
        )
        return g_pred, None, None


def _prepare_into(prefix, weight_fp32, qdata, fprop_scale, dgrad_scale):
    """G-33 / G-42 writing into existing operand tensors (their ``run`` allocates new ones)."""
    kernel(prefix)
    module = _MODULES[prefix]
    p = weight_fp32.data_ptr()
    fn = module._FNS[32 if p % 32 == 0 else 16 if p % 16 == 0 else 4]
    fn(
        weight_fp32.view(-1),
        qdata.view(-1).view(torch.uint8),
        fprop_scale.view(-1).view(torch.uint8),
        dgrad_scale.view(-1).view(torch.uint8),
    )


def install_expert_weight_prep():
    """G-33 (W13) / G-42 (W2): fp32 expert weight -> MXFP8 operands in one kernel.

    Applies to expert weights FSDP does not actually shard (a size-1 expert FSDP mesh), where the
    fp32 shard is the whole weight: the all-gather hook skips its fp32 -> bf16 cast (nothing reads
    the gathered tensor, which FSDP releases once the operands exist) and the operands come from
    the fp32 shard. torchtitan exposes no config node for this FSDP extension, so this patches its
    two hooks.
    """
    from torchtitan.models.common.dist_moe.mxfp8 import (
        _DistMoeMXFP8Operands,
        _DistMoeMXFP8ShardedTensorBase,
        _DistMoeW13ShardedTensor,
        _DistMoeW2ShardedTensor,
    )

    def unsharded_fp32(self, logical_size):
        return (
            self._tensor.dtype == torch.float32
            and not self.mega_fused_inference
            and tuple(self._tensor.shape) == tuple(logical_size)
        )

    stock_pre = _DistMoeMXFP8ShardedTensorBase.fsdp_pre_all_gather

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        if mesh.size() == 1 and unsharded_fp32(self, outer_size):
            dtype = mp_policy.param_dtype or self._tensor.dtype
            return (
                torch.empty(
                    self._tensor.shape, dtype=dtype, device=self._tensor.device
                ),
            ), outer_size
        return stock_pre(self, mesh, outer_size, outer_stride, module, mp_policy)

    _DistMoeMXFP8ShardedTensorBase.fsdp_pre_all_gather = fsdp_pre_all_gather
    for cls, prefix in (
        (_DistMoeW13ShardedTensor, "gri3"),
        (_DistMoeW2ShardedTensor, "adoe"),
    ):
        stock_build = cls._build_operands

        def _build_operands(
            self, logical_tensor, out=None, _prefix=prefix, _stock=stock_build
        ):
            if not unsharded_fp32(self, logical_tensor.shape):
                return _stock(self, logical_tensor, out)
            if out is None:
                qdata, fprop_scale, dgrad_scale = kernel(_prefix)(self._tensor)
                return _DistMoeMXFP8Operands(
                    qdata=qdata, fprop_scale=fprop_scale, dgrad_scale=dgrad_scale
                )
            _prepare_into(
                _prefix, self._tensor, out.qdata, out.fprop_scale, out.dgrad_scale
            )
            return out

        cls._build_operands = _build_operands
