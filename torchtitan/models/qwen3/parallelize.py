# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 parallelization scaffold.

This file is intentionally strategy-free. Autoresearch is expected to replace
``parallelize_qwen3`` with an implementation specialized to the exact train
command, model flavor, and cluster/system it is optimizing for.
"""

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.components.quantization.mx import (
    mxfp8_shared_input_gate_up,
    mxfp8_shared_input_qkv,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.tools.logging import logger


@triton.jit
def _sequential_rope_forward_kernel(
    x_ptr,
    rope_ptr,
    out_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    d = offsets % head_dim
    tmp = offsets // head_dim
    s = (tmp // num_heads) % seq_len
    half = head_dim // 2
    first_half = d < half
    other_offsets = tl.where(first_half, offsets + half, offsets - half)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    other = tl.load(x_ptr + other_offsets, mask=mask).to(tl.float32)
    cos = tl.load(rope_ptr + s * (head_dim * 2) + d, mask=mask).to(tl.float32)
    sin = tl.load(rope_ptr + s * (head_dim * 2) + head_dim + d, mask=mask).to(
        tl.float32
    )
    rotated = tl.where(first_half, -other, other)
    tl.store(out_ptr + offsets, x * cos + rotated * sin, mask=mask)


@triton.jit
def _sequential_rope_backward_kernel(
    grad_ptr,
    rope_ptr,
    out_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    d = offsets % head_dim
    tmp = offsets // head_dim
    s = (tmp // num_heads) % seq_len
    half = head_dim // 2
    first_half = d < half
    other_offsets = tl.where(first_half, offsets + half, offsets - half)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    other_grad = tl.load(grad_ptr + other_offsets, mask=mask).to(tl.float32)
    cos = tl.load(rope_ptr + s * (head_dim * 2) + d, mask=mask).to(tl.float32)
    sin = tl.load(rope_ptr + s * (head_dim * 2) + head_dim + d, mask=mask).to(
        tl.float32
    )
    rotated_grad = tl.where(first_half, other_grad, -other_grad)
    tl.store(out_ptr + offsets, grad * cos + rotated_grad * sin, mask=mask)


class _SequentialRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        rope_cache = rope_cache.contiguous()
        out = torch.empty_like(x)
        _, seq_len, num_heads, head_dim = x.shape
        block = 256
        grid = (triton.cdiv(x.numel(), block),)
        _sequential_rope_forward_kernel[grid](
            x,
            rope_cache,
            out,
            x.numel(),
            seq_len,
            num_heads,
            head_dim,
            BLOCK=block,
        )
        ctx.save_for_backward(rope_cache)
        ctx.shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rope_cache,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out)
        _, seq_len, num_heads, head_dim = ctx.shape
        block = 256
        grid = (triton.cdiv(grad_out.numel(), block),)
        _sequential_rope_backward_kernel[grid](
            grad_out,
            rope_cache,
            grad_x,
            grad_out.numel(),
            seq_len,
            num_heads,
            head_dim,
            BLOCK=block,
        )
        return grad_x, None


def _sequential_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    return _SequentialRoPE.apply(x, rope_cache)


def _qk_norm_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, q_eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, k_eps)
    return _sequential_rope(xq, rope_cache), _sequential_rope(xk, rope_cache)


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _enable_shared_mxfp8_gate_up_input_cast(model: Qwen3Model) -> None:
    patched_count = 0

    def _shared_mxfp8_feed_forward(x: torch.Tensor, *, feed_forward):
        w1_out, w3_out = mxfp8_shared_input_gate_up(
            x,
            feed_forward.w1.weight,
            feed_forward.w3.weight,
        )
        return feed_forward.w2(F.silu(w1_out) * w3_out)

    for layer in model.layers.values():
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            continue
        if not (
            hasattr(feed_forward.w1.weight, "config")
            and hasattr(feed_forward.w3.weight, "config")
        ):
            continue
        feed_forward.forward = lambda x, feed_forward=feed_forward: (
            _shared_mxfp8_feed_forward(x, feed_forward=feed_forward)
        )
        patched_count += 1

    logger.info(
        "Enabled shared MXFP8 input casts for %s Qwen3 feed-forward modules",
        patched_count,
    )


def _enable_shared_mxfp8_qkv_input_cast(model: Qwen3Model) -> None:
    patched_count = 0

    def _shared_mxfp8_qkv(x: torch.Tensor, *, qkv_linear):
        bs, seqlen, _ = x.shape
        xq, xk, xv = mxfp8_shared_input_qkv(
            x,
            qkv_linear.wq.weight,
            qkv_linear.wk.weight,
            qkv_linear.wv.weight,
        )
        xq = xq.view(bs, seqlen, -1, qkv_linear.head_dim)
        xk = xk.view(bs, seqlen, -1, qkv_linear.head_dim)
        xv = xv.view(bs, seqlen, -1, qkv_linear.head_dim)
        return xq, xk, xv

    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        qkv_linear = getattr(attention, "qkv_linear", None)
        if qkv_linear is None:
            continue
        if not all(
            hasattr(proj.weight, "config")
            for proj in (qkv_linear.wq, qkv_linear.wk, qkv_linear.wv)
        ):
            continue
        qkv_linear.forward = lambda x, qkv_linear=qkv_linear: _shared_mxfp8_qkv(
            x,
            qkv_linear=qkv_linear,
        )
        patched_count += 1

    logger.info(
        "Enabled shared MXFP8 input casts for %s Qwen3 attention Q/K/V modules",
        patched_count,
    )


def _compile_qwen3_feed_forward(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    compiled_count = 0
    for layer in model.layers.values():
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            continue
        feed_forward.compile(backend=compile_config.backend, fullgraph=True)
        compiled_count += 1

    logger.info(
        "Compiling %s Qwen3 feed-forward modules with torch.compile",
        compiled_count,
    )


def _compile_qwen3_qkv_linear(model: Qwen3Model, compile_config: CompileConfig) -> None:
    compiled_count = 0
    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        qkv_linear = getattr(attention, "qkv_linear", None)
        if qkv_linear is None:
            continue
        qkv_linear.compile(backend=compile_config.backend, fullgraph=True)
        compiled_count += 1

    logger.info(
        "Compiling %s Qwen3 attention Q/K/V projection modules",
        compiled_count,
    )


def _enable_compiled_block_norms(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    compiled_rms_norm = torch.compile(
        _rms_norm,
        backend=compile_config.backend,
        fullgraph=True,
    )
    patched_count = 0

    def _forward_with_compiled_block_norms(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input = compiled_rms_norm(
            x,
            self.attention_norm.weight,
            self.attention_norm.eps,
        )
        attention_output = self.attention(
            attention_input, freqs_cis, attention_masks, positions
        )
        x = x + attention_output
        ffn_input = compiled_rms_norm(
            x,
            self.ffn_norm.weight,
            self.ffn_norm.eps,
        )

        if self.moe_enabled:
            x = x + self.moe(ffn_input)
        else:
            x = x + self.feed_forward(ffn_input)
        return x

    for layer in model.layers.values():
        layer.forward = types.MethodType(_forward_with_compiled_block_norms, layer)
        patched_count += 1

    logger.info(
        "Compiling Qwen3 block RMSNorm helpers for %s layers",
        patched_count,
    )


def _enable_triton_sequential_rope(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    patched_count = 0
    compiled_qk_norm_rope = None
    if compile_config.enable and "qk_norm_rope" in compile_config.components:
        compiled_qk_norm_rope = torch.compile(
            _qk_norm_rope,
            backend=compile_config.backend,
            fullgraph=True,
        )

    def _forward_with_triton_rope(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.qkv_linear(x)
        rope_applied = False

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            if compiled_qk_norm_rope is not None and self.use_rope:
                xq, xk = compiled_qk_norm_rope(
                    xq,
                    xk,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    self.q_norm.eps,
                    self.k_norm.eps,
                    rope_cache,
                )
                rope_applied = True
            else:
                xq = self.q_norm(xq)
                xk = self.k_norm(xk)

        if self.use_rope and not rope_applied:
            if self.rope_backend != "cos_sin":
                raise NotImplementedError(
                    "Triton sequential RoPE only supports cos_sin"
                )
            else:
                xq = _sequential_rope(xq, rope_cache)
                xk = _sequential_rope(xk, rope_cache)

        if isinstance(attention_masks, dict):
            mask_key = "rope" if self.use_rope else "nope"
            attention_masks = attention_masks[mask_key]

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        output = output.reshape(bs, seqlen, -1)
        return self.wo(output)

    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        if attention is None:
            continue
        attention.forward = types.MethodType(_forward_with_triton_rope, attention)
        patched_count += 1

    logger.info(
        "Enabled Triton sequential RoPE for %s Qwen3 attention modules",
        patched_count,
    )
    if compiled_qk_norm_rope is not None:
        logger.info("Compiling Qwen3 Q/K RMSNorm plus RoPE helper with torch.compile")


def parallelize_qwen3(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """Generated machine-specific Qwen3 parallelization entry point.

    A valid implementation may make narrow assumptions about the train command,
    mesh shape, hardware topology, memory budget, model flavor, and enabled
    TorchTitan features. It does not need to be a universal implementation.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support TP.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support CP.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support PP.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support EP.")

    if skip_dp or not parallel_dims.dp_enabled:
        return model

    if parallel_dims.dp_replicate != 1:
        raise NotImplementedError("Qwen3 baseline FSDP bootstrap does not support HSDP.")
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "Qwen3 baseline FSDP bootstrap does not support CPU offload."
        )

    _enable_triton_sequential_rope(model, compile_config)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)
    _enable_shared_mxfp8_qkv_input_cast(model)
    _enable_shared_mxfp8_gate_up_input_cast(model)
    if compile_config.enable and "qkv_linear" in compile_config.components:
        _compile_qwen3_qkv_linear(model, compile_config)
    if compile_config.enable and "feed_forward" in compile_config.components:
        _compile_qwen3_feed_forward(model, compile_config)
    if compile_config.enable and "block_norms" in compile_config.components:
        _enable_compiled_block_norms(model, compile_config)

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, training.mixed_precision_param),
        reduce_dtype=getattr(torch, training.mixed_precision_reduce),
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    fsdp_config = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
    }

    layers = list(model.layers.values())
    for idx, layer in enumerate(layers):
        layer_fsdp_config = fsdp_config
        if idx == len(layers) - 1:
            layer_fsdp_config = {
                **fsdp_config,
                "reshard_after_forward": False,
            }
        fully_shard(layer, **layer_fsdp_config)
    fully_shard(model.lm_head, **fsdp_config)
    fully_shard(model, **fsdp_config)
    if layers:
        for layer, next_layer in zip(layers, layers[1:]):
            layer.set_modules_to_forward_prefetch([next_layer])
        layers[-1].set_modules_to_forward_prefetch([model.lm_head])
        model.lm_head.set_modules_to_backward_prefetch([layers[-1]])
        for layer, prev_layer in zip(reversed(layers[1:]), reversed(layers[:-1])):
            layer.set_modules_to_backward_prefetch([prev_layer])

    disable_fsdp_gradient_division(model)
    logger.info(
        "Applied baseline Qwen3 FSDP with dp_shard=%s, reshard_after_forward=%s",
        parallel_dims.dp_shard,
        reshard_after_forward,
    )

    return model
