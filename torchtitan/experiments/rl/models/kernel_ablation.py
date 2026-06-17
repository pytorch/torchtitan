# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kernel-level ablation patches for the TorchTitan unified Qwen3 model running
under vLLM, used by the [3/3] "Bridge the gap in inference performance" study.

Each rung in the ablation table corresponds to swapping one TorchTitan kernel
for a hand-optimized equivalent.  ``apply()`` monkeypatches the shared model
components in place before the model is built, so the same model definition is
reused while one kernel at a time is replaced.

Rungs (cumulative, controlled by generate.py flags):
  --rope-kernel helion     RoPE via Helion kernel (pytorch/torchtitan#3651)
  --silu-vllm              FFN SwiGLU via vLLM's fused SiluAndMul kernel
  --rmsnorm-vllm           RMSNorm via vLLM's kernel (vs torch/Quack rms_norm)
  --rope-kernel vllm       RoPE via vLLM's fused rotary_embedding kernel
  --no-double-transpose    zero-copy attention layout (already the default in
                           current torchtitan; flag kept for table parity)

These patches are inference-only and live in the experiments folder; they do
not touch core model code.
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply(
    *,
    rope_kernel: str | None = None,
    silu_vllm: bool = False,
    rmsnorm_vllm: bool = False,
    allreduce_vllm: bool = False,
    merged_gemm: bool = False,
    fused_addnorm: bool = False,
    embed_allreduce_vllm: bool = False,
    dtensor_native: bool = False,
    no_double_transpose: bool = False,
) -> None:
    """Monkeypatch the shared Qwen3 components for the requested kernel rungs.

    No-op when every flag is at its default; safe to call unconditionally.
    """
    if rope_kernel in (None, "torchtitan"):
        pass
    elif rope_kernel == "helion":
        _patch_rope_helion(dtensor_native=dtensor_native)
    elif rope_kernel == "vllm":
        _patch_rope_vllm()
    else:
        raise ValueError(f"Unknown rope_kernel {rope_kernel!r}")

    # Merged QKV (3->1 GEMM) is a standalone patch on QKVLinear.forward; merged
    # gate_up (2->1) is folded into the all-reduce FFN path below.
    if merged_gemm:
        _patch_merged_qkv()

    # The all-reduce rung reimplements the attention/FFN output projection
    # (wo/w2) as local matmul + vLLM all-reduce, so it owns FeedForward.forward.
    # Fold the vLLM SiluAndMul and merged gate_up into it when requested;
    # otherwise apply the standalone SiluAndMul patch.
    if allreduce_vllm:
        _patch_allreduce_vllm(use_vllm_silu=silu_vllm, use_merged_gemm=merged_gemm)
    elif silu_vllm:
        _patch_silu_vllm()

    if rmsnorm_vllm:
        _patch_rmsnorm_vllm()

    if fused_addnorm:
        _patch_fused_addnorm()

    if embed_allreduce_vllm:
        _patch_embed_allreduce_vllm()

    if no_double_transpose:
        # The current VLLMAttentionWrapper already reshapes (bs, seq, heads,
        # dim) -> (bs*seq, heads, dim) with a zero-copy reshape (no double
        # transpose), so this rung is already applied. Kept for table parity.
        logger.info(
            "no_double_transpose: already the default attention layout in "
            "current torchtitan; nothing to patch."
        )


def _patch_rope_helion(dtensor_native: bool = False) -> None:
    """Replace CosSinRoPE.forward with the fused Helion kernel.

    Two modes:
    - default: to_local() the DTensor q/k/cache/positions, run the kernel on
      local shards, from_local() back (per-op DTensor round-trip).
    - dtensor_native: register a DTensor sharding strategy for rope_helion_fwd
      and call it directly on DTensors (q/k Shard(2) on heads; cache/positions
      Replicate), so the forward stays in DTensor with no to_local/from_local
      round-trip (mirrors qwen3_vllm.py's register_sharding approach).
    """
    from torch.distributed.tensor import DTensor, Replicate

    from torchtitan.experiments.rl.models.helion_rope import (
        can_use_helion_rope,
        register_rope_helion_sharding,
        rope_helion_fwd,
    )
    from torchtitan.models.common.rope import CosSinRoPE

    _orig_forward = CosSinRoPE.forward

    if dtensor_native:
        register_rope_helion_sharding()

        def helion_forward(self, query, key, positions=None):
            if positions is None or not isinstance(query, DTensor):
                return _orig_forward(self, query, key, positions)
            mesh = query.device_mesh
            cache = self.cache
            cache_dt = (
                cache
                if isinstance(cache, DTensor)
                else DTensor.from_local(cache, mesh, [Replicate()] * mesh.ndim)
            )
            pos_dt = (
                positions
                if isinstance(positions, DTensor)
                else DTensor.from_local(positions, mesh, [Replicate()] * mesh.ndim)
            )
            # DTensor in -> DTensor out via register_sharding; no round-trip.
            return rope_helion_fwd(query, key, cache_dt, pos_dt)

        CosSinRoPE.forward = helion_forward
        logger.info("rope_kernel=helion (dtensor-native): patched CosSinRoPE.forward")
        return

    def helion_forward(self, query, key, positions=None):
        q_local = query.to_local() if isinstance(query, DTensor) else query
        k_local = key.to_local() if isinstance(key, DTensor) else key
        cache = self.cache
        cache_local = cache.to_local() if isinstance(cache, DTensor) else cache
        pos = positions.to_local() if isinstance(positions, DTensor) else positions

        if pos is None or not can_use_helion_rope(q_local, k_local, cache_local, pos):
            return _orig_forward(self, query, key, positions)

        q_out, k_out = rope_helion_fwd(
            q_local.contiguous(),
            k_local.contiguous(),
            cache_local.contiguous(),
            pos.contiguous(),
        )
        if isinstance(query, DTensor):
            q_out = DTensor.from_local(q_out, query.device_mesh, query.placements)
            k_out = DTensor.from_local(k_out, key.device_mesh, key.placements)
        return q_out, k_out

    CosSinRoPE.forward = helion_forward
    logger.info("rope_kernel=helion: patched CosSinRoPE.forward")


def _patch_fused_addnorm() -> None:
    """Fuse the (x + attn) residual-add with the following ffn_norm via vLLM's
    fused_add_rms_norm, removing the standalone residual-add elementwise + a
    standalone RMSNorm per layer (native fuses these). Within-block fusion:
    attention_norm and the final (x + ffn) add stay standalone.
    """
    from torch.distributed.tensor import DTensor, Replicate

    from torchtitan.experiments.rl.models.vllm_fused_ops import fused_add_rmsnorm
    from torchtitan.models.qwen3.model import Qwen3TransformerBlock

    def _wrap_like(ref, local):
        if isinstance(ref, DTensor):
            mesh = ref.device_mesh
            return DTensor.from_local(local, mesh, [Replicate()] * mesh.ndim)
        return local

    def block_forward(self, x, attention_masks, positions=None):
        if self.moe_enabled:  # fused path targets the dense FFN block only
            x = x + self.attention(self.attention_norm(x), attention_masks, positions)
            return x + self.moe(self.ffn_norm(x))

        attn = self.attention(self.attention_norm(x), attention_masks, positions)
        a_local = attn.to_local() if isinstance(attn, DTensor) else attn
        x_local = x.to_local() if isinstance(x, DTensor) else x
        w = self.ffn_norm.weight
        w_local = w.to_local() if isinstance(w, DTensor) else w
        normed_local, new_x_local = fused_add_rmsnorm(
            a_local, x_local, w_local, self.ffn_norm.eps
        )
        normed = _wrap_like(x, normed_local)
        new_x = _wrap_like(x, new_x_local)
        return new_x + self.feed_forward(normed)

    Qwen3TransformerBlock.forward = block_forward
    logger.info("fused_addnorm: patched Qwen3TransformerBlock.forward")


def _patch_merged_qkv() -> None:
    """Merge the separate Q/K/V projections into a single GEMM (3->1).

    Native vLLM uses one fused QKV GEMM; torchtitan's QKVLinear runs three.
    The per-rank weight shards are concatenated once (cached by data_ptr inside
    the custom op), so cudagraph replays run a single GEMM. Outputs are rewrapped
    as DTensors sharded on heads (Shard(2)) to match the original colwise output.
    """
    from torch.distributed.tensor import DTensor, Shard

    from torchtitan.experiments.rl.models.vllm_fused_ops import merged_linear3
    from torchtitan.models.common.attention import QKVLinear

    def merged_forward(self, x):
        bs, seqlen, _ = x.shape
        x_local = x.to_local() if isinstance(x, DTensor) else x

        def _wl(lin):
            w = lin.weight
            return w.to_local() if isinstance(w, DTensor) else w

        wq, wk, wv = _wl(self.wq), _wl(self.wk), _wl(self.wv)
        qkv = merged_linear3(x_local, wq, wk, wv)
        xq, xk, xv = qkv.split([wq.shape[0], wk.shape[0], wv.shape[0]], dim=-1)
        xq = xq.reshape(bs, seqlen, -1, self.head_dim)
        xk = xk.reshape(bs, seqlen, -1, self.head_dim)
        xv = xv.reshape(bs, seqlen, -1, self.head_dim)
        if isinstance(x, DTensor):
            mesh = x.device_mesh
            xq = DTensor.from_local(xq, mesh, [Shard(2)])
            xk = DTensor.from_local(xk, mesh, [Shard(2)])
            xv = DTensor.from_local(xv, mesh, [Shard(2)])
        return xq, xk, xv

    QKVLinear.forward = merged_forward
    logger.info("merged_gemm: patched QKVLinear.forward (3->1 QKV GEMM)")


def _patch_allreduce_vllm(
    *, use_vllm_silu: bool = False, use_merged_gemm: bool = False
) -> None:
    """Run the attention/FFN output projection (wo/w2) as local matmul + vLLM
    custom all-reduce instead of a DTensor RowwiseParallel NCCL all-reduce.

    Profiling showed the DTensor TP all-reduce (ncclDevKernel ring) is ~45% of
    torchtitan's inference GPU time, while vLLM's native model uses a custom
    one-shot/multimem all-reduce. The wo/w2 inputs are sharded on the projection
    input dim (heads*head_dim / hidden), so local F.linear yields a partial sum
    that vLLM's all-reduce reduces across the TP group; the replicated result is
    rewrapped as a Replicate DTensor to keep the residual stream consistent.
    """
    import torch.nn.functional as F
    from torch.distributed.tensor import DTensor, Replicate

    from torchtitan.experiments.rl.models.vllm_fused_ops import (
        merged_linear2,
        silu_and_mul,
    )
    from torchtitan.models.common.attention import GQAttention
    from torchtitan.models.common.feed_forward import FeedForward

    from vllm.distributed import tensor_model_parallel_all_reduce

    def _to_local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    def _local_linear_allreduce(x, lin):
        # NOTE: no Python-side logging/prints here -- this runs inside the
        # torch.compile region and any such call forces a graph break.
        x_local = x.to_local() if isinstance(x, DTensor) else x
        w = lin.weight
        w_local = w.to_local() if isinstance(w, DTensor) else w
        y = F.linear(x_local, w_local)
        y = tensor_model_parallel_all_reduce(y)
        if lin.bias is not None:
            b = lin.bias
            y = y + (b.to_local() if isinstance(b, DTensor) else b)
        if isinstance(x, DTensor):
            mesh = x.device_mesh
            y = DTensor.from_local(y, mesh, [Replicate()] * mesh.ndim)
        return y

    def attn_forward(self, x, attention_masks, positions=None):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.qkv_linear(x)
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq, xk = self.rope(xq, xk, positions)
        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        output = output.view(bs, seqlen, -1)
        return _local_linear_allreduce(output, self.wo)

    def ffn_forward(self, x):
        # gate_up: one merged GEMM (2->1) or two separate, all on local shards.
        x_local = _to_local(x)
        if use_merged_gemm:
            gate_up = merged_linear2(
                x_local, _to_local(self.w1.weight), _to_local(self.w3.weight)
            )
        else:
            gate_up = torch.cat([_to_local(self.w1(x)), _to_local(self.w3(x))], dim=-1)
        if use_vllm_silu:
            fused = silu_and_mul(gate_up)
        else:
            h = gate_up.shape[-1] // 2
            fused = F.silu(gate_up[..., :h]) * gate_up[..., h:]
        # local w2 + vLLM all-reduce (fused is a local partial-input tensor).
        y = F.linear(fused, _to_local(self.w2.weight))
        y = tensor_model_parallel_all_reduce(y)
        if self.w2.bias is not None:
            y = y + _to_local(self.w2.bias)
        if isinstance(x, DTensor):
            mesh = x.device_mesh
            y = DTensor.from_local(y, mesh, [Replicate()] * mesh.ndim)
        return y

    def fused_swiglu_ffn_forward(self, x):
        # Fused gate_up override (w13): einsum gate_up on local shards, silu,
        # then local w2 + vLLM all-reduce (mirrors ffn_forward for FusedSwiGLU).
        x_local = _to_local(x)
        w13 = _to_local(self.w13)  # [hidden_local, 2, dim]
        gate_up = torch.einsum("...d,hgd->...hg", x_local, w13)  # [.., hidden_local, 2]
        gate, up = gate_up[..., 0], gate_up[..., 1]
        if use_vllm_silu:
            fused = silu_and_mul(torch.cat([gate, up], dim=-1))
        else:
            fused = F.silu(gate) * up
        y = F.linear(fused, _to_local(self.w2.weight))
        y = tensor_model_parallel_all_reduce(y)
        if isinstance(x, DTensor):
            mesh = x.device_mesh
            y = DTensor.from_local(y, mesh, [Replicate()] * mesh.ndim)
        return y

    GQAttention.forward = attn_forward
    FeedForward.forward = ffn_forward
    # GQAttention.attn_forward already uses self.qkv_linear, so FusedQKVLinear is
    # covered; only the fused FFN (FusedSwiGLU) needs its own reimplementation.
    try:
        from torchtitan.overrides.fused_swiglu import FusedSwiGLU

        FusedSwiGLU.forward = fused_swiglu_ffn_forward
    except ImportError:
        pass
    logger.info(
        "allreduce_vllm: patched GQAttention/FeedForward/FusedSwiGLU output "
        "projections with local matmul + vLLM all-reduce (vllm_silu=%s)",
        use_vllm_silu,
    )


def _patch_embed_allreduce_vllm() -> None:
    """Route the vocab-parallel embedding's reduction through vLLM's all-reduce.

    The unified model's ``Embedding`` does a masked local lookup and declares its
    output ``Partial`` (decoder_sharding sets ``out_src = spmd.P``). When that
    Partial flows into the first layer's norm (which wants ``Replicate``), DTensor
    inserts a ``Partial -> Replicate`` reduction over **NCCL ring** -- one extra
    all-reduce per forward step that native does not have (native's
    VocabParallelEmbedding reduces with vLLM's custom AR). Profiling attributes
    ~48.6k us / 133 calls to this leftover NCCL AR ("#3" in the gap accounting).

    This reimplements ``VLLMModelWrapper.forward`` to convert the embedding's
    Partial output to Replicate with ``tensor_model_parallel_all_reduce`` (the
    same one-shot/multimem path used by the wo/w2 rung), so the whole forward
    uses one AR backend and the first layer sees Replicate (no NCCL redistribute).
    """
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.placement_types import Partial

    from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper

    from vllm.distributed import tensor_model_parallel_all_reduce

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        tokens_2d = input_ids.unsqueeze(0)
        h = self.model.tok_embeddings(tokens_2d)

        # Partial -> Replicate via vLLM AR (instead of DTensor's NCCL ring), so
        # the downstream norm sees Replicate and no extra NCCL AR is inserted.
        if isinstance(h, DTensor) and any(isinstance(p, Partial) for p in h.placements):
            mesh = h.device_mesh
            h_local = tensor_model_parallel_all_reduce(h.to_local())
            h = DTensor.from_local(h_local, mesh, [Replicate()] * mesh.ndim)

        positions = positions.unsqueeze(0)
        for layer in self.model.layers.values():
            h = layer(h, attention_masks=None, positions=positions)

        h = self.model.norm(h)
        if isinstance(h, DTensor):
            h = h.full_tensor()
        if h.dim() == 3:
            h = h.view(-1, h.size(-1))
        return h

    VLLMModelWrapper.forward = forward
    logger.info(
        "embed_allreduce_vllm: patched VLLMModelWrapper.forward (embedding "
        "Partial->Replicate via vLLM AR, drops the leftover NCCL all-reduce)"
    )


def _patch_rope_vllm() -> None:
    """Replace CosSinRoPE.forward with vLLM's fused rotary_embedding kernel.

    Below-the-divider comparison rung (torchtitan's RoPE / the Helion rung is
    the optimized path). vLLM's neox rotary runs in-place on flattened
    (num_tokens, n_heads*head_dim) q/k with a (max_pos, head_dim) cache. Under
    TP the q/k are DTensors sharded on heads; the kernel runs on local shards
    and is rewrapped with the input placements.
    """
    from torch.distributed.tensor import DTensor

    from torchtitan.experiments.rl.models.vllm_fused_ops import vllm_rotary
    from torchtitan.models.common.rope import CosSinRoPE

    _orig_forward = CosSinRoPE.forward

    def vllm_forward(self, query, key, positions=None):
        cache = self.cache
        cache_local = cache.to_local() if isinstance(cache, DTensor) else cache
        pos = positions.to_local() if isinstance(positions, DTensor) else positions
        if pos is None or pos.ndim != 2 or pos.shape[:2] != query.shape[:2]:
            return _orig_forward(self, query, key, positions)

        q_is_dt = isinstance(query, DTensor)
        q_local = query.to_local() if q_is_dt else query
        k_local = key.to_local() if isinstance(key, DTensor) else key
        bs, seq, n_q, head_dim = q_local.shape
        n_k = k_local.shape[2]
        q_flat = q_local.reshape(bs * seq, n_q * head_dim).contiguous()
        k_flat = k_local.reshape(bs * seq, n_k * head_dim).contiguous()

        q_out, k_out = vllm_rotary(
            q_flat, k_flat, pos.reshape(-1), cache_local.contiguous(), head_dim
        )
        q_out = q_out.reshape(bs, seq, n_q, head_dim)
        k_out = k_out.reshape(bs, seq, n_k, head_dim)
        if q_is_dt:
            q_out = DTensor.from_local(q_out, query.device_mesh, query.placements)
            k_out = DTensor.from_local(k_out, key.device_mesh, key.placements)
        return q_out, k_out

    CosSinRoPE.forward = vllm_forward
    logger.info("rope_kernel=vllm: patched CosSinRoPE.forward")


def _patch_silu_vllm() -> None:
    """Replace FeedForward.forward's SwiGLU with vLLM's fused SiluAndMul kernel.

    Core path: ``w2(F.silu(w1(x)) * w3(x))`` -- separate silu, mul, and the
    implicit intermediate. Fused path concatenates the gate/up projections and
    runs vLLM's single silu_and_mul kernel. Under TP the projections are
    DTensors sharded on the hidden dim; gate and up share the same shard layout,
    so cat||silu_and_mul on local shards is correct and the result is rewrapped
    with the gate projection's placements before the row-parallel w2.
    """
    from torch.distributed.tensor import DTensor

    from torchtitan.experiments.rl.models.vllm_fused_ops import silu_and_mul
    from torchtitan.models.common.feed_forward import FeedForward

    def fused_forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        g_local = gate.to_local() if isinstance(gate, DTensor) else gate
        u_local = up.to_local() if isinstance(up, DTensor) else up
        fused = silu_and_mul(torch.cat([g_local, u_local], dim=-1))
        if isinstance(gate, DTensor):
            fused = DTensor.from_local(fused, gate.device_mesh, gate.placements)
        return self.w2(fused)

    FeedForward.forward = fused_forward

    # Also handle the fused_swiglu override's FusedSwiGLU (single w13 param) when
    # --fused-qkv-gateup is active: replace its F.silu(gate)*up with vLLM's kernel.
    try:
        from torchtitan.overrides.fused_swiglu import FusedSwiGLU

        def fused_swiglu_silu(self, x):
            gate, up = torch.einsum("...d,hgd->...hg", x, self.w13).unbind(-1)
            g = gate.to_local() if isinstance(gate, DTensor) else gate
            u = up.to_local() if isinstance(up, DTensor) else up
            fused = silu_and_mul(torch.cat([g, u], dim=-1))
            if isinstance(gate, DTensor):
                fused = DTensor.from_local(fused, gate.device_mesh, gate.placements)
            return self.w2(fused)

        FusedSwiGLU.forward = fused_swiglu_silu
    except ImportError:
        pass
    logger.info("silu_vllm: patched FeedForward/FusedSwiGLU with fused silu_and_mul")


def _patch_rmsnorm_vllm() -> None:
    """Replace RMSNorm.forward with vLLM's fused rms_norm kernel.

    The core RMSNorm is an nn.RMSNorm subclass dispatching to torch's fused
    rms_norm (the Quack kernel in recent nightlies); this rung swaps in vLLM's
    _C.rms_norm for a direct comparison. Normalization is over the last dim,
    which is never TP-sharded, so the kernel runs on local shards and is
    rewrapped with the input's placements.
    """
    from torch.distributed.tensor import DTensor

    from torchtitan.experiments.rl.models.vllm_fused_ops import rms_norm
    from torchtitan.models.common.nn_modules import RMSNorm

    def rms_forward(self, x):
        x_local = x.to_local() if isinstance(x, DTensor) else x
        weight = self.weight
        w_local = weight.to_local() if isinstance(weight, DTensor) else weight
        out = rms_norm(x_local.contiguous(), w_local.contiguous(), self.eps)
        if isinstance(x, DTensor):
            out = DTensor.from_local(out, x.device_mesh, x.placements)
        return out

    RMSNorm.forward = rms_forward
    logger.info("rmsnorm_vllm: patched RMSNorm.forward with fused rms_norm")
