# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified-model GDN generation core for vLLM (single FLA-kernel path).

TorchTitan's own Qwen3.5 ``GatedDeltaNet`` runs inside vLLM (the unified model
path), borrowing ONLY vLLM's paged conv+ssm cache. Everything computational uses
the SAME ``fla`` package the trainer uses -- there is no vLLM-vendored-kernel
fallback and no env-var switch. Under batch-invariant mode the generator is
therefore bitwise-identical to the trainer forward.

The core (``_forward``) is decorated with ``@eager_break_during_capture`` so it
composes with cudagraph: it is a graph-split point that runs eager during breakable
piecewise capture while the surrounding compute is captured -- breakable piecewise
for prefill/mixed batches and a full graph for pure decode.

Mechanism (all FLA):
  - conv: fla ``causal_conv1d`` (prefill) / ``causal_conv1d_update`` (decode)
    against vLLM's PAGED conv_state (``kv_cache[0]``). fla's conv state is
    ``[N, conv_dim, W]`` while vLLM stores ``[.., conv_dim, W-1]`` (the last W-1
    pre-conv inputs) and column 0 is a verified don't-care -- so we pad a zero
    column on read and drop it on write. Using the paged conv_state (not a side
    buffer) lets vLLM's prefix cache save/restore the conv history.
  - recurrence: fla ``fused_recurrent_gated_delta_rule`` for BOTH prefill and
    decode (recurrent-everywhere) against the paged ssm_state via gather/scatter
    (upstream fla recurrent has no slot indexing).
  - bitwise parity (only under ``is_in_batch_invariant_mode()``): run the
    recurrence in fp32 and always pass a MATERIALIZED zero/gathered initial state
    (never ``None``) so triton compiles the SAME kernel for fresh-prefill,
    continuation and decode -> the recurrence is resume-exact, i.e.
    decode == prefill == trainer. Outside BI the
    recurrence runs in the model dtype (fast); parity is not required there.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# The SAME fla package the trainer uses -> generator == trainer under BI.
from fla.modules.convolution import (
    causal_conv1d as _fla_causal_conv1d,
    causal_conv1d_update as _fla_causal_conv1d_update,
)
from fla.ops.gated_delta_rule import (
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from torch.distributed.tensor import DTensor

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.protocols.module import Module

# Under breakable cudagraph capture the GDN op must run eager at the split point
# (GDN only supports AttentionCGSupport.UNIFORM_BATCH). eager_break_during_capture
# marks that break and is inert unless VLLM_USE_BREAKABLE_CUDAGRAPH is set.
# Imported from open-source vLLM, matching rl/models/attention.py.
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class VLLMGatedDeltaNetCore(Module, MambaBase):
    """Paged-cache GDN generation core for the unified model path.

    Holds NO learnable parameters -- projections, conv weights, gates, norm and
    out_proj all live in the enclosing ``qwen3_5.model.GatedDeltaNet``, which
    passes the conv weight / A_log / dt_bias into ``_forward`` as tensor
    args. This layer owns only the vLLM cache plumbing (state discovery + paged
    conv/ssm state) and the FLA kernels.

    Non-speculative decoding only (asserts otherwise). Supports TP: the enclosing
    GatedDeltaNet is head-sharded, so this core computes on LOCAL per-rank head
    slices and vLLM's paged conv/ssm state is local-head (get_state_shape divides
    by tp_size); ``forward`` bridges the head-sharded DTensor inputs/outputs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int = 4
        activation: str = "silu"

    def __init__(self, config: Config) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_spec = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )

        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.activation = config.activation

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        # Tensor parallelism: the enclosing GatedDeltaNet is head-sharded (colwise
        # projections + head-sharded fused conv), so the projections/conv/gates this
        # core receives are LOCAL (per-rank) head slices, and vLLM's paged conv/ssm
        # state is local-head too (get_state_shape divides by tp_size). Keep GLOBAL
        # counts for get_state_shape (it takes tp_size + global) and derive LOCAL
        # counts for the compute, split, and gather/scatter.
        if self.num_k_heads % self.tp_size != 0 or self.num_v_heads % self.tp_size != 0:
            raise ValueError(
                f"num_k_heads ({self.num_k_heads}) and num_v_heads "
                f"({self.num_v_heads}) must both be divisible by "
                f"tensor_parallel_size ({self.tp_size})."
            )
        self.local_num_k_heads = self.num_k_heads // self.tp_size
        self.local_num_v_heads = self.num_v_heads // self.tp_size
        self.local_key_dim = self.local_num_k_heads * self.head_k_dim
        self.local_value_dim = self.local_num_v_heads * self.head_v_dim

        # Bitwise parity needs the paged conv AND ssm states in fp32: decode
        # round-trips both through the cache each step, so bf16 drifts from the
        # single-pass prefill. Set on cache_config (not just get_state_dtype) so
        # vLLM's page-size accounting agrees; non-BI serving keeps the defaults.
        if is_in_batch_invariant_mode():
            self.cache_config.mamba_cache_dtype = "float32"
            self.cache_config.mamba_ssm_cache_dtype = "float32"

        # vLLM populates this via the KV-cache allocator: (conv_state, ssm_state).
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.prefix = f"model.layers.{config.layer_idx}.linear_attn"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate GDN layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    # ---- MambaBase contract ------------------------------------------------
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    # ---- helpers -----------------------------------------------------------
    def _split_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused [T, C] conv output into q/k/v with a batch dim.

        Under TP mixed_qkv holds only this rank's heads, so split by the LOCAL
        key_dim / head counts (== global on a single GPU).
        """
        t = mixed_qkv.shape[0]
        kd = self.local_key_dim
        q = (
            mixed_qkv[:, :kd]
            .contiguous()
            .view(1, t, self.local_num_k_heads, self.head_k_dim)
        )
        k = (
            mixed_qkv[:, kd : 2 * kd]
            .contiguous()
            .view(1, t, self.local_num_k_heads, self.head_k_dim)
        )
        v = (
            mixed_qkv[:, 2 * kd :]
            .contiguous()
            .view(1, t, self.local_num_v_heads, self.head_v_dim)
        )
        return q, k, v

    @staticmethod
    def _pad_w1_to_w(state_w1: torch.Tensor) -> torch.Tensor:
        """vLLM paged conv_state [N, C, W-1] -> fla conv state [N, C, W].

        fla's causal_conv1d state's trailing W-1 columns equal vLLM's paged
        conv_state; column 0 is a don't-care (verified), so pad a zero column on
        read (and callers drop it with ``[..., 1:]`` on write).
        """
        pad = state_w1.new_zeros(state_w1.shape[0], state_w1.shape[1], 1)
        return torch.cat([pad, state_w1], dim=-1)

    # ---- core --------------------------------------------------------------
    # @eager_break_during_capture makes this a cudagraph graph-split point: it runs
    # eager (breaking the segment) during breakable piecewise capture -- required
    # because the recurrence has host syncs, data-dependent indexing, and paged-cache
    # mutation. It writes core_attn_out in place (the decorator needs a caller-owned
    # output whose address is stable across replays). Inert unless breakable is on.
    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """conv + gated-delta recurrence against vLLM's paged state, in place.

        Args (flattened token layout, one row per request concatenated):
            mixed_qkv: [num_tokens_padded, C] fused q|k|v projections, PRE-conv.
            a, b: [num_tokens_padded, Hv] gate inputs (in_proj_a/b outputs).
            conv_weight: fused depthwise conv weight [C, W]; conv_bias [C] or None.
            A_log, dt_bias: [Hv] decay params.
            core_attn_out: [num_tokens_padded, Hv, Dv] output buffer, mutated.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # Dummy run (profiling / warmup): no metadata -> leave zeros.
        if attn_metadata is None:
            return
        assert isinstance(attn_metadata, dict)
        gdn_metadata = attn_metadata[self.prefix]
        assert isinstance(gdn_metadata, GDNAttentionMetadata)
        assert (
            gdn_metadata.spec_sequence_masks is None
        ), "VLLMGatedDeltaNetCore does not support speculative decoding"

        num_tokens = gdn_metadata.num_actual_tokens
        if num_tokens == 0:
            return

        mixed_qkv = mixed_qkv[:num_tokens]
        a = a[:num_tokens]
        b = b[:num_tokens]

        non_spec_state_indices = gdn_metadata.non_spec_state_indices_tensor
        assert non_spec_state_indices is not None
        ssm_state = self.kv_cache[1]
        # Paged conv state is (.., C, W-1) in the DS (dim-first) layout, or (.., W-1, C)
        # in SD (vLLM's default, dim innermost); transpose the SD case so the fla conv
        # kernels see (.., C, W-1). See _pad_w1_to_w for the W-1 <-> W bridge.
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_width = self.conv_kernel_size

        # fp32 gate scalars, hoisted (identical fp32 ops -> pure CSE, bitwise-safe).
        A_neg_exp = -torch.exp(A_log.float())
        dt_bias_fp32 = dt_bias.float()

        out = mixed_qkv.new_empty(
            1, num_tokens, self.local_num_v_heads, self.head_v_dim
        )

        def _recurrence(
            conv_out: torch.Tensor,
            segment: slice,
            slot_indices: torch.Tensor,
            cu_seqlens: torch.Tensor,
            *,
            is_decode: bool,
            prefill_has_initial_state: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the fla gated-delta recurrence for this segment's sequences.

            slot_indices: each sequence's paged-cache slot -- where its ssm state is
                read from and written back to.
            cu_seqlens: per-sequence token boundaries; the gaps set each sequence's
                step count T in the fla kernel (T=1 for decode, T=L for prefill).
            is_decode: True for the decode segment (all sequences resume; gather
                every slot, cudagraph-safe). False for prefill (zeros + masked restore).
            prefill_has_initial_state: prefill only, per-sequence bool -- True where a
                sequence resumes a prefix-cached state, None when all are fresh.
            """
            q, k, v = self._split_qkv(conv_out)
            if q.shape[2] != v.shape[2]:  # grouped-value: expand q/k heads
                num_repeats = v.shape[2] // q.shape[2]
                q = q.repeat_interleave(num_repeats, dim=2)
                k = k.repeat_interleave(num_repeats, dim=2)
            # Under BI, run the recurrence in fp32 so decode (T=1) and prefill (T=L)
            # match: bf16 diverges ~1e-5 across the two call shapes, fp32 ~1e-8
            # (vanishes on the bf16 downcast). The trainer upcasts the same way.
            if is_in_batch_invariant_mode():
                q, k, v = q.float(), k.float(), v.float()
            # Eager gate, matching qwen3_5.model.GatedDeltaNet exactly: g in fp32,
            # beta = sigmoid(b) in the model dtype. Under BI, upcast beta to fp32 for
            # the fp32 recurrence (the trainer's _RecurrentFwdChunkBwd upcasts the
            # same way).
            g = (A_neg_exp * F.softplus(a[segment].float() + dt_bias_fp32)).unsqueeze(0)
            beta = torch.sigmoid(b[segment]).unsqueeze(0)
            if is_in_batch_invariant_mode():
                beta = beta.float()
            # Recurrent initial state: ALWAYS a materialized tensor (never None) so
            # triton compiles the SAME kernel for fresh/continuation/decode ->
            # resume-exact. Paged ssm_state is [.., V, K]; fla wants [.., K, V].
            # TODO: drop this paged-state gather/copy once fla recurrent can
            # read/write the paged cache by slot index directly.
            state_dtype = torch.float32
            if is_decode:
                # Decode: every sequence resumes -> int-index gather of all slots.
                # cudagraph-capturable (no data-dependent/boolean indexing).
                initial_state = (
                    ssm_state[slot_indices]
                    .transpose(-1, -2)
                    .to(state_dtype)
                    .contiguous()
                )
            else:
                # Prefill (eager): start from zeros, then restore the prefix-cache
                # continuations by per-seq boolean mask. prefill_has_initial_state is
                # None when every sequence is fresh. Boolean-mask indexing is
                # eager-only, fine as only pure-decode is cudagraph-captured.
                num_seqs = int(cu_seqlens.numel()) - 1
                initial_state = q.new_zeros(
                    num_seqs, q.shape[2], q.shape[3], v.shape[3], dtype=state_dtype
                )
                if prefill_has_initial_state is not None:
                    initial_state[prefill_has_initial_state] = (
                        ssm_state[slot_indices[prefill_has_initial_state]]
                        .transpose(-1, -2)
                        .to(state_dtype)
                    )
            out, final_state = _fla_fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[slot_indices] = final_state.transpose(-1, -2).to(ssm_state.dtype)
            return out

        num_decode_tokens = gdn_metadata.num_decode_tokens

        # Decode segment: 1 token/seq; resume conv + ssm from the paged cache.
        if gdn_metadata.num_decodes > 0:
            decode_slots = non_spec_state_indices[: gdn_metadata.num_decodes]
            conv_cache = self._pad_w1_to_w(conv_state[decode_slots])  # [nd, C, W]
            conv_out, conv_cache = _fla_causal_conv1d_update(
                mixed_qkv[:num_decode_tokens],
                conv_cache,
                weight=conv_weight,
                bias=conv_bias,
                activation=self.activation,
            )
            conv_state[decode_slots] = conv_cache[..., 1:].to(conv_state.dtype)
            out[:, :num_decode_tokens] = _recurrence(
                conv_out,
                slice(0, num_decode_tokens),
                decode_slots,
                gdn_metadata.non_spec_query_start_loc[: gdn_metadata.num_decodes + 1],
                is_decode=True,
            )

        # Prefill segment: fresh AND/OR prefix-cache continuation sequences.
        if gdn_metadata.num_prefills > 0:
            assert gdn_metadata.prefill_state_indices is not None
            prefill_state_indices = gdn_metadata.prefill_state_indices
            # per-seq bool (vLLM's prefill_has_initial_state), or None when fresh.
            prefill_has_initial_state = gdn_metadata.prefill_has_initial_state
            prefill_start = num_decode_tokens if gdn_metadata.num_decodes > 0 else 0
            if gdn_metadata.num_decodes == 0:
                prefill_cu_seqlens = gdn_metadata.non_spec_query_start_loc  # 0-based
            else:
                assert gdn_metadata.prefill_query_start_loc is not None
                prefill_cu_seqlens = (
                    gdn_metadata.prefill_query_start_loc
                    - gdn_metadata.prefill_query_start_loc[0]
                )
            num_prefill_seqs = int(prefill_cu_seqlens.numel()) - 1
            # This core always runs eager at the cudagraph graph-split boundary (the
            # GDN custom op is never captured, in any mode), so the .any() host sync
            # and the boolean-mask restore are safe here.
            has_continuations = prefill_has_initial_state is not None and bool(
                prefill_has_initial_state.any()
            )
            conv_initial_state = mixed_qkv.new_zeros(
                num_prefill_seqs, mixed_qkv.shape[1], conv_width
            )
            if has_continuations:
                continuation_slots = prefill_state_indices[prefill_has_initial_state]
                conv_initial_state[prefill_has_initial_state, :, 1:] = conv_state[
                    continuation_slots
                ]
            conv_out, conv_final = _fla_causal_conv1d(
                mixed_qkv[prefill_start:num_tokens].unsqueeze(0),
                weight=conv_weight,
                bias=conv_bias,
                activation=self.activation,
                cu_seqlens=prefill_cu_seqlens,
                initial_state=conv_initial_state,
                output_final_state=True,
            )
            conv_out = conv_out.squeeze(0)
            conv_state[prefill_state_indices] = conv_final[..., 1:].to(conv_state.dtype)
            out[:, prefill_start:num_tokens] = _recurrence(
                conv_out,
                slice(prefill_start, num_tokens),
                prefill_state_indices,
                prefill_cu_seqlens,
                is_decode=False,
                prefill_has_initial_state=prefill_has_initial_state,
            )

        core_attn_out[:num_tokens] = out[0, :num_tokens].to(core_attn_out.dtype)

    # ---- entry point (called by qwen3_5.model.GatedDeltaNet) ---------------
    def forward(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Allocate the output buffer and dispatch to the vLLM custom op (the
        graph-split boundary). The op re-fetches this core from the forward context and
        runs ``_forward``, which writes ``core_attn_out`` in place.

        Inputs are 3D ``[B, L, ...]`` (shared signature with the training
        ``GatedDeltaCore``); in vLLM B=1, so flatten to the ``[T, ...]`` layout the
        paged op works in and reshape the output back. ``mixed_qkv`` [B, L, C]
        (pre-conv), ``a``/``b`` [B, L, Hv], returns [B, L, Hv, Dv]. Per-sequence
        boundaries come from vLLM's ``attn_metadata`` inside ``_forward`` (no
        cu_seqlens arg); the depthwise conv has no bias (config bias=False).

        Under TP the enclosing GatedDeltaNet is head-sharded, so the inputs arrive as
        head-sharded DTensors. The paged custom op works on LOCAL per-rank tensors
        (paged state is local-head), so convert DTensor -> local before the op and
        rewrap the [B, L, Hv, Dv] output as a head-sharded DTensor for the
        RowwiseParallel out_proj. (Single GPU / TP=1: inputs are plain tensors.)
        """
        mesh = None
        placements = None
        if isinstance(mixed_qkv, DTensor):
            mesh = mixed_qkv.device_mesh
            placements = mixed_qkv.placements  # Shard on the head/channel dim
            mixed_qkv = mixed_qkv.to_local()
            a = a.to_local()
            b = b.to_local()
            conv_weight = conv_weight.to_local()
            A_log = A_log.to_local()
            dt_bias = dt_bias.to_local()
        bs, seqlen, c = mixed_qkv.shape
        t = bs * seqlen
        mixed_qkv = mixed_qkv.reshape(t, c)
        a = a.reshape(t, a.shape[-1])
        b = b.reshape(t, b.shape[-1])
        # zeros (not empty) so padded rows stay defined; see vLLM PR #28182.
        core_attn_out = mixed_qkv.new_zeros(t, self.local_num_v_heads, self.head_v_dim)
        self._forward(
            mixed_qkv,
            a,
            b,
            conv_weight,
            None,  # conv_bias (depthwise conv has no bias)
            A_log,
            dt_bias,
            core_attn_out,
        )
        out = core_attn_out.reshape(bs, seqlen, self.local_num_v_heads, self.head_v_dim)
        if mesh is not None:
            # Rewrap as a head-sharded DTensor (Shard on dim 2 = value heads),
            # reusing the input's placements so multi-axis meshes carry through,
            # for the RowwiseParallel out_proj downstream.
            out = DTensor.from_local(out, mesh, placements)
        return out
