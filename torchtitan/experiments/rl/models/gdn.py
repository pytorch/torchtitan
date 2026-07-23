# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM paged-cache adapter for TorchTitan's Gated DeltaNet.

The enclosing Qwen3.5 module owns all parameters. This adapter runs the same FLA
convolution and recurrence kernels as training while reading and updating vLLM's
paged convolution and SSM states.

Batch-invariant execution has three additional requirements:

* Convolution and SSM cache states use float32. Decode otherwise rounds the state
  through bfloat16 after every token, unlike a single prefill call.
* Recurrence inputs use float32. FLA's recurrent kernel has small shape-dependent
  bfloat16 differences between one-token decode and multi-token prefill.
* Every recurrence receives a materialized initial state and ``cu_seqlens``. This
  keeps FLA's Triton specialization identical for fresh prefill and resumed state.

The recurrent kernel and materialized state are also used outside batch-invariant
mode because vLLM generation must support prefix continuation. Only the cache and
input precision differ by mode.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Using the trainer's FLA kernels is required for batch-invariant parity.
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

# The recurrence mutates paged state and must run eager at a breakable cudagraph
# split point. This decorator is inert when breakable capture is disabled.
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
    """Paged-cache GDN generation core.

    The enclosing ``qwen3_5.model.GatedDeltaNet`` owns all parameters. This
    module owns only vLLM cache plumbing and the FLA kernels.

    The enclosing module and vLLM cache are both head-sharded under tensor
    parallelism. Speculative decoding is not supported.
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
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )

        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.activation = config.activation

        # vLLM's state-shape calculator takes global head counts, while the
        # computation and allocated cache use local head counts.
        if (
            self.num_k_heads % self.tensor_parallel_size != 0
            or self.num_v_heads % self.tensor_parallel_size != 0
        ):
            raise ValueError(
                f"num_k_heads ({self.num_k_heads}) and num_v_heads "
                f"({self.num_v_heads}) must both be divisible by "
                f"tensor_parallel_size ({self.tensor_parallel_size})."
            )
        self.local_num_k_heads = self.num_k_heads // self.tensor_parallel_size
        self.local_num_v_heads = self.num_v_heads // self.tensor_parallel_size
        self.local_key_dim = self.local_num_k_heads * self.head_k_dim

        if is_in_batch_invariant_mode():
            # vLLM's allocator reads these fields for both dtype and page-size
            # accounting, so configure them before it allocates the cache.
            self.cache_config.mamba_cache_dtype = "float32"
            self.cache_config.mamba_ssm_cache_dtype = "float32"

        # vLLM populates this via the KV-cache allocator: (conv_state, ssm_state).
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.prefix = f"model.layers.{config.layer_idx}.linear_attn"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate GDN layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return the (conv_state, ssm_state) cache dtypes.

        Required by vLLM's MambaBase interface: the KV-cache allocator calls this
        to allocate the paged conv and SSM state before the model runs.
        """
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return the per-slot (conv_state, ssm_state) cache shapes.

        Required by vLLM's MambaBase interface: the KV-cache allocator calls this
        to size the paged conv and SSM state before the model runs.
        """
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tensor_parallel_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_speculative_tokens,
        )

    def _split_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split local fused channels and add FLA's singleton batch dim."""
        num_tokens = mixed_qkv.shape[0]
        local_key_dim = self.local_key_dim
        query = (
            mixed_qkv[:, :local_key_dim]
            .contiguous()
            .view(1, num_tokens, self.local_num_k_heads, self.head_k_dim)
        )
        key = (
            mixed_qkv[:, local_key_dim : 2 * local_key_dim]
            .contiguous()
            .view(1, num_tokens, self.local_num_k_heads, self.head_k_dim)
        )
        value = (
            mixed_qkv[:, 2 * local_key_dim :]
            .contiguous()
            .view(1, num_tokens, self.local_num_v_heads, self.head_v_dim)
        )
        return query, key, value

    def _run_recurrence(
        self,
        conv_output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        negative_exp_A: torch.Tensor,
        dt_bias: torch.Tensor,
        initial_state: torch.Tensor,
        ssm_state: torch.Tensor,
        slot_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run FLA's recurrence and update the selected paged SSM slots."""
        query, key, value = self._split_qkv(conv_output)
        # Grouped-value heads: expand q/k to match the value head count.
        if query.shape[2] != value.shape[2]:
            num_repeats = value.shape[2] // query.shape[2]
            query = query.repeat_interleave(num_repeats, dim=2)
            key = key.repeat_interleave(num_repeats, dim=2)

        decay = (negative_exp_A * F.softplus(a.float() + dt_bias)).unsqueeze(0)
        update_gate = torch.sigmoid(b).unsqueeze(0)

        # The batch-invariant trainer passes these recurrence inputs in float32;
        # matching that dtype keeps generator outputs and final states identical.
        if is_in_batch_invariant_mode():
            query = query.float()
            key = key.float()
            value = value.float()
            update_gate = update_gate.float()

        output, final_state = _fla_fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta=update_gate,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[slot_indices] = final_state.transpose(-1, -2).to(ssm_state.dtype)
        return output

    # The decorator makes this an eager graph-split point during breakable capture.
    # The caller-owned output has a stable address across graph replays.
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
        output: torch.Tensor,
    ) -> None:
        """Run convolution and recurrence against vLLM's paged state in place."""
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # vLLM's profiling/warmup runs have no attention metadata; leave the
        # zero-filled output.
        if attn_metadata is None:
            return
        assert isinstance(attn_metadata, dict)
        gdn_metadata = attn_metadata[self.prefix]
        assert isinstance(gdn_metadata, GDNAttentionMetadata)
        assert (
            gdn_metadata.spec_sequence_masks is None
        ), "VLLMGatedDeltaNetCore does not support speculative decoding"

        num_actual_tokens = gdn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        a = a[:num_actual_tokens]
        b = b[:num_actual_tokens]

        state_indices = gdn_metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        ssm_state = self.kv_cache[1]
        # vLLM's default conv-state layout keeps channels innermost (.., W-1, C);
        # FLA needs channels first (.., C, W-1), so transpose that case.
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        negative_exp_A = -torch.exp(A_log.float())
        dt_bias = dt_bias.float()
        num_decodes = gdn_metadata.num_decodes
        num_prefills = gdn_metadata.num_prefills
        num_decode_tokens = gdn_metadata.num_decode_tokens
        num_sequences = num_decodes + num_prefills

        # Convolution is split by request type and writes one contiguous
        # conv_output for the single recurrence below. Decode is FULL-captured in
        # a CUDA graph, so it must use the single-token update kernel: the varlen
        # causal_conv1d prepares chunk indices with host syncs, which capture
        # forbids. Prefill (eager at the graph break) uses the varlen kernel.
        # vLLM orders tokens decode-first, then prefill.
        conv_output = mixed_qkv.new_empty(num_actual_tokens, mixed_qkv.shape[1])

        decode_slots = state_indices[:num_decodes]
        if num_decodes > 0:
            decode_conv_state = conv_state[decode_slots]
            # vLLM stores the trailing W - 1 inputs. FLA expects W entries but
            # ignores the first, so prepend a zero column and drop it on write.
            zero_padding = decode_conv_state.new_zeros(
                decode_conv_state.shape[0], decode_conv_state.shape[1], 1
            )
            conv_cache = torch.cat([zero_padding, decode_conv_state], dim=-1)
            decode_conv_output, conv_cache = _fla_causal_conv1d_update(
                mixed_qkv[:num_decode_tokens],
                conv_cache,
                weight=conv_weight,
                bias=conv_bias,
                activation=self.activation,
            )
            conv_state[decode_slots] = conv_cache[..., 1:].to(conv_state.dtype)
            conv_output[:num_decode_tokens] = decode_conv_output

        prefill_slots = None
        prefill_has_initial_state = None
        if num_prefills > 0:  # prefill, or mixed prefill decode
            assert gdn_metadata.prefill_state_indices is not None
            prefill_slots = gdn_metadata.prefill_state_indices
            prefill_has_initial_state = gdn_metadata.prefill_has_initial_state
            prefill_start = num_decode_tokens if num_decodes > 0 else 0
            # cu_seqlens must be 0-based within the prefill slice that the conv
            # kernel receives (mixed_qkv[prefill_start:]).
            if num_decodes == 0:  # pure prefill
                # No decode tokens in front, so the batch offsets are already
                # 0-based for the prefill slice.
                prefill_cu_seqlens = gdn_metadata.non_spec_query_start_loc
            else:
                # Mixed batch: prefill_query_start_loc holds absolute offsets that
                # start at num_decode_tokens, because decode tokens occupy the front
                # of the batch. Subtract the first offset (which equals
                # num_decode_tokens) to rebase the prefill slice's cu_seqlens to 0.
                assert gdn_metadata.prefill_query_start_loc is not None
                prefill_cu_seqlens = (
                    gdn_metadata.prefill_query_start_loc
                    - gdn_metadata.prefill_query_start_loc[0]
                )
            num_prefill_sequences = int(prefill_cu_seqlens.numel()) - 1
            # This core runs eager at the graph break, so checking whether any
            # prefix state must be restored does not enter a captured graph.
            has_continuations = prefill_has_initial_state is not None and bool(
                prefill_has_initial_state.any()
            )
            conv_initial_state = mixed_qkv.new_zeros(
                num_prefill_sequences, mixed_qkv.shape[1], self.conv_kernel_size
            )
            # Fresh prefills keep zero state; prefix-cache continuations restore
            # only the sequence slots identified by vLLM metadata.
            if has_continuations:
                resumed_slots = prefill_slots[prefill_has_initial_state]
                conv_initial_state[prefill_has_initial_state, :, 1:] = conv_state[
                    resumed_slots
                ]
            prefill_conv_output, conv_final_state = _fla_causal_conv1d(
                mixed_qkv[prefill_start:num_actual_tokens].unsqueeze(0),
                weight=conv_weight,
                bias=conv_bias,
                activation=self.activation,
                cu_seqlens=prefill_cu_seqlens,
                initial_state=conv_initial_state,
                output_final_state=True,
            )
            conv_state[prefill_slots] = conv_final_state[..., 1:].to(conv_state.dtype)
            conv_output[prefill_start:num_actual_tokens] = prefill_conv_output.squeeze(
                0
            )

        # Recurrence over the whole batch in one call. Decode (T=1) and prefill
        # (T>1) sequences run together, delimited by cu_seqlens; each gets a
        # materialized fp32 initial state. torchtitan already runs
        # recurrent-everywhere, and the kernel processes each sequence
        # independently from its cu_seqlens entry and initial-state row, so
        # merging the former per-segment calls is free.
        cu_seqlens = gdn_metadata.non_spec_query_start_loc[: num_sequences + 1]
        if num_prefills == 0:
            # Pure decode is captured in a CUDA graph, which forbids host syncs and
            # data-dependent (boolean-mask) indexing. Every decode sequence resumes,
            # so gather its paged SSM state directly with the integer slot indices.
            all_slots = decode_slots
            initial_state = (
                ssm_state[all_slots].transpose(-1, -2).to(torch.float32).contiguous()
            )
        else:
            # Prefill / mixed runs eager at the graph break, so boolean-mask gather
            # and the host sync it implies are allowed. A sequence resumes from
            # paged state iff it is a decode (always) or a prefill prefix-cache
            # continuation; fresh prefills keep a zero initial state.
            all_slots = (
                torch.cat([decode_slots, prefill_slots])
                if num_decodes > 0
                else prefill_slots
            )
            resumes_from_cache = torch.zeros(
                num_sequences, dtype=torch.bool, device=mixed_qkv.device
            )
            resumes_from_cache[:num_decodes] = True
            if prefill_has_initial_state is not None:
                resumes_from_cache[num_decodes:] = prefill_has_initial_state
            initial_state = conv_output.new_zeros(
                num_sequences,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=torch.float32,
            )
            initial_state[resumes_from_cache] = (
                ssm_state[all_slots[resumes_from_cache]]
                .transpose(-1, -2)
                .to(torch.float32)
            )
        recurrent_output = self._run_recurrence(
            conv_output,
            a,
            b,
            negative_exp_A,
            dt_bias,
            initial_state,
            ssm_state,
            all_slots,
            cu_seqlens,
        )
        output[:num_actual_tokens] = recurrent_output[0, :num_actual_tokens].to(
            output.dtype
        )

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
        """Bridge TorchTitan's layout to the flattened vLLM cache operation."""
        mesh = None
        placements = None
        # The enclosing module is head-sharded, but vLLM's paged cache and FLA
        # kernels operate on the rank-local head tensors.
        if isinstance(mixed_qkv, DTensor):
            mesh = mixed_qkv.device_mesh
            placements = mixed_qkv.placements
            mixed_qkv = mixed_qkv.to_local()
            a = a.to_local()
            b = b.to_local()
            conv_weight = conv_weight.to_local()
            A_log = A_log.to_local()
            dt_bias = dt_bias.to_local()

        batch_size, seq_len, num_channels = mixed_qkv.shape
        num_tokens = batch_size * seq_len
        mixed_qkv = mixed_qkv.reshape(num_tokens, num_channels)
        a = a.reshape(num_tokens, a.shape[-1])
        b = b.reshape(num_tokens, b.shape[-1])
        # Padded rows must remain defined across vLLM graph replays.
        output = mixed_qkv.new_zeros(
            num_tokens, self.local_num_v_heads, self.head_v_dim
        )
        self._forward(
            mixed_qkv,
            a,
            b,
            conv_weight,
            None,
            A_log,
            dt_bias,
            output,
        )
        output = output.reshape(
            batch_size,
            seq_len,
            self.local_num_v_heads,
            self.head_v_dim,
        )
        if mesh is not None:
            return DTensor.from_local(output, mesh, placements)
        return output
