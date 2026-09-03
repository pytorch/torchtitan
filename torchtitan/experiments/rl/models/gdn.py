# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM paged-cache adapter for TorchTitan's Gated DeltaNet.

The enclosing Qwen3.5 module owns all parameters. This adapter runs Attention
Gym's paging-aware convolution and GDN kernels.

Batch-invariant execution has two additional requirements:

* The accumulated SSM cache state uses float32. Decode otherwise rounds the state
  through bfloat16 after every token, unlike a single prefill call. The convolution
  cache stays in model dtype because it only stores trailing input columns.
* Batch-invariant recurrence uses the same Attention Gym scan as the trainer.

Decode and prefill update the paged convolution and SSM state pools directly.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from attn_gym.linear import (
    causal_conv1d_decode,
    l2norm,
    paged_causal_conv1d,
    paged_chunk_gdn,
    recurrent_gdn,
    recurrent_gdn_decode,
)

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


class VLLMInnerGatedDeltaNet(Module, MambaBase):
    """Paged-cache inner GDN implementation.

    The enclosing ``qwen3_5.gdn.GatedDeltaNet`` owns all parameters. This
    module owns only vLLM cache plumbing and kernel dispatch.

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
        # vLLM speculative decoding retains one state per draft position and
        # commits the accepted state; Attention Gym currently mutates one slot.
        if self.num_speculative_tokens != 0:
            raise ValueError("Attention Gym GDN does not support speculative decoding.")

        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size

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

        if is_conv_state_dim_first():
            raise ValueError(
                "Attention Gym GDN requires VLLM_SSM_CONV_STATE_LAYOUT=SD so "
                "the paged convolution history has contiguous channels."
            )

        # Attention Gym's paged kernels mutate the SSM state pool directly and
        # require FP32 state in both regular and batch-invariant execution.
        if self.cache_config.mamba_ssm_cache_dtype not in {"auto", "float32"}:
            raise ValueError(
                "Attention Gym GDN requires mamba_ssm_cache_dtype='float32', "
                f"got {self.cache_config.mamba_ssm_cache_dtype!r}."
            )
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
        """Split local fused channels and add a singleton batch dim."""
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
        ), "VLLMInnerGatedDeltaNet does not support speculative decoding"

        num_actual_tokens = gdn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        a = a[:num_actual_tokens]
        b = b[:num_actual_tokens]

        state_indices = gdn_metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        ssm_state = self.kv_cache[1]
        conv_state = self.kv_cache[0]
        assert conv_bias is None
        dt_bias = dt_bias.float()
        num_decodes = gdn_metadata.num_decodes
        num_prefills = gdn_metadata.num_prefills
        num_decode_tokens = gdn_metadata.num_decode_tokens
        num_sequences = num_decodes + num_prefills

        # Convolution is split by request type and writes one contiguous
        # conv_output for the single recurrence below. Decode uses the specialized
        # single-token state update, while prefill uses the packed multi-token
        # operation. vLLM orders tokens decode-first, then prefill.
        conv_output = mixed_qkv.new_empty(num_actual_tokens, mixed_qkv.shape[1])

        decode_slots = state_indices[:num_decodes]
        if num_decodes > 0:  # pure decode, or mixed prefill decode
            decode_conv_output = causal_conv1d_decode(
                mixed_qkv[:num_decode_tokens].contiguous(),
                conv_weight,
                conv_state,
                activation="silu",
                state_indices=decode_slots,
            )
            conv_output[:num_decode_tokens] = decode_conv_output

        prefill_slots = None
        prefill_has_initial_state = None
        if num_prefills > 0:  # prefill, or mixed prefill decode
            assert gdn_metadata.prefill_state_indices is not None
            prefill_slots = gdn_metadata.prefill_state_indices
            prefill_has_initial_state = gdn_metadata.prefill_has_initial_state
            assert (
                prefill_has_initial_state is not None
            ), "prefill_has_initial_state is required when num_prefills > 0"
            prefill_start = num_decode_tokens if num_decodes > 0 else 0
            # cu_seqlens must be 0-based within the prefill slice that the conv
            # kernel receives (mixed_qkv[prefill_start:]).
            if num_decodes == 0:
                # No decode tokens in front, so the batch offsets are already
                # 0-based for the prefill slice.
                prefill_cu_seqlens = gdn_metadata.non_spec_query_start_loc
            else:
                # Mixed-batch prefill metadata is already rebased to the prefill slice.
                assert gdn_metadata.prefill_query_start_loc is not None
                prefill_cu_seqlens = gdn_metadata.prefill_query_start_loc
            prefill_conv_output = paged_causal_conv1d(
                mixed_qkv[prefill_start:num_actual_tokens].unsqueeze(0),
                conv_weight,
                conv_state,
                prefill_slots,
                activation="silu",
                cu_seqlens=prefill_cu_seqlens,
                has_initial_state=prefill_has_initial_state,
            )
            conv_output[prefill_start:num_actual_tokens] = prefill_conv_output.squeeze(
                0
            )

        # Recurrence over the whole batch in one call. The batch-invariant path
        # addresses the paged SSM pool directly.
        cu_seqlens = gdn_metadata.non_spec_query_start_loc[: num_sequences + 1]
        if num_prefills == 0:
            all_slots = decode_slots
            has_initial_state = None
        else:
            all_slots = (
                torch.cat([decode_slots, prefill_slots])
                if num_decodes > 0
                else prefill_slots
            )
            has_initial_state = torch.ones(
                num_sequences, dtype=torch.bool, device=mixed_qkv.device
            )
            has_initial_state[num_decodes:] = prefill_has_initial_state

        batch_invariant = is_in_batch_invariant_mode()
        if num_prefills == 0 and not batch_invariant:
            # Pure decode fuses QKV splitting, gate activation, normalization,
            # and the recurrent update into one kernel.
            recurrent_gdn_decode(
                conv_output[:num_decode_tokens],
                a[:num_decode_tokens].unsqueeze(0),
                b[:num_decode_tokens].unsqueeze(0),
                A_log.float(),
                dt_bias,
                ssm_state,
                all_slots,
                scale=self.head_k_dim**-0.5,
                out=output[:num_decode_tokens].unsqueeze(0),
            )
            return

        query, key, value = self._split_qkv(conv_output)
        decay = (-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias)).unsqueeze(
            0
        )
        update_gate = torch.sigmoid(b).unsqueeze(0)
        query = l2norm(query, cu_seqlens=cu_seqlens)
        key = l2norm(key, cu_seqlens=cu_seqlens)

        if batch_invariant:
            recurrent_output, _ = recurrent_gdn(
                query,
                key,
                value,
                decay,
                update_gate,
                ssm_state,
                cu_seqlens=cu_seqlens,
                scale=self.head_k_dim**-0.5,
                state_indices=all_slots,
                has_initial_state=has_initial_state,
                # Triton autotuning breaks batch invariance.
                autotune=False,
            )
        else:
            recurrent_output = paged_chunk_gdn(
                query,
                key,
                value,
                decay,
                update_gate,
                ssm_state,
                all_slots,
                cu_seqlens=cu_seqlens,
                has_initial_state=has_initial_state,
                scale=self.head_k_dim**-0.5,
            )
        output[:num_actual_tokens] = recurrent_output[0, :num_actual_tokens].to(
            output.dtype
        )

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        use_varlen_kernels: bool = False,
    ) -> torch.Tensor:
        """Run the flattened vLLM cache operation on rank-local tensors."""
        assert key_head_dim == self.head_k_dim
        assert value_head_dim == self.head_v_dim
        mixed_qkv_TC = torch.cat([query_TC, key_TC, value_TC], dim=-1)
        conv_weight_CW = torch.cat(
            [conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W],
            dim=0,
        ).squeeze(1)
        assert conv_weight_CW.shape[-1] == self.conv_kernel_size

        num_tokens = mixed_qkv_TC.shape[0]
        # Padded rows must remain defined across vLLM graph replays.
        output_THV = mixed_qkv_TC.new_zeros(
            num_tokens, self.local_num_v_heads, self.head_v_dim
        )
        self._forward(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            conv_weight_CW,
            None,
            A_log_H,
            dt_bias_H,
            output_THV,
        )
        return output_THV
