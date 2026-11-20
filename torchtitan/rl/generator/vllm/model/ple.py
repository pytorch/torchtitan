# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generation PLE layer with request-local token and convolution history."""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch.distributed.functional_collectives import (
    all_gather_single,
    reduce_scatter_single,
    wait_tensor,
)
from torch.distributed.tensor import DTensor
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.qwen3_8_flash_next.embedding import NGramTable, PLELayer
from torchtitan.models.qwen3_8_flash_next.row_sharded_embedding import (
    prepare_row_sharded_embedding,
)
from torchtitan.rl.generator.local_compile import inference_local_compile
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionMetadata,
)
from torchtitan.rl.generator.vllm.attention.query_positions import vllm_query_requests

# T = packed tokens, C = residual channels, D = model dimension,
# S = residual streams, N = hash heads, E = embedding dimension,
# B = state slots, W = convolution window rows, I = token-id window entries,
# P = row owners, Q = P * T, R = local table rows, A = requests in the step.


class VLLMPLELayer(PLELayer):
    @dataclass(kw_only=True, slots=True)
    class Config(PLELayer.Config):
        state_metadata_prefix: str

    def __init__(self, config: Config):
        super().__init__(config)
        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Flash-Next PLE request history requires enable_prefix_caching=False"
            )
        if vllm_config.cache_config.mamba_cache_mode != "none":
            raise ValueError(
                "Flash-Next PLE request history follows its GDN state slot, which "
                "requires mamba_cache_mode='none'"
            )
        if vllm_config.speculative_config is not None:
            raise ValueError("Flash-Next PLE does not support speculative decoding")
        model_dtype = vllm_config.model_config.dtype
        assert isinstance(model_dtype, torch.dtype)
        self.model_dtype = model_dtype
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.window = max(self.conv_shifts())
        self.id_window = self.ple_embedding.ngram_size - 1
        self.state_metadata_prefix = config.state_metadata_prefix
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )
        self.conv_history_BWC: torch.Tensor | None = None
        self.id_history_BI: torch.Tensor | None = None

    def allocate_request_states(self) -> None:
        # TODO: File a vLLM issue for per-bucket KV-cache group sizes. vLLM's hybrid
        # KV-cache manager puts the same number of layers in every KV-cache group
        # (the smallest spec bucket), so a one-layer PLE state bucket would force one
        # layer per group for the whole model (73 groups for Flash-Next, each building
        # attention metadata every step). Until vLLM supports per-bucket group sizes,
        # PLE keeps its history in these model-owned buffers indexed by the state
        # slots of the linear-attention layer named by state_metadata_prefix, and
        # should move back to a vLLM KV-cache state once it does.
        source_state = self._static_forward_context[
            self.state_metadata_prefix
        ].kv_cache[0]
        num_slots = source_state.shape[0]
        self.conv_history_BWC = torch.zeros(
            num_slots,
            self.window,
            self.dim * self.num_streams,
            dtype=self.model_dtype,
            device=source_state.device,
        )
        self.id_history_BI = torch.zeros(
            num_slots, self.id_window, dtype=torch.int64, device=source_state.device
        )

    def forward(
        self,
        x_TC,
        *,
        input_ids_T,
        document_positions_T,
        routing=None,
        padding_mask_T=None,
    ):
        attn_metadata = get_forward_context().attn_metadata
        conv_state_BWC, id_state_BI = self.conv_history_BWC, self.id_history_BI
        embedding = self.ple_embedding
        coordinates = writes = None
        if attn_metadata is None:
            hashes_TN = embedding.hash_ids(
                input_ids_T, document_positions_T=document_positions_T
            )
        else:
            assert isinstance(attn_metadata, dict)
            assert conv_state_BWC is not None and id_state_BI is not None
            metadata = attn_metadata[self.state_metadata_prefix]
            assert isinstance(metadata, TorchTitanLinearAttentionMetadata)
            query_start_loc = metadata.non_spec_query_start_loc
            state_indices = metadata.non_spec_state_indices_tensor
            assert query_start_loc is not None and state_indices is not None
            hashes_TN, coordinates, writes = self._decode_hashes(
                input_ids_T,
                document_positions_T,
                query_start_loc,
                state_indices,
                id_state_BI,
                metadata.execution_path is LinearAttentionExecutionPath.PACKED,
            )
        table = embedding.ngram_embedding
        num_tokens = x_TC.shape[0]
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            num_tokens = int(dp_metadata.num_tokens_across_dp_cpu.max())
        if table.row_mesh is not None and num_tokens <= self.max_num_seqs:
            embedding_TE = embedding.gather_features(
                _gather_owned_rows(
                    table, hashes_TN, num_tokens=num_tokens, dtype=x_TC.dtype
                )
            )
        else:
            if table.row_mesh is not None:
                routing = prepare_row_sharded_embedding(
                    hashes_TN,
                    num_embeddings=table.num_embeddings,
                    row_mesh=table.row_mesh,
                    max_ids_per_owner=embedding.max_ids_per_owner(
                        num_tokens, num_owners=table.row_mesh.size()
                    ),
                )
            embedding_TE = embedding.lookup(
                hashes_TN, routing=routing, dtype=x_TC.dtype
            )
        key_TC = self.key_proj(embedding_TE)
        value_TD = self.value_proj(embedding_TE)
        gated_TC, normed_TC = self._decode_gate(
            self._decode_score(key_TC, x_TC), value_TD
        )
        if coordinates is None or writes is None:
            return torch.zeros_like(x_TC)
        assert conv_state_BWC is not None and id_state_BI is not None
        conv_rows, conv_states_T, conv_slots_T, ids_T, id_states_T, id_slots_T = writes
        output_TC, conv_rows_TC = self._decode_conv(
            gated_TC,
            normed_TC,
            document_positions_T,
            *coordinates,
            conv_state_BWC,
            conv_rows,
        )
        conv_state_BWC[conv_states_T, conv_slots_T] = conv_rows_TC
        id_state_BI[id_states_T, id_slots_T] = ids_T
        return output_TC

    @local_compile("ple_gate", batch_invariant=False)
    def _decode_score(self, key_TC, x_TC):
        return self.gate_score(key_TC, x_TC)

    @local_compile("ple_gate", batch_invariant=False)
    def _decode_gate(self, score_TS, value_TD):
        return self.gate_apply(score_TS, value_TD)

    @inference_local_compile(batch_invariant=True)
    def _decode_hashes(
        self,
        input_ids_T,
        positions_T,
        query_start_loc,
        state_indices,
        id_state_BI,
        packed: bool,
    ):
        embedding = self.ple_embedding
        rows_T, request_T, valid_T = vllm_query_requests(
            query_start_loc, state_indices.shape[0], input_ids_T.shape[0]
        )
        state_T = state_indices[request_T].long()
        starts_T = query_start_loc[request_T]
        valid_T = valid_T & (state_T != 0)
        shifted_T = [input_ids_T.long()]
        reset_T = torch.zeros_like(valid_T)
        for shift in range(1, embedding.ngram_size):
            previous_T = torch.where(
                rows_T - shift >= starts_T,
                input_ids_T[(rows_T - shift).clamp_min(0)].long(),
                id_state_BI[state_T, (positions_T - shift) % self.id_window],
            )
            previous_T = torch.where(
                (positions_T >= shift) & ~reset_T,
                previous_T,
                embedding.eos_token_id,
            )
            shifted_T.append(previous_T)
            reset_T = reset_T | (previous_T == embedding.eos_token_id)
        ids_T = shifted_T[0]
        if packed:
            states_A = state_indices.long()
            conv_rows, conv_states, conv_slots = _window_writes(
                positions_T, query_start_loc, states_A, self.window
            )
            id_rows, id_states, id_slots = _window_writes(
                positions_T, query_start_loc, states_A, self.id_window
            )
            writes = (
                conv_rows,
                conv_states,
                conv_slots,
                ids_T[id_rows],
                id_states,
                id_slots,
            )
        else:
            conv_states = state_T.masked_fill(~valid_T, 0)
            writes = (
                None,
                conv_states,
                (positions_T % self.window).masked_fill(~valid_T, 0),
                ids_T,
                conv_states,
                (positions_T % self.id_window).masked_fill(~valid_T, 0),
            )
        return embedding.mix_hashes(shifted_T), (state_T, starts_T, valid_T), writes

    @local_compile(
        "ple_conv",
        batch_invariant=True,
        options={"triton.prefer_nd_tiling": True, "coordinate_descent_tuning": True},
    )
    def _decode_conv(
        self,
        gated_TC,
        normed_TC,
        positions_T,
        state_T,
        starts_T,
        valid_T,
        conv_state_BWC,
        conv_rows,
    ):
        rows_T = torch.arange(gated_TC.shape[0], device=gated_TC.device)
        taps_TC = []
        for shift in self.conv_shifts():
            source_T = positions_T - shift
            shifted_TC = torch.where(
                (rows_T - shift >= starts_T)[:, None],
                normed_TC[(rows_T - shift).clamp_min(0)],
                conv_state_BWC[state_T, source_T % self.window],
            )
            taps_TC.append(shifted_TC.float().masked_fill((source_T < 0)[:, None], 0))
        output_TC = self.conv_output(gated_TC, taps_TC).masked_fill(
            ~valid_T[:, None], 0
        )
        return output_TC, normed_TC if conv_rows is None else normed_TC[conv_rows]


def _window_writes(positions_T, query_start_loc, states_A, window: int):
    offsets_W = torch.arange(window, device=positions_T.device) - window
    rows_AW = query_start_loc[1:, None] + offsets_W
    keep_AW = (rows_AW >= query_start_loc[:-1, None]) & (states_A != 0)[:, None]
    rows = rows_AW.clamp(0, positions_T.shape[0] - 1).flatten()
    keep = keep_AW.flatten()
    states = states_A[:, None].expand_as(rows_AW).flatten().masked_fill(~keep, 0)
    slots = (positions_T[rows] % window).masked_fill(~keep, 0)
    return rows, states, slots


def _gather_owned_rows(
    table: NGramTable,
    hashes_TN: torch.Tensor,
    *,
    num_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert table.row_mesh is not None
    weight_RE = cast(DTensor, table.weight).to_local()
    if table.row_mesh.size() == 1:
        return _owned_rows(hashes_TN, weight_RE, row_start=0, dtype=dtype)
    padding = num_tokens - hashes_TN.shape[0]
    assert padding >= 0
    hashes_PTN = wait_tensor(
        all_gather_single(
            F.pad(hashes_TN, (0, 0, 0, padding)) if padding else hashes_TN,
            0,
            table.row_mesh,
        )
    )
    rows_PTNE = _owned_rows(
        hashes_PTN,
        weight_RE,
        row_start=table.row_mesh.get_local_rank() * weight_RE.shape[0],
        dtype=dtype,
    )
    embedding_TNE = wait_tensor(
        reduce_scatter_single(rows_PTNE, "sum", 0, table.row_mesh)
    )
    return embedding_TNE[: hashes_TN.shape[0]]


@inference_local_compile(batch_invariant=True)
def _owned_rows(
    hashes_QN: torch.Tensor,
    weight_RE: torch.Tensor,
    *,
    row_start: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows_QN = hashes_QN - row_start
    owned_QN = (rows_QN >= 0) & (rows_QN < weight_RE.shape[0])
    rows_QNE = F.embedding(rows_QN.clamp(0, weight_RE.shape[0] - 1), weight_RE)
    return rows_QNE.to(dtype).masked_fill(~owned_QN.unsqueeze(-1), -0.0)
