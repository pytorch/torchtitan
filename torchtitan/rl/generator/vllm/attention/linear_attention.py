# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym's paged linear attention (GDN/KDA) for the native vLLM runner.

One attention backend and metadata builder serve both linear-attention families.
``configure_hybrid_state`` attaches vLLM's model-level recurrent-state interface;
each family supplies its state shapes, dtypes and copy functions.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import auto, Enum
from typing import Any, TypeVar

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode


class LinearAttentionExecutionPath(Enum):
    SINGLE_TOKEN = auto()
    # Without drafts: chunk kernel + decode kernel for 1-token rows. With drafts: prefill kernels on
    # every token, verify kernels on the first S tokens (see `_build_with_drafts`).
    PACKED = auto()
    # Drafts only: every row runs vLLM's verify kernels (rows with drafts, 1-token rows after context).
    VERIFY = auto()


@dataclass(kw_only=True)
class TorchTitanLinearAttentionMetadata(GDNAttentionMetadata):
    """vLLM's GDN metadata, which vLLM also uses for KDA, plus the execution path."""

    execution_path: LinearAttentionExecutionPath
    # Block-table column of each request's running state, ``[num_reqs, 1]``; the same
    # in every group, so ``update_block_table`` reuses it.
    state_block_column: torch.Tensor
    # Read by packed steps only: the decode requests' slots for the decode kernel;
    # the chunk kernel reads the inherited prefill_* fields (see `_stage_decode_split`).
    decode_state_indices: torch.Tensor | None = None
    # Drafts only: True on the tokens of rows the verify kernels run, over the tokens
    # those kernels see (see `TorchTitanLinearAttentionMetadataBuilder._build_with_drafts`).
    spec_token_mask: torch.Tensor | None = None


class TorchTitanLinearAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TORCHTITAN_LINEAR_ATTENTION"

    @staticmethod
    # The upstream abstract method has no return annotation.
    # pyrefly: ignore[bad-override]
    def get_builder_cls() -> type["TorchTitanLinearAttentionMetadataBuilder"]:
        return TorchTitanLinearAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


class TorchTitanLinearAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    """Stage shared request buffers before eager execution or native graph replay.

    A subclass of vLLM's GDN builder, so vLLM's runner passes it the draft and accepted
    counts (`GPUModelRunner._build_attention_metadata`). Every method and buffer it uses is
    its own; the parent's `__init__` sets `num_spec`, `use_spec_decode` and the batch-reorder
    threshold (1 + k with drafts, else 1).
    """

    # TODO: with the move to vLLM's V2 model runner (pytorch/torchtitan#5956), drop the subclass if V2
    # passes draft counts to any builder that asks. V1 passes them only to known builder classes
    # (GPUModelRunner._build_attention_metadata).

    supports_update_block_table = True

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        parallel = vllm_config.parallel_config
        # DP>1 keeps FULL: TorchTitanCudagraphDispatcher.dispatch replays the mixed
        # graph for padded decode ranks. Not in batch-invariant mode: the packed kernels
        # with cached state don't match the decode kernels bitwise (measured for GDN;
        # KDA assumed).
        # TODO: once they match, drop the batch-invariant clauses here and in
        # TorchTitanCudagraphDispatcher.dispatch.
        if parallel.data_parallel_size != 1 and is_in_batch_invariant_mode():
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        compilation = vllm_config.compilation_config
        if (
            compilation.cudagraph_mode.has_full_cudagraphs()
            and compilation.cudagraph_num_of_warmups < 1
        ):
            raise ValueError(
                "Linear attention (GDN/KDA) FULL capture requires at least one kernel warmup"
            )
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise ValueError(
                "Attention Gym linear attention (GDN/KDA) does not support "
                "intermediate prefix-cache checkpoints; use 'none' or 'align'"
            )
        self.num_reqs_capacity = vllm_config.scheduler_config.max_num_seqs
        # Always allocate one extra interval in query_start_loc (cu_seqlens)
        # so any tokens left in the padded buffer belong to the last, null request.
        self.query_start_loc = torch.zeros(
            self.num_reqs_capacity + 2, device=device, dtype=torch.int32
        )
        self.state_indices = torch.zeros(
            self.num_reqs_capacity + 1, device=device, dtype=torch.int32
        )
        self.has_initial_state = torch.zeros(
            self.num_reqs_capacity + 1, device=device, dtype=torch.bool
        )
        # Decode has one sequence per physical row, including null padding.
        self.decode_query_start_loc = torch.arange(
            self.num_reqs_capacity + 1, device=device, dtype=torch.int32
        )
        self.decode_state_indices = torch.zeros_like(self.state_indices)
        self.prefill_state_indices = torch.zeros_like(self.state_indices)
        self.prefill_query_start_loc = torch.zeros_like(self.query_start_loc)
        # "none" mode's state column (always 0), built once instead of in every build().
        self.zero_state_block_column = torch.zeros(
            self.num_reqs_capacity, 1, device=device, dtype=torch.long
        )
        # TorchTitanGPUModelRunner._determine_batch_execution_and_padding sets these before each build:
        # whether the step replays a packed or a uniform FULL graph, and in dummy runs whether the
        # dummy batch is uniform (None on real steps).
        self.replays_packed_graph = False
        self.replays_uniform_graph = False
        self.dummy_run_uniform: bool | None = None
        # TorchTitanGPUModelRunner.initialize_metadata_builders sets this to the runner's accepted
        # counts, which vLLM keeps current on every step but passes to build() only with drafts.
        self.runner_num_accepted_tokens: torch.Tensor | None = None
        if self.use_spec_decode:
            # The parent's spec_* buffers have decode_cudagraph_max_bs rows, which can be fewer
            # than a packed graph's request rows; these replace them under the same names.
            num_rows = self.num_reqs_capacity + 1
            self.spec_query_start_loc = torch.zeros(
                num_rows + 1, device=device, dtype=torch.int32
            )
            self.spec_state_indices_tensor = torch.zeros(
                num_rows, 1 + self.num_spec, device=device, dtype=torch.int32
            )
            self.num_accepted_tokens = torch.ones(
                num_rows, device=device, dtype=torch.int32
            )
            # The verify kernels see at most this many leading tokens (see _build_with_drafts).
            self.max_verify_tokens = (1 + self.num_spec) * self.num_reqs_capacity
            self.spec_token_mask = torch.zeros(
                self.max_verify_tokens, device=device, dtype=torch.bool
            )
            # Block-table columns of a row's 1 + k slots, after its running-state column.
            self.spec_block_offsets = torch.arange(1 + self.num_spec, device=device)

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        *,
        stage_packed: bool = False,
    ) -> TorchTitanLinearAttentionMetadata:
        m = common_attn_metadata
        capacity = min(self.num_reqs_capacity, m.num_actual_tokens)
        if not 0 <= m.num_reqs <= capacity:
            raise ValueError(
                "Linear attention (GDN/KDA) request count exceeds the batch's metadata capacity"
            )
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            max_state_seq_len = (
                m.block_table_tensor.shape[1] * self.kv_cache_spec.block_size
            )
            if m.max_seq_len > max_state_seq_len:
                raise ValueError(
                    "Linear attention (GDN/KDA) sequence length exceeds the state "
                    f"block-table capacity; got {m.max_seq_len} and "
                    f"{max_state_seq_len}. A request is at most max_model_len long, "
                    "and TorchTitanGPUModelRunner._dummy_run caps capture requests there."
                )
        # vLLM's `mamba_get_block_table_tensor`, one column: "align" keeps the
        # running state in the request's last block, "none" in its only one.
        seq_lens = m.seq_lens[: m.num_reqs]
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            state_block_column = (seq_lens - 1) // self.kv_cache_spec.block_size
            # 0-length requests in CUDA graph capture read column 0.
            state_block_column = state_block_column.clamp_(min=0).long().unsqueeze(1)
        else:
            state_block_column = self.zero_state_block_column[: m.num_reqs]
        self._stage_state_indices(m.block_table_tensor, state_block_column)
        self.query_start_loc[: m.num_reqs + 1].copy_(
            m.query_start_loc[: m.num_reqs + 1]
        )
        # Under FULL, num_actual_tokens is the padded capacity, while the last
        # query offset is the real token count. The appended interval is null.
        self.query_start_loc[m.num_reqs + 1 :].fill_(m.num_actual_tokens)
        self.has_initial_state[: m.num_reqs].copy_(
            m.compute_num_computed_tokens()[: m.num_reqs] > 0
        )
        self.has_initial_state[m.num_reqs :].zero_()
        if self.use_spec_decode:
            return self._build_with_drafts(
                m,
                capacity,
                state_block_column,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
            )
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
            if m.num_reqs
            else (0, 0, 0, 0)
        )
        # If every request is processing one token, use the single-token
        # (decode-style) path, unless the runner picked a packed FULL graph for the
        # step. Building it as the packed step makes update_block_table restage its
        # offsets too.
        single_token = m.max_query_len == 1 and not self.replays_packed_graph
        # A FULL dummy run can replay the mixed-batch graph too (stage_packed).
        decode_split = (
            self._stage_decode_split(num_decodes, capacity)
            if stage_packed or not single_token
            else {}
        )
        return TorchTitanLinearAttentionMetadata(
            execution_path=(
                LinearAttentionExecutionPath.SINGLE_TOKEN
                if single_token
                else LinearAttentionExecutionPath.PACKED
            ),
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            # No drafts here: steps with drafts take _build_with_drafts; KDA rejects drafts.
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=m.num_actual_tokens,
            non_spec_query_start_loc=(
                self.decode_query_start_loc[: capacity + 1]
                if single_token
                else self.query_start_loc[: capacity + 2]
            ),
            non_spec_state_indices_tensor=(
                self.state_indices[:capacity]
                if single_token
                else self.state_indices[: capacity + 1]
            ),
            has_initial_state=(
                self.has_initial_state[:capacity]
                if single_token
                else self.has_initial_state[: capacity + 1]
            ),
            state_block_column=state_block_column,
            **decode_split,
        )

    def _build_with_drafts(
        self,
        m: CommonAttentionMetadata,
        capacity: int,
        state_block_column: torch.Tensor,
        num_accepted_tokens: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> TorchTitanLinearAttentionMetadata:
        """Stage a step with drafts so one graph runs any mix of rows.

        vLLM's verify kernels run the rows with drafts, and the 1-token rows after context
        (they resume a state where the last verify left it). Attention Gym's prefill kernels
        run the other rows. Each kernel gets the null slot (0) on the rows it doesn't own.

        Example: k = 3, rows [verify 4, verify 4, 1-token resume, prefill 13], 24 padded
        tokens, capacity 5, so S = min(24, (1 + 3) x 5) = 20:

            query_start_loc          [0, 4, 8, 9, 22, 24, 24]   row 4: padding (null)
            spec_query_start_loc     [0, 4, 8, 9, 20, 20, 20]   clamped at S
            prefill_query_start_loc  [0, 9, 9, 9, 22, 24, 24]   rows 0-2 merged into one null interval
            spec_state_indices       rows 0-2: their 1 + k slots; others 0
            prefill_state_indices    row 3: its slot; others 0
            spec_token_mask          tokens 0-8 True, of the first S

            VERIFY  uniform graph, uniform dummy run, or an eager step with no prefill row
            PACKED  any other step: prefill kernels on every token, verify kernels on the first S

        The batch puts rows with context and at most 1 + k tokens first (the runner's reorder
        threshold, checked in `TorchTitanGPUModelRunner.calculate_reorder_batch_threshold`), so
        the rows that run the verify kernels end within S. Dummy runs ignore vLLM's stale counts:
        null slots, accepted counts of 1.
        """
        num_reqs = m.num_reqs
        num_tokens = m.num_actual_tokens
        num_verify_tokens = min(num_tokens, self.max_verify_tokens)
        query_start_loc_cpu = m.query_start_loc_cpu[: num_reqs + 1].numpy()
        query_lens = np.diff(query_start_loc_cpu)

        # ======== Classify rows on the CPU ========
        is_dummy_run = self.dummy_run_uniform is not None
        if is_dummy_run:
            runs_verify_kernels = np.zeros(num_reqs, dtype=bool)
        else:
            num_decode_draft_tokens = (
                np.full(num_reqs, -1)
                if num_decode_draft_tokens_cpu is None
                else num_decode_draft_tokens_cpu[:num_reqs].numpy()
            )
            # A step whose drafts the scheduler all trimmed: a 1-token row that resumes after
            # accepting j > 1 tokens still reads its state from slot j.
            if num_accepted_tokens is None:
                if self.runner_num_accepted_tokens is None:
                    raise RuntimeError(
                        "Linear attention (GDN) with drafts needs the runner's accepted counts "
                        "on steps without drafts (TorchTitanGPUModelRunner)"
                    )
                num_accepted_tokens = self.runner_num_accepted_tokens

            if m.seq_lens_cpu_upper_bound is None:
                raise RuntimeError(
                    "Linear attention (GDN) with drafts needs seq_lens_cpu_upper_bound "
                    "to find the 1-token rows that resume a state"
                )
            context_lens = m.seq_lens_cpu_upper_bound[:num_reqs].numpy() - query_lens

            no_context_draft_rows = np.flatnonzero(
                (num_decode_draft_tokens >= 0) & (context_lens == 0)
            )
            if no_context_draft_rows.size:
                # TODO: run such a row on the prefill kernels and drop its placeholder drafts' states.
                raise RuntimeError(
                    f"Row {int(no_context_draft_rows[0])} has drafts but no context: a 1-token "
                    "prompt that vLLM's scheduler padded with placeholder drafts. GDN's verify "
                    "kernels would read its slot's old state, so it is not supported"
                )

            runs_verify_kernels = (num_decode_draft_tokens >= 0) | (
                (query_lens == 1) & (context_lens > 0)
            )

            # A uniform graph runs only the verify kernels, so it must not meet another row.
            if self.replays_uniform_graph:
                other_rows = np.flatnonzero(
                    (query_lens > 0) & (num_decode_draft_tokens != self.num_spec)
                )
                if other_rows.size:
                    row = int(other_rows[0])
                    raise RuntimeError(
                        f"A uniform verify graph met row {row} with {int(query_lens[row])} tokens and "
                        f"{int(num_decode_draft_tokens[row])} drafts; it runs only rows with "
                        f"{self.num_spec} drafts (see "
                        "TorchTitanGPUModelRunner._determine_batch_execution_and_padding)"
                    )

            # The verify kernels see only the first num_verify_tokens tokens.
            late_rows = np.flatnonzero(
                runs_verify_kernels & (query_start_loc_cpu[1:] > num_verify_tokens)
            )
            if late_rows.size:
                row = int(late_rows[0])
                raise RuntimeError(
                    f"Row {row} runs GDN's verify kernels (it has drafts, or 1 token after context) "
                    f"but ends at token {int(query_start_loc_cpu[row + 1])}, past the "
                    f"{num_verify_tokens} tokens those kernels see. Rows with context and at most "
                    "1 + k tokens should lead the batch (reorder threshold 1 + k)"
                )
        runs_prefill_kernels = ~runs_verify_kernels & (query_lens > 0)

        # A replayed graph decides first (an idle DP rank's dummy batch can replay the packed graph).
        if self.replays_packed_graph:
            path = LinearAttentionExecutionPath.PACKED
        elif self.replays_uniform_graph or self.dummy_run_uniform:
            path = LinearAttentionExecutionPath.VERIFY
        elif is_dummy_run:
            path = LinearAttentionExecutionPath.PACKED
        else:
            path = (
                LinearAttentionExecutionPath.VERIFY
                if not runs_prefill_kernels.any()
                and num_tokens <= self.max_verify_tokens
                else LinearAttentionExecutionPath.PACKED
            )
        num_rows = (
            num_reqs if path is LinearAttentionExecutionPath.VERIFY else capacity + 1
        )

        # ======== Stage the slots and accepted counts on the GPU ========
        row_masks = np.zeros((2, num_rows), dtype=bool)
        row_masks[0, :num_reqs] = runs_verify_kernels
        row_masks[1, :num_reqs] = runs_prefill_kernels
        runs_verify_kernels_gpu, runs_prefill_kernels_gpu = async_tensor_h2d(
            row_masks, device=self.device
        )
        spec_slots = torch.gather(
            m.block_table_tensor[:num_reqs],
            1,
            state_block_column + self.spec_block_offsets,
        )
        self.spec_state_indices_tensor[:num_reqs].copy_(
            torch.where(runs_verify_kernels_gpu[:num_reqs, None], spec_slots, 0)
        )
        self.spec_state_indices_tensor[num_reqs:].zero_()
        self.prefill_state_indices[:num_rows].copy_(
            torch.where(runs_prefill_kernels_gpu, self.state_indices[:num_rows], 0)
        )
        self.prefill_state_indices[num_rows:].zero_()
        self.num_accepted_tokens.fill_(1)
        if num_accepted_tokens is not None and not is_dummy_run:
            self.num_accepted_tokens[:num_reqs].copy_(
                torch.where(
                    runs_verify_kernels_gpu[:num_reqs],
                    num_accepted_tokens[:num_reqs],
                    1,
                )
            )

        # ======== Stage the offsets and the token mask ========
        torch.clamp(
            self.query_start_loc[: num_rows + 1],
            max=num_verify_tokens,
            out=self.spec_query_start_loc[: num_rows + 1],
        )
        prefill_query_start_loc = np.full(num_rows + 1, num_tokens, dtype=np.int32)
        prefill_query_start_loc[: num_reqs + 1] = query_start_loc_cpu
        # In each run of adjacent rows that run the verify kernels, the later rows start where the
        # run ends, so the chunk kernel sees the run as one null interval (row-aligned, as in
        # _stage_decode_split).
        is_later_in_run = runs_verify_kernels & np.r_[False, runs_verify_kernels[:-1]]
        is_last_in_run = runs_verify_kernels & ~np.r_[runs_verify_kernels[1:], False]
        row_index = np.arange(num_reqs)
        run_last_row = np.minimum.accumulate(
            np.where(is_last_in_run, row_index, num_reqs)[::-1]
        )[::-1]
        prefill_query_start_loc[:num_reqs][is_later_in_run] = query_start_loc_cpu[
            run_last_row[is_later_in_run] + 1
        ]
        self.prefill_query_start_loc[: num_rows + 1].copy_(
            async_tensor_h2d(prefill_query_start_loc, device=self.device)
        )
        spec_token_mask = np.zeros(num_verify_tokens, dtype=bool)
        verify_tokens = np.repeat(runs_verify_kernels, query_lens)[:num_verify_tokens]
        spec_token_mask[: verify_tokens.shape[0]] = verify_tokens
        self.spec_token_mask[:num_verify_tokens].copy_(
            async_tensor_h2d(spec_token_mask, device=self.device)
        )

        return TorchTitanLinearAttentionMetadata(
            execution_path=path,
            num_prefills=int(runs_prefill_kernels.sum()),
            num_prefill_tokens=int(query_lens[runs_prefill_kernels].sum()),
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=int(runs_verify_kernels.sum()),
            num_spec_decode_tokens=int(query_lens[runs_verify_kernels].sum()),
            num_actual_tokens=num_tokens,
            has_initial_state=self.has_initial_state[:num_rows],
            spec_query_start_loc=self.spec_query_start_loc[: num_rows + 1],
            spec_state_indices_tensor=self.spec_state_indices_tensor[:num_rows],
            num_accepted_tokens=self.num_accepted_tokens[:num_rows],
            prefill_query_start_loc=self.prefill_query_start_loc[: num_rows + 1],
            prefill_state_indices=self.prefill_state_indices[:num_rows],
            spec_token_mask=self.spec_token_mask[:num_verify_tokens],
            state_block_column=state_block_column,
        )

    def _stage_decode_split(
        self, num_decodes: int, capacity: int
    ) -> dict[str, torch.Tensor]:
        """Stage the slots and offsets that send decode requests to the decode kernel.

        vLLM puts decode requests (one token each) first. The chunk kernel sees their
        tokens as one sequence with the null slot. Example, 2 decode requests then a
        3-token prefill, slots 5, 7, 9, capacity 4:

            decode_state_indices     [5, 7, 0, 0]        null past the decode requests
            prefill_query_start_loc  [0, 2, 2, 5, 5, 5]  decode tokens as one sequence
            prefill_state_indices    [0, 0, 9, 0, 0]     ... with the null slot

        - FULL graphs need fixed extents, so prefill_* keep every row (vLLM's builder
          slices them to the prefill requests).
        - prefill_has_initial_state stays None: the chunk kernel reads the shared
          ``has_initial_state`` (null slots ignore it).
        - Restaged only by builds that can replay a packed graph (see ``build``).
        """
        prefill_query_start_loc = self.prefill_query_start_loc[: capacity + 2]
        torch.clamp(
            self.query_start_loc[: capacity + 2],
            min=num_decodes,
            out=prefill_query_start_loc,
        )
        prefill_query_start_loc[0] = 0
        self.decode_state_indices.copy_(self.state_indices)
        self.decode_state_indices[num_decodes:].zero_()
        self.prefill_state_indices.copy_(self.state_indices)
        self.prefill_state_indices[:num_decodes].zero_()
        return dict(
            decode_state_indices=self.decode_state_indices[:capacity],
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=self.prefill_state_indices[: capacity + 1],
        )

    def update_block_table(
        self,
        metadata: TorchTitanLinearAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> TorchTitanLinearAttentionMetadata:
        """Reuse another group's metadata for this group's block table.

        vLLM calls this instead of ``build`` for the step's later groups with the same
        spec. Only the state slots differ; the other buffers are copied into this
        builder's own, which its CUDA graphs captured.
        """
        del slot_mapping
        self._stage_state_indices(blk_table, metadata.state_block_column)
        if self.use_spec_decode:
            return self._update_block_table_with_drafts(metadata, blk_table)
        extent = metadata.has_initial_state.shape[0]
        self.has_initial_state[:extent].copy_(metadata.has_initial_state)
        # Single-token metadata runs eager or replays decode graphs, both reading the
        # static decode offsets, so only the packed path copies the query offsets.
        query_start_loc = self.decode_query_start_loc[: extent + 1]
        decode_split = {}
        if metadata.execution_path is LinearAttentionExecutionPath.PACKED:
            query_start_loc = self.query_start_loc[: extent + 1]
            query_start_loc.copy_(metadata.non_spec_query_start_loc)
            decode_split = self._stage_decode_split(
                metadata.num_decodes, metadata.decode_state_indices.shape[0]
            )
        return replace(
            metadata,
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=self.state_indices[:extent],
            has_initial_state=self.has_initial_state[:extent],
            **decode_split,
        )

    def _update_block_table_with_drafts(
        self, metadata: TorchTitanLinearAttentionMetadata, block_table: torch.Tensor
    ) -> TorchTitanLinearAttentionMetadata:
        """`update_block_table` for `_build_with_drafts` metadata: this group's slots on the
        rows where the first group's slots are real, then copies of every other buffer.

            first group:  spec_state_indices [[1, 2, 3, 4], [0, 0, 0, 0]]  prefill_state_indices [0, 5]
            this group:   spec_state_indices [[7, 8, 9, 10], [0, 0, 0, 0]] prefill_state_indices [0, 11]
        """
        num_reqs = metadata.state_block_column.shape[0]
        num_rows = metadata.has_initial_state.shape[0]
        num_verify_tokens = metadata.spec_token_mask.shape[0]
        spec_slots = torch.gather(
            block_table[:num_reqs],
            1,
            metadata.state_block_column + self.spec_block_offsets,
        )
        self.spec_state_indices_tensor[:num_reqs].copy_(
            torch.where(
                metadata.spec_state_indices_tensor[:num_reqs] != 0, spec_slots, 0
            )
        )
        self.spec_state_indices_tensor[num_reqs:].zero_()
        self.prefill_state_indices[:num_rows].copy_(
            torch.where(
                metadata.prefill_state_indices != 0, self.state_indices[:num_rows], 0
            )
        )
        self.prefill_state_indices[num_rows:].zero_()
        copies = {
            "has_initial_state": self.has_initial_state[:num_rows],
            "spec_query_start_loc": self.spec_query_start_loc[: num_rows + 1],
            "num_accepted_tokens": self.num_accepted_tokens[:num_rows],
            "prefill_query_start_loc": self.prefill_query_start_loc[: num_rows + 1],
            "spec_token_mask": self.spec_token_mask[:num_verify_tokens],
        }
        for name, buffer in copies.items():
            buffer.copy_(getattr(metadata, name))
        return replace(
            metadata,
            spec_state_indices_tensor=self.spec_state_indices_tensor[:num_rows],
            prefill_state_indices=self.prefill_state_indices[:num_rows],
            **copies,
        )

    def _stage_state_indices(
        self, block_table: torch.Tensor, state_block_column: torch.Tensor
    ) -> None:
        """Write each request's state slot; zero the rows past the requests (null slot)."""
        num_reqs = state_block_column.shape[0]
        torch.gather(
            block_table[:num_reqs],
            1,
            state_block_column,
            out=self.state_indices[:num_reqs].unsqueeze(1),
        )
        self.state_indices[num_reqs:].zero_()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TorchTitanLinearAttentionMetadata:
        if self.use_spec_decode and self.dummy_run_uniform is None:
            raise RuntimeError(
                "Linear attention (GDN) capture with drafts needs "
                "TorchTitanGPUModelRunner._determine_batch_execution_and_padding to set dummy_run_uniform"
            )
        # vLLM's FULL dummy runs build here too. At DP>1 a one-token dummy can replay
        # the mixed-batch graph at any padded size, so stage the packed buffers.
        metadata = self.build(0, common_attn_metadata, stage_packed=True)
        self.state_indices.zero_()
        self.decode_state_indices.zero_()
        self.prefill_state_indices.zero_()
        self.has_initial_state.zero_()
        return metadata


_StateShapeT = TypeVar("_StateShapeT", bound=tuple)


def unique_state_shape(
    family: str, state_shapes: Iterable[_StateShapeT]
) -> _StateShapeT | None:
    """Return the one state shape shared by every layer of ``family``.

    vLLM exposes one model-level recurrent-state shape, so every field that
    determines it must match across layers. Returns None for no layers.
    """
    unique = set(state_shapes)
    if not unique:
        return None
    if len(unique) != 1:
        raise ValueError(
            f"All {family} layers must use the same state shape, got {unique}"
        )
    (state_shape,) = unique
    return state_shape


def configure_hybrid_state(
    model_cls: type,
    *,
    get_state_shape: Callable[[VllmConfig], tuple],
    get_state_dtype: Callable[[VllmConfig], tuple],
    get_state_copy_func: Callable[[], tuple],
) -> None:
    """Attach vLLM's hybrid recurrent-state interface to ``model_cls``."""

    def get_state_copy_funcs(cls, mamba_types: Iterable[Any]) -> dict[Any, tuple]:
        copy_funcs = cls.get_mamba_state_copy_func()
        return {mamba_type: copy_funcs for mamba_type in mamba_types}

    model_cls.is_hybrid = True
    model_cls.get_mamba_state_shape_from_config = classmethod(
        lambda cls, vllm_config: get_state_shape(vllm_config)
    )
    model_cls.get_mamba_state_dtype_from_config = classmethod(
        lambda cls, vllm_config: get_state_dtype(vllm_config)
    )
    model_cls.get_mamba_state_copy_func = classmethod(lambda cls: get_state_copy_func())
    model_cls.get_mamba_state_copy_funcs = classmethod(get_state_copy_funcs)
