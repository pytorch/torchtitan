# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

from collections.abc import Set
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.utils.math_utils import round_up
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    TorchTitanLinearAttentionBackend,
    TorchTitanLinearAttentionMetadataBuilder,
)
from torchtitan.tools.utils import device_type


class TorchTitanCudagraphDispatcher(CudagraphDispatcher):
    """Give decode batches their own FULL graphs, apart from prefill and mixed batches.

    vLLM at 9f03b51 keys plain FULL graphs by token count only: four one-token
    decodes replay the mixed-batch graph captured with ``max_query_len = 4``.
    Decode needs its own graph:
      - Attention Gym linear attention (GDN/KDA) runs different decode kernels.
        Fresh requests stay on the single-token key and kernels: their zero
        initial state is a GPU mask.
      - FA4 sizes its grid from ``max_query_len``: decode attention runs 1.3-2.0x
        longer from 32 sequences at head dim 64 and from 64 at head dim 128
        (GB300, 16-token pages).
      - With k drafts, a decode batch verifies 1 + k tokens per row: GDN runs only its
        verify kernels, and FA4 its verify-step path (``TorchTitanVarlenInnerAttentionImpl.forward``).

    Uniform-decode batch in FULL mode:
      num_tokens <= max_decode_capture_size              -> decode graph
      larger (DP padding, or a capture-size gap)         -> mixed graph, as in stock FULL
      larger, batch-invariant, eager allowed             -> eager on the decode kernels

    Decode keys come from the native FULL_DECODE_ONLY dispatcher; native capture,
    replay and graph ownership stay intact.

    This costs one additional graph capture per decode-specialized batch size
    (83 at max_num_seqs 1024; with 3 drafts, 145 verify graphs at max_num_seqs 512).
    Without drafts, ``TorchTitanGPUModelRunner.initialize_kv_cache`` keeps them out of
    vLLM's graph-memory estimate.

    TODO: drop this class once vLLM captures FULL decode graphs next to FULL mixed
    graphs without piecewise compilation.
    """

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ) -> None:
        super().initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
        if cudagraph_mode == CUDAGraphMode.FULL:
            self.decode_dispatcher = CudagraphDispatcher(self.vllm_config)
            self.decode_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.FULL_DECODE_ONLY, uniform_decode_query_len
            )
            self.max_decode_capture_size = 0
            for mode, descriptors in self.decode_dispatcher.get_capture_descs():
                for descriptor in descriptors:
                    self.add_cudagraph_key(mode, descriptor)
                    self.max_decode_capture_size = max(
                        self.max_decode_capture_size, descriptor.num_tokens
                    )

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: Set[CUDAGraphMode] | None = None,
        invalid_modes: Set[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        dispatch = super().dispatch
        if (
            self.keys_initialized
            and self.cudagraph_mode == CUDAGraphMode.FULL
            and uniform_decode
        ):
            if num_tokens <= self.max_decode_capture_size:
                dispatch = self.decode_dispatcher.dispatch
            elif is_in_batch_invariant_mode() and (
                not valid_modes or CUDAGraphMode.NONE in valid_modes
            ):
                # GDN/KDA's packed kernels in the mixed graph don't match decode bitwise.
                return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
        return dispatch(
            num_tokens,
            uniform_decode,
            has_lora,
            num_active_loras,
            valid_modes,
            invalid_modes,
        )


class TorchTitanGPUModelRunner(GPUModelRunner):
    """V1 runner with sequence-parallel padding and FULL-graph decode specialization."""

    # The last step's accepted-token count per request, kept by _update_states for _prepare_inputs.
    _accepted_counts_by_request: dict[str, int] | None = None

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        assert not self.cudagraph_dispatcher.keys_initialized
        self.cudagraph_dispatcher = TorchTitanCudagraphDispatcher(self.vllm_config)
        # Before vLLM resolves the mode against the attention backends (capture_model checks it).
        self.requested_cudagraph_mode = self.compilation_config.cudagraph_mode

    def capture_model(self) -> int:
        """Check the graph modes drafts rely on, then capture.

        mode            GDN with drafts must not run inside a piecewise graph: its piecewise
                        capture has no metadata, so it would replay zeros. A requested FULL
                        must stay FULL (vLLM lowers it when a backend lacks ALWAYS support)
        drafter graphs  under a requested FULL, the drafter must be breakable, or it runs
                        eagerly; TorchTitanEagleProposer.dummy_run captures its graphs
        """
        drafts = self.uniform_decode_query_len > 1
        if drafts and self._linear_attention_builders():
            resolved_mode = self.compilation_config.cudagraph_mode
            if resolved_mode.has_piecewise_cudagraphs() or (
                self.requested_cudagraph_mode == CUDAGraphMode.FULL
                and resolved_mode != CUDAGraphMode.FULL
            ):
                raise RuntimeError(
                    f"GDN with drafts can't run under cudagraph_mode {resolved_mode} (requested "
                    f"{self.requested_cudagraph_mode}): it must not run inside piecewise graphs, "
                    "and a requested FULL must stay FULL. vLLM lowers FULL when an attention "
                    "backend lacks AttentionCGSupport.ALWAYS"
                )

        drafter = getattr(self, "drafter", None)
        if (
            drafter is not None
            and self.requested_cudagraph_mode == CUDAGraphMode.FULL
            and not isinstance(drafter.model, BreakableCUDAGraphWrapper)
        ):
            raise RuntimeError(
                "cuda_graph.mode='FULL' with drafts needs VLLM_USE_BREAKABLE_CUDAGRAPH=1 before vLLM "
                "imports its attention op (VLLMEngine sets it); the MTP drafter would run eagerly"
            )
        return super().capture_model()

    def initialize_metadata_builders(
        self, kv_cache_config, kernel_block_sizes: list[int]
    ) -> None:
        super().initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
        # vLLM passes accepted counts to builders only on steps with drafts, but refreshes this buffer in
        # batch order on every step (GPUModelRunner._prepare_inputs, before its drafts branch). GDN with
        # drafts reads it on steps whose drafts were all trimmed.
        for builder in self._linear_attention_builders():
            builder.runner_num_accepted_tokens = self.num_accepted_tokens.gpu

    def calculate_reorder_batch_threshold(self) -> None:
        super().calculate_reorder_batch_threshold()
        # With drafts, GDN's verify kernels read only the first tokens of a step, so rows with
        # at most 1 + k tokens must lead every batch (TorchTitanLinearAttentionMetadataBuilder).
        if (
            self.uniform_decode_query_len > 1
            and self._linear_attention_builders()
            and self.reorder_batch_threshold != self.uniform_decode_query_len
        ):
            raise RuntimeError(
                f"GDN with drafts needs a batch-reorder threshold of "
                f"{self.uniform_decode_query_len} (1 + k); the attention builders give "
                f"{self.reorder_batch_threshold}"
            )

    def _linear_attention_builders(
        self,
    ) -> list[TorchTitanLinearAttentionMetadataBuilder]:
        return [
            builder
            for group in self._attn_group_iterator()
            if group.backend is TorchTitanLinearAttentionBackend
            for builder in group.metadata_builders
        ]

    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        uniform_decode: bool = False,
        profile_seq_lens: int | None = None,
        **kwargs,
    ):
        """Keep each request of a mixed capture dummy within ``max_model_len``.

        vLLM gives every request of a mixed dummy batch a sequence length equal to the
        whole batch. A real step that size packs several requests, each at most
        ``max_model_len`` long, so the captured graph replays them correctly.

            max_model_len=1024, capture 2048 tokens as 8 requests of 256
            vLLM: each seq_len 2048 -> reads 128 KV pages from a 64-page row
            here: each seq_len 1024 -> reads its 64 pages
        """
        # TODO: drop once vLLM caps the dummy's seq_lens at max_model_len in _dummy_run.
        if (
            profile_seq_lens is None
            and not uniform_decode
            and num_tokens > self.max_model_len
        ):
            profile_seq_lens = self.max_model_len
        return super()._dummy_run(
            num_tokens,
            *args,
            uniform_decode=uniform_decode,
            profile_seq_lens=profile_seq_lens,
            **kwargs,
        )

    def initialize_kv_cache(
        self, kv_cache_config, is_profiling: bool = False, **kwargs
    ):
        super().initialize_kv_cache(kv_cache_config, is_profiling, **kwargs)
        # vLLM's graph-memory estimate, taken out of the KV cache, charges every FULL graph like the second largest
        # (a mixed graph). 1-token decode graphs take far less (Qwen3-8B TP2: 11 added 0.03 GiB to 1.57), so profile
        # only the mixed graphs. Verify graphs (1 + k tokens per row) stay charged: left out, 145 of them put the
        # capture 0.79 GiB over the estimate (Qwen3.8-27B, k = 3, GB300).
        # TODO: drop once vLLM's estimate charges decode graphs at their own size.
        dispatcher = self.cudagraph_dispatcher
        if (
            is_profiling
            and dispatcher.cudagraph_mode == CUDAGraphMode.FULL
            and self.uniform_decode_query_len == 1
        ):
            keys = dispatcher.cudagraph_keys
            keys[CUDAGraphMode.FULL] = {
                key for key in keys[CUDAGraphMode.FULL] if not key.uniform
            }

    def _update_states(self, scheduler_output: SchedulerOutput):
        # Under async scheduling, the last step's accepted-token count copy can land before or after vLLM moves
        # rows (condense, the decode/prefill reorder), and vLLM then gathers each row's count by its last-step
        # row. A moved row can then read another request's count. So wait for the copy before any row moves,
        # and keep the counts by request for _prepare_inputs.
        # TODO: drop this and its use in _prepare_inputs on the move to vLLM's model runner V2
        #   (pytorch/torchtitan#5956), which keeps the counts on the GPU (vllm-project/vllm#63373 removed V1).
        self._accepted_counts_by_request = None
        if (
            self.cache_config.mamba_cache_mode == "align"
            and self.use_async_scheduling
            and self.num_accepted_tokens_event is not None
        ):
            self.num_accepted_tokens_event.synchronize()
            counts = self.input_batch.num_accepted_tokens_cpu
            last_step_rows = self.input_batch.prev_req_id_to_index or {}
            self._accepted_counts_by_request = {
                req_id: int(counts[row]) for req_id, row in last_step_rows.items()
            }
        return super()._update_states(scheduler_output)

    def _prepare_inputs(self, scheduler_output: SchedulerOutput, *args, **kwargs):
        # vLLM writes num_decode_draft_tokens only on steps with drafts. Reset it, so a step without
        # drafts doesn't dispatch on the last step's counts (_determine_batch_execution_and_padding).
        if (
            self.uniform_decode_query_len > 1
            and not scheduler_output.scheduled_spec_decode_tokens
        ):
            self.num_decode_draft_tokens.np.fill(-1)
        outputs = super()._prepare_inputs(scheduler_output, *args, **kwargs)

        if self._accepted_counts_by_request is not None:
            # Put back the accepted-token counts kept in _update_states: preprocess_mamba and GDN's verify rows
            # start from the state at offset count - 1. A row that didn't sample in the last step (new, re-added,
            # a prefill in progress) has count 1.
            counts = self.input_batch.num_accepted_tokens_cpu
            last_step_rows = self.input_batch.prev_req_id_to_index or {}
            for row, req_id in enumerate(self.input_batch.req_ids):
                counts[row] = (
                    self._accepted_counts_by_request[req_id]
                    if req_id in last_step_rows
                    else 1
                )
        return outputs

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        *args,
        force_uniform_decode: bool | None = None,
        **kwargs,
    ):
        """Tell the model how many of this step's tokens are real, before padding.

        vLLM calls this once before every forward: from ``execute_model`` for a
        real step, and from ``_dummy_run`` for profiling, warm-up, CUDA-graph
        capture and an idle DP rank's dummy batch (whose rows all count as
        real). ``num_tokens`` is the unpadded count; vLLM applies every padding
        source inside this call (TP rounding, CUDA-graph capture size, DP
        equalization). The model turns the count into a padding mask, which keeps
        those rows off Dist-MoE's network; graph replay reads the value published
        for the real step. Linear attention (GDN/KDA) builders then learn whether
        the step replays a packed or a uniform FULL graph.

        With k drafts, a real step replays a uniform graph only if every row verifies k drafts:

            k = 3, 4 tokens per row, draft counts [3, 3, 3, 3]   -> uniform graph
            k = 3, 4 tokens per row, draft counts [3, 3, 3, -1]  -> not the uniform graph (a 4-token prefill)
        """
        self.get_model().set_num_valid_tokens(num_tokens)

        # With drafts, linear-attention builders read this flag to stage dummy batches (null slots).
        # vLLM passes force_uniform_decode only from _dummy_run; real steps pass None.
        dummy_run_uniform = force_uniform_decode

        # vLLM calls a step uniform when every row has 1 + k tokens. The uniform graph runs only
        # GDN's verify kernels, so a 1 + k token prefill row must not replay it
        # (TorchTitanLinearAttentionMetadataBuilder._build_with_drafts raises otherwise).
        # TODO: drop with this override and the _prepare_inputs reset on the move to vLLM's V2 model runner
        # (pytorch/torchtitan#5956), which keeps steps with a prefill row off decode graphs
        # (decode_graph_eligible on vLLM main, vllm/v1/worker/gpu/model_runner.py).
        if force_uniform_decode is None and self.uniform_decode_query_len > 1:
            num_decode_draft_tokens = self.num_decode_draft_tokens.np[:num_reqs]
            force_uniform_decode = bool(
                (num_decode_draft_tokens == self.uniform_decode_query_len - 1).all()
            )
        result = super()._determine_batch_execution_and_padding(
            num_tokens,
            num_reqs,
            *args,
            force_uniform_decode=force_uniform_decode,
            **kwargs,
        )
        # TODO: drop once vLLM passes the dispatched (cudagraph_mode, batch_descriptor)
        # to AttentionMetadataBuilder.build.
        cudagraph_mode, batch_descriptor = result[:2]
        replays_full_graph = cudagraph_mode == CUDAGraphMode.FULL
        for builder in self._linear_attention_builders():
            builder.dummy_run_uniform = dummy_run_uniform
            builder.replays_packed_graph = (
                replays_full_graph and not batch_descriptor.uniform
            )
            builder.replays_uniform_graph = (
                replays_full_graph and batch_descriptor.uniform
            )
        return result

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_dense_sp = self.compilation_config.pass_config.enable_sp and tp_size > 1
        enable_expert_sp = (
            self.vllm_config.parallel_config.enable_expert_parallel and tp_size > 1
        )
        if enable_dense_sp or enable_expert_sp:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens


class TorchTitanGPUWorker(GPUWorker):
    """V1 worker that constructs :class:`TorchTitanGPUModelRunner`."""

    def _maybe_get_memory_pool_context(self, tag: str):
        # vLLM uses CuMem for model weights and the KV cache.
        # TorchTitan only transfers model weights over RDMA, so leave all other
        # allocations on PyTorch's default allocator.
        # The parent method checks enable_cumem_allocator and returns a no-op
        # context when it is disabled.
        if tag == "weights":
            return super()._maybe_get_memory_pool_context(tag)
        return nullcontext()

    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError(
                "TorchTitan's vLLM integration requires the V1 model runner"
            )

        # GPUWorker imports its runner class inside init_device and provides no
        # runner factory; the runner builds its MTP drafter the same way. Scope the
        # class substitutions to that construction.
        original_runner_cls = vllm_gpu_model_runner.GPUModelRunner
        original_proposer_cls = vllm_gpu_model_runner.EagleProposer
        vllm_gpu_model_runner.GPUModelRunner = TorchTitanGPUModelRunner
        vllm_gpu_model_runner.EagleProposer = TorchTitanEagleProposer
        try:
            super().init_device()
        finally:
            vllm_gpu_model_runner.GPUModelRunner = original_runner_cls
            vllm_gpu_model_runner.EagleProposer = original_proposer_cls


class TorchTitanEagleProposer(EagleProposer):
    """vLLM's MTP drafter, built next to a TorchTitan target.

    model config      the draft's own: the target's is TorchTitan's HF-shaped config
    embed_tokens      shares storage with the target's `tok_embeddings.weight`
    lm_head           shares storage with the target's `lm_head.weight`
    mtp.* layers      the checkpoint's weights: model state dict loads leave them alone
    """

    # TODO: on vLLM's V2 runner (vLLM >= 0.34, pytorch/torchtitan#5956), subclass
    # MTPSpeculator instead and register it with register_speculative_method.

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: GPUModelRunner | None = None,
    ) -> None:
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        # The overrides below assume Qwen3.5's draft layout: `model.embed_tokens` and one
        # top-level `lm_head`. DeepSeek-style drafts keep a head per MTP layer.
        if self.draft_model_config.architecture not in ("Qwen3_5MTP", "Qwen3_5MoeMTP"):
            raise ValueError(
                "num_speculative_tokens supports Qwen3.5-architecture MTP drafts, got "
                f"{self.draft_model_config.architecture}"
            )

    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # vLLM's _dummy_run asks for drafter graphs only while the target captures piecewise
        # graphs, so under a FULL target the first draft step would capture while serving.
        # TODO: drop with the move to vLLM's V2 model runner (pytorch/torchtitan#5956); V1's
        # GPUModelRunner._dummy_run captures drafter graphs only for piecewise targets
        # ("Eagle currently only supports PIECEWISE cudagraphs").
        capture = is_graph_capturing and not self.speculative_config.enforce_eager
        super().dummy_run(
            num_tokens,
            use_cudagraphs=use_cudagraphs or capture,
            is_graph_capturing=is_graph_capturing,
            slot_mappings=slot_mappings,
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        # vLLM's MTP models read their HF config from `vllm_config.model_config`.
        return replace(
            super()._create_draft_vllm_config(),
            model_config=self.draft_model_config,
        )

    # TODO: drop the two overrides below once vLLM's drafter can ask the target for its
    # embedding and lm_head, as SGLang's `get_embed_and_head` does. vLLM finds them by HF
    # attribute names, which TorchTitan's model doesn't have. No vLLM issue filed yet.

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        _alias_draft_weight(
            draft_weight=self.model.model.embed_tokens.weight,
            target_weight=target_language_model.model.tok_embeddings.weight,
        )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        _alias_draft_weight(
            draft_weight=self.model.lm_head.weight,
            target_weight=target_language_model.model.lm_head.weight,
        )


def _alias_draft_weight(
    draft_weight: nn.Parameter, target_weight: torch.Tensor
) -> None:
    """Point `draft_weight` at the target's storage, which model state dict loads update in place."""
    if isinstance(target_weight, DTensor):
        target_weight = target_weight.to_local()
    if (
        target_weight.shape != draft_weight.shape
        or target_weight.dtype != draft_weight.dtype
    ):
        raise ValueError(
            f"MTP draft weight {tuple(draft_weight.shape)} {draft_weight.dtype} does not "
            f"match the target's {tuple(target_weight.shape)} {target_weight.dtype}"
        )
    draft_weight.data = target_weight.detach()


if device_type == "xpu":
    # vLLM's CUDA Worker rejects XPU devices; CUDA installs never import these.
    import vllm.v1.worker.xpu_worker as vllm_xpu_worker
    from vllm.v1.worker.xpu_model_runner import XPUModelRunner

    class TorchTitanXPUModelRunner(TorchTitanGPUModelRunner, XPUModelRunner):
        """:class:`TorchTitanGPUModelRunner`'s overrides on vLLM's XPU runner."""

    class TorchTitanXPUWorker(vllm_xpu_worker.XPUWorker):
        """V1 XPU worker that constructs :class:`TorchTitanXPUModelRunner`."""

        def init_device(self):
            if self.use_v2_model_runner:
                raise ValueError(
                    "TorchTitan's vLLM integration requires the V1 model runner"
                )
            # TorchTitanGPUWorker.init_device's runner swap. No drafter swap: drafts
            # need CUDA (`_validate_platform`).
            original_runner_cls = vllm_xpu_worker.XPUModelRunner
            vllm_xpu_worker.XPUModelRunner = TorchTitanXPUModelRunner
            try:
                super().init_device()
            finally:
                vllm_xpu_worker.XPUModelRunner = original_runner_cls
