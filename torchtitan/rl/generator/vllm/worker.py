# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

from collections.abc import Set
from contextlib import nullcontext

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.utils.math_utils import round_up
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    TorchTitanLinearAttentionBackend,
)


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

    Uniform-decode batch in FULL mode:
      num_tokens <= max_decode_capture_size              -> decode graph
      larger (DP padding, or a capture-size gap)         -> mixed graph, as in stock FULL
      larger, batch-invariant, eager allowed             -> eager on the decode kernels

    Decode keys come from the native FULL_DECODE_ONLY dispatcher; native capture,
    replay and graph ownership stay intact.

    This costs one additional graph capture per decode-specialized batch size
    (83 at max_num_seqs 1024); ``TorchTitanGPUModelRunner.initialize_kv_cache``
    keeps them out of vLLM's graph-memory estimate.

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

    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: SchedulerOutput, vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        transfer = vllm_config.kv_transfer_config
        if transfer is None or transfer.kv_connector != "MooncakeStoreConnector":
            return GPUModelRunner.kv_connector_no_forward(scheduler_output, vllm_config)

        # TODO(acisseJZhong): upstream to vLLM: its kv_connector_no_forward
        # passes wait_for_save=False, so Mooncake never submits finish-time saves
        # (e.g. hybrid tails pinned after the last forward) scheduled on a step
        # without a forward. Submit them before get_finished, as on forward
        # steps: get_finished retires the ledger of finished requests that have
        # no queued save job.
        with (
            set_forward_context(None, vllm_config),
            GPUModelRunner.maybe_get_kv_connector_output(scheduler_output) as output,
        ):
            pass
        return ModelRunnerOutput.with_kv_conn_output_only(output)

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        assert not self.cudagraph_dispatcher.keys_initialized
        self.cudagraph_dispatcher = TorchTitanCudagraphDispatcher(self.vllm_config)

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
        # vLLM's graph-memory estimate, taken out of the KV cache, charges every FULL graph
        # like the second largest (a mixed graph). Decode graphs take far less (Qwen3-8B TP2:
        # 11 decode graphs added 0.03 GiB to 1.57), so profile the mixed graphs only. vLLM drops
        # these profiling keys afterwards; the real call rebuilds them, decode keys included
        # (test_generator_dp_cudagraph.py checks a decode graph still replays).
        # TODO: drop once vLLM's estimate charges decode graphs at their own size.
        dispatcher = self.cudagraph_dispatcher
        if is_profiling and dispatcher.cudagraph_mode == CUDAGraphMode.FULL:
            keys = dispatcher.cudagraph_keys
            keys[CUDAGraphMode.FULL] = {
                key for key in keys[CUDAGraphMode.FULL] if not key.uniform
            }

    def _determine_batch_execution_and_padding(self, num_tokens: int, *args, **kwargs):
        """Tell the model how many of this step's tokens are real, before padding.

        vLLM calls this once before every forward: from ``execute_model`` for a
        real step, and from ``_dummy_run`` for profiling, warm-up, CUDA-graph
        capture and an idle DP rank's dummy batch (whose rows all count as
        real). ``num_tokens`` is the unpadded count; vLLM applies every padding
        source inside this call (TP rounding, CUDA-graph capture size, DP
        equalization). The model turns the count into a padding mask, which keeps
        those rows off Dist-MoE's network; graph replay reads the value published
        for the real step. Linear attention (GDN/KDA) builders then learn whether
        the step replays a packed FULL graph.
        """
        self.get_model().set_num_valid_tokens(num_tokens)
        result = super()._determine_batch_execution_and_padding(
            num_tokens, *args, **kwargs
        )
        # TODO: drop once vLLM passes the dispatched (cudagraph_mode, batch_descriptor)
        # to AttentionMetadataBuilder.build.
        cudagraph_mode, batch_descriptor = result[:2]
        replays_packed_graph = (
            cudagraph_mode == CUDAGraphMode.FULL and not batch_descriptor.uniform
        )
        for group in self._attn_group_iterator():
            if group.backend is TorchTitanLinearAttentionBackend:
                for builder in group.metadata_builders:
                    builder.replays_packed_graph = replays_packed_graph
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
        # runner factory. Scope the class substitution to that construction.
        original_runner_cls = vllm_gpu_model_runner.GPUModelRunner
        vllm_gpu_model_runner.GPUModelRunner = TorchTitanGPUModelRunner
        try:
            super().init_device()
        finally:
            vllm_gpu_model_runner.GPUModelRunner = original_runner_cls
