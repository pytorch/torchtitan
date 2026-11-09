# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

from contextlib import nullcontext

from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.rl.generator.vllm.attention.linear_attention import (
    TorchTitanHybridModelState,
)


class TorchTitanCudaGraphManager(ModelCudaGraphManager):
    """Give decode batches their own FULL graphs, apart from prefill and mixed batches.

    vLLM's FULL mode keys every graph by token count: four one-token decodes replay
    the mixed graph captured with ``max_query_len = 4``. Decode needs its own graph:
      - Attention Gym linear attention (GDN/KDA) runs different decode kernels.
      - FA4 sizes its grid from ``max_query_len``: decode attention runs 1.3-2.0x
        longer from 32 sequences at head dim 64 and from 64 at head dim 128
        (GB300, 16-token pages).

    Under FULL, this manager also captures FULL_DECODE_ONLY's decode graphs, keyed by
    request count. vLLM's ``dispatch`` tries them first for uniform decode batches:

      uniform decode, n requests <= largest decode graph    -> decode graph
      larger (DP padding, or a capture-size gap)            -> mixed graph, as in FULL
      larger, batch-invariant                               -> eager on the decode kernels
      any other batch                                       -> mixed graph

    Native capture, replay and graph ownership stay intact. This costs one more graph
    capture per decode size (83 at max_num_seqs 1024).

    TODO: drop this class once vLLM has a CUDA-graph mode with FULL graphs for both
    mixed and decode batches (FULL has no separate decode routine today), or once
    FULL_AND_PIECEWISE with breakable graphs is as fast on RL steps.
    """

    # The last ``dispatch`` result, which a FULL step always replays. Eager steps can
    # skip ``dispatch`` (``dispatch_cg_and_sync_dp``), so read it on FULL steps only.
    last_dispatched_descriptor: BatchExecutionDescriptor

    def _init_candidates(self) -> None:
        super()._init_candidates()
        if self.cudagraph_mode != CUDAGraphMode.FULL:
            return
        # vLLM builds decode graphs only for modes with a separate decode routine.
        decode_graphs = CudaGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_lens=self.decode_max_reqs,
            lora_capture_cases=self.lora_capture_cases,
            varlen_decode=self.varlen_decode,
            ubatch_runner=self.ubatch_runner,
        )
        self._decode_descs = decode_graphs._decode_descs
        # Widest first, as vLLM captures: narrower graphs reuse the pool's blocks.
        self._capture_descs[CUDAGraphMode.FULL] = sorted(
            decode_graphs._capture_descs.get(CUDAGraphMode.FULL, [])
            + self._capture_descs.get(CUDAGraphMode.FULL, []),
            key=lambda descriptor: descriptor.num_tokens,
            reverse=True,
        )

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        descriptor = super().dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len,
            num_ubatches,
        )
        if (
            uniform_token_count is not None
            and descriptor.cg_mode == CUDAGraphMode.FULL
            and descriptor.uniform_token_count is None
            and is_in_batch_invariant_mode()
        ):
            # GDN/KDA's packed kernels in the mixed graph don't match decode bitwise.
            descriptor = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=descriptor.num_active_loras,
                num_ubatches=num_ubatches,
            )
        self.last_dispatched_descriptor = descriptor
        return descriptor


class TorchTitanGPUModelRunner(GPUModelRunner):
    """vLLM's V2 runner with FULL decode graphs next to FULL mixed graphs."""

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, is_profiling: bool = False, **kwargs
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, is_profiling, **kwargs)
        # With the extensible KV cache off (vLLM's default is on), vLLM sizes the KV
        # cache after estimating graph memory, charging every FULL graph like the
        # second widest. Decode graphs take far less (Qwen3-8B TP2: 11 decode graphs
        # added 0.03 GiB to 1.57), so the estimate runs on vLLM's own manager, which
        # captures the mixed graphs only.
        # TODO: install this manager before the estimate (drop the is_profiling return)
        # once vLLM's _extrapolate_full_graph_memory charges each FULL graph at its own
        # size instead of the second-largest sample's.
        if is_profiling:
            return
        # vLLM builds its manager here (`ModelCudaGraphManager(...)` in
        # GPUModelRunner.initialize_kv_cache), with no factory to override. Rebuild it
        # with the arguments vLLM passed, read off vLLM's manager.
        # TODO: drop the rebuild once GPUModelRunner builds its manager through a method
        # a subclass can override.
        vllm_manager = self.cudagraph_manager
        self.cudagraph_manager = TorchTitanCudaGraphManager(
            vllm_manager.vllm_config,
            vllm_manager.device,
            vllm_manager.cudagraph_mode,
            # The decode_query_lens vLLM passed, capped at max_num_seqs (capping again is a no-op).
            decode_query_lens=vllm_manager.decode_max_reqs,
            lora_capture_cases=vllm_manager.lora_capture_cases,
            varlen_decode=vllm_manager.varlen_decode,
            ubatch_runner=vllm_manager.ubatch_runner,
        )
        if isinstance(self.model_state, TorchTitanHybridModelState):
            self.model_state.cudagraph_manager = self.cudagraph_manager


class TorchTitanGPUWorker(GPUWorker):
    """vLLM worker that builds :class:`TorchTitanGPUModelRunner`."""

    def _maybe_get_memory_pool_context(self, tag: str):
        # vLLM uses CuMem for model weights and the KV cache.
        # TorchTitan only transfers model weights over RDMA, so leave all other
        # allocations on PyTorch's default allocator.
        # The parent method checks enable_cumem_allocator and returns a no-op
        # context when it is disabled.
        if tag == "weights":
            return super()._maybe_get_memory_pool_context(tag)
        return nullcontext()

    def _make_model_runner(self) -> TorchTitanGPUModelRunner:
        return TorchTitanGPUModelRunner(self.vllm_config, self.device)
