import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from deep_ep import Buffer, Config
from deep_ep.utils import EventHandle, EventOverlap

from torchtitan.tools.logging import logger

# =============================================================================
# Buffer is imported from buffer.py - THE single source of truth
# =============================================================================
# NOTE: Both autotune and training MUST use the same buffer instance.
# DeepEP's RDMA connections are tied to the buffer - using different buffers
# between autotune and training causes timeout errors in internode mode.
# =============================================================================
from .buffer import get_buffer, get_hidden_bytes

# Global tuned configs (set by autotune or manually)
_tuned_dispatch_config: Optional[Config] = None
_tuned_combine_config: Optional[Config] = None


def _sync_comm_stream(buffer: Buffer) -> None:
    """Synchronize the DeepEP comm stream with the current (default) stream.

    This function implements explicit event-based synchronization between
    DeepEP's internal communication stream and the current CUDA stream.

    Why this is needed:
        The standard deep_ep.Buffer uses a SEPARATE communication stream
        for dispatch/combine operations. The primus_turbo fork has a parameter
        `use_default_stream_as_comm_stream=True` that makes all ops run on the
        default stream, but our deep_ep version lacks this parameter.

    Without this fix:
        Race conditions can occur between consecutive DeepEP ops.
        Symptoms observed:
        - 1x small batch → full batch: grad_norm 10.39x explosion
        - 2-6x small batches → full batch: NaN gradients
        - 10x small batches → full batch: 1.00x OK (by random timing!)
        The cyclic behavior proves it's a stream sync issue, not data corruption.

    Why torch.cuda.synchronize() alone is not enough:
        synchronize() waits for ALL streams, but doesn't establish proper
        ordering between the comm stream and current stream for subsequent ops.

    The fix:
        1. Get DeepEP's internal comm stream via buffer.get_comm_stream()
        2. Record a CUDA event on the comm stream (marks completion point)
        3. Make current stream wait for that event (establishes ordering)
        4. Full device sync as final safety barrier

    Performance note:
        This adds overhead. For optimal perf, either:
        - Fork DeepEP and add use_default_stream_as_comm_stream support
        - Use primus_turbo's DeepEP directly
    """
    comm_stream = buffer.get_comm_stream()
    event = torch.cuda.Event()
    event.record(comm_stream)
    torch.cuda.current_stream().wait_event(event)
    torch.cuda.synchronize()


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
        use_cuda_num_token_per_expert: bool = False,
        num_worst_tokens: int = 0,
        sync_comm_stream: bool = False,
    ):
        """Forward pass of fused dispatch."""
        previous_event = None

        if async_finish:
            previous_event = EventOverlap(EventHandle())

        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Do MoE dispatch
        # NOTE(phuc): the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        # TODO(deepep-fork, phuc): The local DeepEP does not support num_recv_tokens_per_expert_as_cuda parameter
        # which exists in torchtitan-amd's forked DeepEP. When we fork DeepEP, we should add this parameter
        # back to avoid the tensor conversion overhead below. The parameter allows DeepEP to return a CUDA
        # tensor directly instead of a Python list for num_recv_tokens_per_expert_list.
        dispatch_kwargs = {
            "topk_idx": token_indices,
            "topk_weights": token_probs,  # DeepEP only supports float32 probs
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "previous_event": event,  # wait in deepep::intra/inter_dispatch
            "async_finish": async_finish,
            "allocate_on_comm_stream": allocate_on_comm_stream,
            # num_recv_tokens_per_expert_as_cuda=use_cuda_num_token_per_expert,  # (phuc) Not supported in local DeepEP
            "num_worst_tokens": num_worst_tokens,
        }
        # Use tuned config if available
        if _tuned_dispatch_config is not None:
            dispatch_kwargs["config"] = _tuned_dispatch_config

        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(x, **dispatch_kwargs)

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Optional: DeepEP communication stream synchronization
        # Enabled via deepep.sync_comm_stream config
        # See _sync_comm_stream() docstring for details on why this may be needed
        if sync_comm_stream:
            _sync_comm_stream(buffer)

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.sync_comm_stream = sync_comm_stream

        # WORKAROUND (phuc): for local DeepEP without num_recv_tokens_per_expert_as_cuda support:
        # The local DeepEP always returns num_recv_tokens_per_expert_list as a Python list.
        # The forked DeepEP in torchtitan-amd has num_recv_tokens_per_expert_as_cuda parameter
        # which when True, returns a CUDA tensor directly instead of a Python list.
        # MODIFIED CODE (phuc, workaround for local DeepEP):
        # NOTE(phuc): fix device placement issue - both branches now use device=x.device
        # Previously the first branch created CPU tensor which caused:
        # "RuntimeError: Expected all tensors to be on the same device, but got offs is on cpu"
        if not use_cuda_num_token_per_expert:
            tokens_per_expert = torch.tensor(
                num_recv_tokens_per_expert_list, device=x.device
            )  # list -> CUDA tensor (FIXED)
        else:
            # Manual conversion: list -> CUDA tensor (workaround since DeepEP doesn't do it)
            # TODO(deepep-fork, phuc): Restore original `tokens_per_expert = num_recv_tokens_per_expert_list`
            # when we fork DeepEP and add num_recv_tokens_per_expert_as_cuda support
            tokens_per_expert = torch.tensor(
                num_recv_tokens_per_expert_list, device=x.device
            )

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx,
        grad_output,
        grad_token_indices,
        grad_token_probs,
        grad_tokens_per_expert,
        grad_handle,
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())

        combine_kwargs = {
            "topk_weights": grad_token_probs.float(),
            "previous_event": previous_event,
            "async_finish": ctx.async_finish,
            "allocate_on_comm_stream": ctx.allocate_on_comm_stream,
        }
        # Use tuned config if available
        if _tuned_combine_config is not None:
            combine_kwargs["config"] = _tuned_combine_config

        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(), handle, **combine_kwargs
        )

        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()

        # Optional: DeepEP communication stream synchronization
        if ctx.sync_comm_stream:
            _sync_comm_stream(buffer)

        return grad_x, None, grad_token_probs, None, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        handle,
        async_finish=False,
        allocate_on_comm_stream=False,
        sync_comm_stream: bool = False,
    ):
        """Forward pass of fused combine.

        Args:
            sync_comm_stream: Whether to sync comm stream after combine operation.
        """
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))

        combine_kwargs = {
            "handle": handle,
            "async_finish": async_finish,
            "previous_event": previous_event,
            "allocate_on_comm_stream": allocate_on_comm_stream,
        }
        # Use tuned config if available
        if _tuned_combine_config is not None:
            combine_kwargs["config"] = _tuned_combine_config

        combined_x, _, after_event = buffer.combine(x, **combine_kwargs)

        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        # Optional: DeepEP communication stream synchronization
        if sync_comm_stream:
            _sync_comm_stream(buffer)

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.sync_comm_stream = sync_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))

        dispatch_kwargs = {
            "handle": ctx.handle,
            "previous_event": previous_event,
            "async_finish": ctx.async_finish,
            "allocate_on_comm_stream": ctx.allocate_on_comm_stream,
        }
        # Use tuned config if available
        if _tuned_dispatch_config is not None:
            dispatch_kwargs["config"] = _tuned_dispatch_config

        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(), **dispatch_kwargs
        )

        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()

        # Optional: DeepEP communication stream synchronization
        if ctx.sync_comm_stream:
            _sync_comm_stream(buffer)

        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
        )  # None for: group, handle, async_finish, allocate_on_comm_stream, sync_comm_stream


def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=False,
    allocate_on_comm_stream=False,
    use_cuda_num_token_per_expert: bool = False,
    num_worst_tokens: int = 0,
    sync_comm_stream: bool = False,
):
    """Perform fused dispatch operation if deep_ep is available.
    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        sync_comm_stream: Whether to sync comm stream after dispatch operation
    Returns:
        Result of FusedDispatch
    """
    return FusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
        use_cuda_num_token_per_expert,
        num_worst_tokens,
        sync_comm_stream,
    )


def fused_combine(
    x,
    group,
    handle,
    async_finish=False,
    allocate_on_comm_stream=False,
    sync_comm_stream: bool = False,
):
    """Perform fused combine operation if deep_ep is available.
    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        async_finish: Whether to finish asynchronously
        allocate_on_comm_stream: Whether to allocate on comm stream
        sync_comm_stream: Whether to sync comm stream after combine operation
    Returns:
        Result of FusedCombine
    """
    return FusedCombine.apply(
        x,
        group,
        handle,
        async_finish,
        allocate_on_comm_stream,
        sync_comm_stream,
    )


def set_deepep_num_sms(num_sms):
    """Sets the number of SMs to use for DeepEP"""
    Buffer.set_num_sms(num_sms)


def set_tuned_configs(
    dispatch_config: Optional[Config] = None,
    combine_config: Optional[Config] = None,
):
    """Set the tuned configs for dispatch and combine operations."""
    global _tuned_dispatch_config, _tuned_combine_config
    _tuned_dispatch_config = dispatch_config
    _tuned_combine_config = combine_config


def get_tuned_configs() -> Tuple[Optional[Config], Optional[Config]]:
    """Get the current tuned configs."""
    return _tuned_dispatch_config, _tuned_combine_config


@dataclass
class AutotuneResult:
    """Result from autotuning."""

    dispatch_config: Tuple[
        int, ...
    ]  # (nvl_chunk, nvl_buffer) or (nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)
    combine_config: Tuple[int, ...]
    dispatch_bandwidth_gbps: float
    combine_bandwidth_gbps: float
    is_internode: bool = False
    # Best num_sms values
    best_dispatch_sms: int = 24
    best_combine_sms: int = 24
    # Worst case stats for comparison
    worst_dispatch_config: Optional[Tuple[int, ...]] = None
    worst_dispatch_bandwidth_gbps: float = 0.0
    worst_combine_config: Optional[Tuple[int, ...]] = None
    worst_combine_bandwidth_gbps: float = 0.0


def _bench_fn(fn, warmup: int = 3, repeat: int = 5) -> float:
    """Benchmark a function and return average time in seconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    return elapsed / repeat if elapsed > 0 else float("inf")


def _detect_internode(buffer: Buffer) -> Tuple[bool, int, int]:
    """
    Detect if communication requires internode (RDMA) or is intranode only (NVLink).

    Uses the Buffer's internal detection which queries the actual network topology
    rather than relying on environment variables which may not be set correctly.

    Returns:
        (is_internode, num_local_ranks, num_nodes)
    """
    import os

    num_ranks = buffer.group_size
    # Use DeepEP's internal detection - this is what dispatch/combine actually use
    num_rdma_ranks = buffer.runtime.get_num_rdma_ranks()
    is_internode = num_rdma_ranks > 1

    # Calculate local ranks and nodes for logging
    if is_internode:
        # num_rdma_ranks = number of RDMA peers (nodes)
        # local_ranks = total_ranks / num_rdma_ranks
        num_nodes = num_rdma_ranks
        local_world_size = num_ranks // num_nodes
    else:
        # Fallback to env var for intranode case (all ranks on one node)
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", num_ranks))
        num_nodes = 1

    return is_internode, local_world_size, num_nodes


def _get_gpu_sm_range(default_sms: int = 24) -> list:
    """Auto-detect GPU type and return appropriate SM search range."""
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        # H100 has 132 SMs, B200 has 192 SMs
        if "b200" in gpu_name or "b100" in gpu_name:
            return [24, 32, 48, 64]
        elif "h200" in gpu_name or "h100" in gpu_name:
            return [16, 20, 24, 28, 32]
        elif "a100" in gpu_name:
            return [16, 20, 24, 28, 32]
        else:
            return [default_sms]  # Don't tune SM for unknown GPUs
    except Exception:
        return [default_sms]


def autotune_deepep(
    group: torch.distributed.ProcessGroup,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    nvl_buffer_size: int = 512,
    rdma_buffer_size: int = 128,
    warmup: int = 5,
    repeat: int = 10,
    verbose: bool = False,
) -> AutotuneResult:
    """
    Autotune DeepEP dispatch/combine configs before training.

    Searches over all tunable parameters:
        - num_sms: Auto-detected range based on GPU type
        - nvl_chunk: NVLink chunk size
        - rdma_chunk: RDMA chunk size (for internode only)

    Args:
        group: Process group for EP communication
        num_tokens: Number of tokens per batch (batch_size * seq_len)
        hidden: Hidden dimension
        num_experts: Total number of experts
        num_topk: Top-k experts per token
        nvl_buffer_size: NVLink buffer size (default: 512)
        rdma_buffer_size: RDMA buffer size for internode (default: 128)
        warmup: Warmup iterations (default: 5)
        repeat: Benchmark iterations (default: 10)
        verbose: Print every config result (default: False)

    Returns:
        AutotuneResult with optimal configs
    """
    rank = torch.distributed.get_rank(group)
    num_ranks = group.size()

    # Auto-detect SM range based on GPU
    sms_range = _get_gpu_sm_range()

    # Create test data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs()
        + 1
    )
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]

    # Get buffer using default allocation
    buffer = get_buffer(group, get_hidden_bytes(x))

    # Detect internode vs intranode using buffer's internal detection
    # This queries the actual network topology rather than relying on env vars
    is_internode, num_local_ranks, num_nodes = _detect_internode(buffer)

    # Get dispatch layout
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)

    # ============ SEARCH SPACE (matches original DeepEP test ranges) ============
    # Internode: full Cartesian product (sms × nvl × rdma)
    # Intranode: sms × nvl only
    if is_internode:
        # Original DeepEP test_internode.py ranges:
        # dispatch: nvl 4-44 step 4, rdma 4-32 step 4
        # combine: nvl 1-7 step 1, rdma 8-32 step 4
        nvl_dispatch_range = list(range(4, 48, 4))  # 11 values
        rdma_dispatch_range = list(range(4, 36, 4))  # 8 values
        nvl_combine_range = list(range(1, 10, 1))  # 9 values
        rdma_combine_range = list(range(8, 36, 4))  # 7 values
    else:
        # Original DeepEP test_intranode.py ranges:
        # dispatch: nvl 4-32 step 2
        # combine: nvl 1-16 step 1
        nvl_dispatch_range = list(range(4, 34, 2))  # 15 values
        rdma_dispatch_range = [16]  # dummy for intranode
        nvl_combine_range = list(range(1, 17, 1))  # 16 values
        rdma_combine_range = [16]  # dummy for intranode

    num_dispatch_configs = (
        len(sms_range) * len(nvl_dispatch_range) * len(rdma_dispatch_range)
    )
    num_combine_configs = (
        len(sms_range) * len(nvl_combine_range) * len(rdma_combine_range)
    )
    total_configs = num_dispatch_configs + num_combine_configs

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        mode_str = (
            f"internode ({num_nodes} nodes)"
            if is_internode
            else f"intranode ({num_ranks} GPUs)"
        )
        logger.info(f"[DeepEP Autotune] {mode_str} on {gpu_name}")
        logger.info(
            f"[DeepEP Autotune] tokens={num_tokens}, hidden={hidden}, experts={num_experts}, topk={num_topk}"
        )
        logger.info(f"[DeepEP Autotune] Search space: sms={sms_range}")
        if is_internode:
            logger.info(
                f"  dispatch: nvl={nvl_dispatch_range[0]}-{nvl_dispatch_range[-1]}"
                f" ({len(nvl_dispatch_range)}), rdma={rdma_dispatch_range[0]}"
                f"-{rdma_dispatch_range[-1]} ({len(rdma_dispatch_range)})"
            )
            logger.info(
                f"  combine:  nvl={nvl_combine_range[0]}-{nvl_combine_range[-1]}"
                f" ({len(nvl_combine_range)}), rdma={rdma_combine_range[0]}"
                f"-{rdma_combine_range[-1]} ({len(rdma_combine_range)})"
            )
        else:
            logger.info(
                f"  dispatch: nvl={nvl_dispatch_range[0]}-{nvl_dispatch_range[-1]} ({len(nvl_dispatch_range)})"
            )
            logger.info(
                f"  combine:  nvl={nvl_combine_range[0]}-{nvl_combine_range[-1]} ({len(nvl_combine_range)})"
            )
        logger.info(
            f"[DeepEP Autotune] Total: {num_dispatch_configs} dispatch + {num_combine_configs} combine = {total_configs} configs"
        )
        logger.info(f"[DeepEP Autotune] warmup={warmup}, repeat={repeat}")

    # Config creation helper
    def make_config(sms: int, nvl_chunk: int, rdma_chunk: int = 16) -> Config:
        if is_internode:
            return Config(sms, nvl_chunk, nvl_buffer_size, rdma_chunk, rdma_buffer_size)
        else:
            return Config(sms, nvl_chunk, nvl_buffer_size)

    # Initial dispatch to get handle and calculate bytes
    initial_config = make_config(
        sms_range[0], nvl_dispatch_range[0], rdma_dispatch_range[0]
    )
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=scores.gather(1, topk_idx).float(),
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=initial_config,
    )

    dispatch_recv_bytes = recv_x.numel() * 2  # bfloat16
    combine_send_bytes = recv_x.numel() * 2

    autotune_start = time.time()

    # ============================================================================
    # INTERNODE MODE: Joint tuning of num_sms for best combined dispatch+combine
    # ============================================================================
    # For internode, dispatch handle contains RDMA metadata tied to num_sms.
    # Combine MUST use the same num_sms as dispatch. So we tune num_sms by
    # evaluating combined (dispatch + combine) performance for each value.
    # ============================================================================
    if is_internode:
        if rank == 0:
            logger.info(
                "[DeepEP Autotune] Internode: tuning num_sms jointly for dispatch+combine..."
            )

        best_combined_time = float("inf")
        best_sms = sms_range[0]
        best_dispatch_nvl = nvl_dispatch_range[0]
        best_dispatch_rdma = rdma_dispatch_range[0]
        best_combine_nvl = nvl_combine_range[0]
        best_combine_rdma = rdma_combine_range[0]
        worst_dispatch_time = 0.0
        worst_combine_time = 0.0
        worst_dispatch_nvl = nvl_dispatch_range[0]
        worst_dispatch_rdma = rdma_dispatch_range[0]
        worst_combine_nvl = nvl_combine_range[0]
        worst_combine_rdma = rdma_combine_range[0]

        for sms in sms_range:
            if rank == 0:
                logger.info(f"[DeepEP Autotune] Testing sms={sms}...")

            # Find best dispatch config for this sms
            sms_best_dispatch_time = float("inf")
            sms_best_dispatch_nvl = nvl_dispatch_range[0]
            sms_best_dispatch_rdma = rdma_dispatch_range[0]

            for nvl_chunk in nvl_dispatch_range:
                for rdma_chunk in rdma_dispatch_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def dispatch_fn():
                        buffer.dispatch(
                            x,
                            topk_idx=topk_idx,
                            topk_weights=scores.gather(1, topk_idx).float(),
                            num_tokens_per_rank=num_tokens_per_rank,
                            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                            is_token_in_rank=is_token_in_rank,
                            num_tokens_per_expert=num_tokens_per_expert,
                            config=config,
                        )

                    try:
                        t = _bench_fn(dispatch_fn, warmup, repeat)
                    except RuntimeError:
                        continue

                    if t < sms_best_dispatch_time:
                        sms_best_dispatch_time = t
                        sms_best_dispatch_nvl = nvl_chunk
                        sms_best_dispatch_rdma = rdma_chunk
                    if t > worst_dispatch_time:
                        worst_dispatch_time = t
                        worst_dispatch_nvl = nvl_chunk
                        worst_dispatch_rdma = rdma_chunk

            if sms_best_dispatch_time == float("inf"):
                continue  # Skip this sms if all dispatch configs failed

            # Re-dispatch with best config to get handle for combine
            dispatch_cfg = make_config(
                sms, sms_best_dispatch_nvl, sms_best_dispatch_rdma
            )
            try:
                recv_x, _, _, _, handle, _ = buffer.dispatch(
                    x,
                    topk_idx=topk_idx,
                    topk_weights=scores.gather(1, topk_idx).float(),
                    num_tokens_per_rank=num_tokens_per_rank,
                    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                    is_token_in_rank=is_token_in_rank,
                    num_tokens_per_expert=num_tokens_per_expert,
                    config=dispatch_cfg,
                )
            except RuntimeError:
                continue

            # Find best combine config for this sms
            sms_best_combine_time = float("inf")
            sms_best_combine_nvl = nvl_combine_range[0]
            sms_best_combine_rdma = rdma_combine_range[0]

            for nvl_chunk in nvl_combine_range:
                for rdma_chunk in rdma_combine_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def combine_fn():
                        buffer.combine(recv_x, handle=handle, config=config)

                    try:
                        t = _bench_fn(combine_fn, warmup, repeat)
                    except RuntimeError:
                        continue

                    if t < sms_best_combine_time:
                        sms_best_combine_time = t
                        sms_best_combine_nvl = nvl_chunk
                        sms_best_combine_rdma = rdma_chunk
                    if t > worst_combine_time:
                        worst_combine_time = t
                        worst_combine_nvl = nvl_chunk
                        worst_combine_rdma = rdma_chunk

            if sms_best_combine_time == float("inf"):
                continue

            # Check if this sms gives best combined performance
            combined_time = sms_best_dispatch_time + sms_best_combine_time
            dispatch_bw = dispatch_recv_bytes / 1e9 / sms_best_dispatch_time
            combine_bw = combine_send_bytes / 1e9 / sms_best_combine_time

            if rank == 0:
                logger.info(
                    f"  sms={sms}: dispatch={dispatch_bw:.1f} GB/s,"
                    f" combine={combine_bw:.1f} GB/s,"
                    f" total={combined_time*1000:.2f}ms"
                )

            if combined_time < best_combined_time:
                best_combined_time = combined_time
                best_sms = sms
                best_dispatch_nvl = sms_best_dispatch_nvl
                best_dispatch_rdma = sms_best_dispatch_rdma
                best_dispatch_time = sms_best_dispatch_time
                best_combine_nvl = sms_best_combine_nvl
                best_combine_rdma = sms_best_combine_rdma
                best_combine_time = sms_best_combine_time

        # Set final values
        best_dispatch_sms = best_sms
        best_combine_sms = best_sms

        total_elapsed = time.time() - autotune_start
        if rank == 0:
            best_dispatch_bw = (
                dispatch_recv_bytes / 1e9 / best_dispatch_time
                if best_dispatch_time > 0
                else 0
            )
            best_combine_bw = (
                combine_send_bytes / 1e9 / best_combine_time
                if best_combine_time > 0
                else 0
            )
            logger.info(
                f"[DeepEP Autotune] Internode tuning done in {total_elapsed:.1f}s"
            )
            logger.info(
                f"  Best sms={best_sms}: dispatch nvl={best_dispatch_nvl},"
                f" rdma={best_dispatch_rdma} -> {best_dispatch_bw:.1f} GB/s"
            )
            logger.info(
                f"  Best sms={best_sms}: combine nvl={best_combine_nvl},"
                f" rdma={best_combine_rdma} -> {best_combine_bw:.1f} GB/s"
            )

    # ============================================================================
    # INTRANODE MODE: Independent tuning of dispatch and combine
    # ============================================================================
    else:
        # ============ TUNE DISPATCH ============
        best_dispatch_time = float("inf")
        worst_dispatch_time = 0.0
        best_dispatch_sms = sms_range[0]
        best_dispatch_nvl = nvl_dispatch_range[0]
        best_dispatch_rdma = rdma_dispatch_range[0]
        worst_dispatch_nvl = nvl_dispatch_range[0]
        worst_dispatch_rdma = rdma_dispatch_range[0]

        if rank == 0:
            logger.info(
                f"[DeepEP Autotune] Tuning dispatch ({num_dispatch_configs} configs)..."
            )

        dispatch_start = time.time()
        tested = 0
        skipped = 0

        for sms in sms_range:
            for nvl_chunk in nvl_dispatch_range:
                for rdma_chunk in rdma_dispatch_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def dispatch_fn():
                        buffer.dispatch(
                            x,
                            topk_idx=topk_idx,
                            topk_weights=scores.gather(1, topk_idx).float(),
                            num_tokens_per_rank=num_tokens_per_rank,
                            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                            is_token_in_rank=is_token_in_rank,
                            num_tokens_per_expert=num_tokens_per_expert,
                            config=config,
                        )

                    try:
                        t = _bench_fn(dispatch_fn, warmup, repeat)
                        tested += 1
                    except RuntimeError:
                        skipped += 1
                        continue

                    bw = dispatch_recv_bytes / 1e9 / t if t > 0 else 0

                    if verbose and rank == 0:
                        logger.info(
                            f"  [{tested}/{num_dispatch_configs}] sms={sms:2d}, nvl={nvl_chunk:2d}: {bw:6.1f} GB/s"
                        )

                    if t < best_dispatch_time:
                        best_dispatch_time = t
                        best_dispatch_sms = sms
                        best_dispatch_nvl = nvl_chunk
                        best_dispatch_rdma = rdma_chunk
                    if t > worst_dispatch_time:
                        worst_dispatch_time = t
                        worst_dispatch_nvl = nvl_chunk
                        worst_dispatch_rdma = rdma_chunk

        dispatch_elapsed = time.time() - dispatch_start
        if rank == 0:
            best_bw = (
                dispatch_recv_bytes / 1e9 / best_dispatch_time
                if best_dispatch_time > 0
                else 0
            )
            logger.info(
                f"[DeepEP Autotune] Dispatch done: {tested} tested, {skipped} skipped in {dispatch_elapsed:.1f}s"
            )
            logger.info(
                f"  Best: sms={best_dispatch_sms}, nvl={best_dispatch_nvl} -> {best_bw:.1f} GB/s"
            )

        # Re-dispatch with best config to get handle for combine tuning
        best_dispatch_cfg = make_config(
            best_dispatch_sms, best_dispatch_nvl, best_dispatch_rdma
        )
        recv_x, _, _, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=scores.gather(1, topk_idx).float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=best_dispatch_cfg,
        )

        # ============ TUNE COMBINE ============
        best_combine_time = float("inf")
        worst_combine_time = 0.0
        best_combine_sms = sms_range[0]
        best_combine_nvl = nvl_combine_range[0]
        best_combine_rdma = rdma_combine_range[0]
        worst_combine_nvl = nvl_combine_range[0]
        worst_combine_rdma = rdma_combine_range[0]

        num_combine_configs = (
            len(sms_range) * len(nvl_combine_range) * len(rdma_combine_range)
        )

        if rank == 0:
            logger.info(
                f"[DeepEP Autotune] Tuning combine ({num_combine_configs} configs)..."
            )

        combine_start = time.time()
        tested = 0
        skipped = 0
        current_sms = best_dispatch_sms

        for sms in sms_range:
            # Re-dispatch if sms changed
            if sms != current_sms:
                try:
                    dispatch_cfg = make_config(
                        sms, best_dispatch_nvl, best_dispatch_rdma
                    )
                    recv_x, _, _, _, handle, _ = buffer.dispatch(
                        x,
                        topk_idx=topk_idx,
                        topk_weights=scores.gather(1, topk_idx).float(),
                        num_tokens_per_rank=num_tokens_per_rank,
                        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                        is_token_in_rank=is_token_in_rank,
                        num_tokens_per_expert=num_tokens_per_expert,
                        config=dispatch_cfg,
                    )
                    current_sms = sms
                except RuntimeError:
                    skipped += len(nvl_combine_range) * len(rdma_combine_range)
                    continue

            for nvl_chunk in nvl_combine_range:
                for rdma_chunk in rdma_combine_range:
                    config = make_config(sms, nvl_chunk, rdma_chunk)

                    def combine_fn():
                        buffer.combine(recv_x, handle=handle, config=config)

                    try:
                        t = _bench_fn(combine_fn, warmup, repeat)
                        tested += 1
                    except RuntimeError:
                        skipped += 1
                        continue

                    bw = combine_send_bytes / 1e9 / t if t > 0 else 0

                    if verbose and rank == 0:
                        logger.info(
                            f"  [{tested}/{num_combine_configs}] sms={sms:2d}, nvl={nvl_chunk:2d}: {bw:6.1f} GB/s"
                        )

                    if t < best_combine_time:
                        best_combine_time = t
                        best_combine_sms = sms
                        best_combine_nvl = nvl_chunk
                        best_combine_rdma = rdma_chunk
                    if t > worst_combine_time:
                        worst_combine_time = t
                        worst_combine_nvl = nvl_chunk
                        worst_combine_rdma = rdma_chunk

        combine_elapsed = time.time() - combine_start
        total_elapsed = time.time() - autotune_start
        if rank == 0:
            best_bw = (
                combine_send_bytes / 1e9 / best_combine_time
                if best_combine_time > 0
                else 0
            )
            logger.info(
                f"[DeepEP Autotune] Combine done: {tested} tested, {skipped} skipped in {combine_elapsed:.1f}s"
            )
            logger.info(
                f"  Best: sms={best_combine_sms}, nvl={best_combine_nvl} -> {best_bw:.1f} GB/s"
            )

    # Calculate bandwidths
    best_dispatch_bw = (
        dispatch_recv_bytes / 1e9 / best_dispatch_time if best_dispatch_time > 0 else 0
    )
    worst_dispatch_bw = (
        dispatch_recv_bytes / 1e9 / worst_dispatch_time
        if worst_dispatch_time > 0
        else 0
    )
    best_combine_bw = (
        combine_send_bytes / 1e9 / best_combine_time if best_combine_time > 0 else 0
    )
    worst_combine_bw = (
        combine_send_bytes / 1e9 / worst_combine_time if worst_combine_time > 0 else 0
    )

    # Build result configs
    if is_internode:
        dispatch_cfg_tuple = (
            best_dispatch_nvl,
            nvl_buffer_size,
            best_dispatch_rdma,
            rdma_buffer_size,
        )
        combine_cfg_tuple = (
            best_combine_nvl,
            nvl_buffer_size,
            best_combine_rdma,
            rdma_buffer_size,
        )
        worst_dispatch_cfg = (
            worst_dispatch_nvl,
            nvl_buffer_size,
            worst_dispatch_rdma,
            rdma_buffer_size,
        )
        worst_combine_cfg = (
            worst_combine_nvl,
            nvl_buffer_size,
            worst_combine_rdma,
            rdma_buffer_size,
        )
    else:
        dispatch_cfg_tuple = (best_dispatch_nvl, nvl_buffer_size)
        combine_cfg_tuple = (best_combine_nvl, nvl_buffer_size)
        worst_dispatch_cfg = (worst_dispatch_nvl, nvl_buffer_size)
        worst_combine_cfg = (worst_combine_nvl, nvl_buffer_size)

    result = AutotuneResult(
        dispatch_config=dispatch_cfg_tuple,
        combine_config=combine_cfg_tuple,
        dispatch_bandwidth_gbps=best_dispatch_bw,
        combine_bandwidth_gbps=best_combine_bw,
        is_internode=is_internode,
        best_dispatch_sms=best_dispatch_sms,
        best_combine_sms=best_combine_sms,
        worst_dispatch_config=worst_dispatch_cfg,
        worst_dispatch_bandwidth_gbps=worst_dispatch_bw,
        worst_combine_config=worst_combine_cfg,
        worst_combine_bandwidth_gbps=worst_combine_bw,
    )

    # Set global configs with best num_sms values
    best_dispatch_config = make_config(
        best_dispatch_sms, best_dispatch_nvl, best_dispatch_rdma
    )
    best_combine_config = make_config(
        best_combine_sms, best_combine_nvl, best_combine_rdma
    )
    set_tuned_configs(
        dispatch_config=best_dispatch_config,
        combine_config=best_combine_config,
    )

    # Print nice summary on rank 0
    if rank == 0:
        dispatch_speedup = (
            best_dispatch_bw / worst_dispatch_bw if worst_dispatch_bw > 0 else 1.0
        )
        combine_speedup = (
            best_combine_bw / worst_combine_bw if worst_combine_bw > 0 else 1.0
        )

        logger.info("=" * 70)
        logger.info(f"[DeepEP Autotune] RESULTS (total time: {total_elapsed:.1f}s)")
        logger.info("=" * 70)

        if is_internode:
            logger.info(f"  DISPATCH (num_sms={best_dispatch_sms}):")
            logger.info(
                f"    Best:  nvl={best_dispatch_nvl:2d}, rdma={best_dispatch_rdma:2d} -> {best_dispatch_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_dispatch_nvl:2d}, rdma={worst_dispatch_rdma:2d} -> {worst_dispatch_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {dispatch_speedup:.2f}x over worst config")
            logger.info(f"  COMBINE (num_sms={best_combine_sms}):")
            logger.info(
                f"    Best:  nvl={best_combine_nvl:2d}, rdma={best_combine_rdma:2d} -> {best_combine_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_combine_nvl:2d}, rdma={worst_combine_rdma:2d} -> {worst_combine_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {combine_speedup:.2f}x over worst config")
        else:
            logger.info(f"  DISPATCH (num_sms={best_dispatch_sms}):")
            logger.info(
                f"    Best:  nvl={best_dispatch_nvl:2d} -> {best_dispatch_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_dispatch_nvl:2d} -> {worst_dispatch_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {dispatch_speedup:.2f}x over worst config")
            logger.info(f"  COMBINE (num_sms={best_combine_sms}):")
            logger.info(
                f"    Best:  nvl={best_combine_nvl:2d} -> {best_combine_bw:6.1f} GB/s"
            )
            logger.info(
                f"    Worst: nvl={worst_combine_nvl:2d} -> {worst_combine_bw:6.1f} GB/s"
            )
            logger.info(f"    Speedup: {combine_speedup:.2f}x over worst config")

        logger.info("=" * 70)

    return result
