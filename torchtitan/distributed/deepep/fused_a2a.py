import torch
from deep_ep import Buffer
from deep_ep.utils import EventOverlap, EventHandle

_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.
    Args:
        x (torch.Tensor): Input tensor
    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.
    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed
    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        # NOTE: Original code from torchtitan-amd used primus_turbo.pytorch.deep_ep.Buffer
        # which has use_default_stream_as_comm_stream parameter. However, the standard
        # deep_ep.Buffer API does not have this parameter. Removing it to match the
        # deep_ep API. The default behavior should be equivalent.
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


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
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        # TODO(deepep-fork, phuc): The local DeepEP does not support num_recv_tokens_per_expert_as_cuda parameter
        # which exists in torchtitan-amd's forked DeepEP. When we fork DeepEP, we should add this parameter
        # back to avoid the tensor conversion overhead below. The parameter allows DeepEP to return a CUDA
        # tensor directly instead of a Python list for num_recv_tokens_per_expert_list.
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            # num_recv_tokens_per_expert_as_cuda=use_cuda_num_token_per_expert,  # (phuc) Not supported in local DeepEP
            num_worst_tokens=num_worst_tokens,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream

        # WORKAROUND (phuc): for local DeepEP without num_recv_tokens_per_expert_as_cuda support:
        # The local DeepEP always returns num_recv_tokens_per_expert_list as a Python list.
        # The forked DeepEP in torchtitan-amd has num_recv_tokens_per_expert_as_cuda parameter
        # which when True, returns a CUDA tensor directly instead of a Python list.
        #
        # ORIGINAL CODE (from torchtitan-amd with forked DeepEP):
        # if not use_cuda_num_token_per_expert:
        #     tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)  # list -> CPU tensor
        # else:
        #     tokens_per_expert = num_recv_tokens_per_expert_list  # Already CUDA tensor from DeepEP
        #
        # MODIFIED CODE (phuc, workaround for local DeepEP):
        # NOTE(phuc): claudecode fixed device placement issue - both branches now use device=x.device
        # Previously the first branch created CPU tensor which caused:
        # "RuntimeError: Expected all tensors to be on the same device, but got offs is on cpu"
        if not use_cuda_num_token_per_expert:
            tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, device=x.device)  # list -> CUDA tensor (FIXED)
        else:
            # Manual conversion: list -> CUDA tensor (workaround since DeepEP doesn't do it)
            # TODO(deepep-fork, phuc): Restore original `tokens_per_expert = num_recv_tokens_per_expert_list`
            # when we fork DeepEP and add num_recv_tokens_per_expert_as_cuda support
            tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, device=x.device)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None



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
):
    """Perform fused dispatch operation if deep_ep is available.
    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        previous_event: Previous CUDA event
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
    )

def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
    """Perform fused combine operation if deep_ep is available.
    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        previous_event: Previous CUDA event
    Returns:
        Result of FusedCombine
    """
    return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

def set_deepep_num_sms(num_sms):
    """Sets the number of SMs to use for DeepEP"""
    Buffer.set_num_sms(num_sms)
