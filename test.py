"""
Minimal repro: cuDNN SDPA backward stride mismatch with Context Parallel.

This is a pre-existing PyTorch bug where cuDNN SDPA backward requires
grad_output.strides() == output.strides(), but CP ring attention backward
produces mismatched strides due to _SDPAMerger arithmetic + torch.chunk/cat.

Run with 2 GPUs (must be H100 or other GPU that supports cuDNN SDPA):

    torchrun --nproc_per_node=2 repro_cudnn_cp_stride.py

Expected error:
    RuntimeError: same_strides(o, dO_) INTERNAL ASSERT FAILED at
    .../aten/src/ATen/native/cudnn/MHA.cpp:1628

Root cause:
    In _templated_ring_attention_backward(), the merged `out` from
    _SDPAMerger has non-standard strides (due to arithmetic ops + chunk/cat),
    while `grad_out` has standard contiguous strides from autograd.
    cuDNN backward asserts these must match. The fix is to add .contiguous()
    for out_ and dout before passing to the cuDNN backward op, similar to
    what's already done for logsumexp on line 568 of _attention.py.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._context_parallel import (
    _context_parallel_shard,
    _ContextParallel,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend


class SDPAModule(nn.Module):
    """Thin wrapper so we can parallelize_module with _ContextParallel."""

    def forward(self, q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    device_mesh = init_device_mesh("cuda", (world_size,))
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    # Attention dimensions
    bs = 2
    nheads = 8
    seq_len = 1024  # per-rank seq_len after sharding
    head_dim = 32
    seq_dim = 2  # (bs, nheads, seq, head_dim)

    # Create identical tensors on all ranks
    torch.manual_seed(42)
    full_seq = seq_len * world_size
    q = torch.randn(bs, nheads, full_seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, full_seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, full_seq, head_dim, device=device, dtype=dtype)

    with torch.no_grad():
        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

    # Set up CP via parallelize_module (same pattern as torchtitan)
    cp_plan = _ContextParallel(
        seq_dim=seq_dim,
        attention_type=_ContextParallel.AttentionType.SDPA,
    )
    attention = SDPAModule()
    attention = parallelize_module(attention, device_mesh, cp_plan)

    # Shard Q, K, V along sequence dim with load balancing
    load_balancer = _HeadTailLoadBalancer(full_seq, world_size, device)
    cp_q, cp_k, cp_v = _context_parallel_shard(
        device_mesh, (q, k, v), (seq_dim,) * 3, load_balancer=load_balancer
    )
    cp_q.requires_grad_(True)
    cp_k.requires_grad_(True)
    cp_v.requires_grad_(True)

    _enable_context_parallel_dispatcher()

    # Force cuDNN backend — this triggers the stride mismatch on backward
    if rank == 0:
        print(f"Running CP ring attention with cuDNN SDPA backend...")
        print(f"  world_size={world_size}, bs={bs}, nheads={nheads}, "
              f"seq_len={full_seq} (per-rank: {seq_len}), head_dim={head_dim}")

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = attention(cp_q, cp_k, cp_v, is_causal=True)
        # Backward triggers the stride mismatch assertion in cuDNN
        out.sum().backward()

    if rank == 0:
        print("SUCCESS — no stride mismatch (bug may be fixed!)")


if __name__ == "__main__":
    main()
