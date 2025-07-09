# Tensor Parallel Gradient Flow in DeepSeek V3 k_pe Expansion

## The Problem: Gradient Flow Issue

### Forward Pass (What happens currently)

```
Rank 0 (TP rank 0):                    Rank 1 (TP rank 1):
┌─────────────────────┐                ┌─────────────────────┐
│ k_pe                │                │ k_pe                │
│ (bsz, seqlen, 1, d) │                │ (bsz, seqlen, 1, d) │
│ [SHARED PARAMETER]  │                │ [SHARED PARAMETER]  │
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │ expand(-1,-1,4,-1)                   │ expand(-1,-1,4,-1)
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ k_pe_expanded       │                │ k_pe_expanded       │
│ (bsz, seqlen, 4, d) │                │ (bsz, seqlen, 4, d) │
│ [LOCAL HEADS 0-3]   │                │ [LOCAL HEADS 4-7]   │
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │ concat with k_nope                   │ concat with k_nope
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ k (final)           │                │ k (final)           │
│ (bsz, seqlen, 4, D) │                │ (bsz, seqlen, 4, D) │
└─────────────────────┘                └─────────────────────┘
```

### Backward Pass (THE PROBLEM)

```
Rank 0:                                Rank 1:
┌─────────────────────┐                ┌─────────────────────┐
│ grad_k              │                │ grad_k              │
│ (bsz, seqlen, 4, D) │                │ (bsz, seqlen, 4, D) │
│ [GRAD FOR HEADS 0-3]│                │ [GRAD FOR HEADS 4-7]│
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │ split grad for k_pe part             │ split grad for k_pe part
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ grad_k_pe_expanded  │                │ grad_k_pe_expanded  │
│ (bsz, seqlen, 4, d) │                │ (bsz, seqlen, 4, d) │
│ [GRAD FOR HEADS 0-3]│                │ [GRAD FOR HEADS 4-7]│
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │ standard expand backward             │ standard expand backward
          │ (sum across head dim)                │ (sum across head dim)
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ grad_k_pe           │                │ grad_k_pe           │
│ (bsz, seqlen, 1, d) │                │ (bsz, seqlen, 1, d) │
│ [PARTIAL GRADIENT]  │                │ [PARTIAL GRADIENT]  │
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │ ❌ NO COMMUNICATION ❌               │ ❌ NO COMMUNICATION ❌
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ k_pe parameter      │                │ k_pe parameter      │
│ GETS PARTIAL GRAD   │                │ GETS PARTIAL GRAD   │
│ ❌ INCORRECT! ❌    │                │ ❌ INCORRECT! ❌    │
└─────────────────────┘                └─────────────────────┘
```

**Problem**: Each rank only gets partial gradients for the shared `k_pe` parameter!

## The Solution: Custom Tensor Parallel Expand

### Our Custom Backward Pass

```
Rank 0:                                Rank 1:
┌─────────────────────┐                ┌─────────────────────┐
│ grad_k_pe_expanded  │                │ grad_k_pe_expanded  │
│ (bsz, seqlen, 4, d) │                │ (bsz, seqlen, 4, d) │
│ [GRAD FOR HEADS 0-3]│                │ [GRAD FOR HEADS 4-7]│
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │ sum across head dim                  │ sum across head dim
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ grad_k_pe_local     │                │ grad_k_pe_local     │
│ (bsz, seqlen, 1, d) │                │ (bsz, seqlen, 1, d) │
│ [LOCAL CONTRIBUTION]│                │ [LOCAL CONTRIBUTION]│
└─────────────────────┘                └─────────────────────┘
          │                                      │
          │              ALL-REDUCE              │
          │ ◄──────────────────────────────────► │
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ grad_k_pe_final     │                │ grad_k_pe_final     │
│ (bsz, seqlen, 1, d) │                │ (bsz, seqlen, 1, d) │
│ [COMPLETE GRADIENT] │                │ [COMPLETE GRADIENT] │
│ ✅ CORRECT! ✅     │                │ ✅ CORRECT! ✅     │
└─────────────────────┘                └─────────────────────┘
          │                                      │
          ▼                                      ▼
┌─────────────────────┐                ┌─────────────────────┐
│ k_pe parameter      │                │ k_pe parameter      │
│ GETS FULL GRADIENT  │                │ GETS FULL GRADIENT  │
│ ✅ CORRECT! ✅     │                │ ✅ CORRECT! ✅     │
└─────────────────────┘                └─────────────────────┘
```

## Code Implementation

### Standard Expand (Problematic)
```python
# Forward
k_pe_expanded = k_pe.expand(-1, -1, n_local_heads, -1)

# Backward (automatic, but wrong for TP)
# Each rank gets: grad_k_pe = sum(grad_k_pe_expanded, dim=2, keepdim=True)
# ❌ Missing all-reduce across TP ranks!
```

### Our Custom Tensor Parallel Expand
```python
class TensorParallelExpand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, expand_shape):
        # Same as standard expand
        return input_tensor.expand(expand_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Step 1: Sum across expanded dimension (local aggregation)
        grad_input = torch.sum(grad_output, dim=ctx.expanded_dim, keepdim=True)

        # Step 2: All-reduce across TP ranks (global aggregation)
        if torch.distributed.is_initialized() and get_world_size() > 1:
            all_reduce(grad_input)  # ✅ This is the key fix!

        return grad_input, None
```

## Why This Matters

### Without the Fix:
- Each TP rank computes gradients for only its local heads
- `k_pe` parameter gets incomplete gradients
- Parameter updates are incorrect
- Training becomes unstable or suboptimal

### With the Fix:
- Each TP rank computes gradients for its local heads
- All-reduce aggregates gradients from all TP ranks
- `k_pe` parameter gets complete, correct gradients
- Training is stable and optimal

## Mathematical Explanation

Let's say we have 8 total heads split across 2 TP ranks (4 heads each):

**Without fix:**
- Rank 0: `grad_k_pe = sum(grad_heads_0_to_3)`
- Rank 1: `grad_k_pe = sum(grad_heads_4_to_7)`
- Each rank updates `k_pe` with only partial gradients ❌

**With fix:**
- Rank 0: `local_grad = sum(grad_heads_0_to_3)`
- Rank 1: `local_grad = sum(grad_heads_4_to_7)`
- All-reduce: `final_grad = local_grad_rank0 + local_grad_rank1`
- Both ranks get: `grad_k_pe = sum(grad_heads_0_to_7)` ✅

This ensures that the shared `k_pe` parameter receives the complete gradient from all attention heads across all TP ranks, which is essential for correct training dynamics.
