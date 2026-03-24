# Batch-Invariant Mode for RL Training

## Why Batch Invariance Matters

In GRPO/PPO-style RL training, the **generator** (vLLM) produces rollouts and
log-probabilities, then the **trainer** (TorchTitan) recomputes those same
log-probabilities during the training forward pass.  The importance sampling
ratio is:

```
r(θ) = π_train(a|s) / π_generator(a|s)
```

At the start of each iteration (before any gradient update), the weights are
identical, so this ratio **must** be exactly 1.0.  Any floating-point
discrepancy causes a non-unit ratio, which injects bias into the policy
gradient and breaks the on-policy assumption.

**Batch invariance** means: *the same input sequence produces bit-identical
results regardless of what other sequences are in the batch.*  This is required
because:

- The generator processes sequences with varying batch compositions due to
  continuous batching.
- The trainer processes sequences in fixed batches.
- Both sides must produce identical log-probabilities despite different batch
  compositions.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      Generator (vLLM)                        │
│                                                              │
│  VLLM_BATCH_INVARIANT=1                                      │
│  init_batch_invariance(attention_backend)                     │
│    ├─ ATen overrides: mm, addmm, bmm, log_softmax, mean.dim │
│    └─ Attention: num_splits=1 (FA2) / fixed_split_size (FI)  │
└──────────────────────┬───────────────────────────────────────┘
                       │  rollout log-probs
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Trainer (TorchTitan)                       │
│                                                              │
│  enable_batch_invariant_mode()                               │
│    ├─ ATen overrides: log_softmax, mean.dim (Triton kernels) │
│    ├─ torch.use_deterministic_algorithms(True)  (for matmul) │
│    ├─ Reduced-precision reductions disabled                  │
│    └─ TF32 disabled                                          │
│  Deterministic NCCL: Ring algo, 1 channel, Simple protocol   │
│  Attention: VLLMCompatibleFlashAttention with num_splits=1   │
└──────────────────────┬───────────────────────────────────────┘
                       │  recomputed log-probs
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                verify_logprob_identity()                      │
│  Checks torch.equal() between generator and trainer logprobs │
│  Reports: bitwise_identical, max_delta, log_ratio stats      │
└──────────────────────────────────────────────────────────────┘
```

## Sources of Non-Determinism and How They Are Addressed

### 1. Matmul (GEMM)

**Problem:** cuBLAS may select different algorithms or tile decompositions
depending on matrix dimensions, which change with batch size.

**Trainer solution:**
`enable_batch_invariant_mode()` in
[`batch_invariant.py`](../torchtitan/experiments/rl/batch_invariant.py)
configures:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

**Why not Triton matmul overrides?** During backward, autograd computes
`grad_weight = activation^T @ grad_output` where K = batch * seq.  Different
batch sizes produce different K values, changing the number of K-tiles in a
Triton kernel and therefore the fp32 accumulation boundaries.  cuBLAS
deterministic algorithms handle this correctly.

**Generator solution:** vLLM's `init_batch_invariance()` registers Triton
matmul kernels as ATen overrides for the forward-only inference path (no
backward pass, so the K-dimension issue doesn't apply).

### 2. log_softmax

**Problem:** CUDA's `_log_softmax` implementation may use non-deterministic
reduction order depending on the number of rows (batch * seq).

**Solution (both sides):** Triton kernel registered as an ATen override for
`aten::_log_softmax`.  One program per row with a fixed three-pass sequential
scan (find max, sum exp, compute output).  Since each row is processed
independently, changing the batch size only changes the number of rows, not the
computation for any given row.

```python
_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
```

### 3. mean (RMSNorm)

**Problem:** `aten::mean.dim` is used by RMSNorm.  The CUDA implementation may
use non-deterministic reduction depending on tensor dimensions.

**Solution (both sides):** Triton kernel registered as an ATen override for
`aten::mean.dim`.  Input is viewed as (M, N, K) where N is the reduction dim.
Each program handles one (m, k) pair and accumulates along N in a fixed
sequential order.

```python
_LIB.impl("aten::mean.dim", _mean_batch_invariant, dispatch_key)
```

### 4. Flash Attention

**Problem:** Flash Attention auto-tunes `num_splits` (split-KV parallelism)
based on batch size and sequence length.  Different splits change the
floating-point accumulation order.

**Solution:**
[`vllm_compat_attention.py`](../torchtitan/experiments/rl/models/vllm_compat_attention.py)
forces `num_splits=1` when batch-invariant mode is enabled:

```python
output = flash_attn_varlen_func(
    ...,
    num_splits=1 if is_batch_invariant_mode_enabled() else 0,
)
```

For FlashInfer backend, equivalent parameters `fixed_split_size` and
`disable_split_kv` are used.

The trainer also uses a custom `torch.autograd.Function`
(`FlashAttnWithBackward`) for the backward pass because vLLM's Flash Attention
API doesn't expose a differentiable backward.

### 5. NCCL Collectives (Multi-GPU)

**Problem:** NCCL may use Tree/CollNet algorithms or multiple channels for
all-reduce, changing the floating-point summation order between runs.

**Solution:**
[`trainer.py`](../torchtitan/experiments/rl/actors/trainer.py) sets NCCL
environment variables **before** `init_distributed`:

```python
os.environ["NCCL_ALGO"] = "Ring"           # Fixed reduction topology
os.environ["NCCL_MIN_NCHANNELS"] = "1"     # Single channel to fix
os.environ["NCCL_MAX_NCHANNELS"] = "1"     #   reduction order
os.environ["NCCL_PROTO"] = "Simple"        # Deterministic protocol
os.environ["NCCL_COLLNET_ENABLE"] = "0"    # Disable CollNet
os.environ["NCCL_NVLS_ENABLE"] = "0"       # Disable NVLink SHARP
```

### 6. Seeding and Global Determinism

**Solution:** `set_determinism()` from TorchTitan's distributed utilities:

- `torch.manual_seed(seed)`, `torch.cuda.manual_seed(seed)`
- `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

## Enabling Batch-Invariant Mode

### Trainer Side

```python
# In PolicyTrainer.__init__:

# 1. NCCL deterministic settings (before init_distributed)
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_MIN_NCHANNELS"] = "1"
os.environ["NCCL_MAX_NCHANNELS"] = "1"
os.environ["NCCL_PROTO"] = "Simple"
os.environ["NCCL_COLLNET_ENABLE"] = "0"
os.environ["NCCL_NVLS_ENABLE"] = "0"

# 2. Global determinism (after init_distributed)
set_determinism(parallel_dims, device, DebugConfig(seed=seed, deterministic=True))

# 3. Batch-invariant ATen overrides + cuBLAS settings
from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant_mode
enable_batch_invariant_mode()
```

### Generator Side

```python
# In PolicyGenerator.__init__:
os.environ["VLLM_BATCH_INVARIANT"] = "1"
init_batch_invariance(AttentionBackendEnum[config.attention_backend])
```

### Verification

After every training step, `verify_logprob_identity()` checks bitwise equality
between generator and trainer log-probabilities:

```python
result = verify_logprob_identity(vllm_token_log_probs, batch_token_log_probs)
# result["logprob_bitwise_identical"] == True  ← goal
# result["logprob_max_delta"] == 0.0
```

## Transparency via ATen Dispatch

All overrides use `torch.library.Library("aten", "IMPL")` to register at the
ATen dispatch level.  This means:

- **No model code changes needed.**  `F.linear`, `F.log_softmax`, `tensor.mean()`
  all transparently route to the batch-invariant kernels.
- **Composite ops decompose automatically.**  `aten::linear` decomposes into
  `aten::mm` + `aten::add`; `aten::softmax` decomposes into
  `aten::_log_softmax` + `aten::exp`.  Overriding the primitives covers all
  composite ops.

## Known Limitations

1. **Performance cost:** Deterministic cuBLAS algorithms and `num_splits=1` in
   Flash Attention are slower than auto-tuned defaults.
2. **torch.compile with inductor:** The inductor backend changes numerics.  Use
   `backend="eager"` or `backend="aot_eager"` only.
3. **Attention backward:** `FlashAttnWithBackward` materializes the full N x N
   attention matrix in the backward pass, which is slow and memory-intensive
   compared to fused flash attention backward.
