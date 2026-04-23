# MoE Inference Numerics Debugging — Final Findings

## Root Cause

**The bug is in core TorchTitan's `parallelize_qwen3(inference=True)`, NOT in the
vLLM wrapper.** A pure TorchTitan forward pass (no vLLM, no KV cache, no attention
wrapper) produces garbage output on the real Qwen3-30B-A3B model.

```
torchrun --nproc_per_node=4 test_pretrain_generate.py
→ "The capital of France is0000000000"
```

## Confirmed Facts

### 1. The vLLM wrapper is NOT the cause
- Replacing TorchTitan's MoE with vLLM's FusedMoE → same garbage
- Pure pretraining forward (no vLLM at all) → same garbage
- Dense Qwen3-0.6B with the same wrapper → correct ("Paris...")

### 2. The MoE computation is correct in isolation
- AllToAllTokenDispatcher: standalone test passes (diff=0.000000)
- MoE.forward with DTensor weights: standalone test passes (diff=0.000000)
- `grouped_mm` vs for-loop expert forward: both produce same garbage
- MoE DTensor boundary trace: all cross-rank checksums MATCH at each stage

### 3. Attention is correct
- Per-layer attention output trace: MATCH across all ranks on all 48 layers
- Layer output trace: MATCH across all ranks on all 48 layers
- Dense model (0.6B) with same attention wrapper: correct

### 4. Weights are loaded correctly
- Expert weights: diff=0.000000 vs direct safetensors loading (all ranks)
- Attention weights (Q, K, V, O): diff=0.000000 vs HF checkpoint
- Norm weights: diff=0.000000
- All 579 params non-zero

### 5. Two bugs in `apply_moe_ep_tp` output_layouts
- **Bug A**: `loss_parallel=True` (default) → output layer uses `Shard(-1)` on vocab
  dim. `full_tensor()` all-gather produces compressed logits (std=0.61, range -4 to 3.7).
  Fix: set `disable_loss_parallel=True` for inference.
- **Bug B**: `output_layouts=(Partial(),)` for EP inference → doubles values via
  all-reduce. Fix: use `Replicate()` when `inference=True and ep_mesh is not None`.

### 6. Even with Bug A fixed, output is still garbage
With `disable_loss_parallel=True`, the model consistently generates token 15 ("0")
for every position. The core `TensorParallel()` on MoE experts +
`PrepareModuleInputOutput(Partial → Shard(1))` produces wrong hidden states
that accumulate across 48 layers.

### 7. Debug model works, real model doesn't
- debugmodel_moe (64 experts, 8 layers, random weights): ✅ correct
- Qwen3-30B-A3B (128 experts, 48 layers, trained weights): ❌ garbage

Random weights don't expose the numerical error because predictions are already
random. Trained weights amplify small errors through 48 layers of attention + MoE.

## Layer-by-Layer Diagnostic (`test_layer_by_layer.py`)

### Full Layer Norms (TP=4, disable_loss_parallel=True)

```
[EMB] norm=1.5648
[L00] attn=    1.8028 moe=    3.4007 out=    4.9821
[L01] attn=    1.7005 moe=  122.3890 out=  122.5654    ← MoE EXPLODES
[L02] attn=    1.0347 moe= 1033.3572 out= 1136.2219    ← MoE EXPLODES
[L03] attn=    1.0365 moe=  771.1474 out= 1891.3585    ← MoE still huge
[L04] attn=    0.9954 moe=   11.6984 out= 1891.0930    ← back to normal
[L10] attn=    1.8050 moe=    0.5996 out= 1890.8101
[L20] attn=    3.9084 moe=    7.6511 out= 1890.6615
[L30] attn=    3.2083 moe=    3.9674 out= 1890.4843
[L40] attn=    7.2702 moe=   10.8713 out= 1893.7061
[L45] attn=   10.2838 moe=   20.8424 out= 1878.8853
[L46] attn=   15.5427 moe=  381.9112 out= 1542.9968
[L47] attn= 1561.1478 moe=  720.2589 out=  960.5065
[OUT] argmax=15 (0)
```

### Analysis

- **Attention is fine**: norms are 1-15 across all layers (reasonable)
- **MoE explodes at L01-L03**: L00 MoE norm=3.4 (normal), L01 jumps to 122,
  L02 to 1033. The residual saturates at ~1890 by L03 and stays flat.
- **After saturation**: MoE contribution is tiny relative to ~1890 residual,
  so the model produces near-uniform logits.

### Root Cause Hypothesis

The MoE output at L01 is ~36× larger than L00. This is not a simple TP×4
scaling error. The explosion must be caused by the `TensorParallel()` +
`Partial → Shard(1)` reduce-scatter producing WRONG values at L01 specifically.

### Deep Trace Layer 01 MoE (`test_layer_by_layer.py`)

```
[L01 DEEP] ffn_norm_out:  norm=26.15  place=Shard(1)
[L01 DEEP] moe_input:     norm=26.15  place=Replicate (after all-gather)
[L01 DEEP] router_out:    norm=0.95   plain (routing scores)
[L01 DEEP] experts_out:   norm=2.69   plain (BEFORE post-hook)
[L01 DEEP] moe_output:    norm=122.39 place=Shard(1) (AFTER post-hook) ← 46× inflation!

experts_out local norms per rank: ['2.6871', '3.0298', '2.5595', '119.7212']
moe_output local norms per rank:  ['119.0967', '27.4945', '6.2530', '0.0000']
```

**RANK 3's expert output is 119.72** — 40× larger than other ranks (~2.5-3.0).

The `Partial → Shard(1)` reduce-scatter correctly sums all 4 ranks' partial
results, giving total norm ~122. Rank 3's inflated partial dominates.

The expert output hook captures output of `GroupedExperts.forward()` which
includes `token_dispatcher.combine()` + `scatter_add`. The raw expert
computation (before combine) on rank 3 may be producing inflated values
from its weight shard `w1[:, 576:768, :]`.

### L00 vs L01 Comparison
- L00 MoE norm = 3.4 (all ranks have similar partial norms)
- L01 MoE norm = 122.4 (rank 3 has 40× larger partial norm)

Same code path, same weight sharding. The difference is the INPUT to L01:
`ffn_norm(h)` where `h` has norm 4.98 from L00's output. After RMSNorm,
the input to the MoE is normalized to norm 26.15 — much larger than L00's
input (norm 1.56 from embedding). The larger input interacts with rank 3's
weight shard to produce inflated values.

This suggests a numerical sensitivity in the SwiGLU computation when the
input scale is larger, specifically for the weight shard on rank 3
(hidden dim 576-768).

### Next: Single-GPU Reference Comparison
Need to run the same model without TP to get the reference L01 MoE norm.
If the reference norm is also ~122, the TP computation is correct and the
model just has a large MoE output at L01. If the reference norm is ~3-5,
the TP computation is inflating values incorrectly.

## Environment
- PyTorch 2.13.0, vLLM 0.19.1, CUDA 13.0
- 8x NVIDIA H100 GPUs (testing with 4 GPUs, TP=4)
- Model: Qwen3-30B-A3B (128 experts, 48 layers, dim=2048, top_k=8)
- Checkpoint: `/data/users/jianiw/model/Qwen3-30B-A3B`
- Branch: `rl-moe-config` (rebased on `rl-parallel-plan`)

## Key Test Scripts (in `rl/scripts/`)

| Script | What It Tests |
|--------|--------------|
| `test_pretrain_generate.py` | Core TorchTitan autoregressive generation (no vLLM) |
| `test_pretrain_only.py` | Single forward pass with pretraining parallelize |
| `test_moe_dtensor_hooks.py` | DTensor boundary trace for MoE layers |
| `test_attention_trace.py` | Per-layer attention/MoE output trace |
| `test_ep_numerics.py` | End-to-end native vLLM vs TorchTitan comparison |
| `test_layer0_parity.py` | Layer 0 TP=4 vs reference comparison |
| `test_pretrain_no_moe_tp.py` | Forward without MoE TP (OOMs) |
