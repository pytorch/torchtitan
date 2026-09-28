# ATE-Bench New-Feature results — TorchTitan

> **Status: all 4 architectures reproduced end-to-end (1 run each).** Each was
> integrated into `deepseek_v3` by a fixed headless agent, trained 64 steps on real
> C4, and judged on its 3 architectural rules. **All 4 CORRECT** (loss decreases +
> judge PASS on all rules).

## Setup
- **Framework:** TorchTitan @ branch tip (per-attempt git worktree off the tip).
- **Base model:** `deepseek_v3_debugmodel` (6 layers, 8 experts), mesh `pp=4, ep=2,
  dp_shard=auto`, seq 2048, **global batch 64**, real `allenai/c4`, BF16, 8×H100.
  (gbs=64 not the paper's 1024: 1024 is for 30B models; on the debug model it gives
  grad-accum 64 + a degenerate loss. gbs=64 learns cleanly — see RESULTS_gpu_titan.)
- **Implementing agent:** headless Claude Code (`opus-4-6` default), full tools.
- **Correctness (paper B.3):** (1) 64-step CE loss decreases + finite; (2) an
  independent LLM judge (`judge.py`) PASSes the `git diff` vs base against 3 fixed
  per-feature rules. Both must hold.

## Results

| Task | Architecture | Loss (median first-8 → last-8) | Judge rules | Diff | Verdict |
|---|---|---|---|---|---|
| NF1 | Differential Attention | 4.29 → 2.77 | 3/3 PASS | 157 ln | ✅ CORRECT |
| NF2 | DynMoE | 4.42 → 2.86 | 3/3 PASS | 329 ln | ✅ CORRECT |
| NF3 | MoBA | 4.56 → 2.81 | 3/3 PASS | 151 ln | ✅ CORRECT |
| NF4 | MoE++ | 4.53 → 2.87 | 3/3 PASS | 337 ln | ✅ CORRECT |

**4/4 architectures integrated, trained, and verified.** What the judge confirmed:
- **NF1 Diff:** `q.chunk(2)`/`k.chunk(2)`; learned per-head `lambda_{q1,k1,q2,k2}`
  params; output `softmax(Q1K1^T)V − lambda·softmax(Q2K2^T)V`.
- **NF2 DynMoE:** per-expert sigmoid gate (cosine sim), learned `threshold_E`
  `nn.Parameter` → variable-k via hard mask; fixed-k aux loss disabled.
- **NF3 MoBA:** configurable `block_size` (128); mean-pooled block-key gate
  (`matmul(q, k_mean)`); causal masking (future blocks excluded, current handled).
- **NF4 MoE++:** zero/copy/constant zero-computation experts; router widened
  (`num_experts + num_zc`); heterogeneous load-balance loss for the zero experts.

## Effort (single run; TorchTitan column only)
| Task | Agent turns | Active GPU-time (GPU-s, 8 GPUs) |
|---|---|---|
| NF2 | 55 | 3320 (~7 min wall) |
| NF3 | 33 | 3163 |
| NF4 | 46 | 3499 |
(NF1's turn/GPU counts were lost when its first attempt over-ran the timeout; its
verdict was salvaged from the worktree + log.)

## Caveats
- **1 run, not 3.** The paper medians over 3 attempts. All 4 already PASS at 1 run,
  so the pass/fail signal is solid; 3-run medians would only tighten the *effort*
  numbers (and those are TorchTitan-only — the cross-framework 62%/64% headline
  needs PithTrain/Megatron, absent here).
- **Debug-scale base + gbs=64**, not paper-scale. This validates that each
  architecture integrates and trains; it is not a paper-fidelity numeric match.
- **Judge = `opus-4-6` default**, not the paper's `opus-4-7 @ xhigh`.
