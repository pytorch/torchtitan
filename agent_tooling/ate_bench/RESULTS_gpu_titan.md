# ATE-Bench GPU tasks — validation status (TorchTitan)

This records what was validated end-to-end for the **Operate & Profile** and
**New-Feature** categories on 8×H100, and what blocks a paper-fidelity run.

## ✅ Validated: the fixed mesh trains on 8×H100
`setup/train.sh` with the corrected mapping (`pp=4, dp_shard=-1 (auto→2), ep=2`)
launches and trains the `deepseek_v3_debugmodel` (6 layers, 8 experts):

```
Model deepseek_v3 debugmodel size: 33,014,016 total parameters
PP rank 0 building stage_idx 0 ...   # 4 pipeline stages, 1F1B, 8 microbatches
Applied fully_shard to the model
Trainer initialized: local batch 8, global batch 1024, grad-accum 64, seq 2048
Training starts at step 1
step: 1  loss: ...  grad_norm: 3.43  tps: 2,219  mfu: 0.12%
step: 2  loss: ...  grad_norm: 4.31
```
So the operate/feature harness *can* drive real distributed MoE training — the
mesh-notation fix (EP carved from the data axis, not multiplicative) is confirmed
on hardware.

## ⚠️ The debug model is not a meaningful numeric reproduction
- **Degenerate loss.** Step loss prints as a constant `-128.0` (a valid CE loss
  for a fresh model is ~ln(vocab) ≈ 10). The `33M` debug model on the tiny bundled
  `c4_test` set (which re-loops *within* a single step) does not learn — so the
  new-feature **loss-decreases** check cannot pass on it regardless of the agent's
  change.
- **Slow at the paper config.** `global_batch_size=1024` → grad-accum 64 →
  ~107 s/step on the debug model. The paper's `gbs=1024` targets 30B-class models
  on DCLM, not a debug model.

**Conclusion:** the debug model is good for *plumbing* validation (the mesh, the
launcher, log parsing, GPU-time monitoring) but not for reproducing the paper's
GPU numbers.

## ❌ What a paper-fidelity GPU run needs (not available here)
- A **real MoE model + released checkpoint** (paper: Qwen3-30B-A3B / DeepSeek-V2-
  Lite / GPT-OSS-20B). OP3/OP4 explicitly "resume from the released HF checkpoint";
  NF needs a model that actually learns in 64 steps.
- **vLLM + lm-evaluation-harness** for OP2 (not installed in this env).
- **DCLM/C4 at scale** (only the tiny `c4_test` is bundled).
- Multi-hour compute per task × 3 runs × (operate+feature) × the 3 frameworks the
  paper compares — and PithTrain/Megatron-LM checkouts aren't present, so the
  cross-framework headline (62%/64%) is out of scope here regardless.

## Net
- **Mesh + launcher: validated on 8×H100.** ✅
- **Operate/feature numeric reproduction: blocked** on real model/checkpoint/deps,
  not on the harness. The harness, checks, GPU monitor, and judge are in place and
  unit-tested; point them at a real MoE config + checkpoint to produce real numbers.
- The **Q&A column is the reproducible slice in this environment** (see
  `RESULTS_qa_titan.md`).
