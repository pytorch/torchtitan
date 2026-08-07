# ATE-Bench — New-Feature tasks (4 tasks)

Each task mimics integrating a recently published modeling architecture into a
training framework. The agent is given the same materials an engineer would
consult: the arXiv paper and the reference implementation (`references.md`). From
these, the agent must produce a training script that integrates the new feature
into the base model and runs on C4 (TorchTitan native; paper used DCLM) under the
fixed config below. The harness is **wired up**: prompts (`nf1..nf4`), per-feature
judge rules (`rules/`), runner (`runner/run_feature.py`, worktree-isolated), loss
check (`runner/checks/nf_loss_curve.py`), and LLM judge (`runner/judge.py`).
Running it requires GPUs + C4 data.

Fixed evaluation config (shared with operate-and-profile tasks): parallelism mesh
`PP=4, EP=2, DP=1`, sequence length `2048`, global batch size `1024`, precision
`BF16`, on 8 GPUs.

Coverage rationale: two tasks revise the attention mechanism and two require
changes at the FFN (MoE) layer, so the suite spans the architectural subsystems a
training framework must accommodate. None of the four architectures had been
integrated into any of the three compared frameworks at the time of the study, so
all frameworks start from the same point.

## Tasks (Appendix B.3.1)

### NF1 — Diff (Differential Attention) [Ye et al.]
Differential attention replaces a single softmax-attention map with the difference
of two parallel softmax maps,
`Attn(Q,K,V) = (softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)) V`, where `lambda`
is a learned per-head scalar. Integrating it involves splitting the per-head Q/K
projections in half and introducing `lambda` as a trainable parameter with the
published initialisation schedule.

### NF2 — DynMoE
Standard MoE fixes the number of activated experts `k` per token. DynMoE replaces
top-k selection with a per-expert sigmoid gate and a learned threshold, so the
count of active experts varies per token. Integrating it involves replacing the
top-k routing kernel of the framework and reformulating the load-balancing
auxiliary loss for a variable-k regime.

### NF3 — MoBA (Mixture of Block Attention)
MoBA partitions the key/value sequence into fixed-size blocks and routes each
query to the top-k blocks selected by a learned gate, yielding sub-quadratic
attention cost in the sequence length. Integrating it involves inserting
block-level routing between the Q/K projection and the attention computation,
composing with the existing attention backend, and preserving causal masking
inside each selected block.

### NF4 — MoE++
MoE++ augments a standard MoE expert pool with three zero-computation expert types
(`zero`, `copy`, and `constant`), allowing easy tokens to be routed past the
feed-forward layer entirely. Integrating it involves extending the expert pool
definition with the zero-computation variants, widening the router output to cover
them, and adjusting the load-balancing loss to prevent the zero experts from
absorbing all easy tokens.

## Correctness (Appendix B.3.2)
Two axes, both must hold:
1. **Loss axis:** cross-entropy loss must decrease across the 64-step run and
   remain finite (no explosion, no NaN) — evidence the modified pipeline produces
   a learnable model.
2. **Rule axis:** each task decomposes into three required components (the
   architectural elements that distinguish the new feature from the baseline).
   The rules are fixed before inspecting any attempt. Each attempt is judged by an
   independent `claude-opus-4-7` session at xhigh effort, given the three rules
   and the git diff produced by the agent against `main`; the judge returns
   PASS/FAIL per rule. This guards against a passing-loss reading from an
   unchanged model.
