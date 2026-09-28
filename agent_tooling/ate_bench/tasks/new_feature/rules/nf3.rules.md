# NF3 (MoBA) — correctness rules

Three required components. Each is PASS only if the agent's `git diff` against
`main` clearly implements it; otherwise FAIL.

1. **Block partitioning.** The key/value sequence is partitioned into fixed-size
   blocks of a configurable block size.
2. **Top-k block routing by a learned gate.** Each query is routed to the top-`k`
   blocks selected by a learned gate (e.g. query · mean-pooled-block-key scores),
   inserted between the Q/K projection and the attention computation. Attention is
   computed only over the selected blocks (sub-quadratic), not all keys.
3. **Causal masking preserved per block.** Causal correctness is preserved: a
   query never attends to future tokens, the current block is handled causally
   (always selected / partially masked), and past-block selection respects the
   causal order.
