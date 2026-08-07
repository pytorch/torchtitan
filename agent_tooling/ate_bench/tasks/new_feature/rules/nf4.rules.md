# NF4 (MoE++) — correctness rules

Three required components. Each is PASS only if the agent's `git diff` against
`main` clearly implements it; otherwise FAIL.

1. **Zero-computation experts added.** The expert pool is extended with the three
   zero-computation expert types — `zero` (outputs zero), `copy` (identity), and
   `constant` (a learned constant vector) — alongside the existing FFN experts.
2. **Router widened to cover them.** The router output dimension is widened so the
   gate can select the zero-computation experts for a token (they participate in
   routing, not bypassed).
3. **Load-balancing loss adjusted.** The load-balancing loss is adjusted to
   prevent the zero-computation experts from absorbing all easy tokens (e.g. a
   gating-residual / capped-share term), not the unmodified baseline aux loss.
