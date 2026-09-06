# NF2 (DynMoE) — correctness rules

Three required components. Each is PASS only if the agent's `git diff` against
`main` clearly implements it; otherwise FAIL.

1. **Per-expert sigmoid gate.** Routing uses an independent sigmoid gate per
   expert (each expert gated on its own), replacing the softmax-over-experts
   top-`k` selection in `TokenChoiceTopKRouter`.
2. **Learned threshold → variable-k.** A learned threshold decides which experts
   are active per token, so the number of activated experts varies per token
   (not a fixed `top_k`). The dispatch path must handle a variable number of
   selected experts per token.
3. **Variable-k load-balancing loss.** The auxiliary load-balancing loss is
   reformulated for the variable-`k` regime (it must not assume a constant `k`
   per token, e.g. the fixed-`k` switch/aux loss).
