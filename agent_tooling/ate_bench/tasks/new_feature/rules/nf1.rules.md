# NF1 (Differential Attention) — correctness rules

Three required components. Each is PASS only if the agent's `git diff` against
`main` clearly implements it; otherwise FAIL.

1. **Split Q/K into two halves.** The per-head query and key projections are split
   into two groups (`Q1,Q2` and `K1,K2`), producing two separate attention score
   maps `softmax(Q1 K1^T)` and `softmax(Q2 K2^T)` over the same value tensor.
2. **Learned per-head lambda.** A trainable per-head scalar `lambda` is introduced
   (e.g. via the reparameterisation `lambda = exp(l_q1·l_k1) - exp(l_q2·l_k2) +
   lambda_init`) and initialised on the published schedule (depth-dependent
   `lambda_init`). It must be a registered learnable parameter, not a constant.
3. **Difference of the two maps.** The attention output uses the *difference*
   `softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)` applied to `V` (with the
   published sublayer/group norm and `(1 - lambda_init)` output scaling), not a
   single softmax map.
