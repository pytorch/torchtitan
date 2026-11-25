# RL policy losses

Every policy loss is a `PolicyGradientLoss`: a per-token surrogate on the tokens the
generator sampled. The generator's logprobs are the old policy, so each token's importance
ratio is `r = p / q`, trainer over generator. A subclass defines

- `token_loss(ratio, advantages)`: the per-token loss, and
- `log_token_weight(log_ratio, advantages)`: `log w`, where `-A * w(r) * grad(log p)` is the
  gradient of `token_loss` (`-inf` where the loss has no gradient). For `DAPOLoss`,
  `w = r` unless the clip binds in the advantage's direction, and 0 there.

| Loss | Surrogate |
| --- | --- |
| `DAPOLoss` | PPO clip with separate lower and upper bounds ("clip-higher") |
| `GRPOLoss` | `DAPOLoss` with one symmetric bound |

## Score centering

The trainer's `p` and the generator's `q` always differ a little (kernels, quantization, MoE
routing, stale weights). Then the expected gradient at every prefix gains a term that holds
no reward information: it distills the trainer toward the generator, scaled by how promising
the prefix is, and weight syncs copy the moved trainer back into the generator, so it
compounds. Score centering ([arXiv 2609.20807](https://arxiv.org/abs/2609.20807)) subtracts
that term at every position. It composes with any `PolicyGradientLoss`:

```python
loss_fn=GRPOLoss.Config(
    global_vocab_size=decoder_vocab_size(trainer_model),
    score_centering=ScoreCentering.Config(num_topk_logprobs=32),
)
```

The controller then asks the generator for its top-k logprobs on every training request
(validation requests get none) and the loss receives them as `[T, k]` inputs:

```
generator (vLLM)            batcher                          PolicyGradientLoss
  top-k ids, logprobs  -->    one row per loss token   -->     correction = sum_H c_v * log p_v
  [num_generated, k]          scattered to [T, k] on device    token_loss += A * correction
```

The coefficients `c_v` use the loss's `log_token_weight` (see `ScoreCentering`'s docstring
for the formula and a worked example). They are 0 when the generator's top-k logprobs equal
the trainer's, which batch-invariant mode guarantees: `rl_grpo_qwen3_0_6b_varlen_batch_invariant`
enables score centering, and the RL loss golden checks that it stays a no-op there.

Requirements and limits:

- `VLLMEngine` only for now; `SGLangEngine` and Verifiers rollouts raise.
- Transport costs 8 * k bytes per generated token (k int32 ids and k fp32 logprobs).
- The paper evaluates score centering with REINFORCE, truncated IS and masked IS, not PPO
  clipping; its composition with `DAPOLoss`/`GRPOLoss` follows the same formula.

Metrics: `score_centering/generator_topk_mass/mean` and `score_centering/trainer_topk_mass/mean`
(the probability mass the top-k covers; the tail model approximates the rest) and
`score_centering/coefficient_l1/mean` (the correction's size before the advantage).
