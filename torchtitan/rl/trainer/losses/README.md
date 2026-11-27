# RL policy losses

Every policy loss is a `PolicyGradientLoss`: a per-token surrogate on the tokens the
generator sampled. The generator's logprobs are the old policy, so each token's importance
ratio is `r = p / q`, trainer over generator. A subclass defines

- `token_loss(tokens)`: the per-token loss from the token's `ScoredTokens` (trainer and
  generator logprobs, their ratio, the advantage), and
- `log_token_weight(log_ratio, advantages, trainer_logprobs)`: `log w`, where
  `-A * w * grad(log p)` is the gradient of `token_loss` (`-inf` where the loss has no
  gradient). Every loss but DPPO ignores `trainer_logprobs`.

| Loss | Weight `w` on a token's score | Score centering |
| --- | --- | --- |
| `DAPOLoss` | `r`, 0 where the PPO clip binds in the advantage's direction (separate lower and upper bounds, "clip-higher") | yes |
| `GRPOLoss` | `DAPOLoss` with one symmetric bound | yes |
| `CISPOLoss` | `min(r, max_ratio)`: truncated IS | yes |
| `IcePopLoss` | `r` inside `[ratio_mask_low, ratio_mask_high]`, 0 outside: masked IS | yes |
| `DPPOLoss` | `min(r, max_ratio)`, 0 once `sign(A) * (p - q)` passes a threshold | yes (tail caveat below) |

## Score centering

The trainer's `p` and the generator's `q` always differ a little (kernels, quantization, MoE
routing, stale weights). A token's gradient `-A * w(r) * grad(log p)` then gives, at every
prefix, an expected update `V * sum_v q_v w(r_v) grad(log p_v)` that holds no reward
information: it distills the trainer toward the generator, scaled by how promising the prefix
is (`V`), and weight syncs copy the moved trainer back into the generator, so it compounds.
Where a loss's weight is the ratio itself, `q_v * r_v = p_v` and `sum_v p_v grad(log p_v) = 0`,
so the term is 0: it comes only from the tokens whose weight is clipped (DAPO/GRPO), truncated
(CISPO) or masked (IcePop), and these grow with the mismatch (`loss/ratio_clipped_frac`,
`loss/ratio_truncated_frac`, `loss/ratio_masked_frac`). Score centering
([arXiv 2609.20807](https://arxiv.org/abs/2609.20807)) subtracts that term at every position.
It composes with any `PolicyGradientLoss`:

```python
loss_fn=GRPOLoss.Config(
    global_vocab_size=decoder_vocab_size(trainer_model),
    score_centering=ScoreCentering.Config(num_topk_logprobs=32),
)
```

The controller then asks the generator for its top-k logprobs on every training request
(validation requests get none) and the loss receives them as `[T, k]` inputs:

```
generator (vLLM, SGLang)    TrainingBatchBuilder             PolicyGradientLoss
  top-k ids, logprobs  -->    one row per loss token   -->     correction = sum_H c_v * log p_v
  [num_generated, k]          scattered to [T, k] on device    token_loss += A * correction
```

The coefficients `c_v` use the loss's `log_token_weight` (see `ScoreCentering`'s docstring
for the formula and a worked example). They are 0 when the generator's top-k logprobs equal
the trainer's, which batch-invariant mode guarantees: `rl_grpo_qwen3_0_6b_varlen_batch_invariant`
enables score centering, and the RL loss golden checks that it stays a no-op there.

Requirements and limits:

- `VLLMEngine` and `SGLangEngine`, with the message-env and Verifiers rollouters, when sampling
  from the full vocabulary; with top-k/top-p sampling, see the next section.
- Transport costs 8 * k bytes per generated token (k int32 ids and k fp32 logprobs). A
  Verifiers rollout worker can hold more: glibc keeps the freed rows, ~4.5 KiB per in-flight
  token at k = 128 in a CPU repro (TODO in `verifiers/generation_server.py`).
- Trainer cost: the top-k logprobs come from the label's logprob and k + 1 gathered logits
  (`compute_topk_logprobs`; batch-invariant mode uses a full-vocabulary log-softmax instead,
  as vLLM does). For one loss chunk of 4,096 tokens and a 151,936-token vocabulary
  on one GB300 without TP, with the compiled loss region the recipes use, score centering
  (k = 32) raises the loss's peak memory from 1.16 to 2.32 GiB (about one more bf16
  `[T, V_local]`) and its forward + backward from 2.1 to 2.7 ms (eager: 9.27 GiB either way,
  15 -> 16 ms).
- The paper evaluates score centering with REINFORCE, truncated IS (`CISPOLoss`) and masked
  IS (`IcePopLoss`), not PPO clipping; its composition with `DAPOLoss`/`GRPOLoss` follows the
  same formula. With stale rollouts, the paper finds it best combined with truncated or
  masked IS.
- `DPPOLoss`: its trust region reads each tail token's own `p_v`, so the tail fold is exact
  only while no tail token can move past `divergence_threshold` (see `ScoreCentering`'s
  docstring). With k = 32 that needs `rho < 0.14` with `A > 0`.

Metrics: `score_centering/generator_topk_mass/mean` and `score_centering/trainer_topk_mass/mean`
(the probability mass the top-k covers; the tail model approximates the rest) and
`score_centering/coefficient_l1/mean` (the correction's size before the advantage).

## Top-k / top-p sampling: kept-set replay

With `SamplingConfig.top_k` (and optionally `top_p < 1`), the generator samples each token
from its kept set `S`, the tokens the filters keep, with probabilities renormalized over `S`:
`q_S`. The loss replays `S`: the trainer's logprob is renormalized over the same tokens,
`log p_S(y) = logits_y / T - logsumexp_{v in S}(logits_v / T)`, so the importance ratio
`p_S(y) / q_S(y)` compares the two distributions over the tokens the generator could sample,
and is 1 when the trainer and the generator agree. Without replay, a full-vocabulary `p(y)`
against `q_S(y)` would put every ratio below 1 by the mass the filters drop.

Example (one position, top-p 0.8, trainer equal to the generator):

```
p           = [A 0.50, B 0.30, C 0.15, D 0.05]      S = {A, B}
q_S         = [0.625, 0.375, 0, 0]                  sampled: B
p_S(B)      = 0.30 / 0.80 = 0.375                   ratio p_S(B) / q_S(B) = 1
gradient    : B up, A down; C and D get none
```

The generator returns each token's kept set and `q_S` over it (SGLang's sampling mask, in
"support" mode). They travel packed, and the training batch builder pads each loss token's kept set to a row of
`top_k` (ids padded with 0, logprobs with -inf, which marks the padding; a larger kept set
raises). The trainer receives `[T, top_k]` tensors, which CP, SPMD typing and the chunked loss
treat like the top-k rows:

```
generator (SGLang)          TrainingBatchBuilder            PolicyGradientLoss
  kept ids, q_S, sizes -->    one row per loss token   -->    log p_S over each kept set
  [num_kept], [num_gen]       [T, top_k], -inf padding        score centering over S
```

With score centering, the head is the kept set: it holds all of the generator's probability,
so no tail model is needed and the centering is exact. `score_centering/generator_topk_mass`
and `trainer_topk_mass` then read 1 by construction; `coefficient_l1` is the informative
metric.

Requirements and limits:

- `SGLangEngine`. The controller rejects top-k/top-p sampling with `VLLMEngine` until its V2
  model runner (#5956), whose sampler returns kept sets, and with the Verifiers rollouter (TODO).
- `top_p < 1` needs `top_k`, which bounds each kept set: top-p alone can keep most of the
  vocabulary on a flat distribution. The model's recommended `top_k` is a good default; it only
  binds where top-p would keep more.
- Batch-invariant mode raises: SGLang computes `q_S` from probabilities, not with the trainer's
  `compute_kept_set_logprobs` (TODO).
- `trainer/entropy/mean` is the entropy of `p_S`, the policy the loss trains.
- Costs: from the generator, 8 bytes per kept token (int32 id, fp32 logprob) and 4 per
  generated token (its kept-set size); to the trainer, 8 * top_k bytes per loss token, as for a
  top-k of the same k. The trainer gathers `top_k + 1` logits per token (one TP all-reduce)
  instead of a log-softmax over the vocabulary.
- The gradient reaches only the kept tokens' logits; the LM-head backward is still a dense
  matmul over the vocabulary (TODO in `PolicyGradientLoss`).
