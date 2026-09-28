# MTP Auxiliary-Loss Normalization

## Project objective

Investigate whether TorchTitan should normalize each Multi-Token Prediction (MTP) depth by its own globally reduced number of valid target tokens, instead of normalizing all MTP losses with the ordinary language-model loss denominator.

This phase documents the current behavior and defines the smallest research question for a later implementation and controlled experiment. It does not change the training system.

## Original behavior

`Trainer.train_step()` counts valid ordinary language-model labels for the optimizer step, sums those counts across gradient-accumulation microbatches, and reduces them across the data-parallel mesh. The resulting `global_valid_tokens` is passed to `TrainingEngine.prepare_step()` and to the loss function.

`MTPLoss` computes a sum-reduced main cross-entropy and one sum-reduced cross-entropy for each MTP depth. It weights each MTP depth by:

```text
mtp_scale / num_mtp_layers
```

Before this change, it divided the complete objective by the ordinary `global_valid_tokens` denominator:

```text
L_current =
    (L_main_sum + mtp_scale / D * sum_k(L_mtp_k_sum)) / N_main
```

MTP labels are produced by shifting the original labels. Positions beyond the sequence tail, padding positions, and positions that cross packed-document boundaries are filled with `IGNORE_INDEX`, so they do not contribute to cross entropy.

## Identified problem

The main denominator is the valid-token count before MTP-specific shifting. Each MTP depth can have a different valid-target count after shifting:

```text
N_main != N_mtp_1 != N_mtp_2 ...
```

For a finite unpadded sequence, depth `k` normally loses `k` tail targets. Padding and packed-document boundaries can remove additional targets. Consequently, later MTP depths can receive a smaller average contribution per valid target when all depths are divided by `N_main`.

## Research question

Does normalizing each MTP depth by its own globally reduced valid-target count produce a more correct and stable auxiliary-loss signal than the current shared-denominator objective when target counts differ?

The word `correct` is conditional on the intended objective: the proposed behavior corresponds to equal weighting of the mean loss at each MTP depth. The current behavior is internally consistent with a different objective that sums all terms and divides by the ordinary-token denominator.

## Hypothesis

Per-depth normalization will make each MTP depth contribute its configured share of mean token loss, independent of how many targets remain after shifting. It will increase the per-valid-target gradient scale for deeper MTP depths when those depths have fewer valid targets. This may improve optimization stability or convergence on padded, variable-length, or packed data, but it may also change convergence or final quality because it changes the objective.

## Objective definitions

Let `D` be the number of MTP depths, `N_main` the ordinary valid-token count, and `N_k` the valid-target count at depth `k`.

Current objective:

```text
L_current =
    (L_main_sum + mtp_scale / D * sum_k(L_mtp_k_sum)) / N_main
```

Proposed objective:

```text
L_proposed =
    L_main_sum / N_main
    + mtp_scale / D * sum_k(L_mtp_k_sum / N_k)
```

The existing `mtp_scale / D` weighting remains unchanged.

The Phase 2 implementation now uses the proposed objective for DeepSeek MTP.

## Bug versus objective definition

The current implementation is not automatically a mathematical bug. It faithfully implements a shared-denominator objective and may have been chosen to reuse the ordinary loss normalization and preserve a common training scale.

It is different from the documented interpretation of `mtp_scale * mean(mtp_losses)` when `N_k` differs from `N_main`. Whether that difference is a correctness problem depends on which objective TorchTitan intends to define and optimize.

## Later controlled experiment

The later experiment should compare the current and per-depth objectives with fixed initialization, data order, optimizer, schedule, token budget, precision, model, and parallelism. It should include fixed-length data as a control and padded, variable-length, or packed data where denominators differ.

The experiment is intended to evaluate:

- numerical equivalence in equal-count cases;
- the predicted gradient scaling difference;
- training loss and per-depth loss behavior;
- gradient norms and stability;
- validation or task quality where available;
- throughput and memory overhead from carrying and reducing per-depth counts.

No experiment is part of this documentation phase.
