# Rollout buffer: depth targets the mean policy age, the window caps the max

Two knobs on `AsyncLoopConfig`, both in train steps:

```text
target_offpolicy_steps  S    buffer depth B = (S + 1) * P groups   ->  MEAN age of a training batch ≈ S   (Little's law: S*P groups wait behind the P in training, P consumed per step)
window_batches          n    window W = n * P group ids            ->  MAX age of any group = S + n       (None: no window, no cap)
```

`P` is `num_prompts_per_train_step`. The window is anchored at the oldest unfinished group and does not move when a younger group is taken, so at most `W - 1` younger groups can be trained before the batcher waits for that group.

```text
P = 8, S = 3 (32 slots)     window          who may pass a stuck head            max age
window_batches = 1          8 ids           only the rest of its own batch       4
window_batches = 3 (default) 24 ids         three batches                        6
window_batches = None       no window       everyone; the oldest ready group is taken   unbounded; over-target samples counted + warned
```

## The trade-off

A stuck head with a small window means the trainer waits for it while finished younger groups sit in the buffer (head-of-line blocking): lower max age, more trainer wait. A large window or `None` trains the finished groups and lets the straggler come back older: less wait, a longer age tail. The mean age does not move either way, because the depth is fixed; the order only decides which group carries the age. The choice matters only when generation has a tail; with short, even generation the window never blocks anything.

## Example: a stuck head, `P = 3`, `S = 1`, `window_batches = 1` (6 slots, window 3 ids)

```text
[ 0 ready ][ 1 ready ][ 2 slow ][ 3 ready ][ 4 ready ][ 5 ready ]
```

The batcher takes 0 and 1. The window is now anchored at 2 and covers `[2, 4]`: 3 completes the first batch and 4 is taken next, so 2 gets two more groups' worth of time. 5 is outside the window and waits until 2 finishes. With `window_batches = None`, 5 would be taken as well and 2 would be trained whenever it lands, one step older.

## Where the cap comes from

A group admitted at the back of a full buffer is preceded by `B - 1` older groups and may be passed by the `W - 1` younger ones inside its window, so at most `B + W - 2` groups are trained before it: `max age = (B + W - 2) // P`, which with `B = (S + 1) * P` and `W = n * P` is `S + n` for `P >= 2`. The trainer checks this invariant at consume time (`compute_policy_age_metrics`) and raises if it is ever violated. With `window_batches = None` there is no cap; `train_batch/num_samples_over_target_age` counts the samples older than `S` and a warning names them.

Windowed FIFO originates in Section 6.2.4 of the [MiniMax paper](https://arxiv.org/pdf/2605.26494), which sizes the window as a fraction of the buffer; here it is sized in batches so the max age reads off directly.
