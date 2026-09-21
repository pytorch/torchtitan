# Rollout consumption: greedy by default, windowed FIFO on request

Windowed FIFO follows the rollout scheduling described in Section 6.2.4 of the [MiniMax paper](https://arxiv.org/pdf/2605.26494).

## Symbols

- `P`: prompt groups per train step (`num_prompts_per_train_step`).
- `S`: target steady-state offpolicy steps (`target_offpolicy_steps`).
- `B`: active buffer size in prompt groups (`max_active_rollout_groups`), computed as `B = (S + 1) * P`.
- `n`: FIFO look-ahead window in train batches (`window_batches`); `None` means no window.
- `W`: the same window in group ids (`window_size`), computed as `W = n * P`.

## Greedy consumption (default)

With `window_batches = None` the batcher takes the oldest finalized group anywhere in the buffer. It does not wait for an unfinished older group while a younger finished group is ready.

There is no maximum offpolicy age. A slow group is trained when it lands, however old. Samples older than `S` are counted in `train_batch/num_samples_over_target_age` and logged with a warning; they are never dropped.

The mean offpolicy age is still about `S`: the buffer holds `B = (S + 1) * P` groups and the trainer consumes `P` per step, whatever the order.

## The straggler problem

Why bound the age at all? If the oldest group is slow, greedy consumption keeps training on younger groups and the slow group comes back older every step. The alternatives are:

- **Stall:** wait for the slow group and leave the trainer idle.
- **Drop:** train on younger groups, then drop the slow group when it becomes too old. This wastes completed rollout work and can bias which samples reach training.
- **Increase the target offpoliciness:** enlarge the active buffer so every group may wait longer. This makes all training samples older just to accommodate a small number of stragglers.

Windowed FIFO is a bounded compromise. It lets a limited number of younger groups bypass a slow group, buying time for that group to finish. A bypassed straggler is consumed at most `n` steps older than the target, and the buffer size is not increased for every sample.

The window is anchored at the oldest group still in the buffer. Consuming a younger group does not slide it forward, so no more than `W - 1` younger groups can bypass the oldest group before the batcher waits for it.

## Windowed FIFO configuration

The user configures `S`, `P`, and `n` through `target_offpolicy_steps`, `num_prompts_per_train_step`, and `window_batches`. `window_batches` defaults to `None`.

The controller derives:

```text
B = (S + 1) * P
W = n * P            (None when window_batches is None)
```

`n = 1` is FIFO by batch: at most `P - 1` younger group ids can pass a stuck head. Increasing `n` exposes more younger groups to the scheduler without increasing `B`.

```text
P = 8, S = 3 (B = 32)    W       max offpolicy steps
window_batches = None    none    unbounded
window_batches = 1       8       4
window_batches = 3       24      6
```

## Example

Let `target_offpolicy_steps = 1`, `num_prompts_per_train_step = 3`, and `window_batches = 1`. Then `max_active_rollout_groups = (1 + 1) * 3 = 6`, and `window_size = 1 * 3 = 3`. Groups `0`, `1`, `3`, `4`, and `5` finish quickly, but group `2` is slow:

```text
[ 0 ready ][ 1 ready ][ 2 slow ][ 3 ready ][ 4 ready ][ 5 ready ]
```

The batcher consumes groups `0` and `1`. The window is now anchored at group `2` and covers groups `[2, 4]`:

```text
anchored window
[ 2 slow ][ 3 ready ][ 4 ready ]   [ 5 ready ]
                                      ^ blocked outside the window
```

Group `3` may bypass group `2` and complete the train batch. Group `4` may also be consumed, buying more time for group `2` to finish. Consuming either younger group does not move the anchor, so group `5` remains blocked until group `2` is consumed.

With `window_batches = None` there is no window: group `5` is consumed as well, and group `2` is trained whenever it finishes.

## Offpolicy bound

Given a window size `W`, we can calculate the worst-case offpoliciness. Consider a slow target group `g` that was admitted at the back of a full active buffer:

1. When `g` is admitted, it occupies one of the `B` active slots, so the full buffer can contain at most `B - 1` older groups ahead of it. The trainer consumes those older groups first.
2. Because `B - 1 = S * P + (P - 1)`, this completes `S` train steps and leaves the next batch one group short.
3. Group `g`, which would complete that batch, stalls.
4. The trainer consumes all `W - 1` younger groups that may bypass `g` within the window.

```text
older groups                                  younger groups
[ 0 ][ 1 ] ... [ g-1 ][ g ][ g+1 ] ... [ g+W-1 ]
                       ^    at most W-1 prompt groups can bypass g
                       target group
```

```text
consumed_before = B - 1
consumed_after = W - 1
total_consumed = consumed_before + consumed_after = B + W - 2
```

Each train step consumes `P` groups, so the maximum age of `g` at consumption is:

```text
max_offpolicy_steps = (B + W - 2) // P
```

Substituting `B = (S + 1) * P` and `W = n * P` gives `max_offpolicy_steps = S + n` for `P >= 2`. Windowed FIFO therefore increases the worst-case offpoliciness by `window_batches` steps. Without a window there is no bound.
