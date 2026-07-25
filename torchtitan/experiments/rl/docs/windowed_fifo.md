# Windowed FIFO design

This design follows the windowed FIFO rollout scheduling described in Section 6.2.4 of the [MiniMax paper](https://arxiv.org/pdf/2605.26494).

## Symbols

This document uses these symbols:

- `P`: prompt groups per train step (`num_prompts_per_train_step`).
- `S`: target steady-state offpolicy steps (`target_offpolicy_steps`).
- `f`: fraction of the `RolloutGroupWorkBuffer` size (`window_fraction`).
- `B`: active buffer size in prompt groups (`max_active_rollout_groups`), computed as `B = (S + 1) * P`.
- `W`: FIFO look-ahead window size (`window_size`), computed as `W = max(1, floor(f * B))`; strict FIFO uses `W = 1`.

## Strict FIFO

The simplest scheduling policy is strict FIFO: always consume the oldest prompt group before any younger group. With `P` prompt groups per train step and a target of `S` offpolicy steps, the buffer holds `B = (S + 1) * P` groups. Strict FIFO keeps every group within the target bound because at most `B - 1` older groups can be consumed first.

## The straggler problem

Strict FIFO has head-of-line blocking. If the oldest group is slow, every completed younger group waits behind it and training stalls. Without bounded reordering, the alternatives are:

- **Stall:** wait for the slow group and leave the trainer idle.
- **Greedy drop:** greedily train on any completed group, then drop the slow group when it becomes too old. This wastes completed rollout work and can bias which samples reach training.
- **Increase the target offpoliciness:** enlarge the active buffer so every group may wait longer. This makes all training samples older just to accommodate a small number of stragglers.

Windowed FIFO is a bounded compromise between these choices. It lets a limited number of younger groups bypass a slow group, buying extra time for that group to finish. The tradeoff is selective: a bypassed straggler may be consumed slightly older, but the configured target offpoliciness and active-buffer size are not increased for every sample.

The window remains anchored at the oldest active group. Consuming a younger group does not slide it forward, so no more than `W - 1` younger groups can bypass the oldest group before the scheduler applies backpressure again.

## Windowed FIFO configuration

The user configures `S`, `P`, and `f` through `target_offpolicy_steps`, `num_prompts_per_train_step`, and `window_fraction`. `window_fraction` defaults to `0.3` following the MiniMax paper.

The controller derives:

```text
B = (S + 1) * P
W = max(1, floor(f * B))
```

Set `f` to `None` to use strict FIFO (`W = 1`). Increasing `f` exposes more younger groups to the scheduler without increasing `B`.

## Example

Let `target_offpolicy_steps = 1`, `num_prompts_per_train_step = 3`, and `window_fraction = 0.5`. Then `max_active_rollout_groups = (1 + 1) * 3 = 6`, and `window_size = max(1, floor(0.5 * 6)) = 3`. Groups `0`, `1`, `3`, `4`, and `5` finish quickly, but group `2` is slow:

```text
[ 0 ready ][ 1 ready ][ 2 slow ][ 3 ready ][ 4 ready ][ 5 ready ]
```

With strict FIFO, the batcher consumes groups `0` and `1`, then stalls on group `2`. The trainer has only two of the three groups needed for a train step, so it remains idle even though younger groups are ready.

With `window_size = 3`, once groups `0` and `1` are consumed, the window is anchored at group `2` and covers groups `[2, 4]`:

```text
anchored window
[ 2 slow ][ 3 ready ][ 4 ready ]   [ 5 ready ]
                                      ^ blocked outside the window
```

Group `3` may bypass group `2` and complete the train batch. Group `4` may also be consumed, buying more time for group `2` to finish. Consuming either younger group does not move the anchor, so group `5` remains blocked until group `2` is consumed.

## Offpolicy bound

Given a window size `W`, we can calculate the worst-case extra offpoliciness it introduces. Consider a slow target group `g` that was admitted at the back of a full active buffer:

1. When `g` is admitted, it occupies one of the `B` active slots, so the full buffer can contain at most `B - 1` older groups ahead of it. The trainer consumes those older groups first.
2. Because `B - 1 = S * P + (P - 1)`, this completes `S` train steps and leaves the next batch one group short.
3. Group `g`, which would complete that batch, stalls.
4. The trainer consumes all `W - 1` younger groups that may bypass `g` within the window.

The worst-case ordering for a target group is:

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

Substituting `B = (S + 1) * P` gives:

```text
max_offpolicy_steps
  = floor(((S + 1) * P + W - 2) / P)
  = floor(S + (P + W - 2) / P)
  = S + floor(((W - 1) + (P - 1)) / P)
  = S + ceil((W - 1) / P)
```

Windowed FIFO therefore increases the worst-case offpoliciness by `ceil((W - 1) / P)` steps.
