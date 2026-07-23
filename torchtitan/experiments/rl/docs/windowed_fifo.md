# Windowed FIFO offpolicy bound

The asynchronous controller limits rollout staleness with an active buffer and an anchored FIFO window. Define:

- `S`: target offpolicy steps
- `P`: prompt groups consumed per train step
- `B = (S + 1) * P`: active buffer size
- `W`: FIFO window size; `W = 1` is strict FIFO

Consider a target prompt group `g` sampled at policy version `v`:

```text
older groups                                  younger groups
[ 0 ][ 1 ] ... [ g-1 ][ g ][ g+1 ] ... [ g+W-1 ]
                       ^    at most W-1 prompt groups can bypass g
                       target sampled at policy v
```

At most `B - 1` older active groups can be ahead of `g`. Because the window is anchored at the oldest remaining group, only the next `W - 1` younger groups can bypass `g`. A group farther right cannot enter the window before `g` is consumed.

Therefore, at most

```text
(B - 1) + (W - 1) = B + W - 2
```

groups can be consumed before `g`. Each train step consumes `P` groups, so the maximum age of `g` at consumption is

```text
max_offpolicy_steps = floor((B + W - 2) / P).
```

Substituting `B = (S + 1) * P`:

```text
max_offpolicy_steps
  = floor(((S + 1) * P + W - 2) / P)
  = floor(S + (P + W - 2) / P)
  = S + floor((P + W - 2) / P)
  = S + floor(((W - 1) + (P - 1)) / P)
  = S + ceil((W - 1) / P).
```

For the last equality, let `W - 1 = q * P + r`, where `0 <= r < P`. Then

```text
floor(((W - 1) + (P - 1)) / P)
  = floor((q * P + r + P - 1) / P)
  = q + floor((r + P - 1) / P).
```

If `r = 0`, the last term is `0`; if `r > 0`, it is `1`. This is exactly

```text
ceil((W - 1) / P)
  = q,     if r = 0
  = q + 1, if r > 0.
```

Thus the extra steps allowed by reordering are

```text
max_out_of_order_steps = ceil((W - 1) / P),
max_offpolicy_steps = S + max_out_of_order_steps.
```

The ceiling matters. Under strict FIFO, the `B - 1` older groups already leave `P - 1` groups toward the next train batch. Consequently, even one younger group bypassing `g` can complete that batch and add one offpolicy step.

## Example

Let `S = 2`, `P = 3`, `B = 9`, and `W = 4`. Suppose prompt group `8` is slow while the other prompt groups are ready:

```text
Initial active buffer, sampled at policy v0:

train step 1       train step 2       partial step 3
[ 0 ][ 1 ][ 2 ]   [ 3 ][ 4 ][ 5 ]   [ 6 ][ 7 ][ 8 ] <- tracked prompt group

With W = 4, groups 9, 10, and 11 may bypass group 8:

consumed before tracked prompt group 8:
[ 0 ][ 1 ][ 2 ]   [ 3 ][ 4 ][ 5 ]   [ 6 ][ 7 ][ 9*]   [10*][11*]
      step 1             step 2             step 3         partial

* younger prompt group that bypasses prompt group 8
```

Eleven prompt groups can be consumed before prompt group `8`: eight older groups plus three younger groups. The trainer completes three steps before consuming prompt group `8`, so a prompt group sampled at policy `v0` may be consumed at policy `v3`.

```text
max_out_of_order_steps = ceil((4 - 1) / 3) = 1
max_offpolicy_steps = 2 + 1 = 3
```
