# Inside the RolloutGroupWorkBuffer

Run-ahead FIFO (`_work_by_group_id`, ordered by `group_id`). Entry lifecycle:
`WAITING -> INFLIGHT -> FINALIZED -> removed` (batcher takes it).

Two independent knobs:
- `K = max_offpolicy_steps` -> **capacity** = `(K+1)*P` active slots (how far generation runs ahead).
- `s = window_lookahead_steps` -> **extra off-policy steps** a stuck head may incur. `max age = K + s`.

Both are counted in *optimizer steps*, and **1 step = P groups** (`P = num_prompts_per_train_step`).

## 1. Capacity: active slots = (K+1)*P

Active slot = charged at `add_work`, freed only by `release_active_groups` (so it includes groups
already taken by the batcher but not yet released). Example `K=2, P=2` -> cap 6, shown full:

```mermaid
flowchart TB
    subgraph ACTIVE["active slots 6 / 6  (cap = (K+1)*P)"]
      direction TB
      t8["g8 taken (not released)"]:::taken
      t9["g9 taken (not released)"]:::taken
      g10["g10 INFLIGHT (head)"]:::inflight
      g11["g11 FINALIZED"]:::fin
      g12["g12 WAITING"]:::wait
      g13["g13 WAITING"]:::wait
      t8 --> t9 --> g10 --> g11 --> g12 --> g13
    end
    g14["g14 blocked in wait_for_slot()"]:::pending
    ACTIVE -. full .-> g14
    classDef taken fill:#e6d7f0,stroke:#84a,stroke-dasharray:4 3;
    classDef inflight fill:#fde7c7,stroke:#c80;
    classDef fin fill:#d7f0d7,stroke:#3a3;
    classDef wait fill:#eee,stroke:#888;
    classDef pending fill:#fff,stroke:#bbb,stroke-dasharray:4 3,color:#999;
```

## 2. Windowed FIFO: s = extra off-policy steps tolerated

Same active slots as above (`g8/g9` taken-but-unreleased still count). The **window** is a sub-range
of the *dict entries*, anchored at the head. Head `g10` is a stuck straggler (INFLIGHT); with `s=1`
the batcher may bypass it far enough to complete **1 extra batch = 1 extra step**, no more.

`window_end = h + (s+1)*P - r0 - 1`   (here `s=1, P=2, r0=0` -> `g10 + 4 - 1 = g13`)

```mermaid
flowchart TB
    subgraph ACTIVE["active slots 6 / 6  (cap = (K+1)*P = 6)"]
      direction TB
      t8["g8 taken (not released)"]:::taken
      t9["g9 taken (not released)"]:::taken
      subgraph WIN["window [g10 .. g13] -> up to s=1 extra step"]
        direction TB
        w10["g10 INFLIGHT (head, stuck)"]:::inflight
        w11["g11 FINALIZED -> takeable"]:::fin
        w12["g12 FINALIZED -> takeable"]:::fin
        w13["g13 WAITING"]:::wait
        w10 --> w11 --> w12 --> w13
      end
      t8 --> t9 --> w10
    end
    w14["g14 blocked: full (no slot) AND outside window"]:::blocked
    ACTIVE -. full / window_end .-> w14

    note["take g11 + g12 = P groups = 1 batch = 1 step\n=> head g10 ages by at most +1 => max age = K + s"]:::note

    classDef taken fill:#e6d7f0,stroke:#84a,stroke-dasharray:4 3;
    classDef inflight fill:#fde7c7,stroke:#c80;
    classDef fin fill:#d7f0d7,stroke:#3a3;
    classDef wait fill:#eee,stroke:#888;
    classDef blocked fill:#f7f7f7,stroke:#bbb,stroke-dasharray:4 3,color:#999;
    classDef note fill:#eef,stroke:#557;
```

- **active slots** (whole box, incl. taken `g8/g9`) = capacity `(K+1)*P`; **window** (inner box) = the
  take-ahead range within the current dict entries.
- Take-ability = `FINALIZED` **and** within `[head, window_end]` (position-based, not version-based).
- The window is sized in *groups* (`(s+1)*P - r0`) because `1 step = P groups`; `r0` = trainable
  groups already accumulated toward the in-progress batch when the head became head.
- `s` is snapshotted per head and the window slides right only when the head itself is consumed.
