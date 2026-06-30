# Distributed Muon on FlexShard: Placement Strategies

This doc maps the placement design space for running the Muon optimizer on
FlexShard-sharded transformer weights, organized by the tradeoff that defines it:
**comm-free vs memory-balanced**. The two ends are duals —

- **Comm-free `Owned`** (recommended default, implemented): the optimizer step
  issues **zero collectives** and Newton-Schulz (NS) is **bit-exact**, at the cost
  of **matrix-granular** memory balance (a rank owns whole matrices).
- **Memory-balanced `GroupedRaggedShard`** (when comm-free is relaxed):
  **byte-perfect** balance and no idle ranks, at the cost of an all-gather to
  reconstruct each matrix for NS.

[§ Why Muon is hard to shard](#why-muon-is-hard-to-shard) onward is the core
comm-free design and is fully implemented.
[§ Relaxing comm-free](#relaxing-comm-free-from-owned-to-memory-balanced-raggedshard)
maps the granularity ladder from `Owned` to `RaggedShard`/`GroupedRaggedShard`
for when memory balance matters more than a comm-free step.

## Goal (comm-free path)

Run the Muon optimizer on FlexShard-sharded transformer weights so that the
**optimizer step issues zero collectives** and the Newton-Schulz (NS)
orthogonalization is **mathematically exact** (bit-identical to single-device
Muon). The only communication in the whole training step stays where it already
is — FlexShard's forward unshard and backward gradient reduction — and Muon
adds nothing on top.

The enabling primitive is the existing **`Owned` placement**
(`example/owned.py`). For the *comm-free* goal RaggedShard is not the tool — but
it is the endpoint of the *memory-balance* goal; see
[§ Relaxing comm-free](#relaxing-comm-free-from-owned-to-memory-balanced-raggedshard).

## Why Muon is hard to shard

`torch.optim.Muon` (`pytorch/torch/optim/_muon.py`) replaces each 2D weight's
momentum update with the orthogonal polar factor, computed by the quintic
Newton-Schulz iteration `_zeropower_via_newtonschulz` (l.31-70):

```python
for _ in range(ns_steps):                       # default 5 steps
    gram_matrix = ortho_grad @ ortho_grad.T     # X @ X.T   <-- global reduction
    gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
    ortho_grad  = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
```

Each step forms the Gram matrix `X @ X.Tᵀ`, a contraction over **all columns**
of the **whole** matrix. If `X` is row-sharded (`Shard(0)`), the local
`X_local @ X_localᵀ` is **not** a slice of the true Gram matrix —
`NS(shard) ≠ slice of NS(full)`. The orthogonal factor is a global, non-separable
property of the matrix, so NS must see the complete 2D matrix.

Two more facts from the implementation that matter here:

- Muon **rejects any non-2D parameter at construction** (l.128-133), and has
  **no built-in AdamW fallback** — the caller must route embeddings / norms /
  biases / the LM head to a separate optimizer.
- The per-matrix LR scale `_adjust_lr` reads `param_shape[:2]` (l.73-84). It is
  only correct when the optimizer sees the **full** matrix shape, not a shard's
  shape. This is independent corroboration that Muon must operate on whole
  matrices.
- The only state is one `momentum_buffer = zeros_like(p.grad)` (l.156-160); it
  inherits the gradient's shape/placement.

### The three baselines and what's wrong with each

| Approach | NS correctness | Comm in `step()` | NS compute |
|---|---|---|---|
| **DDP + Muon** (full grad on every rank) | exact | none | **redundant** — every rank runs the full NS, wasting `(N-1)/N` |
| **FSDP2 `Shard(0)` + Muon** (NS on the local row-shard) | **approximate** (NS on a shard ≠ NS on full) | none | sharded |
| **FSDP + exact Muon** (all-gather the full matrix inside `step()`, run NS, re-shard) | exact | **extra all-gather per Muon param every step** | sharded |

We want the union of the good columns: **exact** NS, **no comm** in `step()`,
**non-redundant** NS compute. `Owned` placement delivers exactly that.

## Key insight: `Owned` placement makes `step()` comm-free *and* exact

`Owned(owner_rank)` (`example/owned.py:35`) puts the **full, un-split** matrix on
one rank and an empty `(0,0)` tensor on every other rank. Its collective
lifecycle is **broadcast (forward) / reduce-to-owner (backward)**, not
all-gather/reduce-scatter:

- **Forward unshard** = `dist.broadcast(full_param, src=owner_rank)` — owner ships
  the full matrix to everyone for the layer's forward (owned.py:135-160).
- **Backward reduce** = `dist.reduce(grad, dst=owner_rank, op=SUM)` then the owner
  `grad.div_(world_size)` (owned.py:207-214). After backward the **owner already
  holds the full, batch-averaged gradient**; non-owners hold an empty grad.

Now look at what the owner rank has at `optim.step()` time, with **no extra work**:

1. **Full param** — the owner's persistent local storage *is* the whole matrix
   (`compute_local_shape == global_shape` on the owner).
2. **Full averaged gradient** — already concentrated on the owner by the backward
   reduce that FlexShard performs anyway.
3. **Full momentum buffer** — `zeros_like(p.grad)` is full-shaped on the owner and
   persists there across steps.

So the owner runs the **entire** NS iteration on the **true full matrix, locally**.
No `full_tensor()`, no all-gather, no redistribute inside `step()`. The updated
full matrix sits in the owner's storage; the **next forward's broadcast** — which
FlexShard was going to do regardless — distributes it. **Muon adds zero
communication.**

And because NS runs on the genuine full matrix with the genuine averaged
gradient, the result is **bit-for-bit identical to single-device Muon** on the
same global batch — unlike `Shard(0)+Muon`. Plus, each matrix's NS runs on
exactly **one** rank, and ownership is balanced across ranks, so NS compute is
distributed with **no DDP-style redundancy**.

This is precisely the "structure-preserving parameter sharding + bucketed
broadcast" idea, realized natively: the "bucketed broadcast" is just FlexShard's
existing per-bucket forward unshard, and we never broadcast the *update* — we
apply it on the owner and let the next forward's broadcast carry it.

## Design

### 1. Placement strategy

Split the model into two regions:

- **Transformer layers → `Owned`, one owner per layer (balanced by LPT).** Assign
  each whole `layers.{i}.*` to a single owner rank and make that layer one Owned
  bucket (one broadcast forward / one reduce backward, same granularity as today's
  `transformer_bucket_specs`). The owner runs Muon on the layer's 2D matrices
  (attention `wq/wk/wv/wo`, MLP `w1/w2/w3`) and AdamW on the layer's 1D norms —
  all locally. Choose owners to balance memory; see
  [§2 Memory balance](#2-memory-balance-across-ranks) for the strategies (LPT over
  whole layers is the recommended default).
- **Embeddings + LM head + final norm → `Shard(0)` (FSDP), AdamW.**
  `tok_embeddings`, `pos_embeddings`, `output`, `norm` stay evenly sharded and
  go to AdamW on the local shard. AdamW is element-wise, so its sharded step is
  already comm-free. These are intentionally **excluded from Muon** (Keller
  Jordan's recipe puts embeddings/head/scalars on AdamW), and `Owned` would make
  the huge embedding table a memory hotspot on one rank.

**Why one owner per layer:** a FlexShard bucket must have a **single uniform
placement** — `_validate_bucket_uniform_dtype_and_placement` (`utils.py`) requires
every param in a bucket to share one placement tuple, and `Owned.__eq__` compares
`owner_rank`, so `Owned(0) != Owned(1)`. Two differently-owned params in one bucket
raise *"has mixed placements ... bucket collectives use one placement layout."*
Keeping one owner per layer satisfies the constraint, keeps one collective per
layer, and stays compatible with per-layer `reshard_after_forward`. §2 explains how
to pick those owners for even memory.

### 2. Memory balance across ranks

**What actually has to be balanced.** Under `Owned` there are two memory pools:

- **Persistent (this is what can become imbalanced):** for each `Owned(r)` param,
  rank `r` holds the *full* tensor **and its full optimizer state** (Muon momentum
  = 1× param size; AdamW `exp_avg`+`exp_avg_sq` = 2×); every other rank holds
  `(0,0)`. So per-rank persistent memory = **Σ numel of the params that rank owns**
  (× a per-optimizer state multiplier). Balancing memory ⟺ **partitioning total
  owned numel evenly across ranks** — a number-partitioning problem.
- **Transient (symmetric, not a balance concern):** during a layer's forward every
  rank materializes that layer's full params via broadcast; backward concentrates
  the full grad on the owner. With `reshard_after_forward` these are freed per
  layer. This peak is identical on every rank, so it sets the transient ceiling
  (≈ one layer, like FSDP's all-gather buffer) but creates no imbalance.

So the whole problem reduces to: **partition owned numel evenly, subject to one
owner per bucket.**

**The balancing engine already exists — but it is per-param.**
`_assign_params_to_ranks` (owned.py:229-245) is greedy **Longest-Processing-Time
(LPT)**: sort params by numel descending, repeatedly give the largest remaining
param to the least-loaded rank (min-heap on load) — the standard makespan heuristic
(≤ 4/3 of optimal). `param_boundary_placements` (owned.py:248) wraps it. The catch:
it assigns each param an **independent** owner, so two params in the same layer can
get different owners → violates the uniform-bucket rule. It is therefore only
directly usable with **one param per bucket**.

**Options (all keep one owner per bucket, so all are uniform-bucket-legal):**

| Option | Balance quality | Collectives / layer | Reshard? | When to use |
|---|---|---|---|---|
| **A. Round-robin layers** `owner = i % ws` | optimal **iff** layers homogeneous **and** `num_layers % ws == 0` | 1 | ✅ | simplest; uniform stacks |
| **B. LPT over whole layers** *(recommended default)* | within **one layer** of optimal | 1 | ✅ | heterogeneous layers, uneven `num_layers` |
| **C. Per-matrix buckets + per-param LPT** | near-optimal (`param_boundary_placements` directly) | ~7 | ✅ (finer) | when ±1-layer imbalance is too coarse |
| **D. Intra-layer sub-buckets by owner** | near-optimal within each layer | ≤ `min(#matrices, ws)` | ✅ | small `ws` vs matrices/layer |

- **A vs B:** B treats each layer as one item (weight = total layer numel) and LPT-
  assigns it to the least-loaded rank. For homogeneous layers B *reduces to* A, so B
  has no downside and strictly dominates when layers differ (MoE vs dense, fatter
  first/last blocks) or `num_layers` is not a multiple of `ws`. **Use B as the
  default.** Residual imbalance is bounded by the largest single layer (a layer kept
  in one bucket cannot be split).
- **C** gives each matrix its own bucket, so per-param LPT applies with no uniform-
  bucket conflict; cost is ~7× more (smaller) broadcasts per layer and loss of
  per-layer collective coalescing.
- **D** is the middle ground: LPT a layer's ~7 matrices into per-owner groups and
  emit one sub-bucket (one broadcast) per owner. Balances within the layer while
  staying inside the layer's reshard boundary; most attractive when `ws` is small
  (2-4 matrices land per owner).

**Worked examples (homogeneous layers):**
- **32 layers, 8 ranks:** A = B → each rank owns `{r, r+8, r+16, r+24}` = 4 layers =
  exactly `1/8` of layer params. Balanced.
- **30 layers, 8 ranks:** best possible with whole-layer ownership is
  `4,4,4,4,4,4,3,3` → heavy ranks carry **33% more** than light ranks. Neither A nor
  B can beat ±1 layer; only C/D (sub-layer splitting) closes the gap.

**Two consequences to keep in mind:**
- **Optimizer state rides ownership automatically.** Muon's
  `momentum_buffer = zeros_like(grad)` lives full on the owner; AdamW's
  `exp_avg/exp_avg_sq` live wherever the param does. Because A/B hand each rank
  *whole layers*, every rank gets the same Muon-vs-AdamW param mix, so balancing
  numel ≈ balancing bytes. (Embeddings/output are `Shard(0)`, evenly split by
  construction — outside the Owned balancing problem.)
- **A matrix too large for one rank cannot be `Owned`.** It must fall back to
  `Shard`/`RaggedShard`, and Muon on it is then no longer comm-free (see
  [Alternatives](#alternatives-considered)).

**The per-matrix floor: when a layer has fewer matrices than ranks.** Per-matrix
ownership (option C / `balance="per-matrix"`, via
`assign_matrix_owners_per_layer_balanced` — per-layer LPT + a layer-index
rotation) is still **whole-matrix granular**, so it degrades sharply when a layer
has **fewer 2D matrices than ranks** (`M < ws`). MLA `self_attn` has exactly
**5** 2D matrices — `q_a_proj` (11.0M), `q_b_proj` (18.9M),
`kv_a_proj_with_mqa` (4.1M), `kv_b_proj` (8.4M), `o_proj` (58.7M) — so any
`ws > 5` (or a dense/attention-only layer on a large cluster) hits this. The LPT
assigner then hands out `M` distinct owners and **leaves `ws − M` ranks idle for
that layer**; because it cannot pack, each active rank's load is the **raw matrix
size** — `o_proj`'s 58.7M vs `kv_a`'s 4.1M is a **~14× per-layer spread**. The
broadcast still hits all ranks (everyone needs the full matrix for forward), so
idle ranks save no comm — they just do no NS and hold no momentum for that layer.
The layer-index rotation rescues the *global* totals (over many layers each rank
owns `o_proj` ≈ `num_layers/ws` times and the idle set rotates), but only when
`num_layers ≫ ws`, and it balances *accumulated* state, not *per-step* compute.
This is the floor that motivates
[relaxing comm-free](#relaxing-comm-free-from-owned-to-memory-balanced-raggedshard):
to balance below one matrix you must split matrices across ranks (which makes NS
non-local), and `GroupedRaggedShard` does exactly that.

### 3. Placement helpers (in `example/owned.py`)

`param_boundary_placements` already does per-param LPT (option C). Add a
**fixed-owner-per-bucket** helper (used by A/B/D) and a **layer-LPT** assigner:

```python
def make_owned_placement_fn(owner_rank: int) -> PlacementFn:
    """Place every parameter in the bucket on a single owner rank (options A/B/D)."""
    def placement_fn(named_params, mesh):
        return {fqn: (Owned(owner_rank),) for fqn, _ in named_params}
    return placement_fn


def assign_layer_owners_lpt(layer_numels: list[int], world_size: int) -> list[int]:
    """LPT-balance whole layers across ranks (option B). Returns one owner per layer."""
    loads = [(0, r) for r in range(world_size)]
    heapq.heapify(loads)
    owners = [0] * len(layer_numels)
    for i in sorted(range(len(layer_numels)), key=lambda i: layer_numels[i], reverse=True):
        load, rank = heapq.heappop(loads)
        owners[i] = rank
        heapq.heappush(loads, (load + layer_numels[i], rank))
    return owners
```

Bucket builder (new `example/muon.py`), defaulting to option B and falling back to
round-robin (A):

```python
def comm_free_muon_buckets(
    model, num_layers, world_size, *, reshard_after_forward=True, balance="lpt",
):
    if balance == "lpt":                                            # option B
        layer_numels = [
            sum(p.numel() for _, p in model.layers[i].named_parameters())
            for i in range(num_layers)
        ]
        owners = assign_layer_owners_lpt(layer_numels, world_size)
    else:                                                           # option A
        owners = [i % world_size for i in range(num_layers)]
    layer_buckets = [
        BucketSpec([f"layers.{i}.*"],
                   placement_fn=make_owned_placement_fn(owners[i]),
                   reshard_after_forward=reshard_after_forward)
        for i in range(num_layers)
    ]
    rest_bucket = BucketSpec(
        ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"],
        placement_fn=per_param_placements,            # Shard(0)
        reshard_after_forward=reshard_after_forward)
    return [*layer_buckets, rest_bucket]
```

Options **C** and **D** are alternative bucket builders: C emits one `BucketSpec`
per matrix with `param_boundary_placements`; D groups each layer's matrices by LPT
owner into per-owner sub-buckets.

### 4. Optimizer param-group construction (per-rank, in `example/muon.py`)

After `flex_shard`, `model.parameters()` yields the local views, each annotated
with FlexShard metadata (`get_placements`, `get_global_shape` from
`sharded_param.py`). Partition them **per rank**:

```python
def build_muon_param_groups(model, mesh, *, is_muon_matrix):
    rank = mesh.get_local_rank()
    muon_params, adamw_params = [], []
    for fqn, p in model.named_parameters():
        placements = get_placements(p)
        global_shape = get_global_shape(p)
        owned_here = (
            placements is not None
            and isinstance(placements[0], Owned)
            and placements[0].owner_rank == rank
        )
        if owned_here and len(global_shape) == 2 and is_muon_matrix(fqn) and p.numel() > 0:
            muon_params.append(p)            # full 2D matrix this rank owns -> Muon
        elif p.numel() > 0:                  # any non-empty local tensor -> AdamW
            adamw_params.append(p)
        # drop empty (0,0) non-owned Owned tensors: nothing to update on this rank
    return muon_params, adamw_params
```

- The Muon group on each rank contains **only this rank's owned 2D matrices**
  (full, non-empty). Empty `(0,0)` non-owned tensors are never fed to Muon, so
  Muon's `ndim==2` guard and degenerate NS-on-empty are sidestepped entirely.
- `is_muon_matrix(fqn)` excludes `tok_embeddings/pos_embeddings/output/norm`
  even though some are 2D.
- Everything else with non-empty local storage (norms owned here, embedding /
  output shards) goes to AdamW and steps on its local data — comm-free.

Then build two optimizers (Muon ships in `torch.optim`):

```python
muon  = torch.optim.Muon(muon_params, lr=..., momentum=0.95, weight_decay=...)
adamw = torch.optim.AdamW(adamw_params, lr=..., weight_decay=...)
```

Optionally wrap both in a tiny combined optimizer with `step()/zero_grad()` for
ergonomics. Each rank's two optimizers only ever touch local tensors.

### 5. End-to-end step lifecycle (no extra comm)

```
forward(layer i)   : broadcast full params from owner(i)       [FlexShard unshard]
backward(layer i)  : reduce grads to owner(i), owner /= ws      [FlexShard reduce]
optim.step()       : owner(i) runs Muon NS on full matrix       [LOCAL, no comm]
                     all ranks run AdamW on their local shards  [LOCAL, no comm]
next forward       : broadcast updated params from owner(i)      [FlexShard unshard]
```

### 6. Memory & communication analysis

- **At-rest memory:** with one-owner-per-layer (option A/B), rank `r` holds full
  params + full optimizer state for the layers it owns → `≈ total_layer_params / ws`
  per rank when balanced (option B bounds the imbalance by one layer). Embeddings/
  output are evenly `Shard(0)`-ed, so no hotspot.
- **Transient memory:** with `reshard_after_forward`, only the active layer's full
  params are materialized at a time — same transient footprint as FSDP's all-gather
  buffer, identical on every rank.
- **Communication volume:** per layer, broadcast (owner→all) vs FSDP all-gather
  (all→all) move comparable receive volume (`≈ matrix size`); broadcast concentrates
  the *send* on the owner, but the owner **rotates across layers** (A/B), so send
  load balances across the step. We **eliminate** the extra per-Muon-param all-gather
  that exact-FSDP-Muon would need in `step()`.
- **NS compute:** distributed and non-redundant (vs DDP's full redundant NS), and
  balanced across ranks to the same degree as memory (option B).

## Implementation phases

Phases 1–4 are **implemented** (`example/owned.py`, `example/muon.py`,
`flex_shard/reshard_after_forward.py`, `tests/test_flex_shard_muon.py`).

1. **Placement helpers** *(done)* — `make_owned_placement_fn(owner_rank)` and
   `assign_layer_owners_lpt(layer_numels, world_size)` (option B) in
   `example/owned.py` (exported from `example/__init__.py`). Per-matrix (C) /
   sub-bucket (D) builders remain optional future work.
2. **Comm-free Muon wiring** *(done)* — `example/muon.py` with
   `comm_free_muon_buckets(...)`, `build_muon_param_groups(...)`,
   `build_comm_free_muon_optimizers(...)`, and the `CombinedOptimizer`
   convenience. All in the experiment folder.
3. **Reshard-after-forward for `Owned`** *(done)* — FlexShard emits a semantic
   unshard marker for bucket materialization, so activation checkpointing
   recomputes the marker instead of matching raw c10d collectives. The eager
   unshard still issues `c10d.broadcast_` (Owned) / `c10d.allgather_` (Shard),
   while torch.compile lowers to the functional `_c10d_functional.*` forms; the
   marker keeps those placement details out of the checkpoint policy.
   `comm_free_muon_buckets` defaults `reshard_after_forward=True` and emits one
   bucket **per** rest pattern (a grouped embeddings/LM-head/norm bucket resolves
   to the root module and is rejected by the reshard hook installer).
4. **Tests** *(done)* — `tests/test_flex_shard_muon.py` (details below).
5. **Runnable example + doc** — small example mirroring `example/ragged_shard.py`;
   keep this plan doc updated.

## Validation

The headline correctness claim is **bitwise parity with single-device Muon**.

1. **Single-rank exactness (CPU/1-GPU unit test).** Build the small Transformer
   (`tests/common.make_transformer_model`), run Owned-Muon on world_size=1
   vs a reference unsharded model stepped by `torch.optim.Muon` directly →
   identical params after `step()`.
2. **Multi-GPU parity (world_size=2, `FSDPTest`).** Mirror
   `test_flex_shard_training.py`: reference = unsharded model, grads
   AVG-reduced (`_average_reference_grads`), stepped by single-device Muon on the
   **full** params; FlexShard model = Owned layers + `build_muon_param_groups`,
   stepped by the per-rank Muon+AdamW. Compare full updated params (add an
   `expected_owned` helper analogous to `expected_shard`: full-on-owner /
   empty-elsewhere). Expect **exact** equality (Owned ⇒ exact NS), unlike
   `Shard(0)+Muon`.
3. **Ablation.** Show `Shard(0)+Muon` (NS on the row-shard) does **not** match
   single-device Muon — demonstrates the exactness win of Owned.
4. **Comm audit.** Assert no collective fires during `optim.step()`. Reuse the
   `_record_comm_if_eager` markers (`utils.py`) — all
   `FlexShard::broadcast`/`FlexShard::reduce` events must land in forward/backward,
   none in `step()`.
5. **Checkpoint round-trip.** The Muon `momentum_buffer` lives full on the owner;
   verify DCP save/load round-trips it consistently with the Owned sharding
   (owner = full, others = empty), reusing the `test_flex_shard_state_dict.py`
   pattern.
6. **Loss convergence.** Per `.claude/CLAUDE.md`, this is a computation change
   (a new optimizer path), so validate loss/grad_norm on a representative run via
   `scripts/loss_compare.py` (TensorBoard), not just stdout's 5 digits.

## Open decisions & risks

- **Ownership granularity (see [§2](#2-memory-balance-across-ranks)).** Default to
  **option B (LPT over whole layers)**: one collective per layer, uniform-bucket-
  safe, reshard-friendly, ±1-layer imbalance. Escalate to **C** (per-matrix buckets
  + per-param LPT) or **D** (intra-layer sub-buckets by owner) only when ±1-layer is
  too coarse, accepting more (smaller) broadcasts. **A** (round-robin) is fine for
  homogeneous stacks with `num_layers % ws == 0`. **Decision needed per model.**
- **`num_layers % world_size != 0`** → whole-layer ownership (A/B) leaves a
  ±1-layer imbalance; fall back to option C/D to split the remainder more evenly.
- **Reshard-after-forward + broadcast tagging** *(resolved, phase 3)* — the AC
  policy now tags both the eager (`c10d.broadcast_`) and functional
  (`_c10d_functional.broadcast`) broadcast ops `MUST_RECOMPUTE`, verified by a
  policy unit test and a backward-recompute count test. Each rest pattern is its
  own bucket so reshard hooks stay replayable.
- **Mixed precision.** Owned broadcast/reduce should honor `mp_policy` (reduce in
  fp32); Muon internally casts to bf16 for NS regardless. Verify the interaction
  on the Owned path.
- **Weight tying.** FlexShard rejects shared params; if embeddings are tied to
  the LM head, handle before sharding. (Test model uses `weight_tying=False`.)
- **Large single matrix.** A matrix that doesn't fit on one rank cannot use
  `Owned`; it must be split (then Muon is no longer comm-free for it — see
  Alternatives).
- **`torch.optim.Muon` availability.** Confirm the installed torch exports
  `torch.optim.Muon` (recent addition; present in
  `/data/users/weif/code-review/pytorch`).

## Relaxing comm-free: from `Owned` to memory-balanced `RaggedShard`

`Owned` and per-matrix `Owned` are **comm-free** but **whole-tensor granular**:
they assign indivisible matrices to ranks, so the best any balancer (LPT) can do
is bounded below by the **largest single tensor** (`o_proj` = 58.7M; the 14× MLA
hotspot from §2). You cannot beat that floor with whole-tensor assignment. To go
under it you must **split a matrix across ranks** → no rank holds the full matrix
→ NS cannot run locally → you need a collective to reconstruct it.
**Sub-matrix memory balance ⟺ a split matrix ⟺ comm-free is gone** — it is the
same coin. So once memory balance matters more than a comm-free step, the question
is *how finely can we split*, and the answer is a granularity ladder ending at
`GroupedRaggedShard`.

### The granularity ladder

| placement | split granularity | memory balance | NS in `step()` | comm in `step()` |
|---|---|---|---|---|
| `Owned` (whole layer) | layer | coarse (≤ largest layer) | local, exact | — |
| per-matrix `Owned` | matrix | medium (≤ largest matrix, e.g. `o_proj`) | local, exact | — |
| `RaggedShard` (per-param) | **rows of one param** (uneven units, incl. 0) | fine, per-param | needs all-gather | 1 all-gather **per param** |
| `GroupedRaggedShard` (bucket-global) | **bytes, across param boundaries** | **perfect** (bucket/`ws` ± padding) | needs all-gather | 1 all-gather **per bucket** |

Memory balance improves monotonically down the ladder; comm-freeness holds only
for the two `Owned` rungs.

### `RaggedShard` vs `GroupedRaggedShard` (from the code)

- **`RaggedShard`** (`example/ragged_shard.py:40`) shards a **single** param along
  its flattened prefix dims (`dims`), with `local_units` giving each rank's
  relative row count — *ragged*, so units may differ and may be `0` (e.g. shape
  `[8, hidden]`, `local_units=(1,2,1,0)` → prefix lengths `[2,4,2,0]`, rank 3
  empty). It can finally split `o_proj` across ranks, but it is **row-granular and
  per-param**: small params can't split far, the divisibility rule
  `prefix_numel % sum(local_units) == 0` must hold, and each param is its own
  all-gather unit.
- **`GroupedRaggedShard`** (`ragged_shard.py:422`) plans the **whole bucket** as
  one param-major flat buffer and cuts it into `ws` byte-balanced ranges that
  **cross param boundaries** — a rank can own *"tail rows of `o_proj` + all of
  `kv_a` + head rows of `q_b`"*. Alignment = `lcm` of suffix numels plus padding to
  `alignment · Σunits` make balance **byte-perfect for arbitrary sizes**, and the
  all-gather output is **directly viewable as full params** for NS
  (`finish_prepared_unshard`). **One all-gather / reduce-scatter per bucket.** This
  is precisely the **FSDP2 flat-parameter / uneven-DTensor-shard** model, and it is
  the memory-optimal endpoint.

### One continuum: `local_units` is the knob

These are not separate mechanisms — they are one placement family parameterized by
`local_units` (× bucket grouping):

- `local_units = (0,…,T,…,0)` (all units on one rank) ⇒ **functionally `Owned`**
  (owner holds the full prefix, others empty). The only difference is the
  collective: `Owned` uses `broadcast`/`reduce` (1-source, cheaper); `RaggedShard`
  uses `all_gather`/`reduce_scatter`.
- `local_units = (1,1,…,1)` ⇒ **even FSDP shard** (max balance).
- anything between ⇒ **dial it** — give memory-pressured ranks fewer units, exclude
  a rank with `0`.

So `Owned`, per-matrix, ragged, and FSDP are points on a single axis trading
comm-freeness for memory balance.

### Running NS on a sharded matrix (the optimizer's choice, not the placement's)

Sharding balances **param + grad + momentum** storage, but NS still needs the
**full 2D matrix**. Two strategies — *same placement, different optimizer*:

1. **Gather → redundant NS → keep your shard.** Reuse the unshard hook to
   materialize full matrices (`GroupedRaggedShard.finish_prepared_unshard` already
   returns full params), run NS on every rank, write back only the local shard.
   Simple; memory balanced; NS compute redundant ×`ws`. **Momentum caveat:** to
   keep state balanced, store momentum sharded and **all-gather it too** each step
   (an extra collective), or accept a replicated momentum (no state saving).
   ✅ **Implemented** as `RaggedShardMuon` + `grouped_ragged_shard_muon_buckets`
   (`example/muon.py`). It avoids the momentum caveat: the momentum buffer stays
   **sharded** and is updated element-wise on the shard, so only the **pre-NS update**
   is gathered — **one all-gather per bucket, no momentum gather** — and the step is
   **bit-exact** with single-device `torch.optim.Muon`. The same optimizer also covers
   per-param `RaggedShard` (it dispatches through the placement's own gather), while 3D
   experts stay on comm-free `GroupedMuon`.
2. **Distributed NS** (Gram all-reduce / Dion / Moonshot-style): do the `XᵀX`
   contraction on the row-sharded matrix and all-reduce the small `n×n` Gram, never
   materializing the full matrix and distributing the compute. Comm-reduced (only
   `n×n` crosses the wire, cheap when `m ≫ n`); placement unchanged, NS lives in
   the optimizer.

### Why this resolves the `self_attn` floor

`GroupedRaggedShard` directly dissolves the `M < ws` pathology: the 5 attention
matrices (≈101M) + norms become one bucket split into exactly `101M/ws` per rank —
**no idle ranks** even when `ws > 5` (the cut is byte-granular, not
matrix-granular) and **no `o_proj` hotspot** (its 58.7M is spread across all
ranks). The cost is *dual to `Owned`, not strictly worse*: `Owned` already pays a
forward **broadcast** of each whole matrix; `GroupedRaggedShard` pays a forward
**all-gather** + grad **reduce-scatter** (the FSDP traffic you'd pay anyway). You
are not adding *forward* comm vs `Owned` — you spend *optimizer* comm to buy
**perfect memory + balanced NS compute + zero idle ranks**.

**Bottom line.** The progression is `Owned → per-matrix Owned → RaggedShard →
GroupedRaggedShard`, and once comm-free is relaxed, **`GroupedRaggedShard` is the
endpoint** — the only rung that achieves byte-perfect balance and the only one that
fixes "matrices < world_size," because it is the only one that stops respecting
matrix boundaries. The remaining decision is purely *how* NS runs on the sharded
matrix (gather-redundant vs distributed), an optimizer choice layered on top.

## Alternatives considered

- **RaggedShard / GroupedRaggedShard for Muon.** Not for the *comm-free* goal —
  they split each matrix across ranks, so NS is no longer local/exact. They are the
  **memory-balance** endpoint instead; see
  [§ Relaxing comm-free](#relaxing-comm-free-from-owned-to-memory-balanced-raggedshard)
  for the full treatment (granularity ladder, the `local_units` continuum, and the
  gather-redundant vs Gram/distributed-NS choice). **Recommendation:** `Owned` for
  comm-free Muon; `GroupedRaggedShard` when memory balance — or a matrix too large
  to own on one rank, or `matrices < world_size` — outweighs a comm-free step.
- **All-gather inside `step()` (exact-FSDP-Muon).** Correct but pays an extra
  collective per Muon param every step — exactly the cost `Owned` removes.
- **Patch Muon to skip empty tensors** instead of per-rank filtering — not
  needed; per-rank param-group construction already excludes empties.

## Related work: DeepSpeed Muon (ZeRO)

DeepSpeed ships a Muon optimizer (`deepspeed/runtime/zero/muon/`; blog: *Using
Muon Optimizer with DeepSpeed*), exercised by
[delock/deepspeed_finetune_demo](https://github.com/delock/deepspeed_finetune_demo)
(Moonlight-16B-A3B MoE + AutoEP). It solves the **same** core problem — "NS needs
the full 2D matrix, but sharding flattens it" — with the **opposite** philosophy,
which makes it a useful contrast for this design.

**What DeepSpeed does (from source).** The reference Muon (`original_muon.py`,
Keller Jordan's algorithm) runs *via* the ZeRO engine: `muon_optimizer.py`'s
`MuonWithAuxAdam.step` for `use_muon` groups does **only** `p.add_(p.grad, -lr)` —
*no NS* — because *"the parameter here is a flat version."* The Newton-Schulz is
moved into the ZeRO engine's `get_flat_partition`, **where per-parameter gradients
are still unflattened 2D**. The enabling trick is to **keep the gradient
replicated** — `z3_muon.json` sets `"reduce_scatter": false` (all-reduce the full
gradient instead of scattering it) — so the full 2D matrix is always present
locally and NS is trivially local. Hybrid selection is `use_muon = (ndim >= 2 and
"embed" not in name)` (else fused Adam), with two LRs (`muon_lr`/`adam_lr`). NS is
`gram` (Gram-Schmidt NS, ~2× on rectangular matrices, default) or `standard`, and
`@torch.compile`d. MoE uses **AutoEP** (shard the expert dim; each rank holds
whole experts → per-expert NS); 4D conv is flattened to 2D. Muon state = 1
momentum buffer vs Adam's 2 (~45% state / ~9% peak saving).

**The essential contrast.** DeepSpeed's trick is *don't scatter the gradient*, so
the 2D matrix is never lost and NS is local *for free* — but (a) the gradient
all-reduce is full-size, (b) params stay **replicated** in ZeRO-1/2 (no
param-memory balance), and (c) the optimizer still all-gathers the updated weights,
so the step is **not communication-free**. `Owned` attacks the *other* lever:
concentrate each matrix's grad on one owner (reduce-to-owner), run NS there, and
let the forward broadcast it does anyway carry the update → the step adds **zero
collectives** and stays exact. That comm-free property is this design's distinctive
contribution; DeepSpeed has no equivalent. `GroupedRaggedShard` is the rung that
most resembles DeepSpeed (reconstruct the full matrix, then NS), except it gathers
explicitly per bucket and balances **to the byte, gradient included**, whereas
DeepSpeed gets the full matrix implicitly by replicating the gradient.

| Axis | DeepSpeed Muon (ZeRO) | flex_shard + Muon |
|---|---|---|
| Full matrix for NS | gradient **replicated** (all-reduce, `reduce_scatter:false`) | `Owned`: reduce-to-owner; `GroupedMuon`: whole matrices kept; `GroupedRaggedShard`: all-gather/bucket |
| Step communication | **not comm-free** (full-grad all-reduce + param all-gather) | `Owned`: **zero**; `GroupedRaggedShard`: 1 all-gather/bucket |
| NS redundancy | distributed by partition / round-robin (~1 rank/param) | `Owned`/`GroupedMuon`: 1 rank; `GroupedRaggedShard`: ×`ws` |
| Param memory | ZeRO-1/2 **replicated**; z3 sharded but grads kept full | `Owned`: matrix-granular (LPT); `GroupedRaggedShard`: **byte-perfect** |
| NS kernel | **Gram NS** (~2×) / standard, `torch.compile` | upstream quintic; `GroupedMuon` batched `baddbmm` (no Gram yet) |
| MoE | AutoEP shards expert dim → per-expert NS | `Shard(0)` → `GroupedMuon` (**same idea**, comm-free) |
| Framework | config-driven, coupled to the ZeRO engine, HF out-of-box | explicit placement contract, PyTorch-native, eager |

**Convergent design.** Both land on the same MoE insight (shard the expert dim →
whole experts local → batched/per-expert NS), the same hybrid rule (`ndim >= 2 &
not embed`), the same optimizer-state halving, and NS on the genuine full matrix
(exact).

**Where each leads.** *DeepSpeed:* Gram NS (~2×), `torch.compile`d update, mature
AutoEP/AutoTP/offload + HF/checkpoint tooling, production-validated, zero-code
config. *flex_shard:* the **comm-free exact** `Owned` step, an explicit memory↔comm
**placement spectrum** the user dials, byte-perfect balance *including gradients*,
per-layer LPT owner balancing, and no DeepSpeed/ZeRO coupling (upstream
`torch.optim.Muon`). Concrete cross-pollination: DeepSpeed's **Gram-NS** kernel
would drop straight into `GroupedMuon` / `RaggedShardMuon` for a ~2×
orthogonalization speedup, since both run NS on the reconstructed full matrix.

## Key references

- Muon: `pytorch/torch/optim/_muon.py` — NS `_zeropower_via_newtonschulz`
  (l.31-70), 2D guard (l.128-133), momentum state (l.156-160), `_adjust_lr`
  (l.73-84).
- `Owned`: `example/owned.py:35` (broadcast unshard l.135-160, reduce-to-owner
  l.207-214, `param_boundary_placements` l.248).
- `Shard(0)`: `example/shard.py:337-342` (reduce-scatter AVG l.317-321).
- `BucketSpec` / uniform placement: `flex_shard/flex_shard.py:57`,
  `_validate_bucket_uniform_dtype_and_placement` (`utils.py`).
- Placement metadata: `flex_shard/sharded_param.py` (`get_placements`,
  `get_global_shape`).
- Test harness: `tests/common.py` (`check_flex_shard_parity`, `expected_shard`,
  `transformer_bucket_specs`), `tests/test_flex_shard_training.py`
  (AVG reference grads, `FSDPTest` world_size=2).
- DeepSpeed Muon (ZeRO): `deepspeed/runtime/zero/muon/` —
  `muon_optimizer.py::MuonWithAuxAdam.step` (NS moved out to the engine),
  `original_muon.py` (`muon_update`, `zeropower_via_gram_newtonschulz` /
  `…newtonschulz5`, round-robin + `all_gather` `Muon.step`), NS hooked in the ZeRO
  engine's `get_flat_partition`. Blog:
  `pytorch.org/blog/using-muon-optimizer-with-deepspeed`. Demo:
  `github.com/delock/deepspeed_finetune_demo` (`z3_muon.json` →
  `reduce_scatter:false`; `z2_moonlight_autoep_muon.json` → AutoEP + `ns_method:gram`).
