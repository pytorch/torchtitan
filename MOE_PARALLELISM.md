# MoE Parallelism: a guided code walk

This document explains how Expert Parallelism (EP) works in this codebase,
in detail, with line references. It assumes you've read `WALKTHROUGH.md`
and know the high-level file layout.

The four files you need to understand:

1. **`src/distributed/parallel_dims.py`** — builds the device meshes
2. **`src/models/moe/moe.py`** — the MoE forward pass (router, experts, combine)
3. **`src/distributed/expert_parallel.py`** — the `ExpertParallel` `ParallelStyle` that hooks into the MoE module
4. **`src/models/parallelize.py`** — applies `ExpertParallel` + FSDP to the model

Read this doc first, then read each file with this doc in hand.

---

## Part 1: The mental model (before any code)

### 1.1 The problem EP solves

An MoE layer has `E` experts. Without EP, every GPU holds all `E` experts and
they each process the tokens routed to them locally. This works but doesn't
scale: once `E × expert_size` exceeds a GPU's memory, you're stuck.

EP's trick: **shard the experts across GPUs by the expert dimension**. If
you have `E=64` experts and `ep=4` GPUs, each GPU holds 16 experts.

But there's a catch: the tokens destined for GPU 1's experts are mixed in
with every other GPU's tokens. A token on GPU 0 might need to be processed
by expert 20, which lives on GPU 1. So you need to **route tokens between
GPUs**, process them with the local experts, and **route the outputs back**.

This routing is done with **all-to-all** collectives:

```
Before all-to-all (dispatch):
    GPU 0 has some tokens for each of experts 0..63
    GPU 1 has some tokens for each of experts 0..63
    GPU 2 has some tokens for each of experts 0..63
    GPU 3 has some tokens for each of experts 0..63

After dispatch all-to-all:
    GPU 0 has ALL tokens destined for experts  0..15
    GPU 1 has ALL tokens destined for experts 16..31
    GPU 2 has ALL tokens destined for experts 32..47
    GPU 3 has ALL tokens destined for experts 48..63

    → Each GPU runs its 16 local experts on its local tokens.

After combine all-to-all:
    GPU 0 has the outputs for the tokens it originally held
    (same for all GPUs)
```

That's the whole idea. The rest is bookkeeping.

### 1.2 The two collectives

EP uses two all-to-all calls per MoE forward pass:

1. **Dispatch all-to-all** (before expert computation):
   - Input: routed input tokens grouped by destination expert
   - Output: tokens redistributed so each rank owns the tokens for its local experts

2. **Combine all-to-all** (after expert computation):
   - Input: expert outputs (still on the expert-owning rank)
   - Output: outputs redistributed back to the rank that originally held the token

Both calls are symmetric — the combine call uses the same splits as dispatch
but in the opposite direction.

### 1.3 The mesh picture

With `world_size=4, dp_shard=4, ep=4`:

```
                  dp_shard mesh (size 4)
                  ─────────────────────
    GPU 0         non-expert params sharded across 4 GPUs via FSDP
    GPU 1         non-expert grads reduce-scattered across 4 GPUs
    GPU 2
    GPU 3
                  ─────────────────────

                  ep mesh (size 4)
                  ─────────────────────
    GPU 0         holds experts  0..15 (16 experts)
    GPU 1         holds experts 16..31
    GPU 2         holds experts 32..47
    GPU 3         holds experts 48..63
                  ─────────────────────

                  efsdp mesh (size 1)
                  ─────────────────────
    GPU 0         expert params NOT further sharded after EP took its cut
                  ─────────────────────
```

Both the `dp_shard` mesh and the `ep` mesh happen to be the same 4 GPUs in
this case, but they play different roles: `dp_shard` is the FSDP mesh for
non-expert params, `ep` is the all-to-all mesh for token routing.

---

## Part 2: Building the meshes (`parallel_dims.py`)

Open [`src/distributed/parallel_dims.py`](src/distributed/parallel_dims.py).

The `build_mesh` method (line ~61) does all the work. Here's the sequence:

### 2.1 Compute derived sizes (lines 127–129)

```python
batch = self.dp_replicate * self.dp_shard   # 1 * 4 = 4
fsdp  = self.dp_shard * self.cp             # 4 * 1 = 4
efsdp = fsdp * self.tp // (self.etp * self.ep)  # 4 * 1 / (1 * 4) = 1
```

For our config (`dp_shard=4, ep=4, tp=1, pp=1, cp=1`):
- `batch = 4` (data loading mesh)
- `fsdp = 4` (non-expert FSDP mesh)
- `efsdp = 1` (expert FSDP mesh — size 1 means "don't actually shard experts
  further, EP already did it")

### 2.2 Create the world mesh (line 131)

```python
self._world_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("world",))
```

This is the flat 1D mesh: `[GPU 0, GPU 1, GPU 2, GPU 3]`. Everything else is
derived by reshaping this.

### 2.3 Unflatten into three named views (lines 134–149)

The world mesh is then unflattened into three *different logical views* of
the same 4 physical GPUs:

```python
dataloading_mesh = unflatten_mesh(
    self._world_mesh,
    ("pp", "batch", "cp", "tp"),
    (self.pp, batch, self.cp, self.tp),     # (1, 4, 1, 1)
)

dense_mesh = unflatten_mesh(
    self._world_mesh,
    ("pp", "dp_replicate", "fsdp", "tp"),
    (self.pp, self.dp_replicate, fsdp, self.tp),  # (1, 1, 4, 1)
)

sparse_mesh = unflatten_mesh(
    self._world_mesh,
    ("pp", "dp_replicate", "efsdp", "ep", "etp"),
    (self.pp, self.dp_replicate, efsdp, self.ep, self.etp),  # (1, 1, 1, 4, 1)
)
```

**The `dense_mesh` and `sparse_mesh` are key.** They partition the 4 GPUs
differently:

- `dense_mesh`: the 4 GPUs are arranged as the `fsdp` dimension. FSDP all-gathers
  and reduce-scatters parameters across this dimension.
- `sparse_mesh`: the 4 GPUs are arranged as the `ep` dimension. Expert all-to-all
  routes tokens across this dimension.

Both views cover the same 4 physical GPUs. They're just named/indexed
differently, which lets FSDP and EP grab the collective they need.

### 2.4 Build the flat lookup dict (lines 158–169)

```python
self._meshes = {
    "pp":           dataloading_mesh["pp"],
    "batch":        dataloading_mesh["batch"],
    "loss":         loss_mesh,
    "dp_replicate": dense_mesh["dp_replicate"],
    "fsdp":         dense_mesh["fsdp"],
    "cp":           dataloading_mesh["cp"],
    "tp":           dataloading_mesh["tp"],
    "ep":           sparse_mesh["ep"],     # ← this is the EP all-to-all mesh
    "efsdp":        sparse_mesh["efsdp"],  # ← expert-FSDP mesh (size 1 here)
    "etp":          sparse_mesh["etp"],
}
```

Now `parallel_dims.get_mesh("ep")` returns the 4-GPU sub-mesh of
`sparse_mesh`, and `get_mesh("fsdp")` returns the 4-GPU sub-mesh of
`dense_mesh`. Both are usable as torch `DeviceMesh` objects that you can pass
directly to `fully_shard` or `parallelize_module`.

### 2.5 Verify with validation (line 172)

`_validate_meshes` just asserts that each named mesh has the expected size.
Useful guardrail when you change parallelism settings.

**TL;DR of Part 2:** `ParallelDims` gives you multiple named views of the
same physical GPUs. The `"ep"` mesh is what the all-to-all inside the MoE
layer will talk over. The `"fsdp"` mesh is what FSDP wraps non-expert
parameters with.

---

## Part 3: The MoE forward pass (`moe.py`)

Open [`src/models/moe/moe.py`](src/models/moe/moe.py), jump to
`MoE.forward` at line 462. This is TorchTitan code we kept unchanged.

### 3.1 Step-by-step (reshaping annotated)

```python
def forward(self, x):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)                    # (B*S, dim) — flatten token dim
```

Let `N = B * S` (total tokens).

```python
    top_scores, selected_experts_indices, num_tokens_per_expert = \
        self.router(x, self.expert_bias)
    # top_scores, selected_experts_indices: (N, top_k)
    # num_tokens_per_expert: (num_experts,)  — count per expert across all N*top_k entries
```

The router takes each token, computes a score for every expert, picks the
top-k, and records how many tokens chose each expert.

```python
    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)
```

This is the load-balancing bookkeeping. The buffer is aggregated at each
optimizer step and used to nudge the router bias (no auxiliary loss needed).

```python
    top_scores_experts_sorted, token_indices_experts_sorted, num_tokens_per_expert = \
        self.reorderer(top_scores, selected_experts_indices)
    # All shapes: (N * top_k,) — sorted by expert id so all tokens for expert 0 come first, then expert 1, ...
```

**This is the permutation that makes EP possible.** After reordering, the
input to the experts is a flat `(N*top_k, dim)` tensor where the first chunk
is all tokens going to expert 0, then expert 1, etc. The `num_tokens_per_expert`
tells you how long each chunk is.

```python
    routed_input = x[token_indices_experts_sorted // self.router.top_k]
    # (N*top_k, dim) — the actual tokens, in expert order
```

The `// top_k` is because each token appears `top_k` times in
`token_indices_experts_sorted` (once per expert it's routed to).

```python
    if self.score_before_experts:
        routed_input = (routed_input.to(torch.float32)
                        * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)
```

Pre-scale by the router scores so the expert outputs are already weighted.

```python
    routed_output = self.experts(routed_input, num_tokens_per_expert)
```

**This is the line where EP hooks in.** `self.experts` is a `GroupedExperts`
module that, without EP, just runs a batched matmul locally. *With* EP, the
`ExpertParallel` wrapper has installed forward pre/post hooks that (1)
all-to-all dispatch the `routed_input` across the EP ranks, (2) run the
local experts on the redistributed input, (3) all-to-all combine the outputs.

We'll trace that in Part 4.

```python
    out = self.shared_experts(x) if self.shared_experts is not None else None
```

Shared experts run on *every* token — no routing. Note it runs in parallel
with `self.experts`, letting you overlap compute with the (already kicked-off)
all-to-all communication.

The rest is the unsort + weighted combine:

```python
    routed_output_unsorted = torch.zeros((N * top_k, dim), ...)
    routed_output_unsorted[token_indices_experts_sorted] = routed_output
    routed_output_unsorted = routed_output_unsorted.reshape(-1, top_k, dim)
    out_experts = routed_output_unsorted.sum(dim=1)  # sum over top_k

    return (out + out_experts).reshape(bs, slen, dim)
```

Unsort puts outputs back in original token order, then sums over the top_k
experts chosen for each token, adds the shared output, and returns.

**Important:** without EP, the MoE forward is the same code, just with
`self.experts(routed_input, num_tokens_per_expert)` running everything
locally. EP changes that single line's semantics via hooks, not source.

---

## Part 4: The ExpertParallel hook (`expert_parallel.py`)

Now the interesting part. Open
[`src/distributed/expert_parallel.py`](src/distributed/expert_parallel.py).

`ExpertParallel` is a `ParallelStyle` (the same interface that TP uses).
When you call `parallelize_module(experts, ep_mesh, ExpertParallel())`, PyTorch
does three things via `ExpertParallel._apply` (line 114):

```python
def _apply(self, module, device_mesh):
    return distribute_module(
        module,
        device_mesh,
        partition_fn=self._partition_fn,  # (1) shard the weights
        input_fn=self._token_dispatch,    # (2) pre-forward hook
        output_fn=self._token_combine,    # (3) post-forward hook
    )
```

Let's walk through each.

### 4.1 `_partition_fn` — shard expert weights (line 46)

```python
def _partition_fn(self, name, mod, device_mesh):
    for param_name, param in mod.named_parameters(recurse=False):
        dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
        mod.register_parameter(param_name, dist_param)
```

`GroupedExperts` has parameters `w1`, `w2`, `w3` with shape
`(num_experts, hidden_dim, dim)` or `(num_experts, dim, hidden_dim)`. The
leading dim is the expert dimension.

`Shard(0)` says "shard along dim 0." With `ep=4` and `num_experts=64`, each
rank ends up holding a tensor of shape `(16, ..., ...)` — 16 local experts
per rank. The global logical tensor is still `(64, ..., ...)`, but physically
it's distributed.

### 4.2 `_token_dispatch` — the dispatch all-to-all (line 51)

This runs as a pre-forward hook. The input to `self.experts(...)` in
`MoE.forward` (line 513) is `(routed_input, num_tokens_per_expert)`. This
function replaces that input with a redistributed version before the actual
expert computation runs.

```python
def _token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs
    ep_degree = device_mesh.shape[0]
    num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
```

Example: `num_tokens_per_expert` has shape `(64,)`, `ep_degree=4`, so
`num_local_experts=16`.

#### 4.2a Exchange the expert counts

```python
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert, None, None,
            group=device_mesh.get_group(),
        )
```

This is the **first all-to-all**. `num_tokens_per_expert` holds "how many
tokens on this rank are going to each expert" (64 entries). After the
all-to-all, `num_tokens_per_expert_group` holds "how many tokens from each
rank are coming to our local experts" (also 64 entries but reorganized).

Specifically, if rank 0 says "I have 10 tokens for expert 20" (which lives
on rank 1), that 10 gets sent to rank 1, who now knows "rank 0 is sending
me 10 tokens for expert 20."

#### 4.2b Compute the splits for the payload all-to-all

```python
        input_splits  = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1).cpu()
        output_splits = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1).cpu()
```

Reshape to `(ep_degree, num_local_experts)` = `(4, 16)`. Summing along dim 1
gives a `(4,)` tensor: "how many tokens am I sending to each rank" (input)
and "how many tokens am I receiving from each rank" (output).

These have to go to CPU because `all_to_all_single` expects Python ints for
split sizes.

#### 4.2c The payload all-to-all

```python
    routed_input = all_to_all_single_autograd(
        routed_input,
        self.output_splits,  # receive sizes
        self.input_splits,   # send sizes
        device_mesh.get_group(),
    )
```

This is the **second all-to-all** — the one that actually moves the token
embeddings. `_autograd` variant supports backprop (the reverse all-to-all
happens in the backward pass automatically).

After this, `routed_input` on each rank contains:
- Tokens from rank 0 destined for local experts
- Tokens from rank 1 destined for local experts
- Tokens from rank 2 destined for local experts
- Tokens from rank 3 destined for local experts

…in that order.

#### 4.2d Reorder into per-local-expert order

The previous step gave us tokens grouped by *sender*, not by *destination
local expert*. We need them grouped by local expert (so all tokens for
local expert 0 come first, then local expert 1, etc.) because that's what
`GroupedExperts` expects.

```python
    self.input_shape, routed_input, self.permuted_indices, num_tokens_per_expert_group = \
        _permute(routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts)
```

`_permute` (defined in `src/models/moe/utils.py`) computes the permutation
that reorders tokens from "grouped by sender rank" to "grouped by local
expert." It also pads each expert's token count up to a multiple of 8
(`TOKEN_GROUP_ALIGN_SIZE_M = 8`) for the grouped matmul kernel.

#### 4.2e Return modified inputs

```python
    return routed_input, num_tokens_per_expert_group
```

These two tensors are passed into the actual `self.experts.forward()` call,
replacing the original arguments. From the MoE module's perspective, it just
handed off `(routed_input, num_tokens_per_expert)` and got back a result —
unaware that its input got rewritten and the tokens came from 4 different
GPUs.

### 4.3 The local expert computation

Back in `GroupedExperts.forward` (in the same `moe.py` file, lines ~180), it
does a batched matmul. With `use_grouped_mm=True` this calls
`torch._grouped_mm`, a CUDA kernel that can run `w_i @ x_i` for a variable
number of rows per expert efficiently. This runs purely locally — no
collectives.

### 4.4 `_token_combine` — the combine all-to-all (line 99)

This runs as a post-forward hook. `routed_output` has shape
`(total_local_tokens, dim)` — outputs of the local experts.

```python
def _token_combine(self, mod, routed_output, device_mesh):
    routed_output = _unpermute(
        routed_output, self.input_shape, self.permuted_indices
    )
```

Reverse the `_permute` — put the outputs back into "grouped by sender rank"
order.

```python
    routed_output = all_to_all_single_autograd(
        routed_output,
        self.input_splits,    # NOTE: swapped from dispatch
        self.output_splits,
        device_mesh.get_group(),
    )
    return routed_output
```

The **third all-to-all** — routes outputs back to the rank that originally
held each token. Note `input_splits` and `output_splits` are *swapped*
compared to dispatch: what we sent out, we now receive back, and vice versa.

### 4.5 Summary of collectives per MoE layer

Per forward pass, each rank does:
1. One small all-to-all on `num_tokens_per_expert` (tiny, just metadata)
2. One all-to-all on `routed_input` (the actual data, `(local_N_tokens, dim)`)
3. Local `GroupedExperts.forward` (no collectives)
4. One all-to-all on `routed_output` (the data again, back to senders)

And during backward, all three get their reverse collectives automatically
via `all_to_all_single_autograd`.

---

## Part 5: Putting it together (`parallelize.py`)

Now open [`src/models/parallelize.py`](src/models/parallelize.py).

### 5.1 `apply_moe_ep` (line 145)

```python
def apply_moe_ep(model, ep_mesh):
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue
        experts_plan = ExpertParallel()
        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=ep_mesh,
            parallelize_plan=experts_plan,
        )
```

This loops over every transformer block, and for each MoE layer, calls
`parallelize_module` on the `block.moe.experts` submodule (the
`GroupedExperts`). That invokes `ExpertParallel._apply`, which in turn
partitions the weights and installs the dispatch/combine hooks we saw in
Part 4.

After this runs, every `GroupedExperts` module in the model has its forward
intercepted by the all-to-all dance. The top-level `MoE` module doesn't know
anything changed.

### 5.2 `apply_fsdp` (line 17)

FSDP runs *after* EP in the ordering (EP → AC → compile → FSDP). The key
insight is that `apply_fsdp` wraps things at different granularities
depending on whether they are expert or non-expert:

```python
for layer_id, transformer_block in model.layers.items():
    if transformer_block.moe_enabled and ep_degree > 1:
        fsdp_mod_ep_config = fsdp_config.copy()
        fsdp_mod_ep_config["mesh"] = edp_mesh   # ← separate mesh!

        fully_shard(
            transformer_block.moe.experts,
            **fsdp_mod_ep_config,
            ...
        )
        transformer_block.moe.experts.set_gradient_divide_factor(gradient_divide_factor)

    fully_shard(transformer_block, **fsdp_config)
```

Read this carefully:

1. If this layer has MoE and EP is on: FSDP-wrap `block.moe.experts` on the
   **`edp_mesh`** (= the `efsdp` sub-mesh). This is size 1 in our setup, so
   it doesn't actually shard further — it's only wrapped for mixed-precision
   bookkeeping and for FSDP2's parameter manager to include it.
2. Then FSDP-wrap the *whole* `transformer_block` on the **`dp_mesh`** (= the
   `fsdp` sub-mesh, size 4). This wraps attention, router, shared experts,
   and the MoE module *excluding* `experts` (which was already wrapped).

So non-expert params get FSDP-sharded across all 4 GPUs (full sharding), and
expert params get FSDP-wrapped on their own 1-GPU mesh (no extra sharding,
because EP already split them).

### 5.3 The gradient divide factor hack

```python
transformer_block.moe.experts.set_gradient_divide_factor(gradient_divide_factor)
```

Why? FSDP by default divides gradients by the size of the FSDP mesh. For
non-expert params on a 4-GPU mesh, FSDP divides by 4 → correct.

For expert params on a 1-GPU mesh, FSDP would divide by 1 → too big by a
factor of 4 compared to non-expert params. We need them on the same scale
so the optimizer applies consistent updates.

The `gradient_divide_factor` is manually set to `parallel_dims.fsdp_gradient_divide_factor`
(= `dp_replicate * dp_shard * cp = 4`) so expert grads get the same divisor
as non-expert grads.

### 5.4 Prefetch scheduling (line 89 onward)

The tail of `apply_fsdp` sets up explicit forward/backward prefetching
(`set_modules_to_forward_prefetch`, `set_modules_to_backward_prefetch`). This
is necessary because the EP all-to-all calls do a device→host sync, which
can disrupt FSDP's implicit prefetch scheduling. Explicit prefetching tells
FSDP: "while this block is running, start all-gathering the next block's
parameters."

You don't need to understand this to use EP correctly — it's a perf
optimization that only matters when EP is on.

---

## Part 6: Walk through one MoE forward pass end-to-end

Let's trace what happens for *one* token passing through *one* MoE layer
with 4-way EP. Say the token lives on GPU 0, and the router sends it to
expert 20 (which lives on GPU 1) as its top-1 choice.

1. **Input:** `x = [token_embed]` on GPU 0, shape `(1, dim)`.

2. **Router** (local compute on GPU 0):
   - Computes scores for all 64 experts
   - Picks top_k=6, let's say expert 20 is in the top-6
   - `selected_experts_indices = [..., 20, ...]`

3. **Reorderer** (local compute on GPU 0):
   - Sorts tokens by destination expert
   - Our token ends up in the chunk labeled "expert 20"

4. **`self.experts(routed_input, num_tokens_per_expert)` call** — this is where
   the hooks kick in.

5. **`_token_dispatch` pre-hook** (runs on all 4 GPUs in lockstep):
   - Small all-to-all: GPU 0 tells GPU 1 "I'm sending you 1 token for expert 20"
   - Large all-to-all: token embedding actually moves from GPU 0 → GPU 1
   - `_permute` reorders tokens on GPU 1 so all tokens for expert 16 come
     first, then expert 17, …, expert 20 (where our token is), …, expert 31

6. **`GroupedExperts.forward` (local on GPU 1):**
   - Takes the flat `(total_local_tokens, dim)` buffer
   - For each of its 16 local experts, runs `w1 @ x → silu → w3 @ x → elementwise
     → w2 @ y`
   - Our token gets processed by expert 20's weights
   - Returns `(total_local_tokens, dim)` of outputs, same layout

7. **`_token_combine` post-hook** (runs on all 4 GPUs):
   - `_unpermute` puts outputs back in "grouped by sender" order on GPU 1
   - Large all-to-all in reverse: our token's output moves from GPU 1 → GPU 0
   - `routed_output` on GPU 0 now has the expert-20 output for our token

8. **Back in `MoE.forward`** on GPU 0:
   - `routed_output_unsorted[token_indices_experts_sorted] = routed_output`
     puts our token back in the original `(N, top_k, dim)` layout
   - Sum over `top_k` dim combines outputs from all 6 experts our token hit
   - Add shared-expert output
   - Return

One token = 3 all-to-all round trips per MoE layer. With 27 MoE layers (the
15B config), that's 81 round trips per forward pass. Fortunately each
all-to-all only moves the tokens that need to move, and the shared expert
compute happens in parallel to the combine all-to-all, hiding much of the
latency.

---

## Part 7: Why does the EP correctness check pass exactly?

Run `scripts/ep_correctness.sh`: `EP=1 loss: 7.350637`, `EP=4 loss: 7.350637`,
**difference 0.000000**.

This is exact bit-for-bit equality, not an approximation. It passes exactly
because:

1. `force_load_balance=True` → the router assigns each token deterministically
   via round-robin, independent of the router's actual scores. So the token →
   expert assignment is the same whether EP is 1 or 4.
2. The all-to-all routing just *moves* tokens — it doesn't compute anything.
   So `token → expert_20.forward(token)` is the same computation whether
   expert_20 lives on the same rank as the token or on a different rank.
3. Cross-rank reductions (the grad divide factor fix in `apply_fsdp`) make
   the gradients consistent.

If any of these three things were off, the check would diverge. So if you
ever touch EP code and the check starts failing, one of those three is
broken.

---

## Part 8: If you want to experiment

To see what's happening concretely, add some prints inside
`ExpertParallel._token_dispatch`:

```python
def _token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs
    # ...
    # ADD:
    rank = torch.distributed.get_rank(device_mesh.get_group())
    print(f"[rank {rank}] dispatching {routed_input.shape[0]} tokens, "
          f"will receive {sum(self.output_splits)} tokens")
    # ...
```

Run the tiny smoke test and watch each rank report how many tokens it sends
and receives. With `moe_force_load_balance=True` these numbers should be
roughly equal across ranks; without it, you'll see imbalance.

You can also turn on `logfire.configure(console={"min_log_level": "debug"})`
to get more trace info from the internal collectives.

---

## Cheat sheet: when to look at which file

| I want to understand... | Read... |
|---|---|
| How the meshes are named and built | `src/distributed/parallel_dims.py` lines 60–180 |
| How the router picks experts | `src/models/moe/moe.py` lines 180–349 (`TokenChoiceTopKRouter`) |
| How tokens get sorted before EP | `src/models/moe/moe.py` lines 353–409 (`TokenReorderer`) |
| How EP shards weights | `src/distributed/expert_parallel.py` lines 46–49 |
| How EP dispatches tokens via all-to-all | `src/distributed/expert_parallel.py` lines 51–97 |
| How EP combines outputs | `src/distributed/expert_parallel.py` lines 99–112 |
| How the full MoE forward orchestrates router → dispatch → experts → combine | `src/models/moe/moe.py` lines 462–544 (`MoE.forward`) |
| How EP gets plugged into the model | `src/models/parallelize.py` lines 145–161 (`apply_moe_ep`) |
| How FSDP wraps expert vs non-expert params differently | `src/models/parallelize.py` lines 50–90 (`apply_fsdp`) |
| What order EP / AC / compile / FSDP are applied | `src/trainer.py` around line 130 |

Keep this open in a second window while you walk through the code.
