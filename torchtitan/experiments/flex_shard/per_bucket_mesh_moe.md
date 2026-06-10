# Per-Bucket Mesh: Multi-Mesh MoE Sharding in FlexShard

## Question

Can FlexShard's per-bucket `BucketSpec.mesh` reproduce what `fully_shard`'s
per-parameter mesh (`shard_placement_fn` returning `ShardPlacementResult`) does
for a nested `root -> layer -> MoE` model — i.e. experts on an expert-FSDP axis,
dense params on the data-parallel axis?

Reference: `test/distributed/_composable/fsdp/test_fully_shard_training.py::
TestFullyShardNDTraining::test_shard_placement_fn_tp_ep`.

## Answer

**Yes, for the sharding + grouping** — with one caveat about the *runtime
forward* path (below).

`fully_shard` *derives* communication groups from per-param meshes and splits a
`fully_shard(block, shard_placement_fn=...)` call into multiple `FSDPParamGroup`s
(experts on `efsdp`, dense on `dp`). FlexShard instead *declares* them as flat,
FQN-patterned `BucketSpec`s, each carrying its own 1-D mesh. The `root -> layer ->
MoE` nesting becomes a flat partition of buckets (FlexShard is a single
root-level call; nested wrapping is rejected by `_check_not_already_flex_sharded`).

```python
# world = 4 ranks factored as efsdp(2) x ep(2); same rank set, two 1-D meshes
dp_mesh    = init_device_mesh(dev, (4,), ("dp",))         # dense  -> Shard(0)/dp(4)
efsdp_mesh = init_device_mesh(dev, (2, 2), ("efsdp","ep"))["efsdp"]  # experts -> Shard(0)/efsdp(2)

flex_shard(model, buckets=[
    # root (dense) -> dp
    BucketSpec(["tok_embeddings.*"], placement_fn=per_param_placements, mesh=dp_mesh),
    BucketSpec(["pos_embeddings.*"], placement_fn=per_param_placements, mesh=dp_mesh),
    BucketSpec(["norm.*"],           placement_fn=per_param_placements, mesh=dp_mesh),
    BucketSpec(["output.*"],         placement_fn=per_param_placements, mesh=dp_mesh),
    # per layer: dense attention + norms -> dp; expert FFN stacks -> efsdp
    *[BucketSpec([f"layers.{i}.attention.*", f"layers.{i}.attention_norm.*",
                  f"layers.{i}.ffn_norm.*"], placement_fn=per_param_placements, mesh=dp_mesh)
      for i in ...],
    *[BucketSpec([f"layers.{i}.expert_layer.*"], placement_fn=per_param_placements, mesh=efsdp_mesh)
      for i in ...],
])
```

`_assign_params_to_buckets` requires **exactly one bucket per param** (rejects
overlaps and orphans), so experts are carved out with their own pattern rather
than a lazy `layers.N.*`.

### What does NOT carry over to the exact `fully_shard` test

The reference test is a **DTensor (TP+EP) + 2-D HSDP** composition, both outside
FlexShard's current scope (`_validate_eager_params` rejects DTensor;
`_validate_flex_shard_mesh` requires `ndim == 1`). So the per-bucket mesh matches
the *plain-tensor, 1-D-submesh* version of that sharding, not the test verbatim.
The `efsdp`/`dp` meshes here are axis-factorizations of the same world (every rank
participates in each), which is exactly the case the per-bucket mesh was built for
(disjoint-rank submeshes remain a TODO).

## Test added

`tests/test_flex_shard_buckets.py::TestMultiMeshBuckets` (new `FSDPTest`,
`world_size=4`, `@skip_if_lt_x_gpu(4)`). It reuses the same toy `Transformer`
(`ModelArgs(num_experts=8, weight_tying=False, ...)` from
`torch.testing._internal.distributed._tensor.common_dtensor`) that fully_shard's
`test_shard_placement_fn_tp_ep` uses — so per layer the dense set is attention
`wq/wk/wv/wo` + `attention_norm`/`ffn_norm` (on `dp`) and the experts are the 3-D
FFN stacks `expert_layer.experts.{w1,w2}` (on `efsdp`). `weight_tying=False`
because flex_shard rejects shared params; the toy MoE has no router gate (it runs
and averages all experts), so only the expert FFN weights live on `efsdp`. It
asserts:

1. **Grouping** — each `ShardedBucketStorage._mesh` is `efsdp` for experts, `dp`
   otherwise.
2. **Sharding** — every local shard is byte-exact
   `expected_shard(ref, mesh_local_rank, mesh_size)`: experts<->efsdp,
   dense<->dp.
3. **Leading dim** — expert dim-0 == `num_experts // efsdp_size` (2), not
   `// dp_size` (4).
4. **Runtime gather** — gathering each local shard over its bucket's mesh process
   group reconstructs the full reference param (experts via efsdp(2), dense via
   dp(4)).

Result: **passes** (needs **4 GPUs**; skips otherwise — with fewer, `efsdp(2) <
dp(4)` cannot be formed). It uses the upstream `Transformer`, whose
`Experts.forward` reads `self.w1` twice, so it verifies the gather via the mesh
process groups directly rather than `model.forward` (see the limitation below).

### Full training-loop parity (`test_train_parity`)

A second test runs a real fwd/bwd/SGD loop and asserts numerics against a
single-device reference, with **dense blocks on `dp` and expert FFNs on `efsdp`**.
It uses a small custom `_SiblingMoE` (dense `blocks` and `experts` as sibling
module lists) whose forwards read each managed parameter **once** (see limitation).
All ranks share the input, so per-rank grads are identical and flex's
reduce-scatter (mean) is a no-op + chunk; after each SGD step every local shard
equals the reference param chunked on its bucket's mesh (experts<->efsdp,
dense<->dp). Forward parity (both meshes' all-gathers reconstruct), loss parity,
and per-shard parity hold across 3 steps. **Passes.**

## Limitation: a managed parameter may be read only once per forward

`model.forward` on the upstream model failed:

```
RuntimeError: FlexShard eager mode requires full parameter data from a bucket
unshard hook for parameter '...experts.w1' ... The bucket hook was registered but
did not run before parameter access.
```

**Root cause (NOT multi-bucket / NOT nesting — my earlier note was wrong):**
flex_shard replaces each managed parameter with a property getter backed by a
per-param *unshard slot*; the bucket's pre-forward hook fills the slot and the
post-forward hook clears it. But `UnshardedParamSlot.consume_unsharded_param`
(`flex_shard/unsharded_param_getters.py`) **clears the slot on the first read**,
with an explicit TODO: *"Clearing it here breaks legal modules that read the same
parameter more than once."* So a forward that reads the same `self.w` twice gets
`None` on the second read and raises.

- `nn.Linear` / `LayerNorm` read `self.weight` once → fine (per-layer FSDP works,
  and that is why `transformer_bucket_specs` forward tests pass).
- The upstream `Experts.forward` reads `self.w1` twice — `isinstance(self.w1,
  DTensor)` **and** the `w1, w2 = self.w1, self.w2` assignment — so it raises.
- Earlier expert forwards I wrote (`self.w1.shape[0]` + `self.w1.transpose(...)`)
  hit the same double-read. Multi-bucket-per-layer and sibling/nested buckets were
  red herrings; the blocker is the double-read.

This is **orthogonal to the per-bucket mesh feature** — sharding and per-mesh
all-gather both work.

### Workarounds / fix
- Model side: read each managed param into a local once (`w1 = self.w1; ...`) —
  what `_SiblingMoE` does, which is what unblocks the training-loop test.
- flex_shard side: keep the slot valid for the whole forward (clear only in the
  post-forward hook, per the existing TODO) so arbitrary modules — including the
  upstream `Experts` — can train unmodified.

## Notes

- The reverted repo-wide churn: `pre-commit` here runs pyrefly / auto-fix hooks
  that ignore `--files` and strip unused imports across the whole tree. Use
  `git commit --no-verify`; lint individual files with `black`/`flake8` directly
  rather than via `pre-commit`. The test file itself is lint-clean
  (black/µfmt/flake8/isort pass).
- `tests/test_flex_shard_buckets.py` belongs to PR #3239 (`Introduce FlexShard`);
  the new test can be amended there.
