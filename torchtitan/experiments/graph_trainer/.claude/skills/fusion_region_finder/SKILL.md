---
name: fusion_region_finder
description: Find kernel-fusion regions in a GraphTrainer training step (forward, recompute and backward) from the traced joint graph, a profiler trace and a measured roofline; rank them by gap to roofline, and write one folder per region with a compile.extra_passes graph pass that lifts the region into a call_module, whose body is the fusion target. Use when asked to find fusion opportunities, rank glue ops by distance from roofline, or prepare targets for handwritten kernels in graph_trainer.
---

# Finding fusion regions in a GraphTrainer step

The method: capture one training step as GraphTrainer runs it, with every FX node
labelled in the profiler trace; attribute each GPU kernel to its node; measure how
far each group of fusible nodes is from its roofline; lift the groups into
`call_module` nodes so a kernel can replace them. Kernel problems are written by
substitute_kernels when a lifted region has no kernel yet (section 5). Everything runs from
`scripts/` in this skill and writes to one capture directory under `outputs/`
(gitignored). Nothing under `torchtitan/` is edited.

Terms:
- **Glue:** any node that is not a GEMM, attention or collective: norms, RoPE, casts,
  routing, permutes, scatters, residual adds.
- **Region:** a set of glue nodes (plus, for T3, one GEMM) that one kernel could run.
  A region repeats per layer; each repetition is an **instance**.
- **Phase:** `forward`, `recompute` (forward ops copied before their backward
  consumers by `selective_activation_remat_pass`) or `backward`.
- **Fusion stage:** number of GEMM/attention/collective nodes on the longest path to
  a node. Glue nodes at the same stage have no such op between them.
- **Ideal time:** `max(external bytes / HBM bandwidth, flops / GEMM peak)`, where
  external bytes are the region's inputs read once and outputs written once.
  **Gap** = (measured - ideal) x instances per step.
- **Bitwise tier:** how hard a bitwise-identical kernel is. T0 pointwise/views/casts;
  T1 reductions (order must match); T2 order-sensitive data movement (scatter,
  index_put, sort, top-k); T3 contains a GEMM or attention.

## Workflow

```
recipe + enable_fusion_finder() --> capture dir: graph_dump/, iteration_N/ trace, setup.json
calibrate_roofline.py ------------> roofline.json
parse_trace -> build_node_table -> find_regions -> curate_regions -> validate_regions
                                                      |
                     regions.json, regions_table.md, fusion_regions/ (one folder per region)
                                                      |
                         fusion_regions/g003_rope/: region.json, lift_pass.py
                                                      |
   lifted run: compile.extra_passes = ["lift_fusion_region_g003", ...] -> same loss, per-region timing
```

Effort on DeepSeek-V3 671B with the fake process group on one GB300: capture
~13 min, analysis ~2 min. A debug model takes under a minute end to end.

## 1. Capture

Write a recipe that starts from the model's verified GraphTrainer recipe, changes
only what one fake-PG GPU needs, and passes it to
`enable_fusion_finder(config, out_dir, pipeline=...)`. Copy `scripts/example_recipe.py`:
the verified DeepSeek-V3 671B MXFP8 Dist-MoE recipe (fused MLA and SwiGLU) on FSDP 256
x EP 64 without PP, one microbatch per step. Its comments explain each change. The
helper:
- selects the pass pipeline, which is the default GraphTrainer pipeline plus graph
  dumps (`graph_dump/{pre_inductor,final}_nodes.jsonl.gz`, one JSON record per node:
  target, op schema, args, users, `val` shape/stride/dtype, stack trace, module FQN,
  recompute tag, `autograd_backward`) and `custom_codegen_pass` last, so each node runs under
  `_RecordFunctionFast("node: op (file:line)")`;
- sets `debug.seed = 42`, `debug.deterministic = True`, profiles step 10 after 3
  warmup steps, turns accelerator graphs off (replay hides the labels), and writes
  `setup.json` with the config deltas. The first graph dump adds the world size. TensorBoard goes to `<out>/tb`, the full-precision baseline for
  lifted runs.

Pipelines:
- `fusion_finder`: the default.
- `fusion_finder_recompute_index_put`: also retags `fresh buffer -> index_put_ ->
  consumers` to recompute. Use it when `memory_policy="full"` runs out of memory on an
  MoE model; there, `tag_sac_policy` saves the MoE combine all_to_all output, which is
  10.5 GiB per layer for DeepSeek-V3 671B.

Run from the repo root on an idle GPU (check `nvidia-smi`; another process's kernels
distort timing). Save the command to `<out>/command.txt` so it lands in
`regions.json`:

```bash
SKILL=torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts
OUT=outputs/fusion_finder/dsv3_671b
PYTHONPATH=$SKILL:. NGPU=256 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 FUSION_FINDER_OUT=$PWD/$OUT \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -m torchtitan.train --module example_recipe --config dsv3_671b_fake --comm-backend fake
```

The fake process group runs rank 0 of a large world on one GPU (see the
`fake_distributed_backend` skill). Collectives become local copies: their kernels
are reported as `fake_comm` and excluded from compute. Dist-MoE's fused ops run with
every EP peer aliasing this rank, so their time follows this rank's routing, not real
traffic; they are reported as `fused_moe` and, like GEMMs and attention, are never
glue.

The capture runs one microbatch per step. Work done once per step (FSDP gradient
packing before reduce-scatter, weight casts after all-gather) is then as frequent as
per-microbatch work, but a production schedule with many microbatches amortizes it.
Rank the two groups separately.

## 2. Calibrate the roofline

```bash
CUDA_VISIBLE_DEVICES=1 python $SKILL/calibrate_roofline.py $OUT --gemm 4096x16384x7168 \
    --gemm 16384x8192x8192 --mxfp8
```

Pass one or more representative GEMM shapes (`MxKxN`). GEMM throughput depends on
operand values. Under the fake process group, weights and activations have very low
entropy and large GEMMs run at the near-zero-operand rate (2.36 PFLOP/s vs 1.81 with
random data on GB300), so the default peak is the zero-operand one, which keeps ideal
times a lower bound. With real data, pass `--gemm-peak random`. `--mxfp8` also
measures MXFP8 GEMMs (1x32 block scales, as `MXFP8Linear` runs them; 4.16 PFLOP/s on
GB300); GEMMs with float8 operands are priced at that peak, and the analysis raises if
the graph has them and the roofline does not.

## 3. Analyze

From the repo root:

```bash
export PYTHONPATH=$SKILL:.
python $SKILL/parse_trace.py $OUT        # kernels -> CUDA launch -> enclosing node label
python $SKILL/build_node_table.py $OUT   # node table, step budget, symbolic sizes
python $SKILL/find_regions.py $OUT       # candidates by kind, grouped into instances
python $SKILL/curate_regions.py $OUT [--annotations notes.json] [--gpu-busy]
python $SKILL/validate_regions.py $OUT   # schema, ranking, coverage: must print VALID
```

What the scripts do:
- **Attribution:** a kernel maps to its CUDA runtime launch (correlation id), then to
  the enclosing node label on the launching thread. Kernels outside the graph
  (optimizer, grad clipping) fall back to their outer CPU op. Attribution must cover
  at least 95% of GPU-busy time.
- **Phase:** a name ending in `_recomputed` or `_recomputed_<n>` is recompute;
  otherwise `autograd_backward` means backward; otherwise forward.
- **Sizes:** unbacked sizes (MoE routed rows, `u512 + ...`) are resolved from the
  runtime `Input Dims` the labels record.
- **Candidates:** glue-to-glue tensor edges are classified as intra-module (L1),
  cross-module (XM), cross-layer (XL) or recompute-to-backward (RB). Edges between
  different fusion stages, edges out of allocations, edges into shape-only consumers
  (`zeros_like`, ...) and edges into FSDP bucket packs are cut. Kinds:
  - L1: components;
  - XM-max: components across modules;
  - XM, XL, RB: pairs of L1 components;
  - T3-epi: an L1 component plus the GEMM whose output it reads;
  - T3-pro: an L1 component plus the GEMMs that read all its outputs.

  Candidates with the same ops, shapes and normalized module FQNs are one region with
  N instances.
- **Curation:** selects candidates with no size threshold, drops duplicates and
  subsumed pairs, and merges variants that have the same ops and shapes. A single
  node that launches several kernels or runs at 2x its roofline or worse becomes a
  "single-op rewrite" region. It also assigns scope tags, tiers, blockers and
  overlaps, and lists uncovered memory-bound node classes under `no_opportunity`.
- **Lift checks**, per instance. A region gets a lift pass only if all hold; otherwise
  `regions_table.md` and its `region.json` give the reason:
  - every node exists in the `pre_inductor` graph, which is the graph extra passes see;
  - no path leaves the region and re-enters it. If one does, `blockers` starts with
    `NOT FUSABLE AS FOUND` and says either "not one kernel" (a GEMM, attention or
    collective runs in between) or "incomplete" (only glue on the path; add those
    nodes);
  - instances do not share nodes. Before these checks, a `getitem` of a tuple produced
    outside the instance becomes an input (the submodule gets the tensor it reads, not
    the tuple), and an in-place write into outside storage that nothing in the
    instance reads (a buffer accumulate such as `tokens_per_expert.add_`) stays
    outside, after the call. The submodule outputs only plain tensors: a tuple-valued
    member takes its `getitem`s along, and a view of external inputs alone that is also
    used outside stays outside;
  - lifting cannot reorder side effects. The call runs inside the instance's own span
    and no other node moves, because the joint graph has in-place ops (`add_` grad
    accumulation, `index_put_`, `copy_`) and out= writers (`split_with_sizes_copy.out`)
    whose order no edge records. So outputs must not be used before the last input
    exists; members may mutate only buffers the region allocates (and nothing outside
    reads earlier); and no other op inside the span may mutate storage a member reads
    (storage resolved through views, in-place results and out= results). This is
    `lift_hazard` in `lift_regions.py`, the same function the pass runs on the live
    graph, applied to the `pre_inductor` dump;
  - a topology matcher can be generated that finds every instance (see section 5).

## 4. Review

The scripts find dataflow-connected candidates. These parts need judgment:

- Read the top of `regions_table.md` and the region's folder: the matcher in
  `lift_pass.py` (its comments name the captured nodes and literals).
  For each region worth a kernel, write its title, the dataflow
  that makes it fusible, and its real blockers into an annotations JSON keyed by the
  region key that `curate_regions.py` prints. Then rerun it with `--annotations`.
- **Re-tier by semantics.** The tier comes from op types. An `index_put` or scatter
  whose indices come from `sort`/`argsort`/`cumsum` is a permutation: no duplicates, so
  it is order-free (T0-like) data movement and a gather can replace it. A
  deterministic `index_put(accumulate=True)` with duplicates and rows wider than 32
  rounds to the destination dtype after every add, in stable-sorted original-position
  order; a bitwise kernel must do the same.
- **Fix `NOT FUSABLE AS FOUND` regions.** Extend "incomplete" ones with the path
  nodes. Split "not one kernel" ones at the GEMM or attention node.
- **Sweep the known blind spots by hand:**
  - horizontal (sibling) fusion of ops with no dataflow edge (independent norms,
    per-layer weight-grad casts, independent reductions);
  - fusion with collectives (packing a gather into the all_to_all send buffer,
    gathering into the final parameter layout);
  - anything outside the graph (optimizer, data loading);
  - save/recompute changes. Recompute copies that do no useful work are a memory-policy
    fix, not a kernel.
- Traps seen on DeepSeek-V3:
  - `nn_module_stack` is sometimes a string;
  - backward accumulation nodes have no module FQN (inferred from their inputs);
  - out= writers (`split_with_sizes_copy.out`) have no FX edge to their readers;
  - the FSDP all-gather pack's `val` is the whole all-gather buffer, not the shard it
    writes;
  - tensors over 2^31 elements make eager split every op into several launches;
  - `_local_scalar_dense` nodes on CPU split sizes are host syncs (GPU idle), not
    kernels.

Write `report.md` in the capture directory with these sections:
1. setup: command, commit, config deltas, GPU and its utilization, roofline numbers;
2. step budget by category, attribution coverage and GPU idle time;
3. the ranked table;
4. details of the regions worth a kernel;
5. no-opportunity time with reasons;
6. observations: what was missing, false leads;
7. reproduction.

## 5. Lift the regions into call_module nodes

`curate_regions.py` writes one folder per region into the `fusion_regions/` package,
named by region id and its innermost module (`g003_rope`, `g013_w13_grouped_gemm`).
For a liftable region, `lift_pass.py` registers `lift_fusion_region_g003` with
`@register_graph_pass`. Importing the package registers every pass, and a recipe enables
regions by listing their passes:

```python
import fusion_regions  # the capture dir must be on PYTHONPATH

config.compile.extra_passes = ["lift_fusion_region_g003", "lift_fusion_region_g005"]
# or a set that does not step on each other, largest gap first:
config.compile.extra_passes = list(fusion_regions.DEFAULT_SELECTION)
```

**Matching is by topology, not node names.** Each `lift_pass.py` holds generated
plain-Python matchers. A matcher starts from an anchor node and walks the region: producers through
`args`, and users (`user_of`) for members that are not ancestors of the anchor. At each
node it checks the op and its literal arguments (`is_op`), and it checks that paths
meeting inside the region, or reading the same external input, meet at the same node:

```python
def match_v0(n0):
    """One instance, anchored at its aten.unsqueeze.default node: (members, external inputs) or None."""
    # unsqueeze_2 (as captured)
    if not is_op(n0, 'aten.unsqueeze.default', [N, 0]):
        return None
    n1 = n0.args[0]
    # transpose (as captured)
    if not is_op(n1, 'aten.transpose.int', [N, 0, 1]):
        return None
    n2 = n1.args[0]
    # cat (as captured)
    if not is_op(n2, 'aten.cat.default', [[N, N], -1]):
        return None
    ...
    # getitem_5 (as captured)
    if not is_op(n13, '_operator.getitem', [N, 1]):
        return None
    if n13.args[0] is not n5:
        return None
    return [n7, n5, n3, n13, ...], [x0, x1]


MATCHERS = [('aten.unsqueeze.default', match_v0)]


@register_graph_pass
def lift_fusion_region_g003(gm, example_inputs=None):
    return lift_matches(gm, REGION_ID, MATCHERS)
```

**Sizes and scalar operands match any value; devices match by type.** The op schema
decides, when the matcher is generated:
- `S`: arguments typed `SymInt`, `SymInt[]` or `SymInt?` (view/reshape/expand sizes,
  split sizes, slice bounds, `topk`'s k, the bucketing ops' `group_size`), and the
  `int[]` `normalized_shape` of `_fused_rms_norm` and its backward;
- `V`: numbers passed to `Tensor` or `Scalar` arguments (`mul(x, 256)`,
  `clamp_min(x, 1e-12)`);
- `D('cuda')`: `Device` arguments, any index of that device type (`device=cuda:3`);
- every other literal must be equal: dims, dtypes, bools, strings, `float` arguments
  such as `eps`, custom-op ints such as the MXFP8 block size, and list lengths.

The comment keeps the captured literals:

```python
    # _fused_rms_norm (as captured: [N, [7168], N, 1e-06])
    if not is_op(n4, 'aten._fused_rms_norm.default', [N, [S], N, 1e-06]):
```

Ranks trace the same graph except for literals that `S` and `D` cover (device index,
the bucketing ops' `rank`, FSDP shard offsets in `slice`), so a matcher generated from
the rank-0 capture lifts the same instances on every rank. A region found in one capture
is also found in other variants and layouts of the model where only shapes or config
constants differ: the 671B norm region lifts in the 16B model (hidden size 2048).
Lifting keeps numerics either way. Kernel caches (substitute_kernels) key kernels by the
actual literals, so another shape gets its own kernel problem. Regions whose structure differs (different routing config, a collective
or bucket layout that depends on the sharding) still have to be found in that capture.

`curate_regions.py` generates a matcher from one instance and runs it on the dumped
graph; if some instance is not found (a structural variant), it generates another
matcher from that instance. The pass's docstring records how many sites the matchers
find in the capture against the instances the finder reported. Small patterns often
find more, e.g. the recompute copy of a forward region is the same subgraph.

`lift_matches` (`scripts/lift_regions.py`) finds non-overlapping sites, runs
`lift_hazard` on each against the live graph (the section 3 side-effect rules), skips
unsafe ones with a warning, and outlines the rest. Each instance becomes one
`call_module`; instances with the same structure share one submodule whose body is the
region's original aten ops:

```python
fusion_g003_i0 = self.fusion_G003(getitem_1093, fusion_g117_i0)  # one per site
```

A pass with nothing left to lift raises `ValueError` (the graph changed, or an
overlapping region was lifted first). Passes run in `extra_passes` order and each takes
the sites still unclaimed, so the order matters when patterns overlap.
`DEFAULT_SELECTION` replays the passes in gap order on the dumped graph and keeps those
that still find a safe site. For a hand-picked plan, `plan_passes.py` does the same
replay in the order given and prints the passes that still lift something:

```bash
python $SKILL/plan_passes.py $OUT lift_fusion_region_g006 lift_fusion_region_g020 ...
```
 The passes run after the other GraphTrainer passes and
before the final Inductor passes, which is the graph the `pre_inductor` dump records.
`scripts/example_recipe.py` (`dsv3_671b_fake_lifted`) shows a full lifted recipe.

Check every lifted run:
- **Numerics:** loss and grad_norm must be bitwise identical to the unlifted run.
  Compare full precision from TensorBoard (`TensorBoardLogger`, as
  `scripts/loss_compare.py` does); stdout shows only 5 digits. To cover every pass,
  group the passes so that each still finds a site after the ones before it, and run
  each group.
- **Timing:** with `enable_fusion_finder` the lifted run is also profiled, and each
  `call_module` is one profiler label (`fusion_g003_i0: fusion_G003`). That gives each
  region's measured time directly, and later the measured time after a kernel replaces
  it.

To plug in kernels, list `substitute_with_kernels_pass`
(`graph_trainer/substitute_kernels.py`) after the plan's lift passes, bound to a
`cache_dir`. On the first run it writes every lifted pattern as a problem,
`pat_<l1>/reference.py` plus one `specialization_<l2>.json` per shape; the `autokernel`
skill optimizes that problem set into `<l2>.py` kernels; later runs replace each
`call_module` whose kernel exists. A miss runs the lifted reference, so every run must
stay bitwise identical to the capture. The fusion loop:

```
capture -> analyze -> plan_passes.py -> lifted run + substitute (writes problems)
   ^                                                  |
   |                                     autokernel (bitwise CuTe DSL kernels)
   |                                                  |
   +--------- lifted run + substitute (hits): bitwise check, profile <-----+
```

Each round, re-run the analysis on the new profile: the fused regions drop out and the
next largest gaps lead the next plan.

## Outputs

| file | content |
|---|---|
| `regions.json` | inventory: setup, step budget, regions, no_opportunity ([schema](references/regions_schema.md)) |
| `regions_table.md` | ranked table with tier and the lift pass (or why there is none) |
| `fusion_regions/<folder>/region.json` | the region's `regions.json` entry plus `lift`: pass name, sites, default selection, or why it is not liftable |
| `fusion_regions/<folder>/lift_pass.py` | liftable regions: generated topology matchers (`MATCHERS`) and the registered pass `lift_fusion_region_g<id>` |
| `fusion_regions/__init__.py` | imports every lift pass; `PASSES` (region -> folder, pass) and `DEFAULT_SELECTION` |
| `analysis/node_table.csv.gz` | one row per node: kernel time, phase, FQN, bytes, flops, ideal, ratio |
| `analysis/candidates.json` | all discovery candidates before curation |
| `analysis/step_budget.json` | GPU busy/idle and time by category |

Done when attribution is at least 95%, `validate_regions.py` prints VALID, `report.md` is written, and a lifted run of the
selected regions gives bitwise identical loss and grad_norm.
