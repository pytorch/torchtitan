# Numerics debugger (DebugMode-based) — user guide

A per-op activation capture + comparison toolkit for spotting numerics divergence between two training runs (e.g. eager vs `aot_fx_trace`, FSDP vs no-FSDP, before vs after a refactor).  Two runs being compared should use the **same dtype and seed** — the diff tool matches ops by Shape + float64 L1 norm, so a precision change (e.g. bf16 vs fp32) would make every row diverge and the matcher would degrade to the structural-only `stats` pass.  Use it to find *unintended* numeric drift between runs that should agree bitwise (or within float32 reduction-order noise).

Two pieces:

- **`torchtitan/tools/activation_tracer.py`** — captures during training, driven by `--profiler.dump_numerics`.  Output: `{dump_folder}/numerics/rank_{N}_activations.log`.
- **`torchtitan/tools/compare_numerics.py`** — diffs two logs, produces an HTML report.

## Capturing

```bash
./run_train.sh \
    --dump_folder ./outputs/run_A \
    --training.steps 2 \
    --profiler.dump_numerics \
    --profiler.profile_freq 2 \
    --debug.seed 42 \
    --debug.deterministic \
    --training.mixed_precision_param float32
```

Per-step memory overhead during the capture step: ~10–40% extra (it grows with op count, not tensor size — stats are computed inline in float64, tensors aren't held).  Other steps pay no overhead.

The capture step is **`profile_freq`**.  With `profile_freq=2` and `training.steps=2`, the second step is captured (step 1 lets things warm up, step 2 is the snapshot).

## Comparing

```bash
python -m torchtitan.tools.compare_numerics \
    outputs/run_A/numerics/rank_0_activations.log \
    outputs/run_B/numerics/rank_0_activations.log \
    --name1 run_A \
    --name2 run_B \
    -o diff.html
```

Open `diff.html`.  Each row pairs one op from each run; cells turn red when a stat diverges; the "Match method" chip shows which of the four matching passes (override / same op key / fuzzy op key / stats) found the pair.

## What you can customize

### 1. Excluded ops (`_EXCLUDED_OPS` in `activation_tracer.py`)

A denylist of op names that get dropped at capture time as "infrastructure, not numerics."  Includes view/reshape/permutation/indexing/creation ops, type-casts (`_to_copy`, `to`, `clone`), and `detach`.

When to edit:
- **Add to the list** if a new op is firing that pollutes the diff with structurally identical rows (e.g. a fused kernel that wraps things you already capture upstream).
- **Remove from the list** if you want visibility into something you currently hide.  Collective comms (`all_gather_into_tensor`, `reduce_scatter_tensor`, `wait_tensor`, etc.) are *not* in the default list any more — they show up in the diff because they often diverge between eager and traced.

Op names skipped by the filter that *actually fired* during capture get surfaced as a chip list per side in the HTML.  Use that to discover what you're dropping.

### 2. Numel / dtype filter (`_should_capture` / `min_numel`)

By default we skip tensors smaller than `min_numel=1000` and tensors that aren't `{float32, float16, bfloat16}`.  Adjustable via the `min_numel` and `op_filter` kwargs on `DebugModeTracer` (you'd have to construct one directly; not currently a CLI flag).

### 3. Hash function (`log_tensor_hashes`)

The `output_hash` / `input_hashes` fields come from `DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True)`.  `"norm"` = L1 in float64.  Other options DebugMode supports:

- `"hash_tensor"` — `torch.hash_tensor` (XOR-reduce of the underlying bytes; truly bit-exact, fast, but loses arithmetic-tolerance semantics).
- A custom callable — `lambda t: (t.float().mean().item(), t.float().std().item())` if you want mean+std as a "hash."

To switch, edit the `DebugMode.log_tensor_hashes(hash_fn=..., hash_inputs=True)` call inside `DebugModeTracer.__enter__`.

### 4. Manual overrides (`--override <csv>`)

The four-pass matcher (override / exact key / fuzzy key / stats) handles most cross-mode pairings automatically, but a handful of patterns *don't* match:

- **AC-recompute attribution drift**: eager's selective-activation-checkpoint recompute fires modules directly, so ModTracker only sees `feed_forward` (no layer prefix), while traced sees the full `layers.N.feed_forward`.
- **Per-layer counter shifts**: an in-place collective op (`_allgather_base_`) takes the `op_0` slot in eager's per-layer counter; traced doesn't have it under the layer key, so subsequent ops shift by one.
- **Collective renaming**: eager's in-place `_allgather_base_` ↔ traced's lowered `all_gather_into_tensor_out` + `wait_tensor` chain.

Override file format (`# `-prefixed lines are comments):

```
# AC-recompute
feed_forward/op_7_mul, layers.2.feed_forward/op_3_mul

# Counter shift
layers.1/op_1_add, layers.1/op_0_add

# Collective rename
layers.0/op_0__allgather_base_, layers.0.attention_norm/op_1_all_gather_into_tensor_out
```

Each line force-pairs `<run1_key>, <run2_key>` regardless of stat similarity.

### Common Eager-vs-Traced Override Scenarios with Examples

Confirm each candidate with matching `Shape` and matching or near-matching
`output_hash` before adding it.

- **AC recompute FQN drift**: eager may log recompute under a short FQN, while
  traced keeps the full layer FQN.
  `feed_forward/op_7_mul, layers.2.feed_forward/op_3_mul`
- **Per-layer counter shifts**: eager FSDP collectives can consume early
  `op_N` slots, so later same-layer compute has shifted counters.
  `layers.0/op_2_add, layers.0/op_0_add`
- **Repeated attention math**: Qwen-style rotary/GQA ops repeat similar
  `mul` / `neg` / `add` / `bmm` patterns, so fuzzy matching can drift by one
  repeated block.
  `layers.0.attention/op_0_mul, layers.0.attention/op_8_mul`
- **Backward attribution drift**: backward ops can move to `<none>`, a parent
  module, or a generated backward op name.
  `layers.0.attention/op_12_add, <none>/op_66_add`
- **Communication lowering**: eager `_allgather_base_` /
  `_reduce_scatter_base_` often corresponds to traced `wait_tensor` rows under
  the consumer module.
  `layers.0/op_1__allgather_base_, layers.0.attention_norm/op_1_wait_tensor`

### 5. HTML appearance

- `--name1 / --name2` set the run labels in headings and the side column.
- Cell coloring is log-scale on relative diff (`1e-8` → faint pink, `1e-1` → full red).  Adjust the thresholds in `_hash_cell` and the stat-cell loop in `generate_html` if you want a different gradient.
- Input-hash cells are collapsible (3-line max by default).  Adjust `.ih-cell.collapsed { max-height: 3em }` in the CSS for a different cap.

## When numerics differ — ask the agent

The point of this tool is to surface differences fast and let you (or an agent) reason about them.  Useful agent prompts after generating a diff:

- **"Explain why these two ops don't match"** — show the agent a row (key + both side's hashes + producer info) and ask what kind of mismatch it represents (genuine numeric divergence, FQN attribution drift, counter shift, collective renaming, AC recompute, …).  The hash-and-producer columns usually tell the whole story.

- **"Find override candidates for this run"** — point the agent at the two log files and ask it to look for eager rows that should match a traced row with a different key but the same Shape and `output_hash`.  Auto-generating the overrides file is much faster than hand-crafting.

- **"Why is `output_hash` different but L2 norm matches?"** (or vice versa) — usually float-precision artifacts (`output_hash` is float64, downstream stats are float32 or sub-stats sensitive to reduction order).  Or a structural issue like NaN propagation.  Ask the agent to walk through the values.

- **"What's the producer chain for this input?"** — hover the input value in the HTML for the immediate producer; ask the agent to trace backward through the producer keys to build the full data-flow story leading up to a divergence.

- **"Add `<op_name>` to the excluded list"** / **"remove `<op_name>` from the excluded list and re-run"** — fastest way to iterate on filter granularity.

- **"Convert the per-input hash list into a compact table"** — for in-place ops like `_fused_adamw_` (one row, 285 input hashes), an agent can group hashes by param, paint each as a heatmap, etc.

- **"This row is marked diff but the only diff is `Output Min`. Should I worry?"** — the match status only considers Shape + output_hash + input_hashes, so a `diff` here usually means the hashes diverged.  But sub-stats can drift even when the L1 agrees (e.g. a few outlier elements changed sign without changing the L1 sum).  Ask the agent to triangulate which interpretation fits.
