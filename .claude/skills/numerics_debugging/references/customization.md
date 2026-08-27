# Customizing the capture and diff

## 1. Excluded ops (`_EXCLUDED_OPS` in `activation_tracer.py`)

Denylist of op names dropped at capture time as "infrastructure, not
numerics." Includes view/reshape/permutation/indexing/creation ops,
type-casts (`_to_copy`, `to`, `clone`), and `detach`.

- **Add to the list** if a new op is firing that pollutes the diff with
  structurally identical rows (e.g. a fused kernel that wraps things you
  already capture upstream).
- **Remove from the list** if you want visibility into something you
  currently hide. Collective comms (`all_gather_into_tensor`,
  `reduce_scatter_tensor`, `wait_tensor`, etc.) are *not* in the default
  list — they show up in the diff because they often diverge between eager
  and traced.

Op names skipped by the filter that *actually fired* during capture are
surfaced as a chip list per side in the HTML. Use that to discover what
you're dropping.

## 2. Numel / dtype filter (`_should_capture` / `min_numel`)

By default we skip tensors smaller than `min_numel=1000` and tensors that
aren't `{float32, float16, bfloat16}`. Adjustable via the `min_numel` and
`op_filter` kwargs on `DebugModeTracer` (constructed directly; not a CLI
flag).

## 3. Hash function (`log_tensor_hashes`)

The `output_hash` / `input_hashes` fields come from
`DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True)`. `"norm"` =
L1 in float64. Other options DebugMode supports:

- `"hash_tensor"` — `torch.hash_tensor` (XOR-reduce of the underlying bytes;
  truly bit-exact, fast, but loses arithmetic-tolerance semantics).
- A custom callable —
  `lambda t: (t.float().mean().item(), t.float().std().item())` for
  mean+std as a "hash."

To switch, edit the `DebugMode.log_tensor_hashes(hash_fn=..., hash_inputs=True)`
call inside `DebugModeTracer.__enter__`.

## 4. Manual overrides (`--override <csv>`)

The four-pass matcher (override / exact key / fuzzy key / stats) handles
most cross-mode pairings automatically, but some patterns don't match:

- **AC-recompute attribution drift**: eager's selective-activation-checkpoint
  recompute fires modules directly, so ModTracker only sees `feed_forward`
  (no layer prefix), while traced sees the full `layers.N.feed_forward`.
- **Per-layer counter shifts**: an in-place collective op
  (`_allgather_base_`) takes the `op_0` slot in eager's per-layer counter;
  traced doesn't have it under the layer key, so subsequent ops shift by one.
- **Collective renaming**: eager's in-place `_allgather_base_` ↔ traced's
  lowered `all_gather_into_tensor_out` + `wait_tensor` chain.

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

### Common eager-vs-traced override scenarios

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

## 5. HTML appearance

- `--name1` / `--name2` set the run labels in headings and the side column.
- Cell coloring is log-scale on relative diff (`1e-8` → faint pink, `1e-1` →
  full red). Adjust the thresholds in `_hash_cell` and the stat-cell loop
  in `generate_html` if you want a different gradient.
- Input-hash cells are collapsible (3-line max by default). Adjust
  `.ih-cell.collapsed { max-height: 3em }` in the CSS for a different cap.
