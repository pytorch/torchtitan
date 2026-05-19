# Numerics debugging (DebugMode-based) — agent guide

Per-op activation capture + comparison toolkit for spotting numerics
divergence between two training runs (e.g. eager vs `aot_fx_trace`, FSDP vs
no-FSDP, before vs after a refactor). Two runs being compared should use the
**same dtype and seed** — the diff tool matches ops by Shape + float64 L1
norm, so a precision change (e.g. bf16 vs fp32) would make every row diverge
and the matcher would degrade to the structural-only `stats` pass. Use it to
find *unintended* numeric drift between runs that should agree bitwise (or
within float32 reduction-order noise).

Two pieces:

- `activation_tracer.py` — captures during training (via a profiler hook the
  agent installs into torchtitan, see "Patching torchtitan" below). Output:
  `{dump_folder}/numerics/rank_{N}_activations.log`.
- `compare_numerics.py` — diffs two logs, produces an HTML report.

This folder is intentionally **outside** the torchtitan package. Nothing in
core torchtitan or `graph_trainer` references it; an agent must edit
torchtitan to wire it in before a capture run.

## Patching torchtitan to enable capture

The capture is gated on a profiler flag and a `NumericsDebugger` constructed
inside `Profiler`. Apply these patches before a capture run, then revert
them when you are done (they don't belong on `main`).

### 1. `torchtitan/tools/profiler.py`

Add the import, config field, constructor arg, lifecycle hooks, and builder.

```python
# top of file
from agent_tooling.numerics_debugging.activation_tracer import NumericsDebugger

# inside Profiler.Config — add next to enable_memory_snapshot
dump_numerics: bool = False
"""Dump per-op activation logs for numerics debugging.
Writes ``{dump_folder}/numerics/rank_{rank}_activations.log``
(per-op stats + norm hashes of inputs / outputs)."""

# Profiler.__init__ — add model kwarg and slots
def __init__(
    self,
    config: "Profiler.Config",
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
    model: torch.nn.Module | None = None,  # for numerics debugger
) -> None:
    ...
    self.numerics_debugger = None
    # Numerics debugger registers global module forward hooks on
    # the model so backward ops can recover their owning FQN.
    self._model = model

# Profiler.__enter__ — build the debugger alongside the memory profiler
self.numerics_debugger = self.build_numerics_debugger(
    base_folder=self._base_folder,
)

# Profiler.__exit__ — teardown
if self.numerics_debugger is not None:
    self.numerics_debugger.__exit__(exc_type, exc_val, exc_tb)
    self.numerics_debugger = None

# Profiler.step — drive the capture-step cadence
if self.numerics_debugger is not None:
    self.numerics_debugger.step()

# new method
def build_numerics_debugger(self, *, base_folder: str):
    """Create and return a :class:`NumericsDebugger`, or ``None`` if disabled."""
    cfg = self._config
    if not cfg.dump_numerics or self._model is None:
        return None

    dump_dir = os.path.join(base_folder, "numerics")
    debugger = NumericsDebugger(
        enabled=True,
        model=self._model,
        dump_dir=dump_dir,
        capture_step=cfg.profile_freq,
    )
    debugger.__enter__()
    return debugger
```

### 2. `torchtitan/trainer.py`

Pass the model to the profiler when entering the training loop:

```python
with config.profiler.build(
    global_step=self.step,
    base_folder=config.dump_folder,
    model=self.model_parts[0],  # add this line
) as profiler:
    ...
```

`model_parts[0]` is the eager model rank-0 owns; that is what `NumericsDebugger`
installs forward hooks on so DebugMode's ModTracker can attribute backward ops
to the right FQN.

### 3. graph_trainer only — replay traced graph through FQNInterpreter

When the active path is `--compile.mode aot_fx_trace`, the traced graph is
called as `gm(*flat_inputs)`, which bypasses every `nn.Module.forward`. That
means `DebugMode`'s `ModTracker` can no longer attribute ops to a FQN, and
the log degrades to `<none>/op_N_*` everywhere.

The fix is to walk the graph node-by-node with an FX interpreter that
restores the FQN / stack / phase that the traced commit already stashed in
`node.meta`, so capture gets the same context that eager would.

**3a. `torchtitan/experiments/graph_trainer/debug_utils.py`** — append:

```python
class FQNInterpreter(torch.fx.Interpreter):
    """Interpreter that sets activation tracer context vars from node metadata.

    For each node, reads:
    - ``node.meta["custom"]["module_fqn"]`` → sets _current_module_name
    - ``node.meta["stack_trace"]`` → parsed and set as _current_stack_frames
    - ``node.meta["autograd_backward"]`` → sets _current_phase_override

    This is needed because traced graph replay via ``gm(*inputs)`` bypasses
    module forwards entirely, so DebugMode's ModTracker cannot infer the
    module FQN. FQNInterpreter walks the graph node-by-node, restoring the
    context that eager capture would otherwise get from DebugMode and
    autograd metadata.
    """

    def run_node(self, n: torch.fx.Node):
        from contextvars import Token

        from agent_tooling.numerics_debugging.activation_tracer import (
            _current_module_name,
            _current_phase_override,
            _current_stack_frames,
            _parse_stack_trace,
        )

        fqn = (n.meta.get("custom") or {}).get("module_fqn")
        stack_trace = n.meta.get("stack_trace")
        is_backward = n.meta.get("autograd_backward", False)

        phase = "backward" if is_backward else "forward"

        tokens: list[Token] = []
        if fqn:
            tokens.append(_current_module_name.set(fqn))
        if stack_trace:
            tokens.append(_current_stack_frames.set(_parse_stack_trace(stack_trace)))
        tokens.append(_current_phase_override.set(phase))
        try:
            return super().run_node(n)
        finally:
            for token in reversed(tokens):
                token.var.reset(token)
```

`run_traced` already accepts an `interpreter_cls` kwarg (in
`torchtitan/experiments/graph_trainer/make_fx_tracer.py`), so no patch is
needed there — just pass `FQNInterpreter` through.

**3b. `torchtitan/experiments/graph_trainer/trainer.py`** — pick the
interpreter only when capture is active, and thread it through
`run_traced`:

```python
def _maybe_get_fqn_interpreter(self) -> type | None:
    from agent_tooling.numerics_debugging.activation_tracer import (
        is_numerics_capture_active,
    )

    if is_numerics_capture_active():
        from torchtitan.experiments.graph_trainer.debug_utils import FQNInterpreter

        return FQNInterpreter

    return None

# in forward_backward_step, where run_traced is invoked:
outputs = run_traced(
    ...,
    interpreter_cls=self._maybe_get_fqn_interpreter(),
)
```

The interpreter only kicks in on the capture step (and only under
aot_fx_trace), so steady-state training is untouched.

## Capturing

Once patches 1–2 (and 3 for graph_trainer) are in:

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

Per-step memory overhead during the capture step: ~10–40% extra (it grows
with op count, not tensor size — stats are computed inline in float64,
tensors aren't held). Other steps pay no overhead.

The capture step is `profile_freq`. With `profile_freq=2` and
`training.steps=2`, the second step is captured (step 1 lets things warm
up, step 2 is the snapshot).

## Comparing

```bash
python -m agent_tooling.numerics_debugging.compare_numerics \
    outputs/run_A/numerics/rank_0_activations.log \
    outputs/run_B/numerics/rank_0_activations.log \
    --name1 run_A \
    --name2 run_B \
    -o diff.html
```

Open `diff.html`. Each row pairs one op from each run; cells turn red when a
stat diverges; the "Match method" chip shows which of the four matching
passes (override / same op key / fuzzy op key / stats) found the pair.

## What you can customize

### 1. Excluded ops (`_EXCLUDED_OPS` in `activation_tracer.py`)

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

### 2. Numel / dtype filter (`_should_capture` / `min_numel`)

By default we skip tensors smaller than `min_numel=1000` and tensors that
aren't `{float32, float16, bfloat16}`. Adjustable via the `min_numel` and
`op_filter` kwargs on `DebugModeTracer` (constructed directly; not a CLI
flag).

### 3. Hash function (`log_tensor_hashes`)

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

### 4. Manual overrides (`--override <csv>`)

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

#### Common eager-vs-traced override scenarios

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

- `--name1` / `--name2` set the run labels in headings and the side column.
- Cell coloring is log-scale on relative diff (`1e-8` → faint pink, `1e-1` →
  full red). Adjust the thresholds in `_hash_cell` and the stat-cell loop
  in `generate_html` if you want a different gradient.
- Input-hash cells are collapsible (3-line max by default). Adjust
  `.ih-cell.collapsed { max-height: 3em }` in the CSS for a different cap.

## Triage prompts for an agent looking at a diff

- **"Explain why these two ops don't match"** — show the agent a row (key +
  both sides' hashes + producer info) and ask what kind of mismatch it
  represents (genuine numeric divergence, FQN attribution drift, counter
  shift, collective renaming, AC recompute, …). The hash-and-producer
  columns usually tell the whole story.
- **"Find override candidates for this run"** — point the agent at the two
  log files and ask it to look for eager rows that should match a traced
  row with a different key but the same Shape and `output_hash`.
  Auto-generating the overrides file is much faster than hand-crafting.
- **"Why is `output_hash` different but L2 norm matches?"** (or vice versa)
  — usually float-precision artifacts (`output_hash` is float64,
  downstream stats are float32 or sub-stats sensitive to reduction order).
  Or a structural issue like NaN propagation. Ask the agent to walk through
  the values.
- **"What's the producer chain for this input?"** — hover the input value in
  the HTML for the immediate producer; ask the agent to trace backward
  through the producer keys to build the full data-flow story leading up to
  a divergence.
- **"Add `<op_name>` to the excluded list"** / **"remove `<op_name>` from
  the excluded list and re-run"** — fastest way to iterate on filter
  granularity.
- **"Convert the per-input hash list into a compact table"** — for in-place
  ops like `_fused_adamw_` (one row, 285 input hashes), an agent can group
  hashes by param, paint each as a heatmap, etc.
- **"This row is marked diff but the only diff is `Output Min`. Should I
  worry?"** — the match status only considers Shape + output_hash +
  input_hashes, so a `diff` here usually means the hashes diverged. But
  sub-stats can drift even when the L1 agrees (e.g. a few outlier elements
  changed sign without changing the L1 sum). Ask the agent to triangulate
  which interpretation fits.
