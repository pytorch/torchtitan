---
name: numerics_debugging
description: Capture and compare per-op activations between two TorchTitan runs to spot numerics divergence (eager vs aot_fx_trace, FSDP vs no-FSDP, before vs after a refactor). Use when the user wants to debug bitwise / numeric drift in training, or invokes /numerics_debugging.
---

# Numerics Debugging (DebugMode-based)

Per-op activation capture + comparison toolkit. Captures activations on a
designated step via `torch.utils._debug_mode.DebugMode`, then diffs two
captures into an HTML report to surface numerics divergence between runs
that should agree (bitwise, or within float32 reduction-order noise).

Two pieces live in `agent_tooling/numerics_debugging/`:

- `activation_tracer.py` — runtime capture, driven from `Profiler` by
  `ActivationCaptureProfiler`. Output:
  `{dump_folder}/numerics/rank_{N}_activations.log`.
- `compare_numerics.py` — diffs two logs, produces an HTML report.

`agent_tooling/` sits **outside** the torchtitan package on purpose. Nothing
in core torchtitan or `graph_trainer` references it; an agent must edit
torchtitan to wire it in before a capture run, and revert the edits when
done (they don't belong on `main`).

> The two runs being compared **must use the same dtype and seed**. The
> matcher keys on shape + float64 L1 norm; a precision change (bf16 vs
> fp32) makes every row diverge and the matcher degrades to the
> structural-only `stats` pass.

## Workflow

1. **Patch torchtitan** to wire the capture into `Profiler` (and
   `graph_trainer` if you're capturing the traced path). Full patch set:
   [references/patching.md](references/patching.md).
2. **Capture twice**, once per run you want to compare:
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
   The capture step is `profile_freq`. With `profile_freq=2` and
   `training.steps=2`, step 1 warms up and step 2 is the snapshot. Capture
   adds ~10–40% memory only on the capture step (stats are computed inline
   in float64; tensors aren't held).
3. **Diff** the two logs:
   ```bash
   python -m agent_tooling.numerics_debugging.compare_numerics \
       outputs/run_A/numerics/rank_0_activations.log \
       outputs/run_B/numerics/rank_0_activations.log \
       --name1 run_A --name2 run_B \
       -o diff.html
   ```
   Open `diff.html`. Each row pairs one op from each run; cells turn red
   when a stat diverges; the "Match method" chip shows which of the four
   matching passes (override / exact key / fuzzy key / stats) paired the
   row.

## Customizing

Excluded ops, numel / dtype filter, hash function, the manual-override
file format, and HTML appearance are all tunable. See
[references/customization.md](references/customization.md), which also
catalogs the common eager-vs-traced mismatch patterns (AC-recompute FQN
drift, per-layer counter shifts, collective renaming) you'll see in the
diff and how to express them as overrides.
