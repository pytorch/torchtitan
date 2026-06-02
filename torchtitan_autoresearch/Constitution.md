# Constitution — Llama3-8B pretraining efficiency search

Binding rules for this experiment. The Harness reads, enforces, and exposes this
as `observe().rules`; the Human amends it via `amend_constitution()`. Where this
disagrees with `ARCHITECTURE.md`, this file wins. Format is defined in
`ARCHITECTURE.md` section 4.1.

## Objective

- **Maximize** Harness-measured steady-state **throughput** (tokens/sec) for the
  locked workload.
- **Subject to** a quality guarantee. **v1 policy: faithfulness-only** — a
  candidate is kept only if it is numerically faithful to the golden (its short
  deterministic loss+grad_norm trajectory stays within the golden's own rounding
  noise, non-trending). Changes that move the math are **rejected**; there is no
  held-out eval in v1, so quality is preserved by construction.
- We are **degradation-sensitive**: ties and ambiguous faithfulness resolve to
  **reject**.

## Workload (LOCKED)

- **Model flavor:** Llama 3.1 `8B` via `MODULE=llama3 CONFIG=llama3_8b`. Dense.
  The flavor passed to `model_registry` may not change.
- **Dataset:** real **`c4`**, Llama-3.1-8B tokenizer at `./assets/hf/Llama-3.1-8B`.
- **Sequence length:** `training.seq_len = 4096`. **Locked** and pinned by the
  Harness (note `llama3_8b` defaults to 8192). Raw throughput rewards shorter
  sequences (attention is O(seq^2)); seq_len defines the workload. A
  `WorkloadViolation` is raised for any candidate that changes it, before any run.
- **Hardware / world:** hardware is **not** hardcoded. The harness detects the
  available single-node GPU count at run start, fixes it as the world size for
  the whole run (so throughput stays comparable across candidates), records the
  detected GPU model/count in state, and launches through `./run_train.sh` with
  FSDP over that `fsdp` mesh. Roofline anchors come from the run's printed peak
  FLOPS, not a hardcoded device.

## Quality (LOCKED) — v1: faithfulness-only

- **Golden (the anchor):** the frozen baseline recipe — mainline `llama3_8b` at
  the locked workload, default precision (float32 master weights + bf16 compute),
  SDPA attention, no compile. Captured once at setup; re-captured only if this
  constitution changes the workload. It is the absolute anchor for the
  faithfulness check.
- **Faithfulness IS the quality gate.** A candidate is kept only if its short
  seed-pinned **deterministic** loss AND grad_norm trajectory matches the golden's:
  - **magnitude:** no per-step relative deviation exceeds the band — the golden's
    own deterministic-vs-nondeterministic rounding jitter x `band_headroom`, floored;
  - **non-trending:** the signed mean deviation stays within `trend_factor` x band,
    so a small systematic bias is caught even when each step is within the band.
- **Affecting changes are REJECTED.** Any candidate that fails faithfulness — i.e.
  moves the math beyond the golden's rounding noise (precision / FP8 / MXFP8 /
  `training.dtype`; `local_batch_size` / LR / `weight_decay` / optimizer) — is
  rejected. v1 runs no held-out eval and has no quality floor; a held-out-eval
  track for affecting changes is a future extension.
- **Routing is by measurement, not declaration:** the verify check decides
  faithful vs affecting; the agent never declares a change's class.

## Editable scope

**MAY edit:**
- `torchtitan/models/llama3/parallelize.py`
- `torchtitan/models/llama3/sharding.py`
- `torchtitan/models/llama3/config_registry.py`, only inside `llama3_8b()` and
  only for non-fixed `Trainer.Config` values
- `model_spec` kwargs `attn_backend`, `converters` only
- command-line / config knobs used to launch a candidate
- runtime/environment flags (NCCL, allocator, etc.)

**MAY NOT edit (fixed):**
- `loss`, `hf_assets_path`, `dataloader`, `checkpoint`, `training.seq_len`,
  `metrics.log_freq`, and the model flavor inside `llama3_8b()`
- anything under `torchtitan_autoresearch/` (the Harness)
- `torchtitan/distributed/`, `torchtitan/protocols/`, `torchtitan/models/common/`
- trainer, metrics, data loader, loss, or evaluation code
- dependency files or environment setup

## Measurement protocol (LOCKED)

- `metrics.log_freq = 1` is pinned by the Harness; it is not a search knob.
- **Screening:** `training.steps = 10`, steady-state window = steps 2-10.
- **Contender validation:** `training.steps = 20`, steady-state window = steps
  11-20. Only candidates that beat the champion at screening are validated.
- Throughput is the Harness-computed steady-state **mean over the window** (with
  std/cv), never the last printed `tps`. A run with any other step count, or not
  measured through the Harness, is invalid for comparison.

## Significance policy

- Throughput noise is **heavy-tailed**. Promotion uses a tail-aware test: a
  single-sample improvement must clear the measured **tail quantile**, not merely
  2-sigma.
- Verdicts: `promote` (clears the bar) / `rerun` (promising but in the
  `[1-sigma, tail)` band -> add exactly one sample, then re-decide) / `reject`
  (within noise). Do not rerun by habit.
- **Noise calibration (at setup, and after any large regime change):** repeat the
  current champion to fit the throughput noise model; run the golden deterministic
  vs same-seed non-deterministic to size the per-axis faithfulness bands (loss,
  grad_norm).

## Substrate policy

- **Family time-boxing:** classify every failure (substrate/toolchain vs logic).
  Auto-defer an idea family after **3** consecutive substrate-class failures of
  the same class (**4** for non-substrate failures). Deferred families are
  surfaced in `observe().deferred_families`.
- **Timeouts:** each run uses a timeout appropriate to the command; a hang is
  killed, logged as a crash, and classified.
- **Warm start:** a persistent `TORCHINDUCTOR_CACHE_DIR` is a standing
  environment setting for the whole experiment (not a per-candidate knob).

## Machine-readable rules

The Harness reads the fenced block below as the enforced source of truth; the
prose above is the human explanation of the same rules. Keep them in sync.

```json
{
  "objective": {"maximize": "throughput", "quality_floor": true},
  "workload": {
    "model_flavor": "llama3_8b",
    "config_fn": "llama3_8b",
    "model_registry_flavor": "8B",
    "module": "llama3",
    "dataset": "c4",
    "seq_len": 4096,
    "ngpu": "auto",
    "gpu": "auto",
    "launcher": "run_train.sh"
  },
  "quality": {
    "policy": "faithfulness_only",
    "faithful": {"verify_steps": 8, "band_headroom": 3.0, "trend_factor": 0.5},
    "affecting_action": "reject"
  },
  "editable": {
    "files": [
      "torchtitan/models/llama3/parallelize.py",
      "torchtitan/models/llama3/sharding.py"
    ],
    "config_fn": "llama3_8b",
    "model_spec_kwargs": ["attn_backend", "converters"],
    "allow_cli_knobs": true,
    "allow_env_flags": true
  },
  "fixed_fields": [
    "loss", "hf_assets_path", "dataloader", "checkpoint",
    "training.seq_len", "metrics.log_freq", "model_flavor"
  ],
  "locked_paths": [
    "torchtitan_autoresearch/",
    "torchtitan/distributed/",
    "torchtitan/protocols/",
    "torchtitan/models/common/"
  ],
  "provenance": {
    "base_commit": "HEAD",
    "branch_pattern": "autoresearch/{tag}",
    "allow_resume": false
  },
  "banned_workload_fields": ["seq_len", "dataset", "model_flavor"],
  "measurement": {
    "log_freq": 1,
    "screen": {"steps": 10, "window": [2, 10]},
    "validate": {"steps": 20, "window": [11, 20]}
  },
  "significance": {"z": 2.0, "rerun_band": 1.0, "tail_quantile": 0.9, "calibration_runs": 5},
  "substrate": {"family_defer_substrate": 3, "family_defer_other": 4}
}
```
