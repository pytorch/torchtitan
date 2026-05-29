# Constitution — Qwen3-14B pretraining efficiency search

Binding rules for this experiment. The Harness reads, enforces, and exposes this
as `observe().rules`; the Human amends it via `amend_constitution()`. Where this
disagrees with `ARCHITECTURE.md`, this file wins. Format is defined in
`ARCHITECTURE.md` section 4.1.

## Objective

- **Maximize** Harness-measured steady-state **throughput** (tokens/sec) for the
  locked workload.
- **Subject to** the one-sided quality floor: a candidate's model quality must be
  `>= golden_quality - epsilon`. Quality improvement is always allowed; quality
  is never the thing being climbed, only protected.
- We are **degradation-sensitive**: ties and ambiguous quality evidence resolve
  to **reject**.

## Workload (LOCKED)

- **Model flavor:** Qwen3 `14B` via `MODULE=qwen3 CONFIG=qwen3_14b`. Dense, 40
  layers, dim 5120, 40 heads, 8 KV heads, head dim 128, vocab 151936, no weight
  tying. The flavor passed to `model_registry` may not change.
- **Dataset:** real **`c4`** (not `c4_test`), tokenizer at `./tests/assets/tokenizer`.
- **Sequence length:** `training.seq_len = 4096`. **Locked.** Raw throughput
  rewards shorter sequences (attention is O(seq^2)); seq_len defines the workload.
  A `WorkloadViolation` is raised for any candidate that changes it, before any
  run. (Training-seq curricula are a future, eval-judged extension; not enabled.)
- **Hardware / world:** hardware is **not** hardcoded. The harness detects the
  available single-node GPU count at run start, fixes it as the world size for
  the whole run (so throughput stays comparable across candidates), records the
  detected GPU model/count in state, and launches through `./run_train.sh` with
  FSDP over that `fsdp` mesh. Roofline anchors come from the run's printed peak
  FLOPS, not a hardcoded device.

## Quality (LOCKED)

- **Eval (definition of "good"):** held-out **C4** eval at the reference
  `seq_len = 4096`; metric is mean eval cross-entropy loss (lower is better).
  Quality is whatever this reports. May extend to a vector of eval slices later;
  if so, the floor must hold on every component.
- **Golden (the quality bar):** the frozen high-precision reference recipe —
  BF16 params, **no FP8/MXFP8**, SDPA attention, minimal/no compile — at the
  locked workload. Captured once at setup; re-captured only if this constitution
  changes the workload or eval. Never drifts; it is the absolute anchor for both
  the quality floor and numerical-faithfulness checks.
- **epsilon:** **0.5%** relative eval-loss, **one-sided** (a candidate may raise
  eval loss by at most 0.5% over golden). The Harness must set the eval-run
  length so eval-noise `< epsilon/2`; if that is infeasible at acceptable cost,
  surface it rather than loosen `epsilon` silently.
- **Quality-affecting change classes** (require the held-out eval; never take the
  cheap verify-only path):
  - precision / quantization: FP8, MXFP8, `training.dtype`, reduced-precision
    accumulation;
  - optimization: `local_batch_size`, learning rate, `weight_decay`, optimizer
    choice/betas. **`batch_size` and LR must be changed together as one coupled
    candidate** (raising batch without rescaling LR is a known quality regression).
  - plus: any change that fails the verify faithfulness check is reclassified
    quality-affecting automatically (verify routes; the agent does not declare).

## Editable scope

**MAY edit:**
- `torchtitan/models/qwen3/parallelize.py`
- `torchtitan/models/qwen3/sharding.py`
- `torchtitan/models/qwen3/config_registry.py`, only inside `qwen3_14b()` and
  only for non-fixed `Trainer.Config` values
- `model_spec` kwargs `attn_backend`, `moe_comm_backend`, `converters` only
- command-line / config knobs used to launch a candidate
- runtime/environment flags (NCCL, allocator, etc.)

**MAY NOT edit (fixed):**
- `loss`, `hf_assets_path`, `dataloader`, `checkpoint`, `training.seq_len`,
  `metrics.log_freq`, and the model flavor inside `qwen3_14b()`
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
  current champion 5x to fit the throughput noise model; repeat the golden eval
  to fit the eval-quality noise model and set the eval-run length.

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
    "model_flavor": "qwen3_14b",
    "model_registry_flavor": "14B",
    "dataset": "c4",
    "seq_len": 4096,
    "ngpu": "auto",
    "gpu": "auto",
    "launcher": "run_train.sh"
  },
  "quality": {
    "eval": {"dataset": "c4_heldout", "seq_len": 4096, "metric": "eval_loss"},
    "golden": {"dtype": "bfloat16", "quant": "none", "attn": "sdpa", "compile": "minimal"},
    "epsilon_rel": 0.005,
    "one_sided": true,
    "quality_affecting": [
      "precision", "fp8", "mxfp8", "training.dtype",
      "local_batch_size", "lr", "weight_decay", "optimizer"
    ],
    "coupled": [["local_batch_size", "lr"]]
  },
  "editable": {
    "files": [
      "torchtitan/models/qwen3/parallelize.py",
      "torchtitan/models/qwen3/sharding.py"
    ],
    "config_fn": "qwen3_14b",
    "model_spec_kwargs": ["attn_backend", "moe_comm_backend", "converters"],
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
