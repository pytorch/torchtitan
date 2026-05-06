# Open Discussion Topics

These are open questions / ideas that came up during review of the metrics
PR. Items get removed once they're either resolved in the code or
deferred to a separate PR with an explicit anchor (TODO, follow-up
issue).

---

## 1. The `stats_` prefix → resolved as `_` prefix

**Status:** changed.

`Stats` previously emitted `key/stats_mean`, `key/stats_max`, … Now it
emits `key/_mean`, `key/_max`, `key/_min`, `key/_std`, `key/_sum`. The
leading underscore still keeps Stats outputs from colliding with
`Mean`/`Max`/`Min`/`Std` under the same metric key (`Mean(v)` →
`len/mean`; `Stats(v)` → `len/_mean`) and reads cleaner than
`stats_mean`.

This is open to revisit: alternative stylings considered were
`len.stats/mean` (extra namespace), `len_mean` (drop slash, join with
underscore). If we ever want to drop the underscore prefix entirely
we'd need a separator-per-reduction model (Mean joins with `/`, Stats
joins with `_`); deferred.

---

## 2. `MetricsConfig.log_freq` is exposed but not honored

**Status:** documented + warned.

The field stays on the surface so users / future code can rely on it,
but the controller still logs every step. `MetricLogger.build` now
emits a one-time warning when `log_freq != 1` so users aren't silently
surprised, and the field's docstring carries
`TODO(rl-metrics-log-freq)`.

The trainer-side cost angle is the actually interesting part: the
all_reduce inside `forward_backward` runs every step regardless of
whether anyone reads the result. If logging slows down by enough to
matter, we should honor `log_freq` on the actor side (skip the SUM
pack when `step % log_freq != 0`).

Until someone files a "logging is too noisy / slow" report, leave it.

---

## 3. Static / sticky console output → implemented

**Status:** done in `ConsoleMetricLogger`.

`ConsoleMetricLogger` now keeps a sticky column list. On the first
`log()` it expands the configured `allow_list` regexes against the
incoming metrics dict (in pattern order). New keys seen on later
steps append to the end (so validation/* keys land at the right of
the line on the validation step instead of shuffling everything).
Missing keys render as `--`. Color cycles by column position; the
`step:` prefix is red. TTY auto-detect: piped output (CI, redirects)
gets plain text via `NoColor`.

Open question: do we want to expose a separate
`MetricsConfig.console_static_keys: list[str] | None` that takes
literal keys (no regex) for total order control? The current pattern
expansion derives keys lazily from each step's metrics, which means
the column count can grow as new keys appear. For the headline
allow-list (mostly `^...$` exact-anchored regexes) this is fine. For
broader patterns it could surprise.

---

## 4. Test coverage gaps flagged by the second-opinion review

| Their flag | Status | Action |
| --- | --- | --- |
| `validate()` produces `validation/reward`, `…/response_length`, `…/num_samples`, reward components | **gap** | Add a test that mocks the generator. ~40 lines. Worth doing in this PR. |
| `Completion.finish_reason` plumbing | **gap** | Now exercised end-to-end via `rollout/truncation_rate` (item 5 below); a focused unit test on `Completion.finish_reason` storage is still useful. |
| `PolicyTrainer.optim_step()` returns `train/grad_norm/mean`, `train/lr`, `train/policy_version/mean` | **hard to test cheaply** | Constructing `PolicyTrainer` requires Monarch + a real model. Either mock the optimizer container + scheduler, or accept this is GPU-integration territory. |
| `reduce_verification_metrics()` distributed path: SUM lane + MAX lane + inverted bool across ranks | **gap** | `test_loss_reducer_simulated_two_dp` doubles a tensor uniformly; doesn't actually test unequal-token-count math. Add a per-rank dispatch mock. |

**Recommendation:** the `validate()` and unequal-DP reducer tests are
both ~40 lines and would meaningfully strengthen the coverage.
Optim_step is GPU-only — skip until we have a richer integration
harness.

---

## 5. `rollout/truncation_rate` → implemented

**Status:** added.

`_collect_rollouts` now emits
`rollout/truncation_rate = Mean.from_list([c.finish_reason == "length"
for c in completions])`. A test pins it (with a mix of `stop`/`length`
finish reasons) at `tests/test_grpo_metrics.py`.

The richer `rollout/finish_reason/<reason>/frac` breakdown
(separate Means for `stop`/`length`/`abort`) is not implemented —
truncation is the actionable signal (high → `max_tokens` too low).
Add the others if anyone asks.

---

## 7. Console output: per-call allow_list inside `MetricLogger.log`

**Status:** sketched, not implemented.

Today one `ConsoleMetricLogger` accumulates the union of train + val
keys, so train rows carry `--` for val keys and vice versa.

Make the console output a per-call argument to `MetricLogger.log` and
move the rendering into a free function. Aggregation still runs once
per call; W&B and other backends are unaffected.

```python
@dataclass(kw_only=True, slots=True)
class MetricsConfig:
    log_freq: int = 1
    train_console_allow_list: list[str] | None = None        # was console_allow_list
    validation_console_allow_list: list[str] | None = None   # new
    enable_wandb: bool = False
```

```python
class MetricLogger:
    def log(self, step, metrics, *, console_allow_list=None):
        reduced = aggregate_metrics(metrics)
        if console_allow_list is not None:
            log_to_console(step, reduced, allow_list=console_allow_list)
        for backend in self._backends:           # wandb, ...
            try: backend.log(reduced, step)
            except Exception: ...
```

```python
def log_to_console(step, aggregated, *, allow_list, prefix=""):
    """Render ``step: N | k1: v1 | ...``. Patterns expand against this
    step's keys, in pattern order. Empty match → no print.
    """
```

Controller — single callsite per step, allow_list per context:

```python
self.metric_logger.log(0, pre_metrics,
                       console_allow_list=cfg.metrics.validation_console_allow_list)
self.metric_logger.log(step, step_metrics,
                       console_allow_list=cfg.metrics.train_console_allow_list)
self.metric_logger.log(num_steps, post_metrics,
                       console_allow_list=cfg.metrics.validation_console_allow_list)
```

Each call's console line is built only from the keys actually present
this step; no `--` columns. **`ConsoleMetricLogger` is deleted** — the
backend interface for console renders is replaced entirely by the
stateless `log_to_console` function called inside `MetricLogger.log`.
No sticky state, no class to keep in sync; the column order is
deterministic per context from the `allow_list` pattern order.

---

## 6. `reward/zero_std_frac` (was `degenerate_group_fraction`) → renamed

**Status:** renamed.

`reward/zero_std_frac` matches slime's name and is exact: the fraction
of GRPO groups whose rollouts all got the same reward (group std =
zero, so the GRPO advantage is zero for every member). High values
mean the prompt batch is too easy or too hard for the current policy.
