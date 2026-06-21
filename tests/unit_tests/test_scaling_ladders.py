# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the scaling_ladders experiment.

Covers parameter accounting, the OLMo-ported policy math, batch rounding,
the WSD-S LR curve (cross-checked against an inline copy of OLMo-core's
``WSDS.get_lr``), checkpoint/validator step gating, the ConfigManager default
recipe path, the plan/compare JSON contracts, and run-dir identity.
"""

import itertools
import math
from dataclasses import replace

import pytest

import torch

from torchtitan.config import ConfigManager
from torchtitan.config.configs import ParallelismConfig
from torchtitan.experiments.scaling_ladders import LADDER, RUNGS, WSDSChinchillaPolicy
from torchtitan.experiments.scaling_ladders.checkpoint import LadderCheckpointManager
from torchtitan.experiments.scaling_ladders.ladder import default_ladder
from torchtitan.experiments.scaling_ladders.lr_scheduler import (
    _wsds_multiplier,
    WSDSScheduler,
)
from torchtitan.experiments.scaling_ladders.model import (
    _build_rung_config,
    count_ladder_params,
)
from torchtitan.experiments.scaling_ladders.planner import ComputeSpec, resolve
from torchtitan.experiments.scaling_ladders.validate import LadderValidator

_DENSE_RUNGS = [r for r in RUNGS if r != "debug"]


# ----------------------------------------------------------------------------
# OLMo-core reference, copied verbatim from the cited sources (no olmo_core dep):
# scheduler.py WSDS.get_lr + _linear_warmup/_linear_decay,
# wsds_chinchilla_run_configurator.py formulas. Used only to cross-check.
# ----------------------------------------------------------------------------
def _olmo_target_batch(N):
    return round(2048 * 160 * (N / 108_000_000) ** (2 / 3))


def _olmo_peak_lr(N, mult=1.0):
    return 0.0047 * (N / 108_000_000) ** (-1 / 3) / 2.0 * mult


def _olmo_periods(cm):
    return [2**p for p in range(-1, int(math.log(cm, 2)) + 1)]


def _olmo_wsds_get_lr(
    initial_lr, current, warmup_steps, period_lengths, decay_fraction
):
    adjusted = [period_lengths[0] - warmup_steps] + period_lengths[1:]
    cum = list(itertools.accumulate(adjusted))
    if current < warmup_steps:
        return initial_lr * min(current, warmup_steps) / warmup_steps
    ac = current - warmup_steps
    if ac >= cum[-1]:
        return 0.0
    pidx = next(i for i, e in enumerate(cum) if ac <= e)
    start = 0 if pidx == 0 else cum[pidx - 1]
    length = adjusted[pidx]
    pos = min(max(ac - start, 0), length)
    decay = int(round(decay_fraction * period_lengths[pidx]))
    stable = length - decay
    if pos < stable:
        return initial_lr
    t = pos - stable
    return initial_lr * min(decay - t, decay) / decay


# ----------------------------------------------------------------------------
# Parameter accounting
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("rung", _DENSE_RUNGS)
def test_ladder_params_within_tolerance(rung):
    """ladder_params excludes the input embedding and is within 5% of nominal."""
    N = count_ladder_params(rung)
    shape = RUNGS[rung]
    assert abs(N - shape.nominal_params) / shape.nominal_params <= 0.05
    # ladder_params == total - vocab * dim (OLMo num_non_embedding_params).
    config = _build_rung_config(shape, "flex")
    with torch.device("meta"):
        model = config.build()
    total = sum(p.numel() for p in model.parameters())
    assert N == total - shape.vocab_size * shape.dim


def test_tied_embeddings_not_double_counted():
    """Tying lm_head to the embedding counts the shared matrix once."""
    from dataclasses import replace

    shape = RUNGS["debug"]
    untied = _build_rung_config(shape, "flex")
    tied = replace(untied, enable_weight_tying=True)
    with torch.device("meta"):
        untied_total = sum(p.numel() for p in untied.build().parameters())
        tied_total = sum(p.numel() for p in tied.build().parameters())
    # Tying drops one vocab*dim matrix (the lm_head) from the count.
    assert untied_total - tied_total == shape.vocab_size * shape.dim


# ----------------------------------------------------------------------------
# Policy math (cross-checked against OLMo-core)
# ----------------------------------------------------------------------------
def test_policy_matches_olmo():
    policy = WSDSChinchillaPolicy()
    N = count_ladder_params("100M")
    assert policy.target_token_batch(N) == _olmo_target_batch(N)
    assert policy.train_tokens(N) == int(4 * 20 * N)
    assert policy.warmup_tokens(N) == N
    assert policy.peak_lr(N) == pytest.approx(_olmo_peak_lr(N))
    assert policy.chinchilla_periods() == [float(p) for p in _olmo_periods(4)]
    assert policy.beta2(600_000) == 0.95
    assert policy.beta2(500_000) == 0.99


def test_lr_multiplier_scales_peak():
    N = count_ladder_params("100M")
    base = WSDSChinchillaPolicy().peak_lr(N)
    scaled = WSDSChinchillaPolicy(lr_multiplier=2.0).peak_lr(N)
    assert scaled == pytest.approx(2.0 * base)


def test_policy_validations():
    with pytest.raises(ValueError):
        WSDSChinchillaPolicy(chinchilla_multiple=3)  # not a power of 2
    with pytest.raises(ValueError):
        WSDSChinchillaPolicy(decay_fraction=0.5)  # not < 0.5


# ----------------------------------------------------------------------------
# Planner: batch rounding, duration, period/checkpoint alignment
# ----------------------------------------------------------------------------
def _plan(rung="100M", **policy_kwargs):
    policy = WSDSChinchillaPolicy(**policy_kwargs)
    compute = ComputeSpec(
        world_size=8, parallelism=ParallelismConfig(), local_batch_size=1
    )
    return resolve(rung, policy, compute, seed=0)


def test_batch_rounding_is_valid():
    plan = _plan("100M")
    unit = 1 * plan.dp_degree  # local_batch_size * dp_degree
    assert plan.global_batch_size % unit == 0
    assert plan.global_batch_size >= 1
    assert plan.actual_token_batch == plan.global_batch_size * plan.seq_len
    assert plan.grad_accum_steps == plan.global_batch_size // unit


def test_duration_and_periods_consistent():
    plan = _plan("100M")
    # steps equals the sum of incremental period lengths (and the last
    # cumulative period end), so scheduler and checkpoints stay aligned.
    assert sum(plan.period_lengths) == plan.steps
    assert plan.post_decay_steps[-1] == plan.steps
    assert plan.post_decay_steps == [
        sum(plan.period_lengths[: i + 1]) for i in range(len(plan.period_lengths))
    ]


def test_checkpoint_steps_align_with_decay_boundary():
    """Pre-decay step is the last full-peak step; LR drops one step later."""
    plan = _plan("100M")
    warmup = plan.warmup_steps
    adjusted = [plan.period_lengths[0] - warmup] + plan.period_lengths[1:]
    cum = list(itertools.accumulate(adjusted))
    decays = [round(plan.decay_fraction * p) for p in plan.period_lengths]

    def mult(step):
        return _wsds_multiplier(
            step - 1,
            warmup_steps=warmup,
            adjusted_period_lengths=adjusted,
            cum_period_end=cum,
            decays=decays,
            period_lr_multipliers=None,
        )

    for pre in plan.pre_decay_steps:
        assert mult(pre) == pytest.approx(1.0)  # still at peak
        assert mult(pre + 1) < 1.0  # decay has begun
    for post in plan.post_decay_steps:
        assert mult(post) == pytest.approx(0.0)  # bottom of the decay


# ----------------------------------------------------------------------------
# WSD-S scheduler: full curve equals OLMo-core's get_lr
# ----------------------------------------------------------------------------
def test_wsds_curve_matches_olmo():
    plan = _plan("100M")
    warmup = plan.warmup_steps
    adjusted = [plan.period_lengths[0] - warmup] + plan.period_lengths[1:]
    cum = list(itertools.accumulate(adjusted))
    decays = [round(plan.decay_fraction * p) for p in plan.period_lengths]
    for step in range(1, plan.steps + 2):
        mine = _wsds_multiplier(
            step - 1,
            warmup_steps=warmup,
            adjusted_period_lengths=adjusted,
            cum_period_end=cum,
            decays=decays,
            period_lr_multipliers=None,
        )
        ref = _olmo_wsds_get_lr(
            1.0, step, warmup, plan.period_lengths, plan.decay_fraction
        )
        assert mine == pytest.approx(ref, abs=1e-9), f"step {step}"


def test_wsds_scheduler_build_drives_lambda():
    """build() wires a real LambdaLR whose curve matches _wsds_multiplier."""
    import torch.nn as nn

    from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig

    plan = _plan("100M")
    base_lr = 0.01
    optimizers = OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*", optimizer_name="AdamW", optimizer_kwargs={"lr": base_lr}
            )
        ],
    ).build(model_parts=[nn.Linear(4, 4)])
    scheduler = WSDSScheduler.Config(
        warmup_steps=plan.warmup_steps,
        period_lengths=plan.period_lengths,
        decay_fraction=plan.decay_fraction,
    ).build(optimizers=optimizers, training_steps=plan.steps)
    assert isinstance(scheduler, WSDSScheduler)

    adjusted = [plan.period_lengths[0] - plan.warmup_steps] + plan.period_lengths[1:]
    cum = list(itertools.accumulate(adjusted))
    decays = [round(plan.decay_fraction * p) for p in plan.period_lengths]

    def expected(last_epoch):
        mult = _wsds_multiplier(
            last_epoch,
            warmup_steps=plan.warmup_steps,
            adjusted_period_lengths=adjusted,
            cum_period_end=cum,
            decays=decays,
            period_lr_multipliers=None,
        )
        return base_lr * mult

    assert scheduler.schedulers[0].get_last_lr()[0] == pytest.approx(expected(0))
    scheduler.step()
    assert scheduler.schedulers[0].get_last_lr()[0] == pytest.approx(expected(1))


def test_wsds_config_requires_periods():
    with pytest.raises(ValueError):
        WSDSScheduler.Config(period_lengths=[]).build(optimizers=None, training_steps=1)


# ----------------------------------------------------------------------------
# Checkpoint and validator step gating (logic only, no distributed init)
# ----------------------------------------------------------------------------
def test_checkpoint_fires_only_on_explicit_steps():
    mgr = LadderCheckpointManager.__new__(LadderCheckpointManager)
    mgr.enable = True
    mgr.load_only = False
    mgr.checkpoint_steps = {100, 200}
    # Attributes touched by the inherited __del__ -> close() at GC time.
    mgr.purge_thread = None
    mgr.stager = None
    assert mgr._should_save(100)
    assert mgr._should_save(200)
    assert not mgr._should_save(500)  # a base-interval multiple does NOT fire
    assert not mgr._should_save(150)
    assert mgr._should_save(123, last_step=True)  # final step always saves


def test_validator_fires_on_fixed_steps():
    val = LadderValidator.__new__(LadderValidator)
    val.config = LadderValidator.Config(fixed_steps=[3365, 6729])
    assert val.should_validate(1)  # baseline
    assert val.should_validate(3365)
    assert val.should_validate(6729)
    assert not val.should_validate(5000)


# ----------------------------------------------------------------------------
# ConfigManager default-recipe path (exercises tyro)
# ----------------------------------------------------------------------------
def test_config_manager_loads_ladder_recipe():
    config = ConfigManager().parse_args(
        ["--module", "scaling_ladders", "--config", "llama3_ladder_100m"]
    )
    assert config.model_spec.name == "scaling_ladders/llama3"
    assert config.model_spec.flavor == "100M"
    # The WSD-S subclass configs survive tyro's parse (subclass defaults kept).
    assert isinstance(config.lr_scheduler, WSDSScheduler.Config)
    assert isinstance(config.checkpoint, LadderCheckpointManager.Config)
    assert isinstance(config.validator, LadderValidator.Config)
    assert config.checkpoint.checkpoint_steps


# ----------------------------------------------------------------------------
# Ladder API: plan/run_dir/compare contracts
# ----------------------------------------------------------------------------
def test_plan_dict_schema():
    plan = LADDER.plan("100M")
    for key in (
        "rung",
        "ladder_params",
        "dp_degree",
        "global_batch_size",
        "actual_token_batch",
        "grad_accum_steps",
        "steps",
        "warmup_steps",
        "peak_lr",
        "beta2",
        "checkpoint_steps",
        "pre_decay_steps",
        "post_decay_steps",
        "seed",
    ):
        assert key in plan


def test_run_dir_identity_is_unique():
    base = LADDER.run_dir("100M")
    wd = LADDER.run_dir("100M", weight_decay=0.05)
    seeded = LADDER.run_dir("100M", weight_decay=0.05, seed=1)
    other = LADDER.run_dir("190M")
    assert len({base, wd, seeded, other}) == 4
    assert base.endswith("100M/seed0")
    assert seeded.endswith("100M/wd0.05_seed1")


def test_unknown_override_raises():
    with pytest.raises(ValueError, match="Unknown override"):
        LADDER.plan("100M", not_a_knob=1.0)


def test_sweep_expands_grid():
    specs = LADDER.sweep(["60M", "100M"], {"weight_decay": [0.05, 0.1, 0.2]})
    assert len(specs) == 6
    assert {s["overrides"]["weight_decay"] for s in specs} == {0.05, 0.1, 0.2}
    assert {s["rung"] for s in specs} == {"60M", "100M"}


def test_compare_ranks_by_metric():
    ladder = default_ladder()
    synthetic = {
        0.05: 2.90,
        0.1: 2.81,
        0.2: 2.95,
    }

    def fake_metrics(rung, **overrides):
        wd = overrides["weight_decay"]
        return {
            "rung": rung,
            "ladder_params": 1,
            "checkpoints": [
                {
                    "chinchilla_multiple": 1.0,
                    "phase": "post-decay",
                    "val_loss": synthetic[wd],
                }
            ],
        }

    ladder.metrics = fake_metrics
    runs = ladder.sweep(["100M"], {"weight_decay": [0.05, 0.1, 0.2]})
    result = ladder.compare(runs, "val_loss", 1.0)
    assert result["argmin"]["overrides"]["weight_decay"] == 0.1
    assert [r["overrides"]["weight_decay"] for r in result["ranked"]] == [
        0.1,
        0.05,
        0.2,
    ]


# ----------------------------------------------------------------------------
# Showcase: loss-vs-compute extrapolation (Stage 1)
# ----------------------------------------------------------------------------
def test_fit_loss_vs_compute_recovers_curve():
    from torchtitan.experiments.scaling_ladders.showcase import (
        fit_loss_vs_compute,
        predict_loss,
    )

    E, A, alpha = 2.0, 1e6, 0.3
    computes = [10.0**k for k in range(15, 22)]
    losses = [E + A * c ** (-alpha) for c in computes]
    fit = fit_loss_vs_compute(computes, losses)
    assert fit["rmse"] < 1e-6
    assert fit["E"] == pytest.approx(E, rel=1e-3)
    # Extrapolating beyond the fit range recovers the true curve.
    held = 10.0**22
    assert predict_loss(fit, held) == pytest.approx(E + A * held ** (-alpha), rel=1e-3)


def test_extrapolate_predicts_held_out():
    """extrapolate() fits on small rungs and predicts a held-out rung's loss."""
    from torchtitan.experiments.scaling_ladders import showcase

    E, A, alpha = 2.0, 1e6, 0.3
    params = {
        "60M": 61e6,
        "100M": 99e6,
        "190M": 193e6,
        "370M": 366e6,
        "760M": 763e6,
    }

    class _FakePlan:
        def __init__(self, N):
            self.ladder_params = int(N)
            self.steps = 100

    class _FakeLadder:
        def _resolve(self, rung, overrides):
            return _FakePlan(params[rung])

        def metrics(self, rung, **overrides):
            N = int(params[rung])
            records = []
            for xc in (0.5, 1.0):
                tokens = int(20 * N * xc)
                compute = 6 * N * tokens
                records.append(
                    {
                        "phase": "post-decay",
                        "chinchilla_multiple": xc,
                        "tokens": tokens,
                        "val_loss": E + A * compute ** (-alpha),
                    }
                )
            return {"checkpoints": records}

    result = showcase.extrapolate(
        _FakeLadder(),
        fit_rungs=["60M", "100M", "190M", "370M"],
        held_out_rungs=["760M"],
        execute=False,
    )
    assert len(result["fit_points"]) == 8  # 4 rungs x 2 post-decay checkpoints
    assert result["fit"]["rmse"] < 1e-3
    assert len(result["validations"]) == 2  # 760M at 0.5xC and 1xC
    for validation in result["validations"]:
        assert validation["relative_error"] < 1e-3


def test_compare_variants_pairs_matched_points():
    """compare_variants pairs baseline vs variant at matched (rung, xC)."""
    from torchtitan.experiments.scaling_ladders import showcase

    params = {"60M": 61e6, "100M": 99e6}

    class _FakePlan:
        def __init__(self, N):
            self.ladder_params = int(N)
            self.steps = 100

    def _make(loss_offset):
        class _Ladder:
            def _resolve(self, rung, overrides):
                return _FakePlan(params[rung])

            def metrics(self, rung, **overrides):
                N = int(params[rung])
                recs = []
                for xc in (0.5, 1.0):
                    recs.append(
                        {
                            "phase": "post-decay",
                            "chinchilla_multiple": xc,
                            "tokens": int(20 * N * xc),
                            "val_loss": 3.0 - xc * 0.1 + loss_offset,
                        }
                    )
                return {"checkpoints": recs}

        return _Ladder()

    # variant is uniformly 0.05 lower -> wins everywhere, negative mean delta.
    result = showcase.compare_variants(_make(0.0), _make(-0.05), ["60M", "100M"])
    assert result["total"] == 4  # 2 rungs x 2 xC
    assert result["variant_lower_count"] == 4
    assert result["variant_wins_everywhere"] is True
    assert result["mean_delta"] == pytest.approx(-0.05)


def test_compare_perf_verdict(monkeypatch):
    """compare_perf keeps a faster iso-quality variant, rejects a quality regression."""
    from torchtitan.experiments.scaling_ladders import metrics as metrics_mod, showcase

    params = {"60M": 61e6, "100M": 99e6}

    class _Plan:
        def __init__(self, N):
            self.ladder_params = int(N)
            self.steps = 100

    def _make(loss_offset, root):
        class _Ladder:
            def _resolve(self, rung, overrides):
                return _Plan(params[rung])

            def metrics(self, rung, **overrides):
                N = int(params[rung])
                recs = [
                    {
                        "phase": "post-decay",
                        "chinchilla_multiple": xc,
                        "tokens": int(20 * N * xc),
                        "val_loss": 3.0 + loss_offset,
                    }
                    for xc in (0.5, 1.0)
                ]
                return {"checkpoints": recs}

            def run_dir(self, rung, **overrides):
                return f"/tmp/{root}/{rung}"

        return _Ladder()

    # Variant runs (root contains "var") are 1.3x faster; baseline is 1.0x.
    monkeypatch.setattr(
        metrics_mod,
        "read_run_throughput",
        lambda run_dir, **kw: 1300.0 if "var" in run_dir else 1000.0,
    )
    base = _make(0.0, "base")

    keep = showcase.compare_perf(
        base, _make(0.0, "var"), ["60M", "100M"], quality_eps=0.02
    )
    assert keep["mean_speedup"] == pytest.approx(1.3)
    assert keep["iso_quality"] is True and keep["verdict"] == "keep"

    # Faster but +0.05 worse loss -> exceeds eps -> reject.
    reject = showcase.compare_perf(
        base, _make(0.05, "var2"), ["60M", "100M"], quality_eps=0.02
    )
    assert reject["mean_speedup"] == pytest.approx(1.3)
    assert reject["iso_quality"] is False and reject["verdict"] == "reject"


def test_auto_compute_spec_picks_ddp_for_small_fsdp_for_large():
    """auto_compute_spec: DDP when the model fits on one GPU, FSDP only when not."""
    from torchtitan.experiments.scaling_ladders.planner import auto_compute_spec

    policy = WSDSChinchillaPolicy()
    for rung in ("60M", "100M", "370M", "1B"):
        cs = auto_compute_spec(rung, policy, gpus=8)
        assert cs.world_size == 8
        assert cs.parallelism.data_parallel_shard_degree == 1  # DDP (replicas)
        assert cs.parallelism.data_parallel_replicate_degree == 8
        assert cs.local_batch_size >= 1

    # 8B does not fit on one GPU -> must shard; 3B shards less than 8B.
    big = auto_compute_spec("8B", policy, gpus=8)
    assert big.parallelism.data_parallel_shard_degree > 1
    mid = auto_compute_spec("3B", policy, gpus=8)
    assert (
        mid.parallelism.data_parallel_shard_degree
        <= big.parallelism.data_parallel_shard_degree
    )
    # dp_degree is always the full node, regardless of the shard/replicate split.
    for cs in (big, mid):
        dp = (
            cs.parallelism.data_parallel_replicate_degree
            * cs.parallelism.data_parallel_shard_degree
        )
        assert dp == 8


def test_auto_compute_spec_fits_memory_budget():
    """The chosen shard degree keeps model+optimizer state within the budget frac."""
    from torchtitan.experiments.scaling_ladders.model import count_total_params
    from torchtitan.experiments.scaling_ladders.planner import (
        _BYTES_PER_PARAM,
        _MODEL_MEM_FRAC,
        auto_compute_spec,
    )

    policy = WSDSChinchillaPolicy()
    budget = 183.0 * 0.85
    for rung in ("60M", "760M", "3B", "8B"):
        cs = auto_compute_spec(rung, policy, gpus=8)
        sharded_model_gib = (
            count_total_params(rung)
            * _BYTES_PER_PARAM
            / 2**30
            / cs.parallelism.data_parallel_shard_degree
        )
        assert sharded_model_gib <= budget * _MODEL_MEM_FRAC + 1e-6


class _ProbePlan:
    # seqs_per_dp = local_batch_size * grad_accum_steps = 16.
    local_batch_size = 8
    grad_accum_steps = 2


class _ProbeLadder:
    def __init__(self, run_dir):
        self._dir = run_dir

    def _resolve(self, rung, overrides):
        return _ProbePlan()

    def compute_for(self, rung):
        return ComputeSpec(world_size=8)

    def run_dir(self, rung, **overrides):
        return self._dir


def _lbs_from_flags(flags):
    return int(flags[flags.index("--local-batch-size") + 1])


def test_run_rung_with_backoff_shrinks_on_oom(monkeypatch, tmp_path):
    """OOM at the planned lbs retries at the next-smaller divisor, then succeeds."""
    from torchtitan.experiments.scaling_ladders import ladder as ladder_mod

    calls = []

    def fake_spawn(rung, overrides, ws, flags=(), *, check=True, log_path=None):
        lbs = _lbs_from_flags(flags)
        calls.append(lbs)
        with open(log_path, "w") as handle:
            handle.write("CUDA out of memory" if lbs == 8 else "done")
        return 1 if lbs == 8 else 0

    monkeypatch.setattr(ladder_mod, "_spawn_run", fake_spawn)
    ladder_mod.run_rung_with_backoff(_ProbeLadder(str(tmp_path)), "1B", {})
    # 8 OOMs; largest divisor of 16 strictly below 8 is 4, which succeeds.
    assert calls == [8, 4]


def test_run_rung_with_backoff_raises_on_non_oom(monkeypatch, tmp_path):
    """A non-OOM failure is not retried -- it surfaces immediately."""
    from torchtitan.experiments.scaling_ladders import ladder as ladder_mod

    calls = []

    def fake_spawn(rung, overrides, ws, flags=(), *, check=True, log_path=None):
        calls.append(_lbs_from_flags(flags))
        with open(log_path, "w") as handle:
            handle.write("ValueError: something unrelated broke")
        return 1

    monkeypatch.setattr(ladder_mod, "_spawn_run", fake_spawn)
    with pytest.raises(RuntimeError, match="failed"):
        ladder_mod.run_rung_with_backoff(_ProbeLadder(str(tmp_path)), "1B", {})
    assert calls == [8]  # no retry on a non-OOM error


# ----------------------------------------------------------------------------
# fp8 rowwise wiring
# ----------------------------------------------------------------------------
def test_fp8_converter_swaps_linears_skipping_lm_head():
    """The rowwise Float8 converter swaps every transformer Linear but not lm_head."""
    from torchtitan.components.quantization.float8 import (
        Float8Linear,
        Float8LinearConverter,
    )
    from torchtitan.experiments.scaling_ladders.model import model_registry

    if Float8Linear is None:
        pytest.skip("torchao not installed")

    # emulate=True + compile off makes the converter constructible without SM89+.
    converter = Float8LinearConverter.Config(
        recipe_name="rowwise",
        filter_fqns=["lm_head"],
        emulate=True,
        model_compile_enabled=False,
    )
    spec = model_registry("60M", converters=[converter])
    fp8_fqns = [fqn for fqn, _, _, _ in spec.model.traverse(Float8Linear.Config)]
    # 60M has 5 layers x 6 linears (wq, wkv, wo, w1, w2, w3) = 30 swapped.
    assert len(fp8_fqns) == 30
    assert not any("lm_head" in fqn for fqn in fp8_fqns)


# ----------------------------------------------------------------------------
# Concurrent multi-GPU launcher
# ----------------------------------------------------------------------------
class _FakePopen:
    """Stand-in for subprocess.Popen that enforces the GPU-budget invariant.

    Asserts on construction that its assigned GPUs are disjoint from all other
    live jobs and that the node is never oversubscribed; completes after two
    poll() calls so several jobs are concurrently live during the test.
    """

    in_use: set = set()
    launches: list = []

    def __init__(self, cmd, *, stdout=None, stderr=None, env=None):
        self.cmd = cmd
        self.devs = [int(x) for x in env["CUDA_VISIBLE_DEVICES"].split(",") if x != ""]
        assert not (set(self.devs) & _FakePopen.in_use), "GPU double-allocated"
        assert len(_FakePopen.in_use) + len(self.devs) <= 8, "node oversubscribed"
        _FakePopen.in_use.update(self.devs)
        _FakePopen.launches.append({"cmd": cmd, "devs": list(self.devs)})
        self._stdout = stdout
        self._polls = 0

    def poll(self):
        self._polls += 1
        if self._polls >= 2:
            _FakePopen.in_use.difference_update(self.devs)
            return 0
        return None


def _cmd_lbs(cmd):
    return int(cmd[cmd.index("--local-batch-size") + 1])


def test_concurrent_packing_respects_gpu_budget(monkeypatch, tmp_path):
    """Jobs whose GPU demand exceeds the node run in waves, never oversubscribed."""
    from torchtitan.experiments.scaling_ladders import concurrent as conc
    from torchtitan.experiments.scaling_ladders.ladder import default_ladder

    _FakePopen.in_use, _FakePopen.launches = set(), []
    monkeypatch.setattr(conc.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(conc, "_checkpoint_steps_present", lambda d: [])

    ladder = replace(default_ladder(), base_dump_folder=str(tmp_path))
    jobs = [
        conc.Job("60M", 1),
        conc.Job("100M", 1),
        conc.Job("190M", 1),
        conc.Job("370M", 2),
        conc.Job("760M", 4),
    ]  # total demand 9 > 8 -> must wave
    results = conc.run_jobs_concurrent(ladder, jobs, total_gpus=8, poll_secs=0)
    assert len(results) == 5
    assert all(r["status"] == "done" for r in results)
    assert len(_FakePopen.launches) == 5  # each launched exactly once
    # The 760M (widest) launches first under largest-first packing.
    assert len(_FakePopen.launches[0]["devs"]) == 4


def test_concurrent_oom_requeues_smaller_lbs(monkeypatch, tmp_path):
    """An OOM relaunches the same job at a smaller local_batch_size, same width."""
    from torchtitan.experiments.scaling_ladders import concurrent as conc
    from torchtitan.experiments.scaling_ladders.ladder import default_ladder

    _FakePopen.in_use, _FakePopen.launches = set(), []

    class _OOMOncePopen(_FakePopen):
        launched: int = 0

        def __init__(self, cmd, *, stdout=None, stderr=None, env=None):
            super().__init__(cmd, stdout=stdout, stderr=stderr, env=env)
            self._oom = _OOMOncePopen.launched == 0  # only the first attempt OOMs
            _OOMOncePopen.launched += 1
            if self._oom and stdout is not None:
                stdout.write("torch.OutOfMemoryError: CUDA out of memory")
                stdout.flush()

        def poll(self):
            self._polls += 1
            if self._polls >= 2:
                _FakePopen.in_use.difference_update(self.devs)
                return 1 if self._oom else 0
            return None

    monkeypatch.setattr(conc.subprocess, "Popen", _OOMOncePopen)
    monkeypatch.setattr(conc, "_checkpoint_steps_present", lambda d: [])

    ladder = replace(default_ladder(), base_dump_folder=str(tmp_path))
    results = conc.run_jobs_concurrent(
        ladder, [conc.Job("370M", 2)], total_gpus=8, poll_secs=0
    )
    assert results[0]["status"] == "done"
    lbs_seq = [_cmd_lbs(l["cmd"]) for l in _FakePopen.launches]
    assert len(lbs_seq) == 2 and lbs_seq[1] < lbs_seq[0]  # shrank then succeeded


def test_concurrent_retries_transient_failure(monkeypatch, tmp_path):
    """A transient (e.g. C4 fetch) failure is retried at the same lbs, then succeeds."""
    from torchtitan.experiments.scaling_ladders import concurrent as conc
    from torchtitan.experiments.scaling_ladders.ladder import default_ladder

    _FakePopen.in_use, _FakePopen.launches = set(), []

    class _FlakeOncePopen(_FakePopen):
        launched: int = 0

        def __init__(self, cmd, *, stdout=None, stderr=None, env=None):
            super().__init__(cmd, stdout=stdout, stderr=stderr, env=env)
            self._flake = _FlakeOncePopen.launched == 0
            _FlakeOncePopen.launched += 1
            if self._flake and stdout is not None:
                stdout.write("FileNotFoundError: gzip://c4-train.00000 ...")
                stdout.flush()

        def poll(self):
            self._polls += 1
            if self._polls >= 2:
                _FakePopen.in_use.difference_update(self.devs)
                return 1 if self._flake else 0
            return None

    monkeypatch.setattr(conc.subprocess, "Popen", _FlakeOncePopen)
    monkeypatch.setattr(conc, "_checkpoint_steps_present", lambda d: [])

    ladder = replace(default_ladder(), base_dump_folder=str(tmp_path))
    results = conc.run_jobs_concurrent(
        ladder, [conc.Job("370M", 2)], total_gpus=8, poll_secs=0
    )
    assert results[0]["status"] == "done"
    lbs_seq = [_cmd_lbs(launch["cmd"]) for launch in _FakePopen.launches]
    # Retried once at the SAME lbs (transient retry does not shrink the batch).
    assert len(lbs_seq) == 2 and lbs_seq[0] == lbs_seq[1]


def test_concurrent_mixes_bf16_and_fp8_arms(monkeypatch, tmp_path):
    """A single call packs bf16 and fp8 jobs, each with its own flags/output root."""
    from torchtitan.experiments.scaling_ladders import concurrent as conc
    from torchtitan.experiments.scaling_ladders.ladder import default_ladder

    _FakePopen.in_use, _FakePopen.launches = set(), []
    monkeypatch.setattr(conc.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(conc, "_checkpoint_steps_present", lambda d: [])

    ladder = replace(default_ladder(), base_dump_folder=str(tmp_path), compile=True)
    jobs = [
        conc.Job("60M", 1, base_dump_folder=str(tmp_path / "bf16")),
        conc.Job("60M", 1, fp8=True, base_dump_folder=str(tmp_path / "fp8")),
    ]
    results = conc.run_jobs_concurrent(ladder, jobs, total_gpus=8, poll_secs=0)
    assert all(r["status"] == "done" for r in results)
    by_arm = {
        ("fp8" if "--fp8" in l["cmd"] else "bf16"): l["cmd"]
        for l in _FakePopen.launches
    }
    assert set(by_arm) == {"bf16", "fp8"}
    assert str(tmp_path / "bf16") in by_arm["bf16"]
    assert str(tmp_path / "fp8") in by_arm["fp8"]
    assert "--compile" in by_arm["fp8"]  # fp8 needs compile
