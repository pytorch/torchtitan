# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``Llama3Ladder``: the single source of truth driving the scaling ladder.

The config registry, the CLI, and the run/plan/status/metrics/sweep/compare API
all derive from one ``Llama3Ladder`` so the dry-run plan, the launched run, and
the read-back metrics can never disagree. The read side (plan/status/metrics/
compare) is in-process and builds models only on meta; a run is a torchrun
subprocess whose per-rank body is ``config.build() -> train() -> close()``.
"""

import itertools
import os
from dataclasses import dataclass, field, replace

from torchtitan.config.configs import ParallelismConfig
from torchtitan.trainer import Trainer

from .launcher import _checkpoint_steps_present
from .metrics import _find_event_dir, _read_scalars, _TRAIN_LOSS_TAG, read_run_metrics
from .model import RUNGS
from .planner import (
    auto_compute_spec,
    ComputeSpec,
    resolve,
    ResolvedPlan,
    to_plan_dict,
    to_trainer_config,
)
from .policy import OVERRIDABLE_FIELDS, WSDSChinchillaPolicy

# Override keys that re-run the policy (canonical list in policy.py). Anything
# else (e.g. seed) is handled separately; unknown keys raise so a typo never
# silently no-ops.
_POLICY_OVERRIDES = frozenset(OVERRIDABLE_FIELDS)
_SLUG_ABBREV = {
    "weight_decay": "wd",
    "lr_multiplier": "lrm",
    "chinchilla_multiple": "cm",
    "tokens_per_param": "tpp",
    "decay_fraction": "df",
}


@dataclass(kw_only=True)
class Llama3Ladder:
    policy: WSDSChinchillaPolicy
    compute_specs: dict[str, ComputeSpec]
    default_compute: ComputeSpec
    base_dump_folder: str
    dataset: str
    val_dataset: str
    hf_assets_path: str
    seed: int = 0
    enable_validation: bool = True
    val_steps: int = 50
    log_freq: int = 10
    attn_backend: str = "flex"
    compile: bool = False
    # Gradient all-reduce precision: "float32" (default) or "bfloat16" (halves
    # comm bytes on multi-GPU rungs; loss-affecting, so an iso-quality test).
    reduce_dtype: str = "float32"
    # Model-config converters (e.g. Float8LinearConverter.Config) applied at build
    # time; loss-affecting (fp8 quantization) but checkpoint- and schedule-neutral.
    converters: list = field(default_factory=list)

    def rungs(self) -> list[str]:
        return list(self.compute_specs)

    def compute_for(self, rung: str) -> ComputeSpec:
        return self.compute_specs.get(rung, self.default_compute)

    def _split_overrides(self, overrides: dict) -> tuple[WSDSChinchillaPolicy, int]:
        unknown = set(overrides) - _POLICY_OVERRIDES - {"seed"}
        if unknown:
            raise ValueError(f"Unknown override(s): {sorted(unknown)}")
        seed = overrides.get("seed", self.seed)
        policy = replace(
            self.policy,
            **{k: v for k, v in overrides.items() if k in _POLICY_OVERRIDES},
        )
        return policy, seed

    def _resolve(self, rung: str, overrides: dict) -> ResolvedPlan:
        policy, seed = self._split_overrides(overrides)
        return resolve(rung, policy, self.compute_for(rung), seed=seed)

    def run_dir(self, rung: str, **overrides) -> str:
        _, seed = self._split_overrides(overrides)
        parts = [
            f"{_SLUG_ABBREV.get(k, k)}{overrides[k]}"
            for k in sorted(overrides)
            if k != "seed"
        ]
        parts.append(f"seed{seed}")
        return os.path.join(self.base_dump_folder, rung, "_".join(parts))

    def plan(self, rung: str, **overrides) -> dict:
        return to_plan_dict(self._resolve(rung, overrides))

    def trainer_config(
        self, rung: str, *, local_batch_size: int | None = None, **overrides
    ) -> Trainer.Config:
        # local_batch_size is a memory/throughput override (the OOM probe); it does
        # not change the schedule (resolve snaps it to a divisor of the per-rank
        # batch), so plan/status/metrics stay consistent with the run.
        policy, seed = self._split_overrides(overrides)
        compute = self.compute_for(rung)
        if local_batch_size is not None:
            compute = replace(compute, local_batch_size=local_batch_size)
        return to_trainer_config(
            resolve(rung, policy, compute, seed=seed),
            compute=compute,
            dataset=self.dataset,
            val_dataset=self.val_dataset,
            hf_assets_path=self.hf_assets_path,
            dump_folder=self.run_dir(rung, **overrides),
            enable_validation=self.enable_validation,
            val_steps=self.val_steps,
            log_freq=self.log_freq,
            attn_backend=self.attn_backend,
            compile_enabled=self.compile,
            converters=self.converters or None,
            reduce_dtype=self.reduce_dtype,
        )

    def status(self, rung: str, **overrides) -> dict:
        plan = self._resolve(rung, overrides)
        run_dir = self.run_dir(rung, **overrides)
        present = _checkpoint_steps_present(os.path.join(run_dir, "checkpoint"))
        event_dir = _find_event_dir(run_dir, "tb")
        metric_steps = (
            sorted(_read_scalars(event_dir, _TRAIN_LOSS_TAG)) if event_dir else []
        )
        last = metric_steps[-1] if metric_steps else 0
        return {
            "rung": rung,
            "run_dir": run_dir,
            "total_steps": plan.steps,
            "last_metric_step": last,
            "pct_complete": last / plan.steps if plan.steps else 0.0,
            "checkpoint_steps_present": present,
            "expected_checkpoint_steps": plan.checkpoint_steps,
        }

    def metrics(self, rung: str, **overrides) -> dict:
        plan = self._resolve(rung, overrides)
        return read_run_metrics(self.run_dir(rung, **overrides), plan)

    def sweep(self, rungs: list[str], grid: dict[str, list]) -> list[dict]:
        """Expand a grid into ``(rung, overrides)`` run specs (pure; launch via
        ``launcher.run_jobs``)."""
        keys = sorted(grid)
        return [
            {"rung": rung, "overrides": dict(zip(keys, combo))}
            for rung in rungs
            for combo in itertools.product(*(grid[k] for k in keys))
        ]

    def compare(self, runs: list[dict], metric: str, at_xC: float) -> dict:
        """Rank a sweep's runs by ``metric`` at the matched post-decay xC point."""
        results = []
        for spec in runs:
            records = self.metrics(spec["rung"], **spec["overrides"])["checkpoints"]
            record = next(
                (
                    r
                    for r in records
                    if r["chinchilla_multiple"] == at_xC and r["phase"] == "post-decay"
                ),
                None,
            )
            results.append(
                {
                    "rung": spec["rung"],
                    "overrides": spec["overrides"],
                    metric: record.get(metric) if record else None,
                }
            )
        ranked = sorted(
            (r for r in results if r[metric] is not None), key=lambda r: r[metric]
        )
        return {
            "metric": metric,
            "at_xC": at_xC,
            "ranked": ranked,
            "argmin": ranked[0] if ranked else None,
        }


def default_ladder(
    base_dump_folder: str = "./outputs/scaling_ladders", world_size: int = 8
) -> Llama3Ladder:
    """The showcase Llama3 ladder: C4, Llama3 tokenizer assets.

    Per-rung parallelism and local batch size are derived from each rung's memory
    footprint by ``auto_compute_spec`` (no hardcoded table): small rungs that fit
    on one GPU run as DDP replicas, larger rungs shard (FSDP) only as much as
    needed. The launch-time OOM probe trims an over-optimistic local batch size.
    """
    policy = WSDSChinchillaPolicy()
    compute_specs = {
        rung: auto_compute_spec(rung, policy, gpus=world_size)
        for rung in RUNGS
        if rung != "debug"
    }
    return Llama3Ladder(
        policy=policy,
        compute_specs=compute_specs,
        default_compute=ComputeSpec(
            world_size=world_size, parallelism=ParallelismConfig(), local_batch_size=1
        ),
        base_dump_folder=base_dump_folder,
        dataset="c4",
        val_dataset="c4_validation",
        hf_assets_path="./assets/hf/Llama-3.1-8B",
    )


def debug_ladder(
    base_dump_folder: str = "./outputs/scaling_ladders_debug", world_size: int = 8
) -> Llama3Ladder:
    """Tiny ladder for fake-backend and smoke tests (c4_test, test tokenizer)."""
    return Llama3Ladder(
        policy=WSDSChinchillaPolicy(seq_len=2048),
        compute_specs={
            "debug": ComputeSpec(
                world_size=world_size,
                parallelism=ParallelismConfig(),
                local_batch_size=1,
            )
        },
        default_compute=ComputeSpec(
            world_size=world_size, parallelism=ParallelismConfig(), local_batch_size=1
        ),
        base_dump_folder=base_dump_folder,
        dataset="c4_test",
        val_dataset="c4_test",
        hf_assets_path="./tests/assets/tokenizer",
        enable_validation=False,
        val_steps=10,
        log_freq=1,
    )
