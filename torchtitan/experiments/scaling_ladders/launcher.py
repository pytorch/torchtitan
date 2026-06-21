# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single launcher for ladder runs: resolve once, hand the worker a config file.

ONE scheduler (``run_jobs``) bin-packs jobs onto a node's GPUs; sequential is just
one job at ``total_gpus = its width``. Each job is its own torchrun group (own
rendezvous + masked ``CUDA_VISIBLE_DEVICES``). The launcher resolves the schedule
ONCE (``build_spec``) and writes it -- plan plus all build knobs -- to a JSON spec
in the run dir; the worker (``run_from_spec``, invoked by ``train.py``) loads that
spec and builds the ``Trainer.Config`` directly. Config never round-trips through
argv, so launcher and worker cannot disagree and a new knob is one spec key, not a
flag threaded through five files. Greedy largest-first packing keeps the node full.

Jobs at different GPU counts get different data-parallel degrees and thus slightly
different global batches (batch rounding), so a rung's A/B arms must use the same
``gpus`` to stay iso-comparable; the loss-vs-compute fit is unaffected (it keys off
each checkpoint's actual token count).
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace

from torchtitan.config.configs import ParallelismConfig
from torchtitan.tools.logging import logger

from .model import fp8_converter
from .planner import (
    _largest_divisor_leq,
    auto_compute_spec,
    ComputeSpec,
    resolve,
    ResolvedPlan,
    to_plan_dict,
    to_trainer_config,
)

SPEC_FILE = "launch_spec.json"

# OOM -> shrink local_batch_size and retry. Streaming/network flakes (e.g. an HF
# C4 shard fetch) -> retry at the same batch. A run matching neither is a real
# failure and is not retried.
_OOM_MARKERS = ("out of memory", "outofmemoryerror")
_TRANSIENT_MARKERS = (
    "connectionerror",
    "connectionreset",
    "readtimeout",
    "incompleteread",
    "couldn't reach",
    "503 server",
    "huggingface_hub",
    "c4-train",
)


@dataclass
class Job:
    """One rung run on a fixed GPU count.

    ``fp8`` / ``attn_backend`` / ``reduce_dtype`` / ``base_dump_folder`` make the
    job self-describing so a single scheduler call can mix arms on the node;
    unset fields fall back to the launcher ladder's. ``overrides`` are policy
    overrides (see policy.OVERRIDABLE_FIELDS) plus optional ``seed``.
    """

    rung: str
    gpus: int
    overrides: dict = field(default_factory=dict)
    fp8: bool = False
    fp8_filter: tuple = ("lm_head", "attention")
    attn_backend: str | None = None
    reduce_dtype: str | None = None
    base_dump_folder: str | None = None
    # Initial local_batch_size (snapped to a divisor of the per-rank batch). Use to
    # pin a matched microbatch across A/B arms, or to skip the OOM probe's expensive
    # recompile walk on big rungs. None -> auto (one grad-accum step), then OOM probe.
    lbs: int | None = None


def _tail(path: str, nbytes: int = 4000) -> str:
    try:
        with open(path) as handle:
            return handle.read()[-nbytes:]
    except OSError:
        return ""


def _checkpoint_steps_present(checkpoint_folder: str) -> list[int]:
    import re

    if not os.path.isdir(checkpoint_folder):
        return []
    steps = []
    for name in os.listdir(checkpoint_folder):
        match = re.fullmatch(r"step-(\d+)", name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def build_spec(
    ladder,
    job: Job,
    *,
    lbs: int | None = None,
    max_steps: int | None = None,
    profile: bool = False,
) -> dict:
    """Resolve the job once and produce the complete worker build spec (a dict).

    The spec carries the resolved ``ResolvedPlan`` plus every build knob, so the
    worker never re-resolves. ``lbs`` (set by the OOM backoff) only changes
    gradient accumulation, not the schedule.
    """
    policy, seed = ladder._split_overrides(job.overrides)
    compute = auto_compute_spec(job.rung, policy, gpus=job.gpus)
    if lbs is not None:
        compute = replace(compute, local_batch_size=lbs)
    plan = resolve(job.rung, policy, compute, seed=seed)
    base = job.base_dump_folder or ladder.base_dump_folder
    run_dir = replace(ladder, base_dump_folder=base).run_dir(job.rung, **job.overrides)
    return {
        "plan": to_plan_dict(plan),
        "world_size": compute.world_size,
        "dp_replicate": compute.parallelism.data_parallel_replicate_degree,
        "dp_shard": compute.parallelism.data_parallel_shard_degree,
        "dataset": ladder.dataset,
        "val_dataset": ladder.val_dataset,
        "hf_assets_path": ladder.hf_assets_path,
        "dump_folder": run_dir,
        "enable_validation": ladder.enable_validation,
        "val_steps": ladder.val_steps,
        "log_freq": ladder.log_freq,
        "attn_backend": job.attn_backend or ladder.attn_backend,
        "compile": ladder.compile,
        "reduce_dtype": job.reduce_dtype or ladder.reduce_dtype,
        "fp8": job.fp8,
        "fp8_filter": list(job.fp8_filter),
        "max_steps": max_steps,
        "profile": profile,
    }


def run_from_spec(spec_path: str) -> None:
    """Worker body: load a spec, build the Trainer.Config, train. No re-resolve.

    Asserts NGPU (torchrun WORLD_SIZE) matches the spec's world_size so a
    mis-launched job fails loudly instead of silently mis-sharding.
    """
    with open(spec_path) as handle:
        spec = json.load(handle)
    plan = ResolvedPlan(**spec["plan"])
    compute = ComputeSpec(
        world_size=spec["world_size"],
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=spec["dp_replicate"],
            data_parallel_shard_degree=spec["dp_shard"],
        ),
        local_batch_size=plan.local_batch_size,
    )
    world_size = int(os.environ.get("WORLD_SIZE", compute.world_size))
    if world_size != compute.world_size:
        raise ValueError(
            f"Launched with world_size {world_size} but spec world_size is "
            f"{compute.world_size}."
        )
    converters = (
        [fp8_converter(spec["fp8_filter"], model_compile_enabled=spec["compile"])]
        if spec["fp8"]
        else None
    )
    cfg = to_trainer_config(
        plan,
        compute=compute,
        dataset=spec["dataset"],
        val_dataset=spec["val_dataset"],
        hf_assets_path=spec["hf_assets_path"],
        dump_folder=spec["dump_folder"],
        enable_validation=spec["enable_validation"],
        val_steps=spec["val_steps"],
        log_freq=spec["log_freq"],
        attn_backend=spec["attn_backend"],
        compile_enabled=spec["compile"],
        converters=converters,
        reduce_dtype=spec["reduce_dtype"],
    )
    if spec.get("max_steps") is not None:
        cfg.training.steps = min(cfg.training.steps, spec["max_steps"])
    if spec.get("profile"):
        cfg.profiler.enable_profiling = True
        cfg.profiler.profile_freq = 10
        cfg.profiler.profiler_warmup = 3
        cfg.profiler.profiler_active = 2
    trainer = cfg.build()
    try:
        trainer.train()
    finally:
        trainer.close()


def _argv(spec_path: str, gpus: int, rdzv_id: int) -> list[str]:
    # python -m torch.distributed.run == torchrun, bound to this interpreter so it
    # does not depend on torchrun being on PATH. The worker takes ONE arg: the spec.
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        f"--rdzv-id=ladder-{rdzv_id}",
        "-m",
        "torchtitan.experiments.scaling_ladders.train",
        "--config-file",
        spec_path,
    ]


def run_jobs(
    ladder,
    jobs: list[Job],
    *,
    total_gpus: int = 8,
    poll_secs: float = 30.0,
    max_retries: int = 4,
    max_steps: int | None = None,
    profile: bool = False,
    env: dict | None = None,
) -> list[dict]:
    """Run jobs concurrently, packing onto ``total_gpus``; OOM/transient backoff.

    Already-complete runs (final checkpoint present) are skipped. On OOM a job is
    requeued at the next-smaller divisor local_batch_size (schedule-invariant) on
    the same GPU width; on a transient failure it is retried at the same batch. A
    real failure is recorded, not raised, so one bad rung does not abort the run.
    Returns a per-job result dict list.
    """
    base_env = dict(os.environ if env is None else env)

    states = []
    for job in jobs:
        spec = build_spec(
            ladder, job, lbs=job.lbs, max_steps=max_steps, profile=profile
        )
        seqs_per_dp = (
            spec["plan"]["local_batch_size"] * spec["plan"]["grad_accum_steps"]
        )
        states.append(
            {
                "job": job,
                "lbs": spec["plan"]["local_batch_size"],
                "seqs_per_dp": seqs_per_dp,
                "final_step": spec["plan"]["steps"],
                "run_dir": spec["dump_folder"],
                "retries": 0,
                "status": "pending",
            }
        )

    pending, results = [], []
    for st in states:
        present = _checkpoint_steps_present(os.path.join(st["run_dir"], "checkpoint"))
        if st["final_step"] in present:
            st["status"] = "skipped (complete)"
            results.append(st)
            logger.info("Skipping complete run: %s", st["run_dir"])
        else:
            pending.append(st)
    pending.sort(key=lambda s: s["job"].gpus, reverse=True)

    free = list(range(total_gpus))
    running, rdzv_id = [], 0

    def _launch(st: dict) -> None:
        nonlocal rdzv_id
        job = st["job"]
        subset, free[: job.gpus] = free[: job.gpus], []
        spec = build_spec(
            ladder, job, lbs=st["lbs"], max_steps=max_steps, profile=profile
        )
        os.makedirs(st["run_dir"], exist_ok=True)
        spec_path = os.path.join(st["run_dir"], SPEC_FILE)
        with open(spec_path, "w") as handle:
            json.dump(spec, handle, indent=2)
        log_path = os.path.join(
            st["run_dir"], f"launch_{job.gpus}gpu_lbs{st['lbs']}.log"
        )
        proc_env = dict(base_env, CUDA_VISIBLE_DEVICES=",".join(map(str, subset)))
        handle = open(log_path, "w")
        proc = subprocess.Popen(
            _argv(spec_path, job.gpus, rdzv_id),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=proc_env,
        )
        rdzv_id += 1
        running.append(
            {
                "st": st,
                "subset": subset,
                "proc": proc,
                "handle": handle,
                "log": log_path,
            }
        )
        logger.info(
            "Launched %s on GPUs %s (lbs=%d): %s", job.rung, subset, st["lbs"], log_path
        )

    while pending or running:
        for entry in list(running):
            rc = entry["proc"].poll()
            if rc is None:
                continue
            running.remove(entry)
            entry["handle"].close()
            free.extend(entry["subset"])
            free.sort()
            st, job = entry["st"], entry["st"]["job"]
            if rc == 0:
                st["status"] = "done"
                results.append(st)
                logger.info("Finished %s on GPUs %s", job.rung, entry["subset"])
                continue
            tail = _tail(entry["log"]).lower()
            is_oom = any(m in tail for m in _OOM_MARKERS)
            is_transient = (not is_oom) and any(m in tail for m in _TRANSIENT_MARKERS)
            if is_oom and st["lbs"] > 1 and st["retries"] < max_retries:
                st["lbs"] = _largest_divisor_leq(st["seqs_per_dp"], st["lbs"] - 1)
                st["retries"] += 1
                pending.append(st)
                pending.sort(key=lambda s: s["job"].gpus, reverse=True)
                logger.warning(
                    "OOM on %s; requeue at lbs=%d (grad_accum=%d)",
                    job.rung,
                    st["lbs"],
                    st["seqs_per_dp"] // st["lbs"],
                )
            elif is_transient and st["retries"] < max_retries:
                st["retries"] += 1
                pending.append(st)
                pending.sort(key=lambda s: s["job"].gpus, reverse=True)
                logger.warning(
                    "Transient failure on %s (attempt %d); requeue at lbs=%d",
                    job.rung,
                    st["retries"],
                    st["lbs"],
                )
            else:
                reason = "OOM at lbs=1" if is_oom else f"rc={rc}"
                st["status"] = f"failed ({reason})"
                results.append(st)
                logger.error(
                    "Run %s failed (%s); see %s", job.rung, reason, entry["log"]
                )

        launched = True
        while launched:
            launched = False
            for i, st in enumerate(pending):
                if st["job"].gpus <= len(free):
                    _launch(pending.pop(i))
                    launched = True
                    break

        if not running and pending:
            need = sorted({s["job"].gpus for s in pending})
            raise ValueError(
                f"{len(pending)} job(s) need {need} GPUs but only {total_gpus} "
                f"are available; reduce per-job gpus or raise total_gpus."
            )
        if running:
            time.sleep(poll_secs)

    return results


def run_rung(ladder, rung: str, overrides: dict, *, compile: bool = True) -> dict:
    """Sequential convenience: run one rung on its full GPU width (= run_jobs N=1).

    ``compile`` is taken from the ladder; the arg is kept for call-site
    compatibility and applied via replace() so the worker spec reflects it.
    """
    spec_ladder = (
        replace(ladder, compile=compile) if compile != ladder.compile else ladder
    )
    world_size = spec_ladder.compute_for(rung).world_size
    job = Job(rung, world_size, overrides=dict(overrides))
    return run_jobs(spec_ladder, [job], total_gpus=world_size)[0]
