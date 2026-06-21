# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concurrent multi-GPU launcher: bin-pack rung runs across one node's GPUs.

Each job is its OWN torchrun group (own rendezvous + ``CUDA_VISIBLE_DEVICES``); a
single 8-rank group cannot be partitioned, so spare GPUs are used by launching
several smaller groups side by side. Jobs at different GPU counts get different
data-parallel degrees and therefore slightly different global batches (batch
rounding), so a rung's bf16 and fp8 arms must use the SAME ``gpus`` to stay
iso-comparable. The loss-vs-compute FIT across rungs is unaffected: it keys off
each checkpoint's actual token count, not a fixed step.

Greedy largest-first packing keeps the node full: the few big rungs (which set
the makespan) start first on wide GPU subsets, and small rungs fill the gaps.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace

from torchtitan.tools.logging import logger

from .ladder import _checkpoint_steps_present, _OOM_MARKERS, _tail
from .planner import _largest_divisor_leq, auto_compute_spec, resolve

# Streaming/network failures (e.g. an HF-hub C4 shard fetch flaking) are transient
# and worth retrying at the same batch size; over a long multi-job campaign one of
# these would otherwise silently kill a run.
_TRANSIENT_MARKERS = (
    "filenotfounderror",
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

    ``fp8`` and ``base_dump_folder`` make the job self-describing so a single
    scheduler call can mix bf16 and fp8 arms on the node (both 760M runs side by
    side). ``base_dump_folder=None`` falls back to the scheduler ladder's root.
    ``overrides`` are policy overrides.
    """

    rung: str
    gpus: int
    overrides: dict = field(default_factory=dict)
    fp8: bool = False
    base_dump_folder: str | None = None
    attn_backend: str | None = None
    reduce_dtype: str | None = None


def _build_cmd(
    job: Job,
    *,
    lbs: int,
    base_dump_folder: str,
    fp8: bool,
    compile: bool,
    rdzv_id: int,
    max_steps: int | None = None,
) -> list[str]:
    # python -m torch.distributed.run == torchrun, but bound to this interpreter
    # so it does not depend on torchrun being on PATH.
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={job.gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0",
        f"--rdzv-id=ladder-{rdzv_id}",
        "-m",
        "torchtitan.experiments.scaling_ladders.train",
        "--rung",
        job.rung,
        "--gpus",
        str(job.gpus),
        "--base-dump-folder",
        base_dump_folder,
        "--local-batch-size",
        str(lbs),
    ]
    if fp8:
        cmd.append("--fp8")
    if compile:
        cmd.append("--compile")
    if job.attn_backend is not None:
        cmd += ["--attn-backend", job.attn_backend]
    if job.reduce_dtype is not None:
        cmd += ["--reduce-dtype", job.reduce_dtype]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    for key, value in job.overrides.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    return cmd


def _initial_state(ladder, job: Job) -> dict:
    """Resolve the per-job schedule at its own GPU count (gbs depends on it)."""
    policy, seed = ladder._split_overrides(job.overrides)
    compute = auto_compute_spec(job.rung, policy, gpus=job.gpus)
    plan = resolve(job.rung, policy, compute, seed=seed)
    seqs_per_dp = plan.local_batch_size * plan.grad_accum_steps
    arm_folder = job.base_dump_folder or ladder.base_dump_folder
    run_dir = replace(ladder, base_dump_folder=arm_folder).run_dir(
        job.rung, **job.overrides
    )
    return {
        "job": job,
        "lbs": plan.local_batch_size,
        "seqs_per_dp": seqs_per_dp,
        "final_step": plan.steps,
        "run_dir": run_dir,
        "base_dump_folder": arm_folder,
        "fp8": job.fp8,
        "retries": 0,
        "status": "pending",
    }


def run_jobs_concurrent(
    ladder,
    jobs: list[Job],
    *,
    total_gpus: int = 8,
    poll_secs: float = 30.0,
    max_retries: int = 4,
    max_steps: int | None = None,
    env: dict | None = None,
) -> list[dict]:
    """Run jobs concurrently, packing them onto ``total_gpus``; OOM-backoff each.

    ``ladder`` supplies shared config (policy, compile); each ``Job`` carries its
    own arm (``fp8`` and ``base_dump_folder``) so bf16 and fp8 runs can pack onto
    the node together. Already-complete runs (final checkpoint present) are
    skipped. On an out-of-memory failure a job is requeued at the next-smaller
    divisor local_batch_size (schedule-invariant) on the same GPU width. Returns
    a per-job result dict list; a non-OOM failure is recorded, not raised, so one
    bad rung does not abort the campaign.
    """
    base_env = dict(os.environ if env is None else env)
    states = [_initial_state(ladder, job) for job in jobs]

    pending, results = [], []
    for st in states:
        present = _checkpoint_steps_present(os.path.join(st["run_dir"], "checkpoint"))
        if st["final_step"] in present:
            st["status"] = "skipped (complete)"
            results.append(st)
            logger.info("Skipping complete run: %s", st["run_dir"])
        else:
            pending.append(st)
    # Largest-first: big rungs set the makespan, so start them on wide subsets.
    pending.sort(key=lambda s: s["job"].gpus, reverse=True)

    free = list(range(total_gpus))
    running, rdzv_id = [], 0

    def _launch(st: dict) -> None:
        nonlocal rdzv_id
        job = st["job"]
        subset, free[: job.gpus] = free[: job.gpus], []
        os.makedirs(st["run_dir"], exist_ok=True)
        arm = "fp8" if st["fp8"] else (job.attn_backend or "bf16")
        log_path = os.path.join(st["run_dir"], f"launch_{arm}_{job.gpus}gpu.log")
        cmd = _build_cmd(
            job,
            lbs=st["lbs"],
            base_dump_folder=st["base_dump_folder"],
            fp8=st["fp8"],
            compile=ladder.compile,
            rdzv_id=rdzv_id,
            max_steps=max_steps,
        )
        rdzv_id += 1
        proc_env = dict(base_env, CUDA_VISIBLE_DEVICES=",".join(map(str, subset)))
        handle = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=proc_env
        )
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
            "Launched %s [%s] on GPUs %s (lbs=%d): %s",
            job.rung,
            arm,
            subset,
            st["lbs"],
            log_path,
        )

    while pending or running:
        # 1. Reap finished jobs first so their GPUs are free before we launch.
        for entry in list(running):
            rc = entry["proc"].poll()
            if rc is None:
                continue
            running.remove(entry)
            entry["handle"].close()
            free.extend(entry["subset"])
            free.sort()
            st = entry["st"]
            job = st["job"]
            if rc == 0:
                st["status"] = "done"
                results.append(st)
                logger.info("Finished %s on GPUs %s", job.rung, entry["subset"])
                continue
            tail = _tail(entry["log"]).lower()
            is_oom = any(m in tail for m in _OOM_MARKERS)
            is_transient = any(m in tail for m in _TRANSIENT_MARKERS)
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
                st["status"] = f"failed (rc={rc}, oom={is_oom})"
                results.append(st)
                logger.error(
                    "Run %s failed (rc=%d); see %s", job.rung, rc, entry["log"]
                )

        # 2. Launch every pending job that fits the currently free GPUs.
        launched = True
        while launched:
            launched = False
            for i, st in enumerate(pending):
                if st["job"].gpus <= len(free):
                    _launch(pending.pop(i))
                    launched = True
                    break

        # 3. Nothing running and nothing launchable -> a job exceeds total_gpus.
        if not running and pending:
            need = sorted({s["job"].gpus for s in pending})
            raise ValueError(
                f"{len(pending)} job(s) need {need} GPUs but only {total_gpus} "
                f"are available; reduce per-job gpus or raise total_gpus."
            )

        if running:
            time.sleep(poll_secs)

    return results
