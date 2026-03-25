#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint backward compatibility test.

Verifies that checkpoints saved by one git commit can be loaded by another
and produce bit-identical losses. Runs three training jobs:

  1. Reference run at <load_commit>: train full --steps from scratch.
  2. Save run at <save_commit>: train --resume-step steps, save checkpoint.
  3. Resume run at <load_commit>: resume from save checkpoint to --steps.

All use --debug.deterministic --debug.seed=42. Pass = identical losses.

Examples:
  python scripts/checkpoint_compat_test.py HEAD~1 HEAD
  python scripts/checkpoint_compat_test.py HEAD~1 HEAD --steps=100 --resume-step=50
  python scripts/checkpoint_compat_test.py HEAD~1 HEAD --assert-equal
  python scripts/checkpoint_compat_test.py HEAD~1 . --assert-equal
"""

import argparse
import os
import subprocess
import sys
import tempfile

FIXED_OPTIONS = (
    "--debug.deterministic --debug.seed=42"
    " --metrics.enable_tensorboard --metrics.log_freq=1"
)
TB_LOSS_TAG = "loss_metrics/global_avg_loss"


def log(msg: str = "") -> None:
    print(f"[CKPT_COMPAT] {msg}" if msg else "[CKPT_COMPAT]")


def git_run(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def check_git_clean() -> None:
    lines = git_run("status", "--porcelain").split("\n")
    modified = [l for l in lines if l and not l.startswith("??")]
    if modified:
        log("Error: uncommitted changes to tracked files:")
        for l in modified:
            log(f"  {l}")
        sys.exit(1)


def get_current_ref() -> str:
    ref = git_run("rev-parse", "--abbrev-ref", "HEAD")
    return ref if ref != "HEAD" else git_run("rev-parse", "HEAD")


def checkout(commit: str, label: str) -> None:
    if commit != ".":
        log(f"Checking out {label}: {commit}")
        subprocess.run(["git", "checkout", commit], check=True)


def run_cmd(cmd: str, logfile: str, ngpus: int) -> None:
    """Run training command with real-time output and log capture."""
    log(f"Executing: {cmd}")
    env = {**os.environ, "NGPU": str(ngpus), "PYTHONUNBUFFERED": "1"}
    with open(logfile, "w") as f:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # pyrefly: ignore [not-iterable]
            print(line, end="")
            f.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def build_cmd(
    module: str,
    config: str,
    options: str,
    steps: int,
    dump_folder: str,
    *,
    total_steps: int = 0,
    checkpoint_enable: bool = False,
    checkpoint_interval: int = 0,
) -> str:
    cmd = (
        f"MODULE='{module}' CONFIG='{config}' ./run_train.sh"
        f" --dump_folder={dump_folder} {FIXED_OPTIONS}"
        f" --training.steps={steps} --metrics.save_tb_folder=tb"
    )
    if total_steps > 0:
        # Pin LR schedule so the save run uses the same curve as the full run.
        cmd += f" --lr_scheduler.total_steps={total_steps}"
    if options:
        cmd += f" {options}"
    if checkpoint_enable:
        cmd += f" --checkpoint.enable --checkpoint.interval={checkpoint_interval}"
    return cmd


def extract_tb_losses(tb_base: str) -> dict[int, float]:
    """Extract full-precision losses from all TB subdirs under tb_base.

    Unlike loss_compare.extract_losses_from_tensorboard (which expects a single
    subdirectory), this merges events across multiple subdirs to handle the
    resume case where save and resume runs write to the same TB folder.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    losses: dict[int, float] = {}
    for subdir in sorted(os.listdir(tb_base)):
        path = os.path.join(tb_base, subdir)
        if not os.path.isdir(path):
            continue
        acc = EventAccumulator(path)
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if TB_LOSS_TAG in tags:  # pyrefly: ignore [not-iterable]
            for s in acc.Scalars(TB_LOSS_TAG):
                losses[s.step] = s.value
    log(f"Extracted {len(losses)} steps from {tb_base}")
    return losses


def compare_losses(
    ref: dict[int, float],
    resume: dict[int, float],
    assert_equal: bool,
) -> bool:
    ref_steps = sorted(ref)
    if ref_steps != sorted(resume):
        log(f"Step count mismatch: ref={len(ref)}, resume={len(resume)}")
        return False

    log(f"{'Step':<8} {'Reference':<22} {'Resume':<22} {'Match'}")
    log("-" * 60)

    all_ok = True
    for step in ref_steps:
        ok = ref[step] == resume[step]
        all_ok &= ok
        log(
            f"{step:<8} {ref[step]!r:<22} {resume[step]!r:<22} {'OK' if ok else 'MISMATCH'}"
        )

    log()
    log(
        "PASS: All losses are bit-identical."
        if all_ok
        else "FAIL: Loss mismatch detected!"
    )

    if assert_equal and not all_ok:
        sys.exit(1)
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Test checkpoint backward compatibility between two git commits.",
    )
    p.add_argument("save_commit", help="Commit that saves the checkpoint (old code)")
    p.add_argument("load_commit", help="Commit that loads the checkpoint (new code)")
    p.add_argument("--steps", type=int, default=100, help="Total training steps")
    p.add_argument("--resume-step", type=int, default=50, help="Checkpoint/resume step")
    p.add_argument("--module", default="llama3")
    p.add_argument("--config", default="llama3_debugmodel")
    p.add_argument("--options", default="", help="Extra training CLI args")
    p.add_argument("--ngpus", type=int, default=8)
    p.add_argument("--output-folder", default="")
    p.add_argument(
        "--assert-equal", action="store_true", help="Exit non-zero on mismatch"
    )
    args = p.parse_args()

    if args.resume_step >= args.steps:
        p.error(f"--resume-step ({args.resume_step}) must be < --steps ({args.steps})")
    if not args.output_folder:
        args.output_folder = tempfile.mkdtemp(prefix="ckpt_compat_")

    log("Checkpoint Backward Compatibility Test")
    log(
        f"  save={args.save_commit}  load={args.load_commit}  "
        f"steps={args.steps}  resume_step={args.resume_step}  ngpus={args.ngpus}"
    )
    log(f"  output: {args.output_folder}")

    # Resolve SHAs before any checkout
    save_sha = (
        git_run("rev-parse", args.save_commit) if args.save_commit != "." else "."
    )
    load_sha = (
        git_run("rev-parse", args.load_commit) if args.load_commit != "." else "."
    )

    ref_dump = os.path.join(args.output_folder, "ref_outputs")
    resume_dump = os.path.join(args.output_folder, "resume_outputs")
    os.makedirs(ref_dump, exist_ok=True)
    os.makedirs(resume_dump, exist_ok=True)

    needs_checkout = save_sha != "." or load_sha != "."
    original = get_current_ref() if needs_checkout else None
    if needs_checkout:
        check_git_clean()

    common = dict(module=args.module, config=args.config, options=args.options)

    try:
        # Step 1: Reference run at load_commit (full training from scratch)
        log()
        log("=" * 60)
        log("STEP 1: Reference run")
        log("=" * 60)
        checkout(load_sha, "load_commit")
        cmd = build_cmd(**common, steps=args.steps, dump_folder=ref_dump)
        run_cmd(cmd, os.path.join(args.output_folder, "reference.log"), args.ngpus)

        # Step 2: Save run at save_commit (partial training + checkpoint)
        log()
        log("=" * 60)
        log(f"STEP 2: Save run ({args.resume_step} steps)")
        log("=" * 60)
        checkout(save_sha, "save_commit")
        cmd = build_cmd(
            **common,
            steps=args.resume_step,
            dump_folder=resume_dump,
            total_steps=args.steps,
            checkpoint_enable=True,
            checkpoint_interval=args.resume_step,
        )
        run_cmd(cmd, os.path.join(args.output_folder, "save.log"), args.ngpus)

        # Step 3: Resume run at load_commit (load checkpoint, train to end)
        log()
        log("=" * 60)
        log(f"STEP 3: Resume run (to step {args.steps})")
        log("=" * 60)
        checkout(load_sha, "load_commit")
        cmd = build_cmd(
            **common,
            steps=args.steps,
            dump_folder=resume_dump,
            checkpoint_enable=True,
            checkpoint_interval=args.resume_step,
        )
        run_cmd(cmd, os.path.join(args.output_folder, "resume.log"), args.ngpus)

        # Step 4: Compare
        log()
        ref_losses = extract_tb_losses(os.path.join(ref_dump, "tb"))
        resume_losses = extract_tb_losses(os.path.join(resume_dump, "tb"))
        compare_losses(ref_losses, resume_losses, args.assert_equal)

    finally:
        if original:
            log()
            log(f"Restoring: {original}")
            subprocess.run(["git", "checkout", original], check=True)

    log(f"Logs saved in: {args.output_folder}")


if __name__ == "__main__":
    main()
