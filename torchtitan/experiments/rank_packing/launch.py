# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch bounded native multi-rank training on one physical GPU."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _cleanup(process: subprocess.Popen) -> None:
    if not _group_exists(process.pid):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while _group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--dist-moe", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.steps <= 0 or args.timeout <= 0 or args.seq_len <= 0:
        raise ValueError("Steps, sequence length and timeout must be positive.")
    gpu_ids = args.gpus.split(",")
    if len(gpu_ids) not in (1, 8) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Select one physical GPU or eight distinct GPUs.")
    if args.mps and len(gpu_ids) != 1:
        raise ValueError("Private MPS is scoped to one physical GPU.")
    if args.dist_moe and len(gpu_ids) == 1 and not args.mps:
        raise ValueError("The qualified one-GPU DistMoE run requires private MPS.")
    gpu_mask = args.gpus
    if args.mps:
        gpu_mask = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                args.gpus,
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        mode = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                args.gpus,
                "--query-gpu=compute_mode",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if mode != "Default":
            raise ValueError(f"Private MPS requires Default compute mode, got {mode}.")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            "--module",
            "torchtitan.experiments.rank_packing.worker",
        ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=gpu_mask,
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        NCCL_MULTI_RANK_GPU_ENABLE="1",
        NCCL_NVLS_ENABLE="0",
        NCCL_MIN_CTAS="1",
        NCCL_MAX_CTAS="1",
        NCCL_DEBUG="WARN",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=str(output.parent / "cache" / "inductor"),
        TRITON_CACHE_DIR=str(output.parent / "cache" / "triton"),
        HF_HOME=str(output.parent / "cache" / "huggingface"),
        PACKED_OUTPUT=str(output),
        PACKED_STEPS=str(args.steps),
        PACKED_SEQ_LEN=str(args.seq_len),
        PACKED_DIST_MOE="1" if args.dist_moe else "0",
        PACKED_PROFILE="1" if args.profile else "0",
        PYTHONUNBUFFERED="1",
        PYTHONFAULTHANDLER="1",
    )
    if args.dist_moe:
        env["TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"] = "1"
        env["TORCH_SYMM_MEM_DISABLE_MULTICAST"] = "1"
    started = time.monotonic()
    process = None
    ipc_root = None
    mps_attempted = False
    timed_out = False
    code = 1

    def terminate(signum, frame):
        nonlocal code
        code = 128 + signum
        raise SystemExit(code)

    old_handlers = {
        sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        # TODO: remove the short-path workaround after PyTorch validates IPC names.
        # Native rendezvous copies TMPDIR plus the PID into sockaddr_un.sun_path.
        # Keep the PID suffix intact rather than truncating distinct peer names.
        ipc_root = Path(tempfile.mkdtemp(prefix="tt-rank-packing-", dir="/tmp"))
        if len(os.fsencode(ipc_root / "symm_mem-2147483647")) >= 108:
            raise ValueError(
                "IPC directory exceeds the Linux Unix socket pathname limit."
            )
        env["TMPDIR"] = str(ipc_root)
        if args.mps:
            for name in ("pipe", "log"):
                (ipc_root / name).mkdir(mode=0o700)
            env["CUDA_MPS_PIPE_DIRECTORY"] = str(ipc_root / "pipe")
            env["CUDA_MPS_LOG_DIRECTORY"] = str(ipc_root / "log")
            env.pop("CUDA_MPS_PROTOCOL_VERSION", None)
        (output / "launch.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "options": {
                        **vars(args),
                        "output_dir": str(output),
                        "world_size": 8,
                    },
                    "source_commit": source_commit,
                    "env": {
                        key: value
                        for key, value in env.items()
                        if key.startswith(
                            (
                                "PACKED_",
                                "NCCL_",
                                "TORCHINDUCTOR_",
                                "TORCH_SYMM",
                                "CUDA_MPS_",
                            )
                        )
                        or key in ("CUDA_VISIBLE_DEVICES", "TMPDIR")
                    },
                },
                indent=2,
            )
        )
        if args.mps:
            mps_attempted = True
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"], env=env, check=True, timeout=15
            )
        with (output / "console.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                code = 124
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        cleanup_errors = []
        if process is not None:
            try:
                _cleanup(process)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(f"workers: {error}")
        if mps_attempted:
            try:
                subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input="quit\n",
                    text=True,
                    env=env,
                    check=True,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(f"private MPS: {error}")
        group_gone = process is None or not _group_exists(process.pid)
        if not cleanup_errors and group_gone and ipc_root is not None:
            try:
                shutil.rmtree(ipc_root)
            except OSError as error:
                cleanup_errors.append(f"IPC directory: {error}")
        if cleanup_errors or not group_gone:
            code = 1
        try:
            (output / "outcome.json").write_text(
                json.dumps(
                    {
                        "exit_code": code,
                        "timed_out": timed_out,
                        "elapsed_seconds": time.monotonic() - started,
                        "output": str(output),
                        "process_group_gone": group_gone,
                        "cleanup_errors": cleanup_errors,
                        "ipc_root": str(ipc_root) if ipc_root is not None else None,
                    },
                    indent=2,
                )
            )
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    print((output / "outcome.json").read_text())
    if (output / "console.log").exists():
        lines = (output / "console.log").read_text(errors="replace").splitlines()
        print("\n".join(line[:1000] for line in lines[-15:]))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
