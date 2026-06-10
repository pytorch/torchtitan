#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone torchrun repro for torch.compile with FlexShard.

Run from the repository root, for example:

    torchrun --standalone --nproc_per_node=1 \
      torchtitan/experiments/flex_shard/compile_repro.py --device cpu

    torchrun --standalone --nproc_per_node=2 \
      torchtitan/experiments/flex_shard/compile_repro.py --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, ProfilerActivity, schedule

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.experiments.flex_shard import BucketSpec, flex_shard  # noqa: E402
from torchtitan.experiments.flex_shard.example.shard import (  # noqa: E402
    per_param_placements,
)
from torchtitan.experiments.flex_shard.tests.common import (  # noqa: E402
    make_transformer_model,
    transformer_bucket_specs,
    transformer_inputs,
)


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.relu(self.proj(x)))


class TinyModel(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(dim) for _ in range(depth))
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply FlexShard to a tiny model, then run it through torch.compile."
        )
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device type for the repro. auto picks cuda when available.",
    )
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model",
        choices=("tiny", "transformer"),
        default="tiny",
        help="Model to profile. transformer uses the FlexShard unit-test Transformer.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length for --model transformer.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16,
        help="Vocabulary size for --model transformer.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=2,
        help="Attention heads for --model transformer.",
    )
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--reshard-after-forward",
        action="store_true",
        help="Enable FlexShard reshard-after-forward for each bucket.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help=(
            "Directory for per-rank Chrome profiler traces around the first "
            "compiled forward/backward step."
        ),
    )
    parser.add_argument(
        "--profile-eager",
        action="store_true",
        help="Also write an eager forward/backward profiler trace to --profile-dir.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Include memory events in the profiler trace.",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Include Python stack traces in the profiler trace.",
    )
    parser.add_argument(
        "--memory-snapshot-dir",
        type=Path,
        help="Directory for a rank0 CUDA allocator memory snapshot.",
    )
    parser.add_argument(
        "--allow-compile-failure",
        action="store_true",
        help="Exit 0 after printing a torch.compile failure.",
    )
    parser.add_argument(
        "--skip-fwd-side-effects-under-checkpoint",
        action="store_true",
        help=(
            "Set torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint. "
            "This is needed for reshard-after-forward when tracing the real bucket "
            "pre-forward hook state mutations under activation checkpointing."
        ),
    )
    return parser.parse_args()


def _dist_env_present() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_distributed(device_type: str) -> tuple[int, int, int]:
    """Initialize process group and return rank, world size, local rank."""
    if not _dist_env_present():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if device_type == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend,
            timeout=timedelta(seconds=60),
        )
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _select_device(device_arg: str, local_rank: int) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _bucket_specs(depth: int, mesh, *, reshard_after_forward: bool) -> list[BucketSpec]:
    specs: list[BucketSpec] = []
    for idx in range(depth):
        specs.append(
            BucketSpec(
                [f"blocks.{idx}.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
        )
    specs.append(
        BucketSpec(
            ["output.*"],
            placement_fn=per_param_placements,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )
    )
    return specs


def _grad_norm(model: nn.Module, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _run_step(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    return loss.detach(), _grad_norm(model, x.device).detach()


def _build_tiny_model(
    args: argparse.Namespace,
    mesh,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor]:
    model = TinyModel(args.dim, args.depth).to(device)
    flex_shard(
        model,
        buckets=_bucket_specs(
            args.depth,
            mesh,
            reshard_after_forward=args.reshard_after_forward,
        ),
    )
    x = torch.randn(args.batch_size, args.dim, device=device)
    return model, x


def _build_transformer_model(
    args: argparse.Namespace,
    mesh,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor]:
    model_args, model = make_transformer_model(
        device=device,
        n_layers=args.depth,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        dim=args.dim,
        n_heads=args.heads,
    )
    flex_shard(
        model,
        buckets=transformer_bucket_specs(
            model_args.n_layers,
            mesh,
            reshard_after_forward=args.reshard_after_forward,
        ),
    )
    x = transformer_inputs(model_args, batch_size=args.batch_size, device=device)
    return model, x


def _build_model_and_input(
    args: argparse.Namespace,
    mesh,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor]:
    if args.model == "transformer":
        return _build_transformer_model(args, mesh, device)
    return _build_tiny_model(args, mesh, device)


def _profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _sync_step(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()


def _start_memory_snapshot(args: argparse.Namespace, device: torch.device) -> None:
    if args.memory_snapshot_dir is None or device.type != "cuda":
        return
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
        max_entries=200000,
        device=device,
        clear_history=True,
    )
    torch.cuda.reset_peak_memory_stats(device)


def _dump_memory_snapshot(
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
) -> None:
    if args.memory_snapshot_dir is None or device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    if rank == 0:
        args.memory_snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = (
            args.memory_snapshot_dir / f"flex_shard_memory_rank{rank}.pickle"
        )
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(f"memory snapshot: {snapshot_path.resolve()}")
        print(
            "cuda memory: "
            f"max_allocated={max_allocated}, "
            f"max_reserved={max_reserved}, "
            f"allocated={allocated}, "
            f"reserved={reserved}"
        )
    torch.cuda.memory._record_memory_history(enabled=None, device=device)


def _run_profiled_steps(
    model: nn.Module,
    x: torch.Tensor,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    *,
    trace_label: str,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.profile_dir / f"flex_shard_{trace_label}_rank{rank}.json"
    last_loss = torch.empty((), device=x.device)
    last_grad_norm = torch.empty((), device=x.device)
    prof = profile(
        activities=_profiler_activities(device),
        schedule=schedule(
            wait=1,
            warmup=2,
            active=1,
            repeat=1,
            skip_first=1,
        ),
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=args.profile_with_stack,
    )
    prof.start()
    for _ in range(6):
        _sync_step(device)
        with torch.profiler.record_function(f"FlexShard::{trace_label}_step"):
            last_loss, last_grad_norm = _run_step(model, x)
        _sync_step(device)
        prof.step()
    prof.stop()
    prof.export_chrome_trace(str(trace_path))
    return last_loss, last_grad_norm, trace_path


def _run_maybe_profiled_step(
    model: nn.Module,
    x: torch.Tensor,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    *,
    trace_label: str,
    should_profile: bool,
) -> tuple[torch.Tensor, torch.Tensor, Path | None]:
    if args.profile_dir is None or not should_profile:
        loss, grad_norm = _run_step(model, x)
        return loss, grad_norm, None
    return _run_profiled_steps(
        model,
        x,
        args,
        rank,
        device,
        trace_label=trace_label,
    )


def main() -> int:
    args = _parse_args()
    if args.skip_fwd_side_effects_under_checkpoint:
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    device_type = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_type == "auto":
        device_type = "cpu"

    rank, world_size, local_rank = _init_distributed(device_type)
    device = _select_device(args.device, local_rank)

    try:
        torch.manual_seed(2026)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(2026)
        _start_memory_snapshot(args, device)

        mesh = init_device_mesh(
            device.type,
            (world_size,),
            mesh_dim_names=("fsdp",),
        )
        model, x = _build_model_and_input(args, mesh, device)

        eager_loss, eager_grad_norm, eager_trace_path = _run_maybe_profiled_step(
            model,
            x,
            args,
            rank,
            device,
            trace_label="eager",
            should_profile=args.profile_eager,
        )
        if rank == 0:
            print(
                "eager FlexShard step succeeded: "
                f"loss={eager_loss.item():.6f}, "
                f"grad_norm={eager_grad_norm.item():.6f}"
            )
            if eager_trace_path is not None:
                print(f"eager profiler trace: {eager_trace_path.resolve()}")

        compiled_model = torch.compile(
            model,
            backend=args.backend,
            fullgraph=args.fullgraph,
        )
        (
            compiled_loss,
            compiled_grad_norm,
            compiled_trace_path,
        ) = _run_maybe_profiled_step(
            compiled_model,
            x,
            args,
            rank,
            device,
            trace_label="compile",
            should_profile=True,
        )
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            print(
                "compiled FlexShard step succeeded: "
                f"loss={compiled_loss.item():.6f}, "
                f"grad_norm={compiled_grad_norm.item():.6f}"
            )
            if compiled_trace_path is not None:
                print("profiler traces written under: " f"{args.profile_dir.resolve()}")
                print(f"compiled profiler trace: {compiled_trace_path.resolve()}")
        _dump_memory_snapshot(args, rank, device)
        return 0
    except Exception:
        if rank == 0:
            print("torch.compile FlexShard repro failed with:", file=sys.stderr)
            traceback.print_exc()
        _dump_memory_snapshot(args, rank, device)
        return 0 if args.allow_compile_failure else 1
    finally:
        _destroy_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
