#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical comparison across parallelism configs for Qwen3.5.

Feeds identical fake tokens across configs (no_parallel, FSDP, FSDP+EP,
FSDP+EP+TP) and verifies logits match. Requires up to 8 GPUs.

Usage:
  python scripts/checkpoint_conversion/numerical_tests_qwen3_5_shard.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.qwen3_5 import qwen3_5_configs
from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5

CONFIGS = [
    {"ngpu": 1, "tp": 1, "ep": 1, "label": "no_parallel"},
    {"ngpu": 4, "tp": 1, "ep": 1, "label": "fsdp"},
    {"ngpu": 8, "tp": 1, "ep": 4, "label": "fsdp+ep"},
    {"ngpu": 8, "tp": 2, "ep": 2, "label": "fsdp+ep+tp"},
]


def run_worker(args):
    """Worker entry point — called via torchrun."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    dp_shard = world_size // args.tp

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config = qwen3_5_configs["debugmodel_moe"](
        attn_backend="flex",
        moe_comm_backend="standard",
    )

    parallel_dims = ParallelDims(
        dp_shard=dp_shard,
        dp_replicate=1,
        cp=1,
        tp=args.tp,
        pp=1,
        ep=args.ep,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    parallelism = ParallelismConfig(
        tensor_parallel_degree=args.tp,
        data_parallel_shard_degree=dp_shard,
        expert_parallel_degree=args.ep,
    )
    training = TrainingConfig(
        local_batch_size=1,
        seq_len=128,
        steps=1,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )

    config.update_from_config(
        config=type(
            "C",
            (),
            {
                "training": training,
                "parallelism": parallelism,
                "debug": type("D", (), {"moe_force_load_balance": False})(),
            },
        )(),
    )

    model = config.build()
    model.to_empty(device="cuda")
    model.init_weights(buffer_device=torch.device("cuda"))

    model = parallelize_qwen3_5(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=CompileConfig(),
        ac_config=ActivationCheckpointConfig(),
        dump_folder="/tmp",
    )

    torch.manual_seed(seed)
    seq_len = 128
    tokens = torch.randint(0, 248320, (1, seq_len), device="cuda")
    dist.broadcast(tokens, src=0)

    with torch.no_grad():
        output = model(
            tokens,
            special_tokens={"image_id": 151859, "video_id": 151860},
        )

    if isinstance(output, DTensor):
        output = output.full_tensor()

    logits = output[0, 0, :10].float().tolist()

    if rank == 0:
        with open(args.output, "w") as f:
            json.dump(logits, f)

    dist.destroy_process_group()


def main():
    """Orchestrator — launches torchrun for each config and compares."""
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for cfg in CONFIGS:
            outfile = os.path.join(tmpdir, f"{cfg['label']}.json")
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={cfg['ngpu']}",
                __file__,
                "--worker",
                f"--tp={cfg['tp']}",
                f"--ep={cfg['ep']}",
                f"--output={outfile}",
            ]
            print(
                f"Running {cfg['label']} (ngpu={cfg['ngpu']}, "
                f"tp={cfg['tp']}, ep={cfg['ep']})..."
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"  FAILED:\n{result.stderr[-500:]}")
                return 1

            with open(outfile) as f:
                results[cfg["label"]] = json.load(f)
            print(f"  logits: {[f'{v:.6f}' for v in results[cfg['label']]]}")

        print("\n--- Comparison ---")
        baseline_label = CONFIGS[0]["label"]
        baseline = results[baseline_label]
        all_pass = True

        for cfg in CONFIGS[1:]:
            label = cfg["label"]
            logits = results[label]
            max_diff = max(abs(a - b) for a, b in zip(baseline, logits))
            status = "PASS" if max_diff < 0.02 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {baseline_label} vs {label}: max_diff={max_diff:.6e}  {status}")

        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
        return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        sys.exit(main())
