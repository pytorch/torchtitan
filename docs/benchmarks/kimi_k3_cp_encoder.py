# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare Kimi K3 encoder FSDP strategies with identical synthetic inputs.

Run each mode in a fresh process, for example:
    torchrun --standalone --nproc-per-node=4 -m benchmarks.kimi_k3_cp_encoder \
        --mode invariant --cp 2 --size full --output /tmp/invariant.json

Measures encoder forward/backward and AdamW, including indexed vision fusion;
does not measure the decoder or data loading. Use --snapshot with --size debug
to compare full parameter gradients and updates, and --trace for a rank-0 trace.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.distributed.fsdp import (
    apply_fsdp_to_multimodal_encoder,
    disable_fsdp_gradient_division,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_replicated_parameters,
    spmd_local_context,
)
from torchtitan.models.common.multimodal import (
    gather_vision_embeds,
    replicate_cp_vision_output,
)
from torchtitan.models.common.vision_encoder_sharding import (
    set_vision_encoder_cp_invariant,
)
from torchtitan.models.kimi_k2_7.sharding import set_moonvit_sharding_config
from torchtitan.models.kimi_k3.flavors import _vision_encoder_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "invariant"), required=True)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--size", choices=("debug", "full"), default="debug")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--grid-side", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--typecheck", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("Use at least 10 measured training steps")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    world_size = dist.get_world_size()
    if world_size % (args.cp * args.tp):
        parser.error("World size must be divisible by CP * TP")
    context = ParallelismContext(
        dp_replicate=1,
        dp_shard=world_size // (args.cp * args.tp),
        cp=args.cp,
        tp=args.tp,
        pp=1,
        ep=1,
        world_size=world_size,
        enable_sequence_parallel=False,
    )
    context.build_mesh()
    invariant = args.mode == "invariant"
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    full = args.size == "full"
    text_dim = 7168 if full else 256
    config = _vision_encoder_config(
        text_dim=text_dim,
        dim=1024 if full else 256,
        qkv_dim=1536 if full else 512,
        hidden_dim=4096 if full else 512,
        num_layers=27 if full else 2,
        num_heads=12 if full else 4,
        init_pos_emb_height=64 if full else 32,
        init_pos_emb_width=64 if full else 32,
    )
    set_moonvit_sharding_config(config, projector_norm="post_norm")
    if invariant:
        set_vision_encoder_cp_invariant(config)
    with torch.device("cuda"):
        encoder = config.build()
        encoder.init_states()
    parameter_count = sum(p.numel() for p in encoder.parameters())
    with context.activate_spmd():
        annotate_replicated_parameters(encoder, context)
        encoder._parallelize(context)
        mesh, axes = resolve_fsdp_mesh(context, shard_cp=not invariant)
        apply_fsdp_to_multimodal_encoder(
            encoder,
            mesh,
            param_dtype=getattr(torch, args.dtype),
            reduce_dtype=torch.float32,
            dp_mesh_dims=axes,
        )
        disable_fsdp_gradient_division(encoder)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, foreach=True)
    dp_mesh = context.get_optional_mesh("dp")
    cp_mesh = context.get_optional_mesh("cp")
    dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else 0
    cp_rank = cp_mesh.get_local_rank() if cp_mesh is not None else 0
    generator = torch.Generator(device="cuda").manual_seed(1234 + dp_rank)
    num_patches = args.images * args.grid_side**2
    bank_rows = num_patches // 4
    if bank_rows % args.cp:
        parser.error("Vision token count must be divisible by CP")
    pixels = torch.randn(num_patches, 588, generator=generator, device="cuda")
    grids = torch.tensor(
        [[1, args.grid_side, args.grid_side]] * args.images, device="cuda"
    )
    indices = torch.arange(bank_rows, device="cuda").chunk(args.cp)[cp_rank]
    # Interleave text placeholders without changing the bank's CP ownership.
    indices = torch.stack((indices, torch.full_like(indices, -1)), dim=1).flatten()
    text = torch.zeros(
        indices.numel(), text_dim, device="cuda", dtype=getattr(torch, args.dtype)
    )
    target = torch.randn(
        bank_rows * 2, text_dim, generator=generator, device="cuda"
    ).chunk(args.cp)[cp_rank]
    normalizer = bank_rows * text_dim * context.dp_shard
    loss_history = []
    gradient_snapshot = {}

    def step(capture: bool = False) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with context.activate_spmd(typechecking=args.typecheck), spmd_local_context(
            "dp"
        ):
            if args.typecheck:
                spmd.assert_type(
                    pixels,
                    {"dp": spmd.V, "cp": spmd.I if invariant else spmd.R, "tp": spmd.I},
                )
                spmd.assert_type(
                    grids,
                    {"dp": spmd.V, "cp": spmd.I if invariant else spmd.R, "tp": spmd.I},
                )
            bank = encoder(pixels, grid_thw=grids)
            if invariant:
                bank = replicate_cp_vision_output(bank)
            # The harness checks encoder types; token fusion has separate tests.
            with spmd.no_typecheck():
                fused = gather_vision_embeds(
                    text, vision_bank_VD=bank, vision_bank_indices_T=indices
                )
                loss = (fused.float() - target).square().sum() / normalizer
                loss.backward()
        if capture:
            for name, parameter in encoder.named_parameters():
                value = parameter.grad.full_tensor().cpu()
                if dist.get_rank() == 0:
                    gradient_snapshot[name] = value
        optimizer.step()
        return loss.detach()

    for i in range(args.warmup):
        step(capture=args.snapshot is not None and i == 0)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
        loss_history.append(loss)
        dist.barrier()
    elapsed = torch.tensor(times, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    losses = torch.stack(loss_history)
    loss_mesh = context.get_optional_mesh("loss")
    if loss_mesh is not None:
        dist.all_reduce(losses, group=loss_mesh.get_group())
    memory = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda")
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    result = {
        "mode": args.mode,
        "scope": "Kimi K3 encoder + indexed fusion + backward + AdamW (no decoder)",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "world_size": world_size,
        "dp_shard": context.dp_shard,
        "cp": args.cp,
        "tp": args.tp,
        "size": args.size,
        "dtype": args.dtype,
        "parameters": parameter_count,
        "images_per_dp_rank": args.images,
        "patches_per_dp_rank": num_patches,
        "vision_tokens_per_dp_rank": bank_rows,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "median_step_ms": statistics.median(elapsed.tolist()),
        "mean_step_ms": statistics.mean(elapsed.tolist()),
        "max_allocated_gib": memory.item() / 2**30,
        "step_ms": elapsed.tolist(),
        "losses": losses.tolist(),
    }
    if args.snapshot is not None:
        parameters = {
            name: p.full_tensor().cpu() for name, p in encoder.named_parameters()
        }
        if dist.get_rank() == 0:
            torch.save(
                {"gradients": gradient_snapshot, "parameters": parameters},
                args.snapshot,
            )
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if args.trace is not None:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=2, active=1, repeat=1, skip_first=1
            ),
            on_trace_ready=(lambda p: p.export_chrome_trace(str(args.trace)))
            if dist.get_rank() == 0
            else None,
        )
        prof.start()
        for _ in range(6):
            torch.cuda.synchronize()
            dist.barrier()
            step()
            torch.cuda.synchronize()
            dist.barrier()
            prof.step()
        prof.stop()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
