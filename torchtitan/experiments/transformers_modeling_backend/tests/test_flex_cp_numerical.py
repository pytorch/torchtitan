# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Numerical equivalence test for flex attention + context parallelism.
#
# Builds the flex debug model, runs a full-sequence forward as the reference,
# then runs the same weights under CP with the input/positions/BlockMask sharded
# on the sequence axis (as the trainer does). Reconstructs full logits in global
# order (a global-index tensor rides through cp_shard to undo any load-balancer
# permutation) and compares against the reference. Correct CP attention matches
# to fp32 noise (rel < 1e-4); bf16 mixed precision masks this, so we force fp32.
#
# Run: torchrun --nproc_per_node=2 tests/test_flex_cp_numerical.py \
#          [--balancer none|headtail|ptrr]

import argparse
import os

import torch
import torch.distributed as dist

from torchtitan.config import CompileConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.experiments.transformers_modeling_backend.config_registry import (
    transformers_modeling_backend_debugmodel,
    transformers_modeling_backend_debugmodel_flex,
    transformers_modeling_backend_debugmodel_moe,
    transformers_modeling_backend_debugmodel_moe_flex,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="default")
    parser.add_argument("--hf_model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument(
        "--balancer", default="none", choices=["none", "headtail", "ptrr"]
    )
    parser.add_argument("--moe", action="store_true", help="use the flex MoE model")
    parser.add_argument(
        "--no_flex",
        action="store_true",
        help="use SDPA attention (ring CP) instead of flex",
    )
    args = parser.parse_args()
    balancer = None if args.balancer == "none" else args.balancer

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}")
    torch.set_default_dtype(torch.float32)

    cp = world  # cp = world_size (dp=tp=pp=1)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=cp,
        tp=1,
        pp=1,
        ep=1,
        world_size=world,
        spmd_backend=args.backend,
    )
    dist_utils.set_spmd_backend(args.backend)

    # Build the job config, tweak for a small deterministic run. --no_flex
    # selects the SDPA debug config (ring CP) instead of the flex one.
    if args.no_flex:
        cfg = (
            transformers_modeling_backend_debugmodel_moe()
            if args.moe
            else transformers_modeling_backend_debugmodel()
        )
    else:
        cfg = (
            transformers_modeling_backend_debugmodel_moe_flex()
            if args.moe
            else transformers_modeling_backend_debugmodel_flex()
        )
    cfg.hf_model = args.hf_model
    cfg.training.seq_len = args.seq_len
    cfg.training.local_batch_size = args.bs
    # fp32 compute so any CP discrepancy isn't masked by bf16 FSDP mixed precision.
    cfg.training.mixed_precision_param = "float32"
    cfg.parallelism.context_parallel_degree = cp
    cfg.parallelism.spmd_backend = args.backend
    cfg.debug.seed = 42
    cfg.debug.deterministic = True

    def build_model(swap_moe=False):
        model_config = cfg.model_spec.model
        model_config.update_from_config(config=cfg)
        with torch.device(device):
            m = model_config.build()
        m.to(device)
        torch.manual_seed(42)
        m.init_weights(buffer_device=device)
        if swap_moe:
            # The reference is not parallelized, so the titan MoE swap (which
            # runs inside parallelize for the CP model) would not apply. Swap it
            # here too so both sides use the identical MoE implementation; both
            # were seeded identically, so the swap is deterministic.
            from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
                build_and_swap_native_moe,
            )

            build_and_swap_native_moe(m, parallel_dims)
            m.to(device)  # swap builds native experts on CPU; move them back
        return m, model_config

    # --- Reference: full-sequence forward, no parallelism ---
    ref_model, _ = build_model(swap_moe=args.moe)
    ref_model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (args.bs, args.seq_len), device=device)
    positions = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.bs, -1)
        .contiguous()
    )
    full_mask = ref_model.get_attention_masks(positions)
    with torch.no_grad():
        ref_logits = ref_model(
            input_ids, positions=positions, attention_masks=full_mask
        )
    del ref_model
    torch.cuda.empty_cache()

    # --- CP model: same deterministic init (seed 42) built before parallelize,
    # so weights match the reference; parallelize preserves values. ---
    cp_model, model_config = build_model()
    from torchtitan.experiments.transformers_modeling_backend.parallelize import (
        parallelize_hf_transformers,
    )

    parallelize_hf_transformers(
        cp_model,
        parallel_dims=parallel_dims,
        training=cfg.training,
        parallelism=cfg.parallelism,
        compile_config=CompileConfig(),
        ac_config=None,
        dump_folder="/tmp/flex_cp_spike",
    )
    cp_model.eval()

    # Shard input / positions / mask on the sequence axis (trainer's role).
    # A global-index tensor rides along so we can undo any load-balancer
    # permutation when reconstructing full logits for the comparison.
    cp_mesh = parallel_dims.get_mesh("cp")
    full_mask_cp = cp_model.get_attention_masks(positions)
    gidx = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.bs, -1)
        .contiguous()
    )
    (loc_input, loc_pos, loc_gidx), loc_mask = cp_shard(
        cp_mesh,
        (input_ids, positions, gidx),
        full_mask_cp,
        load_balancer_type=balancer,
    )
    _fm = tuple(full_mask_cp.shape) if full_mask_cp is not None else None
    _lm = tuple(loc_mask.shape) if loc_mask is not None else None
    print(
        f"[rank {rank}] balancer={args.balancer} loc_input={tuple(loc_input.shape)} "
        f"full_mask={_fm} loc_mask={_lm}"
    )

    with torch.no_grad():
        loc_logits = cp_model(loc_input, positions=loc_pos, attention_masks=loc_mask)
    loc_logits = (
        loc_logits.to_local() if hasattr(loc_logits, "to_local") else loc_logits
    )

    # Reconstruct full logits in global order via all-gather + index scatter.
    gathered_logits = [torch.empty_like(loc_logits) for _ in range(cp)]
    gathered_gidx = [torch.empty_like(loc_gidx) for _ in range(cp)]
    dist.all_gather(gathered_logits, loc_logits.contiguous())
    dist.all_gather(gathered_gidx, loc_gidx.contiguous())
    full = torch.zeros_like(ref_logits)
    for lg, gi in zip(gathered_logits, gathered_gidx):
        full[:, gi[0].long(), :] = lg.float()

    max_abs = (full.float() - ref_logits.float()).abs().max().item()
    ref_scale = ref_logits.float().abs().max().item()
    rel = max_abs / max(ref_scale, 1e-9)
    # Dense flex+CP is bit-exact up to fp32 noise. MoE adds grouped_mm f32
    # accumulation-order differences (documented in numerical_equivalence.py) plus
    # FSDP expert-shard reduction order, so it uses a looser tolerance; the
    # residual stays flat across CP degrees (a real CP bug would give rel ~O(1)).
    tol = 2e-3 if args.moe else 1e-4
    passed = rel < tol
    if rank == 0:
        verdict = "PASS" if passed else "FAIL"
        print(
            f"\n==== FLEX+CP {verdict} (backend={args.backend} balancer={args.balancer}): "
            f"max_abs_diff={max_abs:.3e} ref_scale={ref_scale:.3e} rel={rel:.3e} ===="
        )
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
