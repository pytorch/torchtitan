# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed train step capture prototype.

Extends train_prototype.py to capture distributed operations (TP + FSDP via
simple_fsdp) in the FX graph. The captured graph includes _c10d_functional
collective ops (all-gather, reduce-scatter).

Three comm modes:
  - fake:         Fake PGs, collectives are no-ops. Single process, fast
                  iteration for graph structure verification.
  - local_tensor: Fake backend + LocalTensorMode. Single process, collectives
                  simulated locally for eager correctness on toy models.
  - real:         Real NCCL via torchrun for final validation on real GPUs.

Two paths:
  - eager:  simple_fsdp model + standard PyTorch training (ground truth)
  - traced: same simple_fsdp model, train step captured via make_fx

Usage:
  # Fake PGs (single process, fast iteration)
  python -m torchtitan.experiments.compiler_toolkit.prototypes.distributed_train_prototype \\
      --model toy --steps 5 --comm-mode fake --dp-degree 4

  # Fake PGs with TP + FSDP
  python -m torchtitan.experiments.compiler_toolkit.prototypes.distributed_train_prototype \\
      --model llama3 --steps 5 --comm-mode fake --tp-degree 2 --dp-degree 2

  # LocalTensor for eager correctness (toy only)
  python -m torchtitan.experiments.compiler_toolkit.prototypes.distributed_train_prototype \\
      --model toy --steps 5 --comm-mode local_tensor --dp-degree 4

  # Real multi-GPU validation
  torchrun --nproc_per_node=4 -m torchtitan.experiments.compiler_toolkit.prototypes.distributed_train_prototype \\
      --model llama3 --steps 10 --comm-mode real --tp-degree 2 --dp-degree 2
"""

import argparse
import dataclasses as _dc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.experiments.compiler_toolkit.prototypes.train_prototype import (
    run_eager,
    run_traced,
    ToyMLP,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def setup_distributed(
    comm_mode: str,
    tp_degree: int,
    dp_degree: int,
    device_type: str = "cuda",
):
    """
    Initialize process groups and build ParallelDims / device meshes.

    Args:
        comm_mode: One of "fake", "local_tensor", "real".
        tp_degree: Tensor parallelism degree.
        dp_degree: Data parallelism (FSDP shard) degree.
        device_type: Device type for mesh creation.

    Returns:
        parallel_dims: ParallelDims with meshes built.
        local_tensor_mode: The LocalTensorMode context (entered) if
            comm_mode == "local_tensor", else None. Caller should
            __exit__ when done.
    """
    from torchtitan.distributed import ParallelDims

    world_size = tp_degree * dp_degree
    local_tensor_mode = None

    if comm_mode == "fake":
        torch.distributed.init_process_group(
            "fake", rank=0, world_size=world_size
        )
    elif comm_mode == "local_tensor":
        torch.distributed.init_process_group(
            "fake", rank=0, world_size=world_size
        )
        from torch.distributed._local_tensor import LocalTensorMode

        local_tensor_mode = LocalTensorMode(world_size)
        local_tensor_mode.__enter__()
    elif comm_mode == "real":
        torch.distributed.init_process_group(backend="nccl")
        actual_ws = torch.distributed.get_world_size()
        assert actual_ws == world_size, (
            f"torchrun world_size ({actual_ws}) != tp_degree*dp_degree ({world_size})"
        )
    else:
        raise ValueError(f"Unknown comm_mode: {comm_mode}")

    parallel_dims = ParallelDims(
        dp_shard=dp_degree,
        dp_replicate=1,
        cp=1,
        tp=tp_degree,
        pp=1,
        ep=1,
        etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    return parallel_dims, local_tensor_mode


def cleanup_distributed(local_tensor_mode=None):
    """Tear down process groups and exit LocalTensorMode if active."""
    if local_tensor_mode is not None:
        local_tensor_mode.__exit__(None, None, None)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Model setup with distributed parallelisms
# ---------------------------------------------------------------------------


def setup_distributed_model(
    model: nn.Module,
    parallel_dims,
    device: torch.device,
    *,
    apply_tp_fn=None,
    on_meta: bool = True,
):
    """
    Apply parallelisms to a model in the standard order:
    1. TP (if tp_degree > 1 and apply_tp_fn provided)
    2. simple_fsdp data_parallel (if dp_shard > 1)
    3. Materialize from meta device + init_weights

    Args:
        model: The model (may be on meta device).
        parallel_dims: ParallelDims with meshes built.
        device: Target device for materialization.
        apply_tp_fn: Callable to apply TP (e.g., apply_tp from llama3).
        on_meta: Whether model is on meta device and needs to_empty().
    """
    from torchtitan.experiments.simple_fsdp.simple_fsdp import (
        data_parallel,
        disable_active_parametrization,
    )

    # 1. TP
    if parallel_dims.tp_enabled and apply_tp_fn is not None:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp_fn(model, tp_mesh)

    # 2. FSDP via simple_fsdp
    if parallel_dims.dp_shard_enabled:
        dp_mesh = parallel_dims.get_mesh("fsdp")
        data_parallel(model, dp_mesh, mode="fully_shard")

    # 3. Materialize and init weights
    if on_meta:
        model.to_empty(device=device)

    with disable_active_parametrization():
        if hasattr(model, "init_weights"):
            model.init_weights()
        else:
            # Fallback for models without init_weights (e.g. ToyMLP)
            _default_init_weights(model)

    model.train()
    return model


def _default_init_weights(model: nn.Module):
    """Simple weight init for models that don't have init_weights()."""
    for name, param in model.named_parameters():
        if param.is_meta:
            raise RuntimeError(
                f"Parameter {name} is still on meta device after to_empty(). "
                "Ensure to_empty(device=...) was called before init."
            )
        # Parameters should already be materialized (zeros) from to_empty.
        # Apply a small random init for non-trivial training.
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif param.dim() == 1:
            nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Graph structure verification
# ---------------------------------------------------------------------------


def verify_graph_structure(gm: torch.fx.GraphModule) -> dict:
    """
    Inspect a traced GraphModule for distributed collective ops.

    Returns a dict with counts of key ops.
    """
    counts = {
        "all_gather": 0,
        "reduce_scatter": 0,
        "all_reduce": 0,
        "wait": 0,
        "total_nodes": 0,
    }

    for node in gm.graph.nodes:
        counts["total_nodes"] += 1
        if node.op == "call_function":
            name = str(node.target)
            if "all_gather_into_tensor" in name:
                counts["all_gather"] += 1
            elif "reduce_scatter_tensor" in name:
                counts["reduce_scatter"] += 1
            elif "all_reduce" in name and "_c10d_functional" in name:
                counts["all_reduce"] += 1
            elif "wait_tensor" in name:
                counts["wait"] += 1

    return counts




# ---------------------------------------------------------------------------
# Setup: Toy model (distributed)
# ---------------------------------------------------------------------------


def setup_toy_distributed(parallel_dims, device, seed=42):
    """Set up distributed ToyMLP + fake data."""
    torch.manual_seed(seed)

    # Build on meta device, then parallelize, then materialize
    with torch.device("meta"):
        model = ToyMLP(64, 128, 64)

    model = setup_distributed_model(
        model,
        parallel_dims,
        device,
        apply_tp_fn=None,  # No TP for toy model
        on_meta=True,
    )

    def loss_fn(pred, labels):
        return F.mse_loss(pred, labels)

    config = TrainConfig(
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
        max_norm=1.0,
        total_steps=10,
        has_global_valid_tokens=False,
    )

    # Pregenerate fixed data
    torch.manual_seed(seed + 1)
    data = []
    for _ in range(config.total_steps):
        x = torch.randn(8, 64, device=device)
        labels = torch.randn(8, 64, device=device)
        data.append((x, labels))

    return model, loss_fn, config, data


# ---------------------------------------------------------------------------
# Setup: Llama3 debugmodel (distributed)
# ---------------------------------------------------------------------------


def setup_llama3_distributed(parallel_dims, device, seed=42):
    """Set up distributed llama3 debugmodel + c4_test data."""
    from torchtitan.components.loss import cross_entropy_loss
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.experiments.simple_fsdp.llama3 import _simple_fsdp_configs
    from torchtitan.experiments.simple_fsdp.llama3.model import SimpleFSDPLlama3Model
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from torchtitan.models.llama3.parallelize import apply_tp

    torch.manual_seed(seed)

    seq_len = 2048
    model_config = _simple_fsdp_configs["debugmodel"]
    model_config = _dc.replace(
        model_config, rope=_dc.replace(model_config.rope, max_seq_len=seq_len)
    )

    # Build on meta device
    with torch.device("meta"):
        model = SimpleFSDPLlama3Model(model_config)

    # Apply parallelisms: TP first, then FSDP
    def _apply_tp(m, tp_mesh):
        apply_tp(
            m, tp_mesh,
            loss_parallel=False,
            enable_float8_tensorwise_tp=False,
        )

    model = setup_distributed_model(
        model,
        parallel_dims,
        device,
        apply_tp_fn=_apply_tp if parallel_dims.tp_enabled else None,
        on_meta=True,
    )

    config = TrainConfig(
        lr=8e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
        max_norm=1.0,
        seq_len=seq_len,
        local_batch_size=8,
        total_steps=10,
        has_global_valid_tokens=True,
    )

    # Set up tokenizer and data loader
    dp_rank = 0
    dp_world_size = 1
    if parallel_dims.dp_shard_enabled:
        dp_mesh = parallel_dims.get_mesh("fsdp")
        dp_rank = dp_mesh.get_local_rank()
        dp_world_size = dp_mesh.size()

    tokenizer = HuggingFaceTokenizer(tokenizer_path="./tests/assets/tokenizer")
    dl_config = HuggingFaceTextDataLoader.Config(dataset="c4_test", infinite=True)
    dataloader = HuggingFaceTextDataLoader(
        dl_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=config.local_batch_size,
    )

    # Preload data for reproducibility
    data = []
    data_iter = iter(dataloader)
    for _ in range(config.total_steps):
        input_dict, labels_batch = next(data_iter)
        tokens = input_dict["input"].to(device)
        labels_batch = labels_batch.to(device)
        global_valid_tokens = (labels_batch != -100).sum().float()
        data.append((tokens, labels_batch, global_valid_tokens))

    def loss_fn(pred, labels):
        return cross_entropy_loss(pred, labels)

    return model, loss_fn, config, data


# ---------------------------------------------------------------------------
# Setup: DeepSeek V3 debugmodel (distributed)
# ---------------------------------------------------------------------------


def setup_deepseekv3_distributed(parallel_dims, device, seed=42):
    """Set up distributed DeepSeek V3 debugmodel (MoE + MLA) + c4_test data."""
    from torchtitan.components.loss import cross_entropy_loss
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.experiments.simple_fsdp.deepseek_v3 import _simple_fsdp_configs
    from torchtitan.experiments.simple_fsdp.deepseek_v3.model import (
        SimpleFSDPDeepSeekV3Model,
    )
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from torchtitan.models.deepseek_v3.parallelize import apply_non_moe_tp

    torch.manual_seed(seed)

    # Reduced seq_len and batch to fit make_fx replay in GPU memory
    # (MoE + MLA produce ~2x the intermediates of dense models)
    seq_len = 512
    local_batch_size = 2
    model_config = _simple_fsdp_configs["debugmodel"]
    # Sync RoPE max_seq_len and replicate update_from_config logic:
    # Attention.softmax_scale depends on rope_max_seq_len, rope_factor,
    # and rope_original_seq_len being synced into layer.attention.
    model_config = _dc.replace(
        model_config,
        rope=_dc.replace(model_config.rope, max_seq_len=seq_len),
        layer=_dc.replace(
            model_config.layer,
            attention=_dc.replace(
                model_config.layer.attention,
                rope_max_seq_len=seq_len,
                rope_factor=model_config.rope.rope_factor,
                rope_original_seq_len=model_config.rope.original_seq_len,
            ),
            # torch._grouped_mm requires CUDA SM90+; disable for CPU compat
            moe=_dc.replace(model_config.layer.moe, use_grouped_mm=False),
        ),
    )

    with torch.device("meta"):
        model = SimpleFSDPDeepSeekV3Model(model_config)

    # TP: non-MoE only (skip MoE EP/TP for prototype)
    def _apply_tp(m, tp_mesh):
        apply_non_moe_tp(
            m,
            tp_mesh,
            loss_parallel=False,
            enable_float8_tensorwise_tp=False,
            cp_enabled=False,
        )

    model = setup_distributed_model(
        model,
        parallel_dims,
        device,
        apply_tp_fn=_apply_tp if parallel_dims.tp_enabled else None,
        on_meta=True,
    )

    config = TrainConfig(
        lr=8e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
        max_norm=1.0,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        total_steps=10,
        has_global_valid_tokens=True,
    )

    dp_rank = 0
    dp_world_size = 1
    if parallel_dims.dp_shard_enabled:
        dp_mesh = parallel_dims.get_mesh("fsdp")
        dp_rank = dp_mesh.get_local_rank()
        dp_world_size = dp_mesh.size()

    tokenizer = HuggingFaceTokenizer(tokenizer_path="./tests/assets/tokenizer")
    dl_config = HuggingFaceTextDataLoader.Config(dataset="c4_test", infinite=True)
    dataloader = HuggingFaceTextDataLoader(
        dl_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
    )

    data = []
    data_iter = iter(dataloader)
    for _ in range(config.total_steps):
        input_dict, labels_batch = next(data_iter)
        tokens = input_dict["input"].to(device)
        labels_batch = labels_batch.to(device)
        global_valid_tokens = (labels_batch != -100).sum().float()
        data.append((tokens, labels_batch, global_valid_tokens))

    def loss_fn(pred, labels):
        return cross_entropy_loss(pred, labels)

    return model, loss_fn, config, data


# ---------------------------------------------------------------------------
# Setup: Qwen3 debugmodel (distributed)
# ---------------------------------------------------------------------------


def setup_qwen3_distributed(parallel_dims, device, seed=42):
    """Set up distributed Qwen3 debugmodel (dense, weight-tying) + c4_test data.

    No SimpleFSDP wrapper exists for Qwen3 — uses base Qwen3Model directly.
    setup_distributed_model() wraps init_weights() in disable_active_parametrization().
    FSDP only, no TP (Qwen3 has no standalone apply_tp).
    """
    from torchtitan.components.loss import cross_entropy_loss
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from torchtitan.models.qwen3 import qwen3_configs
    from torchtitan.models.qwen3.model import Qwen3Model

    torch.manual_seed(seed)

    # Reduced seq_len to fit make_fx replay in GPU memory
    seq_len = 512
    local_batch_size = 2
    model_config = qwen3_configs["debugmodel"]
    model_config = _dc.replace(
        model_config, rope=_dc.replace(model_config.rope, max_seq_len=seq_len)
    )

    with torch.device("meta"):
        model = Qwen3Model(model_config)

    model = setup_distributed_model(
        model,
        parallel_dims,
        device,
        apply_tp_fn=None,  # No standalone TP for Qwen3
        on_meta=True,
    )

    config = TrainConfig(
        lr=8e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
        max_norm=1.0,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        total_steps=10,
        has_global_valid_tokens=True,
    )

    dp_rank = 0
    dp_world_size = 1
    if parallel_dims.dp_shard_enabled:
        dp_mesh = parallel_dims.get_mesh("fsdp")
        dp_rank = dp_mesh.get_local_rank()
        dp_world_size = dp_mesh.size()

    tokenizer = HuggingFaceTokenizer(tokenizer_path="./tests/assets/tokenizer")
    dl_config = HuggingFaceTextDataLoader.Config(dataset="c4_test", infinite=True)
    dataloader = HuggingFaceTextDataLoader(
        dl_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
    )

    data = []
    data_iter = iter(dataloader)
    for _ in range(config.total_steps):
        input_dict, labels_batch = next(data_iter)
        tokens = input_dict["input"].to(device)
        labels_batch = labels_batch.to(device)
        global_valid_tokens = (labels_batch != -100).sum().float()
        data.append((tokens, labels_batch, global_valid_tokens))

    def loss_fn(pred, labels):
        return cross_entropy_loss(pred, labels)

    return model, loss_fn, config, data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Distributed train step capture prototype"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["toy", "llama3", "deepseek_v3", "qwen3"],
        default="toy",
        help="Model to use",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training steps",
    )
    parser.add_argument(
        "--comm-mode",
        type=str,
        choices=["fake", "local_tensor", "real"],
        default="fake",
        help="Communication mode",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=1,
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=1,
        help="Data parallelism (FSDP shard) degree. -1 = auto from world_size / tp_degree",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-eager",
        action="store_true",
        help="Skip eager reference",
    )
    parser.add_argument(
        "--skip-traced",
        action="store_true",
        help="Skip traced path",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    elif args.comm_mode == "real":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device_type = device.type

    # Auto dp-degree
    if args.dp_degree == -1:
        if args.comm_mode == "real":
            ws = int(os.environ.get("WORLD_SIZE", 1))
        else:
            ws = args.tp_degree  # default to 1 dp shard if not specified
        args.dp_degree = ws // args.tp_degree
        assert args.dp_degree >= 1, "Auto dp_degree < 1"

    world_size = args.tp_degree * args.dp_degree
    rank = int(os.environ.get("RANK", 0))

    print(f"{'=' * 70}")
    print(f"Distributed Train Step Prototype")
    print(f"{'=' * 70}")
    print(f"  Model:     {args.model}")
    print(f"  Comm mode: {args.comm_mode}")
    print(f"  TP degree: {args.tp_degree}")
    print(f"  DP degree: {args.dp_degree}")
    print(f"  World size: {world_size}")
    print(f"  Device:    {device}")
    print(f"  Steps:     {args.steps}")
    print(f"  Rank:      {rank}")
    print()

    # --- Setup distributed ---
    parallel_dims, local_tensor_mode = setup_distributed(
        args.comm_mode, args.tp_degree, args.dp_degree, device_type,
    )

    try:
        # --- Setup model ---
        if args.model == "toy":
            model, loss_fn, config, data = setup_toy_distributed(
                parallel_dims, device, args.seed,
            )
        elif args.model == "llama3":
            model, loss_fn, config, data = setup_llama3_distributed(
                parallel_dims, device, args.seed,
            )
        elif args.model == "deepseek_v3":
            model, loss_fn, config, data = setup_deepseekv3_distributed(
                parallel_dims, device, args.seed,
            )
        elif args.model == "qwen3":
            model, loss_fn, config, data = setup_qwen3_distributed(
                parallel_dims, device, args.seed,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        config.total_steps = args.steps
        assert len(data) >= args.steps, (
            f"Not enough data: have {len(data)}, need {args.steps}"
        )

        # --- Print model info ---
        num_params = sum(
            p.numel() for p in model.parameters()
        )
        print(f"Model parameters: {num_params:,}")
        # Show parameter shapes (DTensors have special repr)
        for name, p in model.named_parameters():
            shape_info = str(p.shape)
            dtype_info = str(p.dtype)
            is_dtensor = "DTensor" if hasattr(p, "_spec") else "Tensor"
            print(f"  {name}: {shape_info} ({dtype_info}, {is_dtensor})")
        print()

        # --- Run eager reference ---
        eager_losses = None
        if not args.skip_eager:
            print(f"{'=' * 70}")
            print("Eager reference (standard PyTorch + simple_fsdp)")
            print(f"{'=' * 70}")
            eager_losses = run_eager(
                model, loss_fn, config, data, device,
                fused_optimizer=False,  # fused AdamW doesn't support DTensor
            )
            for i, loss in enumerate(eager_losses):
                print(f"  Step {i:3d}: loss = {loss:.6f}")
            print()

        # --- Run traced ---
        traced_losses = None
        traced_gm = None
        if not args.skip_traced:
            print(f"{'=' * 70}")
            print("Traced (make_fx capture with distributed ops)")
            print(f"{'=' * 70}")
            traced_losses, traced_gm = run_traced(
                model, loss_fn, config, data, device
            )
            for i, loss in enumerate(traced_losses):
                print(f"  Step {i:3d}: loss = {loss:.6f}")
            print()

        # --- Verification ---
        print(f"{'=' * 70}")
        print("Verification")
        print(f"{'=' * 70}")

        # Eager vs traced comparison
        if eager_losses is not None and traced_losses is not None:
            print("\nEager vs Traced:")
            all_close = True
            for i in range(args.steps):
                diff = abs(eager_losses[i] - traced_losses[i])
                if args.comm_mode == "fake":
                    # With fake PGs, both paths go through no-op collectives
                    # so should be very close (or bitwise match)
                    tol = 1e-5
                else:
                    # Real/local_tensor may have small numerical diffs
                    tol = 1e-4
                status = "MATCH" if diff < tol else "DIFF"
                if diff >= tol:
                    all_close = False
                print(
                    f"  Step {i:3d}: eager={eager_losses[i]:.6f}  "
                    f"traced={traced_losses[i]:.6f}  diff={diff:.2e}  {status}"
                )

            if all_close:
                print(f"  Overall: ALL MATCH (within tolerance)")
            else:
                print(f"  Overall: NUMERICAL DIFFERENCES DETECTED")
                print(
                    "  (Expected: eager uses PyTorch's native Adam which may differ"
                )
                print(
                    "   from the functional Adam implementation due to op ordering.)"
                )

        # Convergence check
        if traced_losses is not None:
            print("\nConvergence check (traced):")
            if traced_losses[0] > traced_losses[-1]:
                print(
                    f"  Loss decreased: {traced_losses[0]:.6f} -> "
                    f"{traced_losses[-1]:.6f}  CONVERGING"
                )
            else:
                print(
                    f"  Loss increased: {traced_losses[0]:.6f} -> "
                    f"{traced_losses[-1]:.6f}  NOT CONVERGING"
                )

        # Graph structure summary
        if traced_gm is not None:
            print("\nGraph structure:")
            counts = verify_graph_structure(traced_gm)
            print(f"  Total nodes:        {counts['total_nodes']}")
            print(f"  All-gather ops:     {counts['all_gather']}")
            print(f"  Reduce-scatter ops: {counts['reduce_scatter']}")
            print(f"  All-reduce ops:     {counts['all_reduce']}")
            print(f"  Wait ops:           {counts['wait']}")

            has_collectives = (
                counts["all_gather"] > 0 or counts["reduce_scatter"] > 0
            )
            if has_collectives:
                print("  Distributed ops detected in graph: YES")
            else:
                print("  Distributed ops detected in graph: NO")
                if world_size > 1:
                    print(
                        "  WARNING: Expected distributed ops for world_size > 1"
                    )

    finally:
        cleanup_distributed(local_tensor_mode)


if __name__ == "__main__":
    main()
