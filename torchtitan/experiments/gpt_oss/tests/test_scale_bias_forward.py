# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run CMD:
# torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
#   --local-ranks-filter 0,1,2,3,4,5,6,7 \
#   torchtitan/experiments/gpt_oss/tests/test_scale_bias_forward.py

import os

LOG_INTERNAL = False
if LOG_INTERNAL:
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_LOGS"] = "+torch.distributed.tensor"
LOG_DEBUG = True
LOG_RANK = [0]

import torch
import torch.distributed as dist
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.utils._debug_mode import DebugMode
from torchtitan.experiments.gpt_oss.model.moe import ScaleBiasForward


def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, local_rank, device


def p_fmt(x):
    return f"Tensor[{x.placements}]" + (
        f", Grad[{x.grad.placements}]" if x.grad is not None else ", Grad[None]"
    )


def norm_fmt(x):
    if isinstance(x, DTensor):
        return f"Tensor[{x.full_tensor().norm().item():.6f}]" + (
            f", Grad[{x.grad.full_tensor().norm().item():.6f}]"
            if x.grad is not None
            else ", Grad[None]"
        )
    return f"Tensor[{x.norm().item():.6f}]" + (
        f", Grad[{x.grad.norm().item():.6f}]" if x.grad is not None else ", Grad[None]"
    )


M, N, O = 16, 32, 8


def linear_regular(rank, verbose=False):
    """Baseline: Regular tensor computation without any parallelism."""
    x = torch.arange(M * N, dtype=torch.float32).reshape(M, N).requires_grad_(True)
    x.retain_grad()
    w = torch.arange(N * O, dtype=torch.float32).reshape(N, O).requires_grad_(True)
    w.retain_grad()
    b = torch.arange(O, dtype=torch.float32, requires_grad=True)
    b.retain_grad()

    # Forward & backward
    z = x @ w
    z.retain_grad()
    o = z + b
    o.retain_grad()
    loss = o.sum()
    loss.retain_grad()
    loss.backward()

    if rank == 0 and verbose:
        print("=" * 50 + "\nregular_tensor\n" + "=" * 50)
        print(f"Loss: norm={norm_fmt(loss)}")
        print(f"Output: norm={norm_fmt(o)}")
        print(f" (z): norm={norm_fmt(z)}")
        print(f"Operands:\n (x): norm={norm_fmt(x)}")
        print(f" (w): norm={norm_fmt(w)}")
        print(f" (b): norm={norm_fmt(b)}")

    return x.grad.norm().item(), w.grad.norm().item(), b.grad.norm().item()


def linear_with_scale_bias_forward(verbose=False):
    """DTensor computation using ScaleBiasForward to handle Partial + Replicate."""
    rank, world_size, local_rank, device = init_distributed()
    x_grad_norm, w_grad_norm, b_grad_norm = linear_regular(rank, verbose=verbose)

    # Setup device mesh
    assert torch.cuda.is_available()
    mesh_device_type = "cuda"
    mesh = DeviceMesh(mesh_device_type, list(range(world_size)), mesh_dim_names=("tp",))

    # Build DTensors
    torch.manual_seed(0)
    x = torch.arange(M * N, dtype=torch.float32).reshape(M, N).requires_grad_(True)
    x_dtensor = distribute_tensor(x, mesh, [Shard(1)])
    x_dtensor.retain_grad()

    x_local = x_dtensor.to_local()
    x_local.retain_grad()

    w = torch.arange(N * O, dtype=torch.float32).reshape(N, O).requires_grad_(True)
    w_dtensor = distribute_tensor(w, mesh, [Shard(0)])
    w_dtensor.retain_grad()

    w_local = w_dtensor.to_local()
    w_local.retain_grad()

    b = torch.arange(O, dtype=torch.float32, requires_grad=True)
    b_dtensor = distribute_tensor(b, mesh, [Replicate()])
    b_dtensor.retain_grad()

    b_local = b_dtensor.to_local()
    b_local.retain_grad()

    # Forward computation

    # Convert to local tensors (loses placement info)
    z_local = x_local @ w_local  # the meaning of the tensor should be Partial()
    z_local.retain_grad()

    # Use ScaleBiasForward to scale bias in forward but not in backward
    b_scaled = ScaleBiasForward.apply(b_local, world_size)
    b_scaled.retain_grad()

    # # z_scaled = ScaleProductBackward.apply(z_local, world_size)
    # z_scaled = z_local
    # z_scaled.retain_grad()

    # Add scaled bias to partial sum
    o_local = z_local + b_scaled
    o_local.retain_grad()

    # Convert back to DTensor with Partial placement
    o_dtensor = DTensor.from_local(o_local, mesh, [Partial()])
    o_dtensor.retain_grad()

    loss_dtensor = o_dtensor.sum()
    loss_dtensor.retain_grad()

    with DebugMode(
        record_realtensor=False, record_tensor_attributes=["variable"]
    ) as debug_mode:
        loss_dtensor.backward()

    verbose_map = {
        "loss_dtensor": p_fmt(loss_dtensor) + f", norm={norm_fmt(loss_dtensor)}",
        "o_dtensor": p_fmt(o_dtensor) + f", norm={norm_fmt(o_dtensor)}",
        # "z_dtensor": p_fmt(z_dtensor) + f", norm={norm_fmt(z_dtensor)}",
        "x_dtensor": p_fmt(x_dtensor) + f", norm={norm_fmt(x_dtensor)}",
        "w_dtensor": p_fmt(w_dtensor) + f", norm={norm_fmt(w_dtensor)}",
        "b_dtensor": p_fmt(b_dtensor) + f", norm={norm_fmt(b_dtensor)}",
        "grad_norm": (
            f"x_grad_norm={x_dtensor.grad.full_tensor().norm().item()}, "
            f"w_grad_norm={w_dtensor.grad.full_tensor().norm().item()}, "
            f"b_grad_norm={b_dtensor.grad.full_tensor().norm()}"
        ),
    }
    if rank == 0 and verbose:
        print("=" * 50 + "\nManual: Partial() + Replicate() \n" + "=" * 50)
        print("Loss: " + verbose_map["loss_dtensor"])
        print("Output: " + verbose_map["o_dtensor"])
        # print(" (z): " + verbose_map["z_dtensor"])
        print("Operands:\n (x): " + verbose_map["x_dtensor"])
        print(" (w): " + verbose_map["w_dtensor"])
        print(" (b): " + verbose_map["b_dtensor"])
        print(
            f"Local Operands:\n (z_local): shape={z_local.shape}, norm={norm_fmt(z_local)}"
        )
        print(f" (b_local): shape={b_local.shape}, norm={norm_fmt(b_local)}")
        print("Grad Norms: " + verbose_map["grad_norm"])
    if rank == 0 and verbose:
        # print(f"{x_dtensor=} \n {x_dtensor.grad=}")
        print(
            "=" * 50
            + f"\nDebugMode[Rank-{rank} BWD]:\n"
            + "=" * 50
            + f"\n{debug_mode.debug_string()}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    linear_with_scale_bias_forward(verbose=True)
