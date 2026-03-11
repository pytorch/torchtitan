# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone train step capture prototype.

Captures a full train step (fwd + bwd + optimizer) as a single FX graph,
using torchtitan's real model configs and optimizer hyperparameters.
Verifies bitwise matching between eager and traced execution.

Three modes:
  - eager:           Standard PyTorch training (ground truth)
  - functional-eager: Functional ops run eagerly (debug/verification)
  - traced:          Functional ops captured via make_fx

Usage:
  python -m torchtitan.experiments.compiler_toolkit.prototypes.train_prototype --model toy --steps 10
  python -m torchtitan.experiments.compiler_toolkit.prototypes.train_prototype --model llama3 --steps 10 --device cuda
"""

import argparse
import functools
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx


# ---------------------------------------------------------------------------
# ToyMLP (reused from trace_train_step.py)
# ---------------------------------------------------------------------------


class ToyMLP(nn.Module):
    def __init__(self, dim_in: int = 64, dim_hidden: int = 128, dim_out: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Functional Adam update (all out-of-place, traceable)
# ---------------------------------------------------------------------------


def functional_adam_update(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    step_t: torch.Tensor,
    lr_t: torch.Tensor,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    torch.Tensor,
]:
    """Pure-functional AdamW. lr_t is a scalar tensor for traceability."""
    new_step = step_t + 1.0

    new_params = []
    new_exp_avgs = []
    new_exp_avg_sqs = []

    for p, g, ea, eas in zip(params, grads, exp_avgs, exp_avg_sqs):
        # Decoupled weight decay
        p_decayed = p * (1.0 - lr_t * weight_decay)

        # Moment updates
        new_ea = beta1 * ea + (1.0 - beta1) * g
        new_eas_val = beta2 * eas + (1.0 - beta2) * g * g

        # Bias correction
        bias_correction1 = 1.0 - torch.pow(
            torch.tensor(beta1, device=step_t.device), new_step
        )
        bias_correction2 = 1.0 - torch.pow(
            torch.tensor(beta2, device=step_t.device), new_step
        )
        step_size = lr_t / bias_correction1
        denom = (new_eas_val.sqrt() / bias_correction2.sqrt()) + eps

        # Parameter update
        new_p = p_decayed - step_size * new_ea / denom

        new_params.append(new_p)
        new_exp_avgs.append(new_ea)
        new_exp_avg_sqs.append(new_eas_val)

    return new_params, new_exp_avgs, new_exp_avg_sqs, new_step


# ---------------------------------------------------------------------------
# Functional grad clipping (out-of-place)
# ---------------------------------------------------------------------------


def functional_grad_clip(
    grads: List[torch.Tensor], max_norm: float
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Out-of-place L2 norm clipping."""
    total_norm = torch.sqrt(
        torch.stack([g.pow(2).sum() for g in grads]).sum()
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    clipped = [g * clip_coef for g in grads]
    return clipped, total_norm


# ---------------------------------------------------------------------------
# LR schedule (standalone replication of torchtitan's linear_warmup_stable_decay)
# ---------------------------------------------------------------------------


def compute_lr(
    step: int,
    base_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    decay_type: str = "linear",
    min_lr_factor: float = 0.0,
) -> float:
    """
    Compute LR at a given step. Matches LambdaLR convention (0-indexed step).
    Replicates torchtitan's linear_warmup_stable_decay from lr_scheduler.py.
    """
    warmup_stable_steps = warmup_steps + stable_steps
    if step < warmup_steps:
        # Linear warmup (0-indexed, hence +1)
        curr_adjustment = float((step + 1) / warmup_steps)
    elif step < warmup_stable_steps:
        curr_adjustment = 1.0
    else:
        # Decay phase (0-indexed, hence +1)
        progress = float((step + 1) - warmup_stable_steps) / decay_steps
        if decay_type == "linear":
            curr_adjustment = 1 - progress
        elif decay_type == "sqrt":
            curr_adjustment = 1 - math.sqrt(progress)
        elif decay_type == "cosine":
            curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")
        curr_adjustment = min_lr_factor + (1 - min_lr_factor) * curr_adjustment

    return base_lr * curr_adjustment


# ---------------------------------------------------------------------------
# Optimizer state helpers (reused from trace_train_step.py)
# ---------------------------------------------------------------------------


def extract_optimizer_state(
    optimizer: torch.optim.Optimizer,
    params: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Extract (exp_avgs, exp_avg_sqs, steps) from a pre-initialized Adam optimizer."""
    exp_avgs = []
    exp_avg_sqs = []
    steps = []
    for p in params:
        state = optimizer.state[p]
        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])
        steps.append(state["step"])
    return exp_avgs, exp_avg_sqs, steps


def force_optimizer_state_init(optimizer: torch.optim.Optimizer) -> None:
    """Force lazy optimizer state allocation with correct initial values."""
    import unittest.mock

    if not hasattr(type(optimizer), "_init_group"):
        raise NotImplementedError(
            f"{type(optimizer).__name__} does not implement _init_group."
        )

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    saved_params = {
        p: p.data.clone()
        for group in optimizer.param_groups
        for p in group["params"]
    }

    orig_init_group = type(optimizer)._init_group
    captured_init_state = {}

    def capturing_init_group(self, *args, **kwargs):
        result = orig_init_group(self, *args, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                if p not in captured_init_state and p in self.state:
                    captured_init_state[p] = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in self.state[p].items()
                    }
        return result

    with unittest.mock.patch.object(
        type(optimizer), "_init_group", capturing_init_group
    ):
        optimizer.step()

    for p, init_state in captured_init_state.items():
        for k, v in init_state.items():
            if isinstance(v, torch.Tensor):
                optimizer.state[p][k].copy_(v)
            else:
                optimizer.state[p][k] = v

    for p, data in saved_params.items():
        p.data.copy_(data)

    optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Build train step function (factory for traced path)
# ---------------------------------------------------------------------------


def unpack_train_step_args(flat_args, num_params, num_buffers, has_global_valid_tokens):
    """
    Unpack the flat argument tuple used by train step functions.

    The flat layout is:
      [*params, *buffers, *exp_avgs, *exp_avg_sqs, step, lr, input, labels, [gvt]]

    Returns a dict with keys: params, buffers, exp_avgs, exp_avg_sqs,
    step, lr, input, labels, global_valid_tokens.
    """
    def take(n):
        nonlocal offset
        result = list(flat_args[offset : offset + n])
        offset += n
        return result

    def take1():
        nonlocal offset
        result = flat_args[offset]
        offset += 1
        return result

    offset = 0
    result = {
        "params": take(num_params),
        "buffers": take(num_buffers),
        "exp_avgs": take(num_params),
        "exp_avg_sqs": take(num_params),
        "step": take1(),
        "lr": take1(),
        "input": take1(),
        "labels": take1(),
        "global_valid_tokens": take1() if has_global_valid_tokens else None,
    }
    return result


def build_train_step_fn(
    model: nn.Module,
    param_names: List[str],
    buffer_names: List[str],
    loss_fn,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_norm: float,
    has_global_valid_tokens: bool = False,
):
    """
    Factory that returns a flat function suitable for make_fx tracing.

    Signature of returned fn:
      fn(*params, *buffers, *exp_avgs, *exp_avg_sqs, step_t, lr_t, input, labels, [global_valid_tokens])
      -> (loss, *new_params, *new_eas, *new_eass, new_step)
    """
    num_params = len(param_names)
    num_buffers = len(buffer_names)

    def train_step_fn(*flat_args):
        args = unpack_train_step_args(
            flat_args, num_params, num_buffers, has_global_valid_tokens
        )
        params = args["params"]
        buffers = args["buffers"]

        # Build params+buffers dict for functional_call
        params_buffers_dict = dict(zip(param_names, params))
        params_buffers_dict.update(zip(buffer_names, buffers))

        # Forward via functional_call
        logits = torch.func.functional_call(model, params_buffers_dict, (args["input"],))

        # Loss
        loss_sum = loss_fn(logits, args["labels"])
        gvt = args["global_valid_tokens"]
        loss = loss_sum / gvt if gvt is not None else loss_sum

        # Backward via autograd.grad
        grads = list(torch.autograd.grad(loss, params))

        # Functional grad clipping
        clipped_grads, _total_norm = functional_grad_clip(grads, max_norm)

        # Functional AdamW
        new_params, new_eas, new_eass, new_step = functional_adam_update(
            params,
            clipped_grads,
            args["exp_avgs"],
            args["exp_avg_sqs"],
            args["step"],
            args["lr"],
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )

        return (loss, *new_params, *new_eas, *new_eass, new_step)

    return train_step_fn


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 2
    decay_ratio: float = 0.8
    decay_type: str = "linear"
    min_lr_factor: float = 0.0
    max_norm: float = 1.0
    seq_len: int = 2048
    local_batch_size: int = 8
    total_steps: int = 10
    has_global_valid_tokens: bool = False


# ---------------------------------------------------------------------------
# Setup: Toy model
# ---------------------------------------------------------------------------


def setup_toy_config(device: torch.device, seed: int = 42):
    """Set up ToyMLP + fake data for fast iteration."""
    torch.manual_seed(seed)

    model = ToyMLP(64, 128, 64).to(device)

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

    # Pregenerate fixed data for reproducibility
    torch.manual_seed(seed + 1)
    data = []
    for _ in range(config.total_steps):
        x = torch.randn(8, 64, device=device)
        labels = torch.randn(8, 64, device=device)
        data.append((x, labels))

    return model, loss_fn, config, data


# ---------------------------------------------------------------------------
# Setup: Llama3 debugmodel
# ---------------------------------------------------------------------------


def setup_llama3_config(device: torch.device, seed: int = 42):
    """Set up llama3 debugmodel + c4_test data."""
    import dataclasses as _dc

    from torchtitan.components.loss import cross_entropy_loss
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
    from torchtitan.models.llama3 import llama3_configs

    torch.manual_seed(seed)

    # Get model config and sync RoPE max_seq_len to training seq_len
    seq_len = 2048
    model_config = llama3_configs["debugmodel"]
    model_config.rope = _dc.replace(model_config.rope, max_seq_len=seq_len)

    # Build model on device
    model = model_config.build().to(device)
    model.init_weights()

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
    tokenizer = HuggingFaceTokenizer(tokenizer_path="./tests/assets/tokenizer")
    dl_config = HuggingFaceTextDataLoader.Config(dataset="c4_test", infinite=True)
    dataloader = HuggingFaceTextDataLoader(
        dl_config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=config.local_batch_size,
    )

    # Preload data for reproducibility
    data = []
    data_iter = iter(dataloader)
    for _ in range(config.total_steps):
        input_dict, labels = next(data_iter)
        tokens = input_dict["input"].to(device)
        labels = labels.to(device)
        global_valid_tokens = (labels != -100).sum().float()
        data.append((tokens, labels, global_valid_tokens))

    def loss_fn(pred, labels):
        return cross_entropy_loss(pred, labels)

    return model, loss_fn, config, data


# ---------------------------------------------------------------------------
# Eager reference (standard PyTorch training - ground truth)
# ---------------------------------------------------------------------------


def run_eager(model, loss_fn, config, data, device, *, fused_optimizer=None):
    """
    Standard PyTorch training loop: model(x), loss.backward(),
    clip_grad_norm_, optimizer.step(). This is the ground truth.

    Works with both regular and distributed (simple_fsdp) models — the
    training loop is the same; distributed ops are triggered by the model's
    parametrizations.

    Args:
        fused_optimizer: Whether to use fused AdamW. Default: True on CUDA,
            False otherwise. Set to False for DTensor params.
    """
    import copy

    # Deep copy model so we don't mutate the original
    model = copy.deepcopy(model)
    model.train()

    # Compute LR schedule parameters (same as torchtitan)
    total_steps = config.total_steps
    warmup_steps = config.warmup_steps
    decay_steps = round(total_steps * config.decay_ratio)
    if warmup_steps + decay_steps > total_steps:
        decay_steps = total_steps - warmup_steps
    # +1 virtual last step to prevent LR dropping to 0
    stable_steps = total_steps + 1 - warmup_steps - decay_steps

    if fused_optimizer is None:
        fused_optimizer = device.type == "cuda"

    # Standard PyTorch optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=fused_optimizer,
    )

    # LR scheduler matching torchtitan's schedule
    lr_lambda = functools.partial(
        _lr_lambda,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        decay_type=config.decay_type,
        min_lr_factor=config.min_lr_factor,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []
    for step in range(config.total_steps):
        optimizer.zero_grad()

        if config.has_global_valid_tokens:
            tokens, labels, global_valid_tokens = data[step]
        else:
            tokens, labels = data[step]
            global_valid_tokens = None

        # Forward
        pred = model(tokens)

        # Loss
        loss_sum = loss_fn(pred, labels)
        if global_valid_tokens is not None:
            loss = loss_sum / global_valid_tokens
        else:
            loss = loss_sum

        # Backward
        loss.backward()

        # Grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        losses.append(loss.detach().item())

    return losses


def _lr_lambda(
    current_step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    decay_type: str,
    min_lr_factor: float,
) -> float:
    """LambdaLR-compatible LR multiplier. Matches torchtitan's schedule."""
    warmup_stable_steps = warmup_steps + stable_steps
    if current_step < warmup_steps:
        current_step += 1
        return float(current_step / warmup_steps)
    elif current_step < warmup_stable_steps:
        return 1.0
    else:
        current_step += 1
        progress = float(current_step - warmup_stable_steps) / decay_steps
        if decay_type == "linear":
            curr_adjustment = 1 - progress
        elif decay_type == "sqrt":
            curr_adjustment = 1 - math.sqrt(progress)
        elif decay_type == "cosine":
            curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")
        return min_lr_factor + (1 - min_lr_factor) * curr_adjustment


# ---------------------------------------------------------------------------
# Functional-eager (debug/verification, NOT ground truth)
# ---------------------------------------------------------------------------


def run_functional_eager(model, loss_fn, config, data, device):
    """
    Same functional ops as traced path but run eagerly.
    Useful as intermediate debug step to isolate tracing issues.
    """
    import copy

    model_ref = copy.deepcopy(model)
    model_ref.eval()  # We use functional_call, don't need train mode on ref

    param_names = [n for n, _ in model_ref.named_parameters()]
    buffer_names = [n for n, _ in model_ref.named_buffers()]

    # Extract initial params and buffers
    params = [p.detach().clone().requires_grad_(True) for p in model_ref.parameters()]
    buffers = [b.detach().clone() for b in model_ref.buffers()]

    # Initialize optimizer state (zeros)
    exp_avgs = [torch.zeros_like(p) for p in params]
    exp_avg_sqs = [torch.zeros_like(p) for p in params]
    step_t = torch.tensor(0.0, device=device)

    # Compute LR schedule parameters
    total_steps = config.total_steps
    warmup_steps = config.warmup_steps
    decay_steps = round(total_steps * config.decay_ratio)
    if warmup_steps + decay_steps > total_steps:
        decay_steps = total_steps - warmup_steps
    stable_steps = total_steps + 1 - warmup_steps - decay_steps

    train_step_fn = build_train_step_fn(
        model_ref,
        param_names,
        buffer_names,
        loss_fn,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        max_norm=config.max_norm,
        has_global_valid_tokens=config.has_global_valid_tokens,
    )

    num_params = len(params)
    losses = []

    for step in range(config.total_steps):
        lr = compute_lr(
            step, config.lr, warmup_steps, stable_steps, decay_steps,
            config.decay_type, config.min_lr_factor,
        )
        lr_t = torch.tensor(lr, device=device)

        # Build flat input
        if config.has_global_valid_tokens:
            tokens, labels, gvt = data[step]
            flat_args = (*params, *buffers, *exp_avgs, *exp_avg_sqs, step_t, lr_t, tokens, labels, gvt)
        else:
            tokens, labels = data[step]
            flat_args = (*params, *buffers, *exp_avgs, *exp_avg_sqs, step_t, lr_t, tokens, labels)

        # Run eagerly
        out = train_step_fn(*flat_args)

        # Unpack
        loss = out[0]
        new_params = list(out[1 : 1 + num_params])
        new_eas = list(out[1 + num_params : 1 + 2 * num_params])
        new_eass = list(out[1 + 2 * num_params : 1 + 3 * num_params])
        new_step = out[1 + 3 * num_params]

        losses.append(loss.detach().item())

        # Update state for next step
        params = [p.detach().clone().requires_grad_(True) for p in new_params]
        exp_avgs = [ea.detach().clone() for ea in new_eas]
        exp_avg_sqs = [eas.detach().clone() for eas in new_eass]
        step_t = new_step.detach().clone()

    return losses


# ---------------------------------------------------------------------------
# Traced (functional ops captured via make_fx)
# ---------------------------------------------------------------------------


def run_traced(model, loss_fn, config, data, device):
    """
    Functional ops captured via make_fx, then replayed as a GraphModule.
    """
    import copy

    model_ref = copy.deepcopy(model)
    model_ref.eval()

    param_names = [n for n, _ in model_ref.named_parameters()]
    buffer_names = [n for n, _ in model_ref.named_buffers()]

    params = [p.detach().clone().requires_grad_(True) for p in model_ref.parameters()]
    buffers = [b.detach().clone() for b in model_ref.buffers()]

    exp_avgs = [torch.zeros_like(p) for p in params]
    exp_avg_sqs = [torch.zeros_like(p) for p in params]
    step_t = torch.tensor(0.0, device=device)

    # Compute LR schedule parameters
    total_steps = config.total_steps
    warmup_steps = config.warmup_steps
    decay_steps = round(total_steps * config.decay_ratio)
    if warmup_steps + decay_steps > total_steps:
        decay_steps = total_steps - warmup_steps
    stable_steps = total_steps + 1 - warmup_steps - decay_steps

    train_step_fn = build_train_step_fn(
        model_ref,
        param_names,
        buffer_names,
        loss_fn,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        max_norm=config.max_norm,
        has_global_valid_tokens=config.has_global_valid_tokens,
    )

    num_params = len(params)
    losses = []
    gm = None  # Will be traced on first step

    for step in range(config.total_steps):
        lr = compute_lr(
            step, config.lr, warmup_steps, stable_steps, decay_steps,
            config.decay_type, config.min_lr_factor,
        )
        lr_t = torch.tensor(lr, device=device)

        # Build flat input
        if config.has_global_valid_tokens:
            tokens, labels, gvt = data[step]
            flat_args = (*params, *buffers, *exp_avgs, *exp_avg_sqs, step_t, lr_t, tokens, labels, gvt)
        else:
            tokens, labels = data[step]
            flat_args = (*params, *buffers, *exp_avgs, *exp_avg_sqs, step_t, lr_t, tokens, labels)

        # Trace on first step, reuse GraphModule thereafter
        if gm is None:
            print("  Tracing with make_fx (tracing_mode='real')...")
            gm = make_fx(train_step_fn, tracing_mode="real")(*flat_args)
            print(f"  Graph node count: {len(list(gm.graph.nodes))}")

        # Run the traced graph
        out = gm(*flat_args)

        # Unpack
        loss = out[0]
        new_params = list(out[1 : 1 + num_params])
        new_eas = list(out[1 + num_params : 1 + 2 * num_params])
        new_eass = list(out[1 + 2 * num_params : 1 + 3 * num_params])
        new_step = out[1 + 3 * num_params]

        losses.append(loss.detach().item())

        # Update state for next step
        params = [p.detach().clone().requires_grad_(True) for p in new_params]
        exp_avgs = [ea.detach().clone() for ea in new_eas]
        exp_avg_sqs = [eas.detach().clone() for eas in new_eass]
        step_t = new_step.detach().clone()

    return losses, gm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train step capture prototype: eager vs traced comparison"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["toy", "llama3"],
        default="toy",
        help="Model to use (toy or llama3)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
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
        help="Skip eager reference (useful for debugging traced path only)",
    )
    parser.add_argument(
        "--include-functional-eager",
        action="store_true",
        help="Include functional-eager intermediate verification",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps}")
    print()

    # Setup
    if args.model == "toy":
        model, loss_fn, config, data = setup_toy_config(device, args.seed)
    elif args.model == "llama3":
        model, loss_fn, config, data = setup_llama3_config(device, args.seed)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config.total_steps = args.steps

    # Ensure we have enough data
    assert len(data) >= args.steps, (
        f"Not enough data: have {len(data)}, need {args.steps}"
    )

    # --- Run eager reference ---
    eager_losses = None
    if not args.skip_eager:
        print("=" * 70)
        print("Eager reference (standard PyTorch training)")
        print("=" * 70)
        eager_losses = run_eager(model, loss_fn, config, data, device)
        for i, loss in enumerate(eager_losses):
            print(f"  Step {i:3d}: loss = {loss:.6f}")
        print()

    # --- Run functional-eager (optional debug) ---
    func_eager_losses = None
    if args.include_functional_eager:
        print("=" * 70)
        print("Functional-eager (debug/verification)")
        print("=" * 70)
        func_eager_losses = run_functional_eager(model, loss_fn, config, data, device)
        for i, loss in enumerate(func_eager_losses):
            print(f"  Step {i:3d}: loss = {loss:.6f}")
        print()

    # --- Run traced ---
    print("=" * 70)
    print("Traced (make_fx capture)")
    print("=" * 70)
    traced_losses, _traced_gm = run_traced(model, loss_fn, config, data, device)
    for i, loss in enumerate(traced_losses):
        print(f"  Step {i:3d}: loss = {loss:.6f}")
    print()

    # --- Verification ---
    print("=" * 70)
    print("Verification")
    print("=" * 70)

    # Functional-eager vs traced (should be bitwise match)
    if func_eager_losses is not None:
        print("\nFunctional-eager vs Traced (expect bitwise match):")
        all_pass = True
        for i in range(args.steps):
            diff = abs(func_eager_losses[i] - traced_losses[i])
            status = "PASS" if diff == 0.0 else "FAIL"
            if diff != 0.0:
                all_pass = False
            print(f"  Step {i:3d}: func_eager={func_eager_losses[i]:.6f}  traced={traced_losses[i]:.6f}  diff={diff:.2e}  {status}")
        print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # Eager vs traced (may have numerical differences due to different Adam impl)
    if eager_losses is not None:
        print("\nEager (ground truth) vs Traced:")
        all_close = True
        for i in range(args.steps):
            diff = abs(eager_losses[i] - traced_losses[i])
            # Use a tolerance: eager uses fused/in-place Adam, traced uses functional
            status = "MATCH" if diff < 1e-5 else "DIFF"
            if diff >= 1e-5:
                all_close = False
            print(f"  Step {i:3d}: eager={eager_losses[i]:.6f}  traced={traced_losses[i]:.6f}  diff={diff:.2e}  {status}")

        if all_close:
            print("  Overall: ALL MATCH (within 1e-5)")
        else:
            print("  Overall: NUMERICAL DIFFERENCES DETECTED")
            print("  (Expected: eager uses PyTorch's native Adam which may differ")
            print("   from the functional Adam implementation due to op ordering,")
            print("   lerp_, addcmul_ vs out-of-place equivalents, etc.)")

    # Convergence check
    print("\nConvergence check (traced):")
    if traced_losses[0] > traced_losses[-1]:
        print(f"  Loss decreased: {traced_losses[0]:.6f} -> {traced_losses[-1]:.6f}  CONVERGING")
    else:
        print(f"  Loss increased: {traced_losses[0]:.6f} -> {traced_losses[-1]:.6f}  NOT CONVERGING")


if __name__ == "__main__":
    main()
