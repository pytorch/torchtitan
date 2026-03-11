# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Prototype: Capturing a full train step (fwd + bwd + optimizer) in a single FX graph.

Four phases, selectable via --phase:
  Phase 1: make_fx + manual forward + manual backward + functional AdamW
  Phase 2: make_fx + functional_call + pre-initialized optimizer state
  Phase 3: aot_export_joint (safe fwd+bwd capture) + make_fx (optimizer wrapping)
  Phase 4: make_fx + native optimizer via _single_tensor_adam(capturable=True)

Run with:
  python -m torchtitan.experiments.compiler_toolkit.prototypes.trace_train_step --phase 1
  python -m torchtitan.experiments.compiler_toolkit.prototypes.trace_train_step --phase 2
  python -m torchtitan.experiments.compiler_toolkit.prototypes.trace_train_step --phase 3
  python -m torchtitan.experiments.compiler_toolkit.prototypes.trace_train_step --phase 4 --device cuda
"""

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx


# ---------------------------------------------------------------------------
# Shared: ToyMLP
# ---------------------------------------------------------------------------


class ToyMLP(nn.Module):
    def __init__(self, dim_in: int = 64, dim_hidden: int = 128, dim_out: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Shared: Functional AdamW update (pure tensor ops, all out-of-place)
# ---------------------------------------------------------------------------


def functional_adam_update(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    step_t: torch.Tensor,
    *,
    lr: float = 1e-3,
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
    """Pure-functional AdamW matching _single_tensor_adam. All out-of-place."""
    new_step = step_t + 1.0

    new_params = []
    new_exp_avgs = []
    new_exp_avg_sqs = []

    for p, g, ea, eas in zip(params, grads, exp_avgs, exp_avg_sqs):
        # Weight decay (decoupled)
        p_decayed = p * (1.0 - lr * weight_decay)

        # Moment updates
        new_ea = beta1 * ea + (1.0 - beta1) * g
        new_eas = beta2 * eas + (1.0 - beta2) * g * g

        # Bias correction
        bias_correction1 = 1.0 - torch.pow(torch.tensor(beta1, device=step_t.device), new_step)
        bias_correction2 = 1.0 - torch.pow(torch.tensor(beta2, device=step_t.device), new_step)
        step_size = lr / bias_correction1
        denom = (new_eas.sqrt() / bias_correction2.sqrt()) + eps

        # Parameter update
        new_p = p_decayed - step_size * new_ea / denom

        new_params.append(new_p)
        new_exp_avgs.append(new_ea)
        new_exp_avg_sqs.append(new_eas)

    return new_params, new_exp_avgs, new_exp_avg_sqs, new_step


# ---------------------------------------------------------------------------
# Shared: Extract optimizer state from a pre-initialized optimizer
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
    """Force lazy optimizer state allocation with correct initial values.

    PyTorch optimizers lazily initialize per-parameter state (exp_avg, step,
    etc.) on the first call to step().  For tracing, we need those tensors to
    exist *before* tracing begins so they can be passed as explicit graph
    inputs.

    A naive approach — running a dummy step then zeroing everything — fails
    because not all state starts at zero.  For example NAdam's ``mu_product``
    initializes to 1.0, ASGD's ``mu`` to 1.0 and ``eta`` to the learning
    rate, and Rprop's ``step_size`` to ``lr``.  A generic ``zero_()`` reset
    would corrupt these, producing wrong results on the first real step.

    Instead, we monkey-patch ``_init_group`` (the method every standard
    optimizer uses to allocate state) to snapshot the freshly-initialized
    values *before* the step math runs.  After the dummy step completes we
    restore both the parameter data and the captured initial state.

    Limitations:
    - Requires the optimizer to implement ``_init_group``.  All standard
      PyTorch optimizers do (Adam, AdamW, SGD, NAdam, RAdam, ASGD, Adadelta,
      Adagrad, Adamax, RMSprop, Rprop) except SparseAdam and LBFGS.
    - ``_init_group`` is a private API; its signature varies across
      optimizers.  We only intercept its entry/exit, not its arguments,
      so this is robust to signature changes.
    """
    import unittest.mock

    if not hasattr(type(optimizer), "_init_group"):
        raise NotImplementedError(
            f"{type(optimizer).__name__} does not implement _init_group. "
            "State must be initialized manually for this optimizer."
        )

    # Set zero grads so every param is processed by _init_group.
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

    # Save param data (the step will mutate params via weight decay etc.)
    saved_params = {p: p.data.clone() for group in optimizer.param_groups for p in group["params"]}

    # Intercept _init_group to snapshot state right after allocation,
    # before the step math runs.
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

    with unittest.mock.patch.object(type(optimizer), "_init_group", capturing_init_group):
        optimizer.step()

    # Restore initial state values and param data.
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
# Verification helper
# ---------------------------------------------------------------------------


def verify(traced_loss: torch.Tensor, eager_loss: torch.Tensor, phase: int):
    """Compare traced vs eager loss values."""
    print(f"\n--- Phase {phase} Verification ---")
    print(f"  Traced loss: {traced_loss.item():.6f}")
    print(f"  Eager  loss: {eager_loss.item():.6f}")
    diff = (traced_loss - eager_loss).abs().item()
    print(f"  Abs diff:    {diff:.2e}")
    if diff < 1e-5:
        print("  PASS")
    else:
        print("  FAIL (diff too large)")


# ===========================================================================
# Phase 1: make_fx + manual everything
# ===========================================================================


def run_phase1(device: torch.device):
    print("=" * 70)
    print("Phase 1: make_fx + manual forward + manual backward + functional AdamW")
    print("=" * 70)

    torch.manual_seed(42)
    model = ToyMLP().to(device)

    # Extract parameters as a flat list: [fc1.weight, fc1.bias, fc2.weight, fc2.bias]
    param_names = []
    param_tensors = []
    for name, p in model.named_parameters():
        param_names.append(name)
        param_tensors.append(p.detach().clone())

    num_params = len(param_tensors)

    # Initialize optimizer state (zeros)
    exp_avgs = [torch.zeros_like(p) for p in param_tensors]
    exp_avg_sqs = [torch.zeros_like(p) for p in param_tensors]
    step_t = torch.tensor(0.0, device=device)

    # Sample data
    x = torch.randn(8, 64, device=device)
    labels = torch.randn(8, 64, device=device)

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01

    # ----- Define the full train step as a flat function -----
    def train_step_fn(
        fc1_w, fc1_b, fc2_w, fc2_b,
        ea0, ea1, ea2, ea3,
        eas0, eas1, eas2, eas3,
        step_t,
        x, labels,
    ):
        params = [fc1_w, fc1_b, fc2_w, fc2_b]
        eas_list = [ea0, ea1, ea2, ea3]
        eass_list = [eas0, eas1, eas2, eas3]

        # Manual forward
        h = F.linear(x, fc1_w, fc1_b)
        h = F.relu(h)
        logits = F.linear(h, fc2_w, fc2_b)
        loss = F.mse_loss(logits, labels)

        # Backward via autograd.grad (make_fx can't trace .backward())
        grads = torch.autograd.grad(loss, params)

        # Functional AdamW
        new_params, new_eas, new_eass, new_step = functional_adam_update(
            params, list(grads), eas_list, eass_list, step_t,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd,
        )

        return (
            loss,
            *new_params,
            *new_eas,
            *new_eass,
            new_step,
        )

    # ----- Build flat inputs -----
    # Parameters need requires_grad for autograd.grad
    trace_params = [p.clone().requires_grad_(True) for p in param_tensors]
    trace_eas = [ea.clone() for ea in exp_avgs]
    trace_eass = [eas.clone() for eas in exp_avg_sqs]
    trace_step = step_t.clone()

    all_inputs = (
        *trace_params,
        *trace_eas,
        *trace_eass,
        trace_step,
        x.clone(), labels.clone(),
    )

    # ----- Trace with make_fx -----
    print("\nTracing with make_fx (tracing_mode='real')...")
    gm = make_fx(train_step_fn, tracing_mode="real")(*all_inputs)

    print("\n--- FX Graph (Phase 1) ---")
    gm.print_readable()
    print(f"\nNode count: {len(list(gm.graph.nodes))}")

    # ----- Run traced graph for correctness -----
    run_params = [p.clone().requires_grad_(True) for p in param_tensors]
    run_eas = [ea.clone() for ea in exp_avgs]
    run_eass = [eas.clone() for eas in exp_avg_sqs]
    run_inputs = (
        *run_params,
        *run_eas,
        *run_eass,
        step_t.clone(),
        x.clone(), labels.clone(),
    )
    traced_out = gm(*run_inputs)
    traced_loss = traced_out[0]

    # ----- Eager execution for comparison -----
    eager_params = [p.clone().requires_grad_(True) for p in param_tensors]
    eager_eas = [ea.clone() for ea in exp_avgs]
    eager_eass = [eas.clone() for eas in exp_avg_sqs]
    eager_inputs = (
        *eager_params,
        *eager_eas,
        *eager_eass,
        step_t.clone(),
        x.clone(), labels.clone(),
    )
    eager_out = train_step_fn(*eager_inputs)
    eager_loss = eager_out[0]

    verify(traced_loss, eager_loss, phase=1)


# ===========================================================================
# Phase 2: make_fx + functional_call + real optimizer pre-init
# ===========================================================================


def run_phase2(device: torch.device):
    print("=" * 70)
    print("Phase 2: make_fx + functional_call + pre-initialized optimizer state")
    print("=" * 70)

    torch.manual_seed(42)
    model = ToyMLP().to(device)

    # Sample data
    x = torch.randn(8, 64, device=device)
    labels = torch.randn(8, 64, device=device)

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01

    # ----- Pre-init optimizer state -----
    param_list = list(model.parameters())
    param_names = [n for n, _ in model.named_parameters()]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd,
    )

    # Force lazy state allocation with correct initial values.
    force_optimizer_state_init(optimizer)

    # Extract the fresh (but allocated) state tensors
    exp_avgs, exp_avg_sqs, steps = extract_optimizer_state(optimizer, param_list)

    # Detach parameters for tracing (we'll pass them as explicit inputs)
    detached_params = [p.detach().clone() for p in param_list]

    # Use first step tensor as the shared step (they should all be the same)
    step_t = steps[0].clone().to(dtype=torch.float32)

    num_params = len(detached_params)

    # ----- Define train step using functional_call -----
    def train_step_fn(*flat_args):
        # Unpack flat args
        params = list(flat_args[:num_params])
        offset = num_params
        eas = list(flat_args[offset : offset + num_params])
        offset += num_params
        eass = list(flat_args[offset : offset + num_params])
        offset += num_params
        step_val = flat_args[offset]
        offset += 1
        inp = flat_args[offset]
        offset += 1
        labs = flat_args[offset]

        # Build params dict for functional_call
        params_dict = {}
        for name, p in zip(param_names, params):
            params_dict[name] = p

        # Forward via functional_call
        logits = torch.func.functional_call(model, params_dict, (inp,))
        loss = F.mse_loss(logits, labs)

        # Backward via autograd.grad
        grads = torch.autograd.grad(loss, params)

        # Functional AdamW
        new_params, new_eas, new_eass, new_step = functional_adam_update(
            params, list(grads), eas, eass, step_val,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd,
        )

        return (loss, *new_params, *new_eas, *new_eass, new_step)

    # ----- Build flat inputs -----
    trace_params = [p.clone().requires_grad_(True) for p in detached_params]
    trace_eas = [ea.detach().clone() for ea in exp_avgs]
    trace_eass = [eas.detach().clone() for eas in exp_avg_sqs]

    all_inputs = (
        *trace_params,
        *trace_eas,
        *trace_eass,
        step_t.clone(),
        x.clone(), labels.clone(),
    )

    # ----- Trace with make_fx -----
    print("\nTracing with make_fx (tracing_mode='real')...")
    gm = make_fx(train_step_fn, tracing_mode="real")(*all_inputs)

    print("\n--- FX Graph (Phase 2) ---")
    gm.print_readable()
    print(f"\nNode count: {len(list(gm.graph.nodes))}")

    # ----- Run traced graph for correctness -----
    run_params = [p.clone().requires_grad_(True) for p in detached_params]
    run_eas = [ea.detach().clone() for ea in exp_avgs]
    run_eass = [eas.detach().clone() for eas in exp_avg_sqs]
    run_inputs = (
        *run_params,
        *run_eas,
        *run_eass,
        step_t.clone(),
        x.clone(), labels.clone(),
    )
    traced_out = gm(*run_inputs)
    traced_loss = traced_out[0]

    # ----- Eager execution for comparison -----
    eager_params = [p.clone().requires_grad_(True) for p in detached_params]
    eager_eas = [ea.detach().clone() for ea in exp_avgs]
    eager_eass = [eas.detach().clone() for eas in exp_avg_sqs]
    eager_inputs = (
        *eager_params,
        *eager_eas,
        *eager_eass,
        step_t.clone(),
        x.clone(), labels.clone(),
    )
    eager_out = train_step_fn(*eager_inputs)
    eager_loss = eager_out[0]

    verify(traced_loss, eager_loss, phase=2)


# ===========================================================================
# Phase 3: aot_export_joint (safe capture) + make_fx (optimizer wrapping)
# ===========================================================================


class ModelWithLoss(nn.Module):
    """Wraps model + loss so the joint graph returns a scalar loss."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return F.mse_loss(logits, labels)


def run_phase3(device: torch.device):
    print("=" * 70)
    print("Phase 3: aot_export_joint + make_fx (optimizer wrapping)")
    print("=" * 70)

    from torchtitan.experiments.compiler_toolkit.graph_utils import (
        aot_export_joint_with_descriptors_alone,
    )

    torch.manual_seed(42)
    model = ToyMLP().to(device)

    # Sample data
    x = torch.randn(8, 64, device=device)
    labels = torch.randn(8, 64, device=device)

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01

    # ----- Pre-init optimizer state (same as Phase 2) -----
    param_list = list(model.parameters())
    param_names = [n for n, _ in model.named_parameters()]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd,
    )
    force_optimizer_state_init(optimizer)

    exp_avgs, exp_avg_sqs, steps = extract_optimizer_state(optimizer, param_list)
    detached_params = [p.detach().clone() for p in param_list]
    step_t = steps[0].clone().to(dtype=torch.float32)

    num_params = len(detached_params)

    # ----- Stage A: Capture joint fwd+bwd graph -----
    print("\nStage A: Capturing joint fwd+bwd graph with aot_export_joint...")
    model_with_loss = ModelWithLoss(model)

    # aot_export_joint expects the model's forward args as a tuple
    joint_wd = aot_export_joint_with_descriptors_alone(
        model_with_loss,
        (x.clone(), labels.clone()),
    )

    joint_gm = joint_wd.graph_module

    print("\n--- Joint FX Graph (Stage A) ---")
    joint_gm.print_readable()
    print(f"\nJoint graph node count: {len(list(joint_gm.graph.nodes))}")

    # Inspect the joint graph signature:
    # inputs = [*params, *buffers, *user_inputs, *tangents]
    # outputs = [*fwd_outputs, *param_gradients]
    num_params_spec = len(joint_wd.params_spec)
    num_buffers_spec = len(joint_wd.buffers_spec) if hasattr(joint_wd, 'buffers_spec') else 0

    print(f"\n  params_spec count: {num_params_spec}")
    print(f"  buffers_spec count: {num_buffers_spec}")

    # ----- Stage B: Wrap joint graph + optimizer in make_fx -----
    print("\nStage B: Tracing full train step (joint graph + optimizer) with make_fx...")

    # The joint graph signature (from aot_export_joint):
    #   forward(self, primals: list, tangents: list) -> tree_unflatten(flat_out)
    #   primals = [*params, *buffers, *user_inputs]
    #   tangents = [tangent_for_loss]
    #   flat_out = [loss, *grad_for_each_primal]  (None for non-param primals)
    #
    # make_fx can trace through the joint GraphModule's pytree ops and
    # inline all operations into a single flat graph.

    tangent = torch.ones((), device=device)

    def full_train_step(*flat_args):
        # Unpack: params, optimizer state, x, labels
        params = list(flat_args[:num_params])
        offset = num_params
        eas = list(flat_args[offset : offset + num_params])
        offset += num_params
        eass = list(flat_args[offset : offset + num_params])
        offset += num_params
        step_val = flat_args[offset]
        offset += 1
        inp = flat_args[offset]
        offset += 1
        labs = flat_args[offset]

        # Build joint graph inputs as (primals_list, tangents_list)
        primals = [*params, inp, labs]
        tangents_list = [tangent]

        # Call the joint fwd+bwd graph
        joint_out = joint_gm(primals, tangents_list)

        # Flatten the output to get [loss, grad0, grad1, ...]
        flat_out = torch.utils._pytree.tree_leaves(joint_out)
        loss = flat_out[0]
        grads = flat_out[1:1 + num_params]

        # Functional AdamW
        new_params, new_eas, new_eass, new_step = functional_adam_update(
            params, list(grads), eas, eass, step_val,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd,
        )

        return (loss, *new_params, *new_eas, *new_eass, new_step)

    # Build flat inputs for tracing
    trace_params = [p.clone().requires_grad_(False) for p in detached_params]
    trace_eas = [ea.detach().clone() for ea in exp_avgs]
    trace_eass = [eas.detach().clone() for eas in exp_avg_sqs]

    all_inputs = (
        *trace_params,
        *trace_eas,
        *trace_eass,
        step_t.clone(),
        x.clone(), labels.clone(),
    )

    gm = make_fx(full_train_step, tracing_mode="real")(*all_inputs)

    print("\n--- Full Train Step FX Graph (Stage B) ---")
    gm.print_readable()
    print(f"\nFull graph node count: {len(list(gm.graph.nodes))}")

    # ----- Correctness check -----
    run_inputs = (
        *[p.clone().requires_grad_(False) for p in detached_params],
        *[ea.detach().clone() for ea in exp_avgs],
        *[eas.detach().clone() for eas in exp_avg_sqs],
        step_t.clone(),
        x.clone(), labels.clone(),
    )
    traced_out = gm(*run_inputs)
    traced_loss = traced_out[0]

    eager_out = full_train_step(*run_inputs)
    eager_loss = eager_out[0]

    verify(traced_loss, eager_loss, phase=3)


# ===========================================================================
# Phase 4: make_fx + native optimizer via _single_tensor_adam(capturable=True)
# ===========================================================================


def run_phase4(device: torch.device):
    print("=" * 70)
    print("Phase 4: make_fx + native _single_tensor_adam(capturable=True)")
    print("=" * 70)

    if device.type not in ("cuda", "xpu", "hpu", "xla"):
        print(f"\n  SKIPPED: capturable=True requires an accelerator device, got {device}")
        print("  Run with --device cuda")
        return

    from torch.optim.adam import _single_tensor_adam

    torch.manual_seed(42)
    model = ToyMLP().to(device)

    # Sample data
    x = torch.randn(8, 64, device=device)
    labels = torch.randn(8, 64, device=device)

    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01

    # ----- Pre-init optimizer state -----
    param_list = list(model.parameters())
    param_names = [n for n, _ in model.named_parameters()]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd,
    )
    force_optimizer_state_init(optimizer)

    # capturable=True requires step tensors on the same device as params
    for p in param_list:
        optimizer.state[p]["step"] = optimizer.state[p]["step"].to(device=device)

    exp_avgs, exp_avg_sqs, steps = extract_optimizer_state(optimizer, param_list)
    detached_params = [p.detach().clone() for p in param_list]

    num_params = len(detached_params)

    # ----- Define train step using native optimizer -----
    # The graph will contain in-place ops (mul_, addcmul_, etc.) from
    # _single_tensor_adam. This is expected — downstream compilers (Inductor,
    # XLA) run their own functionalization pass that converts these to
    # out-of-place equivalents before lowering.
    def train_step_fn(*flat_args):
        params = list(flat_args[:num_params])
        offset = num_params
        eas = list(flat_args[offset : offset + num_params])
        offset += num_params
        eass = list(flat_args[offset : offset + num_params])
        offset += num_params
        step_vals = list(flat_args[offset : offset + num_params])
        offset += num_params
        inp = flat_args[offset]
        offset += 1
        labs = flat_args[offset]

        # Forward via functional_call
        grad_params = [p.detach().requires_grad_(True) for p in params]
        params_dict = dict(zip(param_names, grad_params))
        logits = torch.func.functional_call(model, params_dict, (inp,))
        loss = F.mse_loss(logits, labs)

        # Backward via autograd.grad
        grads = list(torch.autograd.grad(loss, grad_params))

        # Native optimizer step (in-place, capturable=True avoids .item())
        _single_tensor_adam(
            params, grads, eas, eass,
            [],  # max_exp_avg_sqs (unused without amsgrad)
            step_vals,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=wd,
            eps=eps,
            maximize=False,
            capturable=True,
            differentiable=False,
            has_complex=False,
            grad_scale=None,
            found_inf=None,
            decoupled_weight_decay=True,
        )

        return (loss, *params, *eas, *eass, *step_vals)

    # ----- Build flat inputs -----
    # Per-parameter step tensors (all on device for capturable=True)
    trace_inputs = (
        *[p.clone() for p in detached_params],
        *[ea.detach().clone() for ea in exp_avgs],
        *[eas.detach().clone() for eas in exp_avg_sqs],
        *[s.clone() for s in steps],
        x.clone(),
        labels.clone(),
    )

    # ----- Trace with make_fx -----
    print("\nTracing with make_fx (tracing_mode='real')...")
    gm = make_fx(train_step_fn, tracing_mode="real")(*trace_inputs)

    print("\n--- FX Graph (Phase 4) ---")
    gm.print_readable()
    print(f"\nNode count: {len(list(gm.graph.nodes))}")

    # Count in-place ops to show the graph's nature — downstream compilers
    # (Inductor, XLA) will functionalize these before lowering.
    inplace_count = sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function"
        and hasattr(node.target, "_schema")
        and node.target._schema.is_mutable
    )
    print(f"In-place (mutable) ops: {inplace_count}")

    # ----- Correctness check -----
    # Note: the traced graph mutates its inputs (in-place optimizer ops).
    # We clone inputs for each run to avoid interference.
    def make_inputs():
        return (
            *[p.clone() for p in detached_params],
            *[ea.detach().clone() for ea in exp_avgs],
            *[eas.detach().clone() for eas in exp_avg_sqs],
            *[s.clone() for s in steps],
            x.clone(),
            labels.clone(),
        )

    traced_out = gm(*make_inputs())
    eager_out = train_step_fn(*make_inputs())

    verify(traced_out[0], eager_out[0], phase=4)

    # Also verify parameter updates match
    print("  Param diffs:")
    for i in range(num_params):
        diff = (traced_out[1 + i] - eager_out[1 + i]).abs().max().item()
        print(f"    param {i}: {diff:.2e}")


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prototype: capture full train step in FX graph",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Which phase to run (1, 2, 3, or 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")

    if args.phase == 1:
        run_phase1(device)
    elif args.phase == 2:
        run_phase2(device)
    elif args.phase == 3:
        run_phase3(device)
    elif args.phase == 4:
        run_phase4(device)


if __name__ == "__main__":
    main()
