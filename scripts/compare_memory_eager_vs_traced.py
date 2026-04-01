#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare peak GPU memory and verify bitwise correctness:
eager (no AC) vs eager (selective AC) vs traced (no AC) vs traced (graph SAC).

Usage:
    python scripts/compare_memory_eager_vs_traced.py
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.experiments.graph_trainer.common_utils import annotate_ac_regions
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.passes import apply_ac_remat_pass
from torchtitan.models.llama3 import llama3_configs, Llama3Model


DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH_SIZE = 2
SEQ_LEN = 2048
NUM_WARMUP = 2
NUM_STEPS = 3
LR = 1e-4


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


def make_train_step(loss_fn):
    def train_step(model, tokens, labels):
        logits = model(tokens)
        loss = loss_fn(logits, labels)
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return train_step


def create_model(config):
    model = Llama3Model(config).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(DEVICE))
    return model


def run_steps(model, tokens, labels, num_steps, *, mode="eager"):
    """Run training steps in the given mode. Returns per-step (loss, grads) and peak memory.

    Args:
        mode: "eager", "eager_ac", "traced", or "traced_ac".
    """
    if mode == "eager_ac":
        apply_ac(model, ActivationCheckpointConfig(mode="selective"))

    if mode in ("traced", "traced_ac"):
        if mode == "traced_ac":
            annotate_ac_regions(model)
        train_step_fn = make_train_step(get_loss)
        traced = minimal_fx_tracer(train_step_fn, (model, tokens, labels))
        if mode == "traced_ac":
            apply_ac_remat_pass(traced)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    results = []

    for step in range(num_steps):
        if step == NUM_WARMUP:
            torch.cuda.reset_peak_memory_stats()

        if mode in ("traced", "traced_ac"):
            result = traced(model, tokens, labels)
            loss = result[0]
            grads = result[1:]
            for p, g in zip(model.parameters(), grads, strict=True):
                p.grad = g
        else:
            logits = model(tokens)
            loss = get_loss(logits, labels)
            loss.backward()
            grads = [p.grad for p in model.parameters()]

        results.append((
            loss.detach().clone().cpu(),
            [g.clone().cpu() for g in grads],
        ))
        opt.step()
        opt.zero_grad()

    peak = torch.cuda.max_memory_allocated() / 1e9
    return results, peak


def run_and_cleanup(config, state, tokens, labels, total_steps, *, mode):
    """Create model, run steps, cleanup GPU memory, return results + peak."""
    model = create_model(config)
    model.load_state_dict(state)
    results, peak = run_steps(model, tokens, labels, total_steps, mode=mode)
    del model
    torch.cuda.empty_cache()
    return results, peak


def verify_bitwise(name_a, results_a, name_b, results_b):
    """Check that loss and grads match bitwise across all steps."""
    all_ok = True
    for step, ((loss_a, grads_a), (loss_b, grads_b)) in enumerate(
        zip(results_a, results_b, strict=True)
    ):
        if not torch.equal(loss_a, loss_b):
            print(
                f"  MISMATCH step {step}: {name_a} loss={loss_a.item():.6f} "
                f"vs {name_b} loss={loss_b.item():.6f}"
            )
            all_ok = False
        for i, (ga, gb) in enumerate(zip(grads_a, grads_b, strict=True)):
            if not torch.equal(ga, gb):
                max_diff = (ga - gb).abs().max().item()
                print(
                    f"  MISMATCH step {step}: grad[{i}] max_diff={max_diff:.2e}"
                )
                all_ok = False
                break
    if all_ok:
        print(f"  PASS: {name_a} vs {name_b} — bitwise identical across all steps")
    return all_ok


def main():
    config = llama3_configs["1B"]
    total_steps = NUM_WARMUP + NUM_STEPS
    print(f"Model: Llama3 1B (dim={config.dim}, n_layers={config.n_layers})")
    print(f"Batch: {BATCH_SIZE} x {SEQ_LEN}, dtype: {DTYPE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Steps: {NUM_WARMUP} warmup + {NUM_STEPS} measured")
    print()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokens = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    labels = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    # Create reference weights on CPU to avoid holding GPU memory.
    torch.manual_seed(42)
    ref_model = create_model(config)
    state = {k: v.cpu() for k, v in ref_model.state_dict().items()}
    del ref_model
    torch.cuda.empty_cache()

    # Run each mode, cleaning up GPU between runs.
    print("Running eager (no AC)...")
    results_eager, peak_eager = run_and_cleanup(
        config, state, tokens, labels, total_steps, mode="eager"
    )
    print("Running eager (selective AC)...")
    results_ac, peak_ac = run_and_cleanup(
        config, state, tokens, labels, total_steps, mode="eager_ac"
    )
    print("Running traced (graph SAC)...")
    results_traced_ac, peak_traced_ac = run_and_cleanup(
        config, state, tokens, labels, total_steps, mode="traced_ac"
    )
    print("Running traced (no AC)...")
    results_traced, peak_traced = run_and_cleanup(
        config, state, tokens, labels, total_steps, mode="traced"
    )

    # --- Correctness ---
    print()
    print("Correctness:")
    verify_bitwise("eager", results_eager, "traced", results_traced)
    verify_bitwise("eager", results_eager, "traced_ac", results_traced_ac)
    print()

    # --- Memory ---
    print("=" * 60)
    print(f"{'Mode':<25} {'Peak Memory (GB)':>15}")
    print("-" * 60)
    print(f"{'Eager (no AC)':<25} {peak_eager:>15.2f}")
    print(f"{'Eager (selective AC)':<25} {peak_ac:>15.2f}")
    print(f"{'Traced (no AC)':<25} {peak_traced:>15.2f}")
    print(f"{'Traced (graph SAC)':<25} {peak_traced_ac:>15.2f}")
    print("=" * 60)
    print()
    print(
        f"AC savings vs eager:  {peak_eager - peak_ac:.2f} GB "
        f"({(1 - peak_ac / peak_eager) * 100:.1f}%)"
    )
    ratio = peak_traced_ac / peak_ac
    print(
        f"Traced SAC vs eager AC:  {peak_traced_ac - peak_ac:+.2f} GB "
        f"(ratio {ratio:.2f}x)"
    )


if __name__ == "__main__":
    main()
