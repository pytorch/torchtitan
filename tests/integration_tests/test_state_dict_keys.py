# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Contract test for model and optimizer state dict keys.

This test verifies that the FQNs produced by ModelWrapper.state_dict() and
OptimizersContainer.state_dict() match the expected hard-coded values.
The FQNs are based on Feb 25 2026 snapshot of TorchTitan and PyTorch llama3
debug model with AdamW optimizer.

It catches regressions from refactoring or upstream PyTorch changes.

Requires GPU. Run with:
    torchrun --nproc_per_node=8 tests/integration_tests/test_state_dict_keys.py
"""

import os
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3 import model_registry
from torchtitan.protocols.model_converter import ModelConvertersContainer


def _build_layer_keys(layer_id: int) -> list[str]:
    prefix = f"layers.{layer_id}"
    return [
        f"{prefix}.attention.wq.weight",
        f"{prefix}.attention.wk.weight",
        f"{prefix}.attention.wv.weight",
        f"{prefix}.attention.wo.weight",
        f"{prefix}.feed_forward.w1.weight",
        f"{prefix}.feed_forward.w2.weight",
        f"{prefix}.feed_forward.w3.weight",
        f"{prefix}.attention_norm.weight",
        f"{prefix}.ffn_norm.weight",
    ]


# Hard-coded expected model state dict keys for llama3 debugmodel
# (dim=256, n_layers=6, vocab_size=2048, n_heads=16)
# These should NOT have _orig_mod. prefix (stripped by clean_model_state_dict).
EXPECTED_MODEL_KEYS: list[str] = sorted(
    ["tok_embeddings.weight"]
    + [key for i in range(6) for key in _build_layer_keys(i)]
    + ["norm.weight", "output.weight"]
)

# AdamW state keys per parameter (step, exp_avg, exp_avg_sq)
_ADAM_STATE_NAMES = ["step", "exp_avg", "exp_avg_sq"]

# Expected optimizer state keys: state.{fqn}.{state_name} for all params
EXPECTED_OPTIM_STATE_KEYS: list[str] = sorted(
    f"state.{fqn}.{state_name}"
    for fqn in EXPECTED_MODEL_KEYS
    for state_name in _ADAM_STATE_NAMES
)


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    dump_folder = tempfile.mkdtemp(prefix="test_state_dict_keys_")

    try:
        # Build model on meta device
        model_spec = model_registry("debugmodel")
        model_config = model_spec.model
        with torch.device("meta"):
            model = model_config.build()

        # Set up ParallelDims for FSDP-only
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=world_size,
        )
        parallel_dims.build_mesh()

        # Apply compile + FSDP2
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=TrainingConfig(),
            model_converters=ModelConvertersContainer.Config(),
            parallelism=ParallelismConfig(),
            compile_config=CompileConfig(enable=True),
            ac_config=ActivationCheckpointConfig(mode="none"),
            dump_folder=dump_folder,
        )

        # Materialize weights
        model.to_empty(device=device)
        with torch.no_grad():
            model.init_weights(buffer_device=device)
        model.train()

        model_parts = [model]

        # Build ModelWrapper and check model state dict keys
        model_wrapper = ModelWrapper(model_parts)
        model_sd = model_wrapper.state_dict()
        actual_model_keys = sorted(model_sd.keys())

        if actual_model_keys != EXPECTED_MODEL_KEYS:
            missing = set(EXPECTED_MODEL_KEYS) - set(actual_model_keys)
            extra = set(actual_model_keys) - set(EXPECTED_MODEL_KEYS)
            msg = "Model state dict keys mismatch!\n"
            if missing:
                msg += f"  Missing keys: {sorted(missing)}\n"
            if extra:
                msg += f"  Extra keys: {sorted(extra)}\n"
            raise AssertionError(msg)

        if rank == 0:
            print(f"PASS: Model state dict has {len(actual_model_keys)} expected keys")

        # Build OptimizersContainer and check optimizer state dict keys
        optim_config = OptimizersContainer.Config(lr=8e-4)
        optimizers = OptimizersContainer(optim_config, model_parts=model_parts)
        optim_sd = optimizers.state_dict()
        actual_optim_keys = sorted(optim_sd.keys())

        # Split into state keys and param_groups keys
        actual_state_keys = sorted(
            k for k in actual_optim_keys if k.startswith("state.")
        )
        actual_pg_keys = sorted(
            k for k in actual_optim_keys if k.startswith("param_groups.")
        )

        # Verify state keys match expected
        if actual_state_keys != EXPECTED_OPTIM_STATE_KEYS:
            missing = set(EXPECTED_OPTIM_STATE_KEYS) - set(actual_state_keys)
            extra = set(actual_state_keys) - set(EXPECTED_OPTIM_STATE_KEYS)
            msg = "Optimizer state keys mismatch!\n"
            if missing:
                msg += f"  Missing keys: {sorted(missing)}\n"
            if extra:
                msg += f"  Extra keys: {sorted(extra)}\n"
            raise AssertionError(msg)

        # Verify param_groups keys exist for every expected FQN
        pg_fqns = set()
        for k in actual_pg_keys:
            # key format: param_groups.{fqn}.{pg_key}
            # We need to match FQNs that may contain dots
            for fqn in EXPECTED_MODEL_KEYS:
                prefix = f"param_groups.{fqn}."
                if k.startswith(prefix):
                    pg_fqns.add(fqn)
                    break

        missing_pg_fqns = set(EXPECTED_MODEL_KEYS) - pg_fqns
        if missing_pg_fqns:
            raise AssertionError(
                f"param_groups missing for FQNs: {sorted(missing_pg_fqns)}"
            )

        if rank == 0:
            print(
                f"PASS: Optimizer state dict has {len(actual_state_keys)} state keys "
                f"and {len(actual_pg_keys)} param_groups keys"
            )

        # Verify round-trip: save then load
        optimizers.load_state_dict(optim_sd)
        optim_sd_after = optimizers.state_dict()
        if sorted(optim_sd_after.keys()) != actual_optim_keys:
            raise AssertionError("Optimizer state dict keys changed after round-trip!")

        if rank == 0:
            print("PASS: Optimizer state dict round-trip preserves keys")
            print("All state dict key contract tests passed!")
    finally:
        shutil.rmtree(dump_folder, ignore_errors=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
