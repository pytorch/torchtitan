# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Contract test for model and optimizer state dict keys.

Verifies that the FQNs produced by ModelWrapper.state_dict() and
OptimizersContainer.state_dict() match the expected hard-coded values
for the llama3 debugmodel with AdamW optimizer.

Catches regressions from refactoring or upstream PyTorch changes.

Runs on CPU without distributed — no GPU or torchrun required.
"""

import unittest

import torch

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model


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
# These should NOT have _orig_mod. prefix (stripped by canonical_model_state_dict).
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


class TestStateDictKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_config = llama3_configs["debugmodel"]
        model = Llama3Model(model_config)
        model.init_weights(buffer_device=torch.device("cpu"))

        # Apply torch.compile to introduce _orig_mod prefixes, using the
        # eager backend so no GPU is needed.
        model = torch.compile(model, backend="eager")
        cls.model = model
        cls.model_parts = [model]

    def test_model_state_dict_keys(self):
        model_wrapper = ModelWrapper(self.model_parts)
        model_sd = model_wrapper.state_dict()
        actual_keys = sorted(model_sd.keys())

        self.assertEqual(
            actual_keys,
            EXPECTED_MODEL_KEYS,
            f"Model state dict keys mismatch!\n"
            f"  Missing: {sorted(set(EXPECTED_MODEL_KEYS) - set(actual_keys))}\n"
            f"  Extra: {sorted(set(actual_keys) - set(EXPECTED_MODEL_KEYS))}",
        )

    def test_optimizer_state_dict_keys(self):
        optim_config = OptimizersContainer.Config(lr=8e-4, implementation="for-loop")
        optimizers = OptimizersContainer(optim_config, model_parts=self.model_parts)
        optim_sd = optimizers.state_dict()
        actual_keys = sorted(optim_sd.keys())

        actual_state_keys = sorted(k for k in actual_keys if k.startswith("state."))
        actual_pg_keys = sorted(k for k in actual_keys if k.startswith("param_groups."))

        self.assertEqual(
            actual_state_keys,
            EXPECTED_OPTIM_STATE_KEYS,
            f"Optimizer state keys mismatch!\n"
            f"  Missing: {sorted(set(EXPECTED_OPTIM_STATE_KEYS) - set(actual_state_keys))}\n"
            f"  Extra: {sorted(set(actual_state_keys) - set(EXPECTED_OPTIM_STATE_KEYS))}",
        )

        # Verify param_groups keys exist for every expected FQN
        pg_fqns = set()
        for k in actual_pg_keys:
            for fqn in EXPECTED_MODEL_KEYS:
                prefix = f"param_groups.{fqn}."
                if k.startswith(prefix):
                    pg_fqns.add(fqn)
                    break

        missing_pg_fqns = set(EXPECTED_MODEL_KEYS) - pg_fqns
        self.assertFalse(
            missing_pg_fqns,
            f"param_groups missing for FQNs: {sorted(missing_pg_fqns)}",
        )

    def test_optimizer_state_dict_roundtrip(self):
        optim_config = OptimizersContainer.Config(lr=8e-4, implementation="for-loop")
        optimizers = OptimizersContainer(optim_config, model_parts=self.model_parts)
        optim_sd = optimizers.state_dict()
        original_keys = sorted(optim_sd.keys())

        optimizers.load_state_dict(optim_sd)
        after_keys = sorted(optimizers.state_dict().keys())

        self.assertEqual(
            original_keys,
            after_keys,
            "Optimizer state dict keys changed after round-trip!",
        )

    def test_flat_optim_state_dict_roundtrip_values(self):
        """Verify flatten/unflatten round-trip preserves optimizer state values.

        This tests the actual DCP flow: get_flat_optim_state_dict produces a
        flat dict, load_flat_optim_state_dict restores it into a fresh optimizer,
        and all state tensor values match exactly.
        """
        from torchtitan.components.checkpoint_utils import (
            get_flat_optim_state_dict,
            load_flat_optim_state_dict,
        )

        # Build a separate model to avoid polluting shared state with grads
        model_config = llama3_configs["debugmodel"]
        model = Llama3Model(model_config)
        model.init_weights(buffer_device=torch.device("cpu"))
        model = torch.compile(model, backend="eager")
        model_parts = [model]

        optim_config = OptimizersContainer.Config(lr=8e-4, implementation="for-loop")
        optimizers = OptimizersContainer(optim_config, model_parts=model_parts)

        # Run a real training step to populate optimizer state with non-zero values
        dummy_input = torch.randint(0, 2048, (2, 16))
        output = model(dummy_input)
        output.sum().backward()
        optimizers.step()
        optimizers.zero_grad()

        # Flatten state from each inner optimizer
        flat_sd: dict = {}
        for optim in optimizers.optimizers:
            flat_sd.update(get_flat_optim_state_dict(optim))

        # Build a fresh optimizer and load the flat state dict
        optimizers2 = OptimizersContainer(optim_config, model_parts=model_parts)
        for optim in optimizers2.optimizers:
            load_flat_optim_state_dict(optim, flat_sd)

        # Re-flatten and compare keys and values
        flat_sd2: dict = {}
        for optim in optimizers2.optimizers:
            flat_sd2.update(get_flat_optim_state_dict(optim))

        self.assertEqual(sorted(flat_sd.keys()), sorted(flat_sd2.keys()))

        for key in flat_sd:
            v1, v2 = flat_sd[key], flat_sd2[key]
            if isinstance(v1, torch.Tensor):
                self.assertTrue(
                    torch.equal(v1, v2),
                    f"Tensor mismatch at {key}: max diff = {(v1 - v2).abs().max()}",
                )
            else:
                self.assertEqual(v1, v2, f"Value mismatch at {key}: {v1} vs {v2}")


if __name__ == "__main__":
    unittest.main()
