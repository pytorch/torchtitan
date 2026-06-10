# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract test for model and optimizer state dict keys (llama3 debugmodel).

Verifies the key contract that distributed checkpointing relies on:
- ModelWrapper.state_dict() strips wrapper prefixes (_orig_mod,
  _checkpoint_wrapped_module) so keys are canonical FQNs.
- OptimizersContainer.state_dict() is keyed by FQN as state.{fqn}.{state_name}
  and param_groups.{fqn}.{key}, matching the model's parameter FQNs.
- The flatten/unflatten round-trip preserves optimizer state values exactly.

Ground-truth keys are derived from the model itself (robust to model evolution),
while a few hard-coded structural anchors catch unexpected model refactors. This
runs on CPU using torch.compile(backend="eager") to introduce the _orig_mod.
prefixes without needing a GPU.
"""

import unittest

import torch

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model

# AdamW creates these per-parameter state tensors (no amsgrad).
_ADAMW_STATE_NAMES = ("step", "exp_avg", "exp_avg_sq")

# Well-known llama3 debugmodel parameter FQNs used as structural anchors. These
# are checked as a subset, so extra keys (biases, buffers) never cause a false
# failure, but a rename in the model would.
_TOP_LEVEL_ANCHORS = ("tok_embeddings.weight", "norm.weight", "lm_head.weight")
_LAYER0_ANCHORS = (
    "layers.0.attention.qkv_linear.wq.weight",
    "layers.0.attention.qkv_linear.wk.weight",
    "layers.0.attention.qkv_linear.wv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w1.weight",
    "layers.0.feed_forward.w2.weight",
    "layers.0.feed_forward.w3.weight",
    "layers.0.attention_norm.weight",
    "layers.0.ffn_norm.weight",
)


def _build_debugmodel() -> Llama3Model:
    config = llama3_configs["debugmodel"](attn_backend="flex")
    model = Llama3Model(config)
    model.init_states()
    return model


def _debugmodel_optimizer_config() -> OptimizersContainer.Config:
    # for-loop keeps the test on CPU (no fused/foreach CUDA path).
    return OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": 8e-4,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            )
        ],
    )


class TestStateDictKeys(unittest.TestCase):
    def setUp(self) -> None:
        # Ground-truth canonical keys come from the uncompiled model.
        model = _build_debugmodel()
        self.clean_state_keys = set(model.state_dict().keys())
        self.clean_param_fqns = {
            name for name, p in model.named_parameters() if p.requires_grad
        }
        # torch.compile (eager backend, no GPU) introduces _orig_mod. prefixes.
        self.compiled_parts = [torch.compile(model, backend="eager")]

    def test_model_structure_anchors(self) -> None:
        """The debugmodel exposes the expected, well-known parameter FQNs."""
        for anchor in _TOP_LEVEL_ANCHORS + _LAYER0_ANCHORS:
            self.assertIn(anchor, self.clean_param_fqns)

    def test_model_state_dict_keys_are_canonical(self) -> None:
        """ModelWrapper strips wrapper prefixes to recover canonical FQNs."""
        keys = set(ModelWrapper(self.compiled_parts).state_dict().keys())
        for key in keys:
            self.assertNotIn("_orig_mod", key)
            self.assertNotIn("_checkpoint_wrapped_module", key)
        self.assertEqual(keys, self.clean_state_keys)

    def test_optimizer_state_dict_keys(self) -> None:
        optimizers = _debugmodel_optimizer_config().build(
            model_parts=self.compiled_parts
        )
        optim_sd = optimizers.state_dict()

        state_keys = {k for k in optim_sd if k.startswith("state.")}
        expected_state_keys = {
            f"state.{fqn}.{name}"
            for fqn in self.clean_param_fqns
            for name in _ADAMW_STATE_NAMES
        }
        self.assertEqual(state_keys, expected_state_keys)

        # Every parameter FQN has param_groups entries, and pattern never leaks.
        pg_keys = {k for k in optim_sd if k.startswith("param_groups.")}
        for fqn in self.clean_param_fqns:
            self.assertTrue(
                any(k.startswith(f"param_groups.{fqn}.") for k in pg_keys),
                f"missing param_groups entries for {fqn}",
            )
        self.assertFalse(any(k.endswith(".pattern") for k in pg_keys))

    def test_optimizer_state_dict_roundtrip_keys(self) -> None:
        optimizers = _debugmodel_optimizer_config().build(
            model_parts=self.compiled_parts
        )
        optim_sd = optimizers.state_dict()
        optimizers.load_state_dict(optim_sd)
        self.assertEqual(set(optim_sd.keys()), set(optimizers.state_dict().keys()))

    def test_flat_optim_roundtrip_values(self) -> None:
        """get/load_flat_optim_state_dict preserve state values exactly."""
        optimizers = _debugmodel_optimizer_config().build(
            model_parts=self.compiled_parts
        )
        # Populate optimizer state with non-zero values via a real step.
        for p in self.compiled_parts[0].parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        optimizers.step()

        flat: dict = {}
        for optim in optimizers.optimizers:
            flat.update(get_flat_optim_state_dict(optim))

        # Load into a fresh container (second model, matching FQNs).
        optimizers2 = _debugmodel_optimizer_config().build(
            model_parts=[_build_debugmodel()]
        )
        for optim in optimizers2.optimizers:
            init_optim_state(optim)
            load_flat_optim_state_dict(optim, flat)

        flat2: dict = {}
        for optim in optimizers2.optimizers:
            flat2.update(get_flat_optim_state_dict(optim))

        self.assertEqual(set(flat), set(flat2))
        for key, v1 in flat.items():
            v2 = flat2[key]
            if isinstance(v1, torch.Tensor):
                self.assertTrue(
                    torch.equal(v1, v2),
                    f"value mismatch at {key}: max diff {(v1 - v2).abs().max()}",
                )
            else:
                self.assertEqual(v1, v2, f"value mismatch at {key}")


if __name__ == "__main__":
    unittest.main()
