# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract test for model and optimizer state dict keys (llama3 debugmodel).

Verifies the key contract that distributed checkpointing relies on:
- ModelWrapper.state_dict() is keyed by canonical FQNs. This holds without any
  key cleaning because model.state_dict() is already canonical: the activation
  checkpoint wrapper strips its own _checkpoint_wrapped_module prefix via a
  state_dict hook, and torchtitan applies torch.compile in place (no segment).
- OptimizersContainer.state_dict() is keyed as state.{fqn}.{state_name} and
  param_groups.{fqn}.{key}, matching the model FQNs. The optimizer is built from
  named_parameters(), which (unlike state_dict()) keeps the wrapper prefix, so
  canonical_fqn must strip it during construction.
- The flatten/unflatten round-trip preserves optimizer state values exactly.

The test wraps each layer with the activation checkpoint wrapper to reproduce the
prefix asymmetry on CPU (named_parameters prefixed, state_dict clean), and derives
ground-truth keys from the unwrapped model plus a few hard-coded structural
anchors that catch unexpected model refactors.
"""

import unittest

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model

_WRAPPER_PREFIX = "_checkpoint_wrapped_module"

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


def _wrap_layers_with_ac(model: Llama3Model) -> None:
    """Wrap each transformer block with the AC wrapper, in place (like apply_ac).

    This makes named_parameters() carry the _checkpoint_wrapped_module prefix
    while state_dict() stays clean (the wrapper strips it via a state_dict hook).
    """
    layers = model.get_submodule("layers")
    for layer_id, block in list(layers.named_children()):
        layers.register_module(layer_id, ptd_checkpoint_wrapper(block))


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
        # Ground-truth canonical keys come from the unwrapped model.
        model = _build_debugmodel()
        self.clean_state_keys = set(model.state_dict().keys())
        self.clean_param_fqns = {
            name for name, p in model.named_parameters() if p.requires_grad
        }
        # Wrapping introduces the _checkpoint_wrapped_module prefix on
        # named_parameters() but not on state_dict() (stripped by a hook).
        _wrap_layers_with_ac(model)
        self.model_parts = [model]

    def test_model_structure_anchors(self) -> None:
        """The debugmodel exposes the expected, well-known parameter FQNs."""
        for anchor in _TOP_LEVEL_ANCHORS + _LAYER0_ANCHORS:
            self.assertIn(anchor, self.clean_param_fqns)

    def test_ac_prefix_only_on_named_parameters(self) -> None:
        """named_parameters() keeps the AC prefix; state_dict() is already clean.

        This is the asymmetry that justifies cleaning FQNs for the optimizer but
        not for the model state dict.
        """
        model = self.model_parts[0]
        self.assertTrue(
            any(_WRAPPER_PREFIX in n for n, _ in model.named_parameters()),
            "expected the AC wrapper prefix on named_parameters()",
        )
        self.assertFalse(
            any(_WRAPPER_PREFIX in k for k in model.state_dict()),
            "model.state_dict() should already be free of the AC wrapper prefix",
        )

    def test_model_state_dict_keys_are_canonical(self) -> None:
        """ModelWrapper.state_dict() yields canonical FQNs with no cleaning."""
        keys = set(ModelWrapper(self.model_parts).state_dict().keys())
        for key in keys:
            self.assertNotIn(_WRAPPER_PREFIX, key)
        self.assertEqual(keys, self.clean_state_keys)

    def test_optimizer_state_dict_keys(self) -> None:
        optimizers = _debugmodel_optimizer_config().build(model_parts=self.model_parts)
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
        optimizers = _debugmodel_optimizer_config().build(model_parts=self.model_parts)
        optim_sd = optimizers.state_dict()
        optimizers.load_state_dict(optim_sd)
        self.assertEqual(set(optim_sd.keys()), set(optimizers.state_dict().keys()))

    def test_flat_optim_roundtrip_values(self) -> None:
        """get/load_flat_optim_state_dict preserve state values exactly."""
        optimizers = _debugmodel_optimizer_config().build(model_parts=self.model_parts)
        # Populate optimizer state with non-zero values via a real step.
        for p in self.model_parts[0].parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        optimizers.step()

        flat: dict = {}
        for optim in optimizers.optimizers:
            flat.update(get_flat_optim_state_dict(optim))

        # Load into a fresh container (second model, matching FQNs).
        model2 = _build_debugmodel()
        _wrap_layers_with_ac(model2)
        optimizers2 = _debugmodel_optimizer_config().build(model_parts=[model2])
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
