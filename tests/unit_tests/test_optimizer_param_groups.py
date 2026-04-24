# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
import torch.nn as nn
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
    ParamGroupConfig,
)


class SimpleModel(nn.Module):
    """A small model with diverse parameter names for testing param groups."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleDict(
            {
                "0": nn.ModuleDict(
                    {
                        "attention": nn.Linear(16, 16),
                        "norm": nn.LayerNorm(16),
                        "ff": nn.Linear(16, 16),
                    }
                ),
            }
        )
        self.output = nn.Linear(16, 32)

    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.layers["0"]["attention"](x)
        x = self.layers["0"]["norm"](x)
        x = self.layers["0"]["ff"](x)
        return self.output(x)


def _get_param_names_in_group(model, group):
    """Return the set of parameter FQNs in an optimizer param group."""
    param_to_name = {p: n for n, p in model.named_parameters()}
    return {param_to_name[p] for p in group["params"]}


class TestParamGroupConfig(unittest.TestCase):
    def test_default_no_param_groups(self):
        """Empty param_groups produces a single group with all params."""
        model = SimpleModel()
        config = OptimizersContainer.Config(lr=1e-3, weight_decay=0.1)
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)

        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        self.assertEqual(len(groups), 1)
        all_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(groups[0]["params"]), len(all_params))
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertEqual(groups[0]["weight_decay"], 0.1)

    def test_single_pattern_weight_decay_zero(self):
        """Pattern matching bias params with weight_decay_multiplier=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Should have 2 groups: default + bias group
        self.assertEqual(len(groups), 2)

        # Default group: non-bias params
        default_names = _get_param_names_in_group(model, groups[0])
        for name in default_names:
            self.assertFalse(name.endswith(".bias"), f"{name} should not be in default")
        self.assertEqual(groups[0]["weight_decay"], 0.1)
        self.assertEqual(groups[0]["lr"], 1e-3)

        # Bias group
        bias_names = _get_param_names_in_group(model, groups[1])
        for name in bias_names:
            self.assertTrue(name.endswith(".bias"), f"{name} should end with .bias")
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[1]["lr"], 1e-3)

    def test_embed_tokens_pattern(self):
        """Pattern matching embed_tokens with weight_decay_multiplier=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    weight_decay_multiplier=0.0,
                ),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        self.assertEqual(len(groups), 2)

        embed_names = _get_param_names_in_group(model, groups[1])
        self.assertTrue(
            all("embed_tokens" in n for n in embed_names),
            f"Expected embed_tokens params, got {embed_names}",
        )
        self.assertEqual(groups[1]["weight_decay"], 0.0)

    def test_lr_multiplier(self):
        """lr_multiplier correctly scales the base lr."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    lr_multiplier=0.1,
                ),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Default group keeps base lr
        self.assertEqual(groups[0]["lr"], 1e-3)
        # Embed group gets 10% of base lr
        self.assertAlmostEqual(groups[1]["lr"], 1e-4)

    def test_first_match_wins(self):
        """When patterns overlap, the first match wins."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                # First pattern: all norm params get wd=0
                ParamGroupConfig(pattern=r".*norm.*", weight_decay_multiplier=0.0),
                # Second pattern: broader match that also covers norm
                ParamGroupConfig(pattern=r".*layers.*", lr_multiplier=0.5),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Norm params should be in group index 0 (the norm pattern), not group 1
        norm_group = groups[1]  # first matched group (after default)
        norm_names = _get_param_names_in_group(model, norm_group)
        self.assertTrue(all("norm" in n for n in norm_names))
        # Norm group should have weight_decay=0 (from first pattern)
        self.assertEqual(norm_group["weight_decay"], 0.0)
        # And default lr (lr_multiplier=1.0 from first pattern)
        self.assertEqual(norm_group["lr"], 1e-3)

    def test_betas_override(self):
        """Per-group betas override works correctly."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            param_groups=[
                # Override both betas
                ParamGroupConfig(pattern=r"embed_tokens\.", beta1=0.85, beta2=0.99),
                # Override only beta2
                ParamGroupConfig(pattern=r".*\.bias$", beta2=0.999),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Default group keeps global betas
        self.assertEqual(groups[0]["betas"], (0.9, 0.95))
        # Embed group: both overridden
        self.assertEqual(groups[1]["betas"], (0.85, 0.99))
        # Bias group: only beta2 overridden, beta1 stays global
        self.assertEqual(groups[2]["betas"], (0.9, 0.999))

    def test_warning_on_zero_matches(self):
        """Patterns that match no parameters emit a warning."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(pattern=r"nonexistent_layer"),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)

        with self.assertLogs(level=logging.WARNING) as cm:
            groups = OptimizersContainer._build_param_groups(
                model, config, default_kwargs
            )

        self.assertTrue(
            any("nonexistent_layer" in msg for msg in cm.output),
            f"Expected warning about unmatched pattern, got: {cm.output}",
        )
        # All params should be in the default group
        self.assertEqual(len(groups), 1)

    def test_all_params_covered(self):
        """Every requires_grad param appears in exactly one group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
                ParamGroupConfig(pattern=r".*norm.*", weight_decay_multiplier=0.0),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        all_grouped_params = []
        for g in groups:
            all_grouped_params.extend(g["params"])
        all_model_params = [p for p in model.parameters() if p.requires_grad]

        self.assertEqual(len(all_grouped_params), len(all_model_params))
        self.assertEqual(
            set(id(p) for p in all_grouped_params), set(id(p) for p in all_model_params)
        )


class TestOptimizersContainerWithParamGroups(unittest.TestCase):
    def test_build_optimizer_with_param_groups(self):
        """End-to-end: build OptimizersContainer with param groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersContainer)

        # Should have 1 optimizer (one model_part)
        self.assertEqual(len(container.optimizers), 1)
        opt = container.optimizers[0]
        # Should have 2 param groups
        self.assertEqual(len(opt.param_groups), 2)

    def test_build_optimizer_default_groups(self):
        """Empty param_groups produces standard single-group behavior."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
        )
        container = config.build(model_parts=[model])
        opt = container.optimizers[0]
        self.assertEqual(len(opt.param_groups), 1)


class TestOptimizersInBackwardWithParamGroups(unittest.TestCase):
    def test_build_with_param_groups(self):
        """OptimizersInBackwardContainer respects param groups."""
        model = SimpleModel()
        config = OptimizersInBackwardContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersInBackwardContainer)

        # Check that bias params have weight_decay=0
        param_to_name = {p: n for n, p in model.named_parameters()}
        for opt in container.optimizers:
            for pg in opt.param_groups:
                for p in pg["params"]:
                    name = param_to_name.get(p, "")
                    if name.endswith(".bias"):
                        self.assertEqual(
                            pg["weight_decay"],
                            0.0,
                            f"Bias param {name} should have weight_decay=0",
                        )


class TestDCPWithParamGroups(unittest.TestCase):
    def test_state_dict_round_trip(self):
        """Optimizer state_dict save/load works with multiple param groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
                ParamGroupConfig(pattern=r"embed_tokens\.", lr_multiplier=0.1),
            ],
        )
        container = config.build(model_parts=[model])

        # Run a step to populate optimizer state
        dummy_input = torch.randint(0, 32, (2, 4))
        output = model(dummy_input)
        output.sum().backward()
        container.step()

        # Save state dict
        state_dict = container.state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertTrue(len(state_dict) > 0)

        # Load into a fresh container
        model2 = SimpleModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)

        # Verify state was restored by checking optimizer states match
        state_dict2 = container2.state_dict()
        self.assertEqual(set(state_dict.keys()), set(state_dict2.keys()))

        for key in state_dict:
            v1 = state_dict[key]
            v2 = state_dict2[key]
            if isinstance(v1, torch.Tensor):
                self.assertTrue(
                    torch.equal(v1, v2),
                    f"State mismatch for key {key}",
                )


if __name__ == "__main__":
    unittest.main()
