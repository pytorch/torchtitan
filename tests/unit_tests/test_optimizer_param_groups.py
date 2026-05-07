# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import torch
import torch.nn as nn
from torchtitan.components.lr_scheduler import LRSchedulersContainer
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


def _get_default_groups(model, config):
    """Helper: build param groups and return the default optimizer's groups."""
    default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
    groups_by_opt = OptimizersContainer._build_param_groups(
        model, config, default_kwargs
    )
    return groups_by_opt.get(config.name, [])


class TestParamGroupConfig(unittest.TestCase):
    def test_default_no_param_groups(self):
        """Empty param_groups produces a single group with all params."""
        model = SimpleModel()
        config = OptimizersContainer.Config(lr=1e-3)

        groups = _get_default_groups(model, config)

        self.assertEqual(len(groups), 1)
        all_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(groups[0]["params"]), len(all_params))
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertEqual(groups[0]["weight_decay"], 0.1)

    def test_single_pattern_weight_decay_zero(self):
        """Pattern matching bias params with weight_decay=0 via optimizer_kwargs."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
            ],
        )
        groups = _get_default_groups(model, config)

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
        """Pattern matching embed_tokens with weight_decay=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
            ],
        )
        groups = _get_default_groups(model, config)

        self.assertEqual(len(groups), 2)

        embed_names = _get_param_names_in_group(model, groups[1])
        self.assertTrue(
            all("embed_tokens" in n for n in embed_names),
            f"Expected embed_tokens params, got {embed_names}",
        )
        self.assertEqual(groups[1]["weight_decay"], 0.0)

    def test_lr_override(self):
        """lr override via optimizer_kwargs correctly sets the lr."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_kwargs={"lr": 1e-4},
                ),
            ],
        )
        groups = _get_default_groups(model, config)

        # Default group keeps base lr
        self.assertEqual(groups[0]["lr"], 1e-3)
        # Embed group gets overridden lr
        self.assertAlmostEqual(groups[1]["lr"], 1e-4)

    def test_first_match_wins(self):
        """When patterns overlap, the first match wins."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                # First pattern: all norm params get wd=0
                ParamGroupConfig(
                    pattern=r".*norm.*",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
                # Second pattern: broader match that also covers norm
                ParamGroupConfig(
                    pattern=r".*layers.*",
                    optimizer_kwargs={"lr": 5e-4},
                ),
            ],
        )
        groups = _get_default_groups(model, config)

        # Norm params should be in group index 1 (first matched group after default)
        norm_group = groups[1]
        norm_names = _get_param_names_in_group(model, norm_group)
        self.assertTrue(all("norm" in n for n in norm_names))
        # Norm group should have weight_decay=0 (from first pattern)
        self.assertEqual(norm_group["weight_decay"], 0.0)
        # And default lr (not overridden by first pattern)
        self.assertEqual(norm_group["lr"], 1e-3)

    def test_betas_override(self):
        """Per-group betas override via optimizer_kwargs works correctly."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_kwargs={"betas": (0.85, 0.99)},
                ),
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_kwargs={"betas": (0.9, 0.999)},
                ),
            ],
        )
        groups = _get_default_groups(model, config)

        # Default group keeps global betas
        self.assertEqual(groups[0]["betas"], (0.9, 0.95))
        # Embed group: both overridden
        self.assertEqual(groups[1]["betas"], (0.85, 0.99))
        # Bias group: overridden
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
            groups_by_opt = OptimizersContainer._build_param_groups(
                model, config, default_kwargs
            )

        self.assertTrue(
            any("nonexistent_layer" in msg for msg in cm.output),
            f"Expected warning about unmatched pattern, got: {cm.output}",
        )
        # All params should be in the default group
        groups = groups_by_opt.get(config.name, [])
        self.assertEqual(len(groups), 1)

    def test_all_params_covered(self):
        """Every requires_grad param appears in exactly one group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r".*norm.*",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups_by_opt = OptimizersContainer._build_param_groups(
            model, config, default_kwargs
        )

        all_grouped_params = []
        for opt_groups in groups_by_opt.values():
            for g in opt_groups:
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
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersContainer)

        # Should have 1 optimizer (one model_part, same optimizer type)
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
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
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
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_kwargs={"lr": 1e-4},
                ),
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


class TestMixedOptimizers(unittest.TestCase):
    def test_mixed_optimizer_types(self):
        """Different optimizer for a param group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 5e-4, "momentum": 0.9},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        opt_types = {type(opt).__name__ for opt in container.optimizers}
        self.assertEqual(opt_types, {"AdamW", "SGD"})

        sgd = next(
            opt for opt in container.optimizers if isinstance(opt, torch.optim.SGD)
        )
        self.assertEqual(sgd.param_groups[0]["lr"], 5e-4)
        self.assertEqual(sgd.param_groups[0]["momentum"], 0.9)

    def test_mixed_optimizers_all_params_covered(self):
        """All parameters should be covered across all optimizers."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 5e-4, "momentum": 0.9},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        all_opt_params = set()
        for opt in container.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    all_opt_params.add(id(p))
        model_params = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(all_opt_params, model_params)

    def test_model_part_indices(self):
        """_model_part_indices correctly maps optimizers to model parts."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 5e-4, "momentum": 0.9},
                ),
            ],
        )
        container = config.build(model_parts=[model1, model2])
        self.assertEqual(len(container.optimizers), 4)
        self.assertEqual(container._model_part_indices[0], 0)
        self.assertEqual(container._model_part_indices[1], 0)
        self.assertEqual(container._model_part_indices[2], 1)
        self.assertEqual(container._model_part_indices[3], 1)

    def test_param_group_labels(self):
        """Param groups have correct _label for logging."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_kwargs={"weight_decay": 0.0},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        adamw = container.optimizers[0]
        self.assertEqual(adamw.param_groups[0]["_label"], "default")
        self.assertEqual(adamw.param_groups[1]["_label"], "output.")

    def test_mixed_optimizer_state_dict_round_trip(self):
        """State dict save/load works with mixed optimizer types."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 5e-4, "momentum": 0.9},
                ),
            ],
        )
        container = config.build(model_parts=[model])

        dummy_input = torch.randint(0, 32, (2, 4))
        output = model(dummy_input)
        output.sum().backward()
        container.step()

        state_dict = container.state_dict()
        self.assertTrue(len(state_dict) > 0)

        model2 = SimpleModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)

        state_dict2 = container2.state_dict()
        self.assertEqual(set(state_dict.keys()), set(state_dict2.keys()))
        for key in state_dict:
            v1 = state_dict[key]
            v2 = state_dict2[key]
            if isinstance(v1, torch.Tensor):
                self.assertTrue(torch.equal(v1, v2), f"State mismatch for key {key}")


class TestLRSchedulerWithMixedOptimizers(unittest.TestCase):
    def _build_scheduler(self, config, lr_config, model, training_steps=100):
        container = config.build(model_parts=[model])
        for opt in container.optimizers:
            opt._opt_called = True
        return (
            lr_config.build(
                optimizers=container,
                training_steps=training_steps,
            ),
            container,
        )

    def test_default_schedule(self):
        """Default schedule should work the same as before."""
        model = SimpleModel()
        config = OptimizersContainer.Config(lr=1e-3, implementation="for-loop")
        lr_config = LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_type="linear",
        )
        scheduler, container = self._build_scheduler(config, lr_config, model)
        self.assertEqual(len(scheduler.schedulers), 1)
        for _ in range(10):
            scheduler.step()
        lr = scheduler.schedulers[0].optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lr, 1e-3, places=6)

    def test_mixed_optimizer_gets_separate_schedulers(self):
        """Mixed optimizers should each get their own scheduler."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 5e-4, "momentum": 0.9},
                ),
            ],
        )
        lr_config = LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_type="linear",
        )
        scheduler, container = self._build_scheduler(config, lr_config, model)
        self.assertEqual(len(scheduler.schedulers), 2)

    def test_mixed_optimizer_same_schedule_different_base_lr(self):
        """Same schedule applied to different base lrs produces different absolute lrs."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="SGD",
                    optimizer_kwargs={"lr": 5e-4, "momentum": 0.9},
                ),
            ],
        )
        lr_config = LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_type="linear",
        )
        scheduler, container = self._build_scheduler(config, lr_config, model)
        for _ in range(10):
            scheduler.step()
        for opt in container.optimizers:
            base_lr = opt.param_groups[0]["lr"]
            if isinstance(opt, torch.optim.SGD):
                self.assertAlmostEqual(base_lr, 5e-4, places=6)
            else:
                self.assertAlmostEqual(base_lr, 1e-3, places=6)


if __name__ == "__main__":
    unittest.main()
