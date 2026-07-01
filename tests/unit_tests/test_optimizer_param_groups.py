# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import (
    default_adamw,
    OptimizersContainer,
    ParamGroupConfig,
    register_moe_load_balancing_hook,
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


class FakeMoE(nn.Module):
    def __init__(self, load_balance_coeff, tokens):
        super().__init__()
        self.load_balance_coeff = load_balance_coeff
        self.register_buffer("tokens_per_expert_E", torch.tensor(tokens))
        if load_balance_coeff is not None:
            self.register_buffer("expert_bias_E", torch.zeros(len(tokens)))
        else:
            self.expert_bias_E = None


class FakeMoEBlock(nn.Module):
    def __init__(self, load_balance_coeff, tokens):
        super().__init__()
        self.moe_enabled = True
        self.moe = FakeMoE(load_balance_coeff, tokens)


class FakeMoEModel(nn.Module):
    def __init__(self, load_balance_coeffs=(0.1, 0.2)):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.layers = nn.ModuleDict(
            {
                "0": FakeMoEBlock(load_balance_coeffs[0], [10, 0]),
                "1": FakeMoEBlock(load_balance_coeffs[1], [0, 10]),
            }
        )


class FakeParallelDims:
    spmd_backend = "none"

    def get_optional_mesh(self, name):
        return None


# Default AdamW param group for catch-all
_DEFAULT_ADAMW = ParamGroupConfig(
    pattern=r".*",
    optimizer_name="AdamW",
    optimizer_kwargs={
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    },
)


def _get_param_names_in_group(model, group):
    """Return the set of parameter FQNs in an optimizer param group."""
    param_to_name = {p: n for n, p in model.named_parameters()}
    return {param_to_name[p] for p in group["params"]}


def _get_default_groups(model, config):
    """Helper: build param groups and return the AdamW optimizer's groups."""
    impl_kwargs = OptimizersContainer._build_impl_kwargs(config)
    param_groups = config.param_groups
    groups_by_opt, _ = OptimizersContainer._build_param_groups(
        model, param_groups, impl_kwargs
    )
    return groups_by_opt.get("AdamW", [])


class TestParamGroupConfig(unittest.TestCase):
    def test_default_no_param_groups(self):
        """Empty param_groups produces a single group with all params."""
        model = SimpleModel()
        config = default_adamw(lr=1e-3)

        groups = _get_default_groups(model, config)

        self.assertEqual(len(groups), 1)
        all_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(groups[0]["params"]), len(all_params))
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertEqual(groups[0]["weight_decay"], 0.1)

    def test_default_adam(self):
        """All params can use Adam via param_groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="Adam",
                    optimizer_kwargs={"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertEqual(len(container.optimizers), 1)
        adam = container.optimizers[0]
        self.assertIsInstance(adam, torch.optim.Adam)
        self.assertEqual(adam.param_groups[0]["lr"], 1e-2)
        self.assertEqual(adam.param_groups[0]["betas"], (0.9, 0.95))

    def test_moe_load_balancing_updates_all_enabled_layers(self):
        model = FakeMoEModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        register_moe_load_balancing_hook(
            container,
            [model],
            FakeParallelDims(),
        )

        container.step()

        torch.testing.assert_close(
            model.layers["0"].moe.expert_bias_E,
            torch.tensor([-0.1, 0.1]),
        )
        torch.testing.assert_close(
            model.layers["1"].moe.expert_bias_E,
            torch.tensor([0.2, -0.2]),
        )
        torch.testing.assert_close(
            model.layers["0"].moe.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )
        torch.testing.assert_close(
            model.layers["1"].moe.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )

    def test_moe_load_balancing_rejects_inconsistent_coeffs(self):
        model = FakeMoEModel(load_balance_coeffs=(None, 0.2))
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        with self.assertRaisesRegex(
            ValueError, "load_balance_coeff must be configured consistently"
        ):
            register_moe_load_balancing_hook(
                container,
                [model],
                FakeParallelDims(),
            )

    def test_single_pattern_weight_decay_zero(self):
        """Pattern matching bias params with weight_decay=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "betas": (0.9, 0.95),
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    },
                ),
                _DEFAULT_ADAMW,
            ],
        )
        groups = _get_default_groups(model, config)

        self.assertEqual(len(groups), 2)

        # Bias group (first match)
        bias_names = _get_param_names_in_group(model, groups[0])
        for name in bias_names:
            self.assertTrue(name.endswith(".bias"), f"{name} should end with .bias")
        self.assertEqual(groups[0]["weight_decay"], 0.0)

        # Default group (catch-all)
        default_names = _get_param_names_in_group(model, groups[1])
        for name in default_names:
            self.assertFalse(name.endswith(".bias"), f"{name} should not be in default")
        self.assertEqual(groups[1]["weight_decay"], 0.1)

    def test_lr_override(self):
        """Different lr for a param group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 1e-4,
                        "betas": (0.9, 0.95),
                        "eps": 1e-8,
                        "weight_decay": 0.1,
                    },
                ),
                _DEFAULT_ADAMW,
            ],
        )
        groups = _get_default_groups(model, config)

        # Embed group
        self.assertAlmostEqual(groups[0]["lr"], 1e-4)
        # Default group
        self.assertEqual(groups[1]["lr"], 1e-3)

    def test_base_lr_with_lr_mult(self):
        """A base lr scales each group by its lr_mult."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"weight_decay": 0.1},
                    lr_mult=0.25,
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"weight_decay": 0.1},
                    lr_mult=1.0,
                ),
            ],
        )
        opt = config.build(model_parts=[model]).optimizers[0]

        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-3 * 0.25)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1e-3 * 1.0)

    def test_base_lr_default_lr_mult_unscaled(self):
        """lr_mult defaults to 1.0 so a single base lr applies unscaled."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            lr=5e-4,
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"weight_decay": 0.1},
                ),
            ],
        )
        opt = config.build(model_parts=[model]).optimizers[0]

        self.assertAlmostEqual(opt.param_groups[0]["lr"], 5e-4)

    def test_base_lr_rejects_group_lr(self):
        """Setting lr in optimizer_kwargs while a base lr is set raises ValueError."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.1},
                ),
            ],
        )
        with self.assertRaises(ValueError):
            config.build(model_parts=[model])

    def test_first_match_wins(self):
        """When patterns overlap, the first match wins."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            param_groups=[
                # First pattern: all norm params get wd=0
                ParamGroupConfig(
                    pattern=r".*norm.*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "betas": (0.9, 0.95),
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    },
                ),
                # Second pattern: broader match that also covers norm
                ParamGroupConfig(
                    pattern=r".*layers.*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 5e-4,
                        "betas": (0.9, 0.95),
                        "eps": 1e-8,
                        "weight_decay": 0.1,
                    },
                ),
                _DEFAULT_ADAMW,
            ],
        )
        groups = _get_default_groups(model, config)

        norm_group = groups[0]
        norm_names = _get_param_names_in_group(model, norm_group)
        self.assertTrue(all("norm" in n for n in norm_names))
        self.assertEqual(norm_group["weight_decay"], 0.0)
        self.assertEqual(norm_group["lr"], 1e-3)

    def test_betas_override(self):
        """Per-group betas override."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "betas": (0.85, 0.99),
                        "eps": 1e-8,
                        "weight_decay": 0.1,
                    },
                ),
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_name="AdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": 0.1,
                    },
                ),
                _DEFAULT_ADAMW,
            ],
        )
        groups = _get_default_groups(model, config)

        self.assertEqual(groups[0]["betas"], (0.85, 0.99))
        self.assertEqual(groups[1]["betas"], (0.9, 0.999))
        self.assertEqual(groups[2]["betas"], (0.9, 0.95))

    def test_error_on_zero_matches(self):
        """Patterns that match no parameters raise ValueError."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r"nonexistent_layer",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3},
                ),
                _DEFAULT_ADAMW,
            ],
        )
        impl_kwargs = OptimizersContainer._build_impl_kwargs(config)

        with self.assertRaises(ValueError):
            OptimizersContainer._build_param_groups(
                model, config.param_groups, impl_kwargs
            )

    def test_all_params_covered(self):
        """Every requires_grad param appears in exactly one group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r".*norm.*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
                ),
                _DEFAULT_ADAMW,
            ],
        )
        impl_kwargs = OptimizersContainer._build_impl_kwargs(config)
        groups_by_opt, _ = OptimizersContainer._build_param_groups(
            model, config.param_groups, impl_kwargs
        )

        all_grouped_params = []
        for opt_groups in groups_by_opt.values():
            for g in opt_groups:
                all_grouped_params.extend(g["params"])
        all_model_params = [p for p in model.parameters() if p.requires_grad]

        self.assertEqual(len(all_grouped_params), len(all_model_params))
        self.assertEqual(
            set(id(p) for p in all_grouped_params),
            set(id(p) for p in all_model_params),
        )

    def test_uncovered_params_raises(self):
        """Missing catch-all pattern raises on uncovered params."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3},
                ),
            ],
        )
        with self.assertRaises(AssertionError):
            config.build(model_parts=[model])


class TestOptimizersContainerWithParamGroups(unittest.TestCase):
    def test_build_optimizer_with_param_groups(self):
        """End-to-end: build OptimizersContainer with param groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersContainer)
        self.assertEqual(len(container.optimizers), 1)
        opt = container.optimizers[0]
        self.assertEqual(len(opt.param_groups), 2)

    def test_build_optimizer_default_groups(self):
        """default_adamw produces standard single-group behavior."""
        model = SimpleModel()
        config = default_adamw(lr=1e-3)
        config.implementation = "for-loop"
        container = config.build(model_parts=[model])
        opt = container.optimizers[0]
        self.assertEqual(len(opt.param_groups), 1)


class TestDCPWithParamGroups(unittest.TestCase):
    def test_state_dict_round_trip(self):
        """Optimizer state_dict save/load works with multiple param groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*\.bias$",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.1},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
                ),
            ],
        )
        container = config.build(model_parts=[model])

        dummy_input = torch.randint(0, 32, (2, 4))
        output = model(dummy_input)
        output.sum().backward()
        container.step()

        state_dict = container.state_dict()
        self.assertIsInstance(state_dict, dict)
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
                self.assertTrue(
                    torch.equal(v1, v2),
                    f"State mismatch for key {key}",
                )


class TestMixedOptimizers(unittest.TestCase):
    def test_mixed_optimizer_types(self):
        """Different optimizer for a param group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="Adam",
                    optimizer_kwargs={"lr": 5e-4, "betas": (0.9, 0.95), "eps": 1e-8},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        opt_types = {type(opt).__name__ for opt in container.optimizers}
        self.assertEqual(opt_types, {"AdamW", "Adam"})

        adam = next(
            opt for opt in container.optimizers if type(opt) is torch.optim.Adam
        )
        self.assertEqual(adam.param_groups[0]["lr"], 5e-4)
        self.assertEqual(adam.param_groups[0]["betas"], (0.9, 0.95))

    def test_pattern_not_leaked_to_state_dict(self):
        """Pattern is logging-only; it must not enter the optimizer or state dict."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
                ),
            ],
        )
        container = config.build(model_parts=[model])
        # Pattern is popped before optimizer construction, so it never reaches
        # the optimizer's param groups or the saved (flat) state dict.
        for opt in container.optimizers:
            for group in opt.param_groups:
                self.assertNotIn("pattern", group)
        self.assertFalse(any(".pattern" in key for key in container.state_dict()))

    def test_mixed_optimizer_state_dict_round_trip(self):
        """State dict save/load works with mixed optimizer types."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="Adam",
                    optimizer_kwargs={"lr": 5e-4, "betas": (0.9, 0.95), "eps": 1e-8},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
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
        config = default_adamw(lr=1e-3)
        config.implementation = "for-loop"
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

    def test_get_metrics_reports_lr_per_group(self):
        """get_metrics reports a learning rate per optimizer param group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
                ),
            ],
        )
        lr_config = LRSchedulersContainer.Config(warmup_steps=10, decay_type="linear")
        scheduler, _ = self._build_scheduler(config, lr_config, model)
        metrics = scheduler.get_metrics()
        # One AdamW optimizer with two param groups -> indexed lr keys.
        self.assertEqual(set(metrics), {"lr/AdamW/0", "lr/AdamW/1"})

    def test_mixed_optimizer_gets_separate_schedulers(self):
        """Mixed optimizers should each get their own scheduler."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="Adam",
                    optimizer_kwargs={"lr": 5e-4, "betas": (0.9, 0.95), "eps": 1e-8},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
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
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"output\.",
                    optimizer_name="Adam",
                    optimizer_kwargs={"lr": 5e-4, "betas": (0.9, 0.95), "eps": 1e-8},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
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
            if type(opt) is torch.optim.Adam:
                self.assertAlmostEqual(base_lr, 5e-4, places=6)
            else:
                self.assertAlmostEqual(base_lr, 1e-3, places=6)


class TestViTMuPParamGroups(unittest.TestCase):
    def test_mup_group_scales_lr_and_wd_across_widths(self):
        """muP scales the hidden-matmul group by 1/m and its wd by m; catch-all unscaled."""
        from torchtitan.experiments.path.config_registry import (
            _vit_optimizer_config,
            BASE_WIDTH,
            MUP_PATTERN,
        )

        base_lr, base_wd = 3e-4, 0.0125
        for flavor, width in (("w256", 256), ("w512", 512)):
            config = _vit_optimizer_config(flavor, mup=True, lr=base_lr, wd=base_wd)
            mup_group, catch_all = config.param_groups[0], config.param_groups[-1]

            self.assertEqual(mup_group.pattern, MUP_PATTERN)
            self.assertAlmostEqual(mup_group.lr_mult, BASE_WIDTH / width)
            self.assertAlmostEqual(
                mup_group.optimizer_kwargs["weight_decay"], base_wd * width / BASE_WIDTH
            )

            self.assertEqual(catch_all.pattern, r".*")
            self.assertAlmostEqual(catch_all.lr_mult, 1.0)
            self.assertAlmostEqual(catch_all.optimizer_kwargs["weight_decay"], base_wd)

    def test_mup_pattern_matches_hidden_matmuls_exactly(self):
        """MUP_PATTERN selects exactly the per-block attention/mlp matmul weights."""
        import re

        from torchtitan.experiments.path.config_registry import (
            _vit_model_config,
            MUP_PATTERN,
        )

        model = _vit_model_config("w256", mup=True).build()
        param_names = {name for name, _ in model.named_parameters()}
        expected = {
            f"blocks.{i}.{submodule}.{leaf}.weight"
            for i in range(len(model.blocks))
            for submodule, leaf in (
                ("attention", "c_attn"),
                ("attention", "c_proj"),
                ("mlp", "c_fc"),
                ("mlp", "c_proj"),
            )
        }
        self.assertEqual(len(expected), 4 * len(model.blocks))
        self.assertTrue(
            expected <= param_names,
            f"hidden matmul weights missing from model: {expected - param_names}",
        )

        matched = {name for name in param_names if re.search(MUP_PATTERN, name)}
        self.assertEqual(matched, expected)


if __name__ == "__main__":
    unittest.main()
