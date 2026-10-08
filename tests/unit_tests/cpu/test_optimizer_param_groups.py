# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torchtitan.components.optim import (
    Adam,
    AdamW,
    BaseOptimizer,
    LRSchedulersContainer,
    OptimizersContainer,
)
from torchtitan.models.common.moe import register_moe_load_balancing_hook


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


class FakeRouter(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.register_buffer("tokens_per_expert_E", torch.tensor(tokens))


class FakeMoE(nn.Module):
    def __init__(self, load_balance_coeff, tokens):
        super().__init__()
        self.load_balance_coeff = load_balance_coeff
        self.router = FakeRouter(tokens)
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
    def __init__(self, load_balance_coeffs=(0.1, 0.2), mtp_load_balance_coeff=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.layers = nn.ModuleDict(
            {
                "0": FakeMoEBlock(load_balance_coeffs[0], [10, 0]),
                "1": FakeMoEBlock(load_balance_coeffs[1], [0, 10]),
            }
        )
        self.mtp_layers = nn.ModuleList()
        if mtp_load_balance_coeff is not None:
            self.mtp_layers.append(FakeMoEBlock(mtp_load_balance_coeff, [3, 1]))


class FakeParallelismContext:
    ep_enabled = False
    tp = 1

    def __init__(self, *, loss_mesh=None):
        self.loss_mesh = loss_mesh

    def get_optional_mesh(self, name):
        return self.loss_mesh if name == "loss" else None


# Default AdamW configuration for catch-all
_DEFAULT_ADAMW = AdamW.Config(
    pattern=r".*",
    lr=1e-3,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=False,
)


def _get_param_names_in_group(model, group):
    """Return the set of parameter FQNs in an optimizer param group."""
    param_to_name = {p: n for n, p in model.named_parameters()}
    return {param_to_name[p] for p in group["params"]}


def _get_default_groups(model, config):
    """Helper: build param groups and return the AdamW optimizer's groups."""
    container = config.build(model_parts=[model])
    return [
        group
        for optimizer in container.optimizers
        if isinstance(optimizer, AdamW)
        for group in optimizer.param_groups
    ]


class TestOptimizerConfig(unittest.TestCase):
    def test_external_optimizer_subclass(self):
        class ExternalSGD(torch.optim.SGD, BaseOptimizer):
            @dataclass(kw_only=True, slots=True)
            class Config(BaseOptimizer.Config):
                lr: float

            def __init__(self, config: Config, *, params) -> None:
                super().__init__(params, lr=config.lr)

        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[ExternalSGD.Config(pattern=r".*", lr=0.25)]
        )

        container = config.build(model_parts=[model])

        self.assertIsInstance(container.optimizers[0], ExternalSGD)
        self.assertEqual(container.optimizers[0].param_groups[0]["lr"], 0.25)

    def test_bfloat16_moments_require_fused_adam(self):
        with self.assertRaisesRegex(ValueError, "require fused=True"):
            AdamW.Config(
                pattern=r".*",
                fused=False,
                moment_dtype="bfloat16",
            )

    def test_catch_all_optimizer(self):
        """A catch-all optimizer selects every trainable parameter."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3, fused=False)]
        )

        groups = _get_default_groups(model, config)

        self.assertEqual(len(groups), 1)
        all_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(groups[0]["params"]), len(all_params))
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertEqual(groups[0]["weight_decay"], 0.1)

    def test_default_adam(self):
        """All parameters can use a configured Adam optimizer."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                Adam.Config(
                    pattern=r".*",
                    fused=False,
                    lr=1e-2,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.2,
                ),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertEqual(len(container.optimizers), 1)
        adam = container.optimizers[0]
        self.assertIsInstance(adam, torch.optim.Adam)
        self.assertEqual(adam.param_groups[0]["lr"], 1e-2)
        self.assertEqual(adam.param_groups[0]["betas"], (0.9, 0.95))
        self.assertEqual(adam.param_groups[0]["weight_decay"], 0.2)

    def test_moe_load_balancing_updates_all_enabled_layers(self):
        model = FakeMoEModel()
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(pattern=r".*", fused=False, lr=0.0, weight_decay=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        register_moe_load_balancing_hook(
            container,
            [model],
            FakeParallelismContext(),
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
            model.layers["0"].moe.router.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )
        torch.testing.assert_close(
            model.layers["1"].moe.router.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )

    def test_moe_load_balancing_halves_full_ac_counts(self):
        """FullAC replays the router forward, so its counts are halved."""
        model = FakeMoEModel()
        model.layers["0"].checkpoint_impl = CheckpointImpl.NO_REENTRANT
        model.layers["0"].moe.router.tokens_per_expert_E.copy_(torch.tensor([20, 0]))
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(pattern=r".*", fused=False, lr=0.0, weight_decay=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        register_moe_load_balancing_hook(container, [model], FakeParallelismContext())

        # Capture the counts the hook reduces over before it zeroes the buffers.
        vstack_inputs = []
        vstack = torch.vstack

        def recording_vstack(tensors):
            vstack_inputs.append([t.clone() for t in tensors])
            return vstack(tensors)

        with unittest.mock.patch("torch.vstack", recording_vstack):
            container.step()

        torch.testing.assert_close(
            vstack_inputs,
            [[torch.tensor([10, 0]), torch.tensor([0, 10])]],
        )

    def test_moe_load_balancing_rejects_inconsistent_coeffs(self):
        model = FakeMoEModel(load_balance_coeffs=(None, 0.2))
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(pattern=r".*", fused=False, lr=0.0, weight_decay=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        with self.assertRaisesRegex(
            ValueError, "load_balance_coeff must be configured consistently"
        ):
            register_moe_load_balancing_hook(
                container,
                [model],
                FakeParallelismContext(),
            )

    def test_moe_load_balancing_updates_mtp_layers(self):
        model = FakeMoEModel(mtp_load_balance_coeff=0.3)
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(pattern=r".*", fused=False, lr=0.0, weight_decay=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        register_moe_load_balancing_hook(container, [model], FakeParallelismContext())

        container.step()

        torch.testing.assert_close(
            model.mtp_layers[0].moe.expert_bias_E,
            torch.tensor([-0.3, 0.3]),
        )
        torch.testing.assert_close(
            model.mtp_layers[0].moe.router.tokens_per_expert_E,
            torch.tensor([0, 0]),
        )

    def test_single_pattern_weight_decay_zero(self):
        """Pattern matching bias params with weight_decay=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*\.bias$",
                    fused=False,
                    lr=1e-3,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
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
            optimizers=[
                AdamW.Config(
                    pattern=r"embed_tokens\.",
                    fused=False,
                    lr=1e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.1,
                ),
                _DEFAULT_ADAMW,
            ],
        )
        groups = _get_default_groups(model, config)

        # Embed group
        self.assertAlmostEqual(groups[0]["lr"], 1e-4)
        # Default group
        self.assertEqual(groups[1]["lr"], 1e-3)

    def test_first_match_wins(self):
        """When patterns overlap, the first match wins."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                # First pattern: all norm params get wd=0
                AdamW.Config(
                    pattern=r".*norm.*",
                    fused=False,
                    lr=1e-3,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
                ),
                # Second pattern: broader match that also covers norm
                AdamW.Config(
                    pattern=r".*layers.*",
                    fused=False,
                    lr=5e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.1,
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
            optimizers=[
                AdamW.Config(
                    pattern=r"embed_tokens\.",
                    fused=False,
                    lr=1e-3,
                    betas=(0.85, 0.99),
                    eps=1e-8,
                    weight_decay=0.1,
                ),
                AdamW.Config(
                    pattern=r".*\.bias$",
                    fused=False,
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.1,
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
            optimizers=[
                AdamW.Config(pattern=r"nonexistent_layer", fused=False, lr=1e-3),
                _DEFAULT_ADAMW,
            ],
        )
        with self.assertRaises(ValueError):
            config.build(model_parts=[model])

    def test_all_params_covered(self):
        """Every requires_grad param appears in exactly one group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*\.bias$",
                    fused=False,
                    lr=1e-3,
                    weight_decay=0.0,
                ),
                AdamW.Config(
                    pattern=r".*norm.*",
                    fused=False,
                    lr=1e-3,
                    weight_decay=0.0,
                ),
                _DEFAULT_ADAMW,
            ],
        )
        container = config.build(model_parts=[model])
        all_grouped_params = [
            param
            for optimizer in container.optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]
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
            optimizers=[
                AdamW.Config(pattern=r"output\.", fused=False, lr=1e-3),
            ],
        )
        with self.assertRaises(ValueError):
            config.build(model_parts=[model])


class TestOptimizersContainerWithParamGroups(unittest.TestCase):
    def test_build_optimizer_with_param_groups(self):
        """End-to-end: build a container with multiple optimizer patterns."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*\.bias$",
                    fused=False,
                    lr=1e-3,
                    weight_decay=0.0,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersContainer)
        self.assertEqual(len(container.optimizers), 2)
        self.assertTrue(
            all(len(optimizer.param_groups) == 1 for optimizer in container.optimizers)
        )

    def test_build_optimizer_default_groups(self):
        """A catch-all AdamW config produces standard single-group behavior."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3, fused=False)]
        )
        container = config.build(model_parts=[model])
        opt = container.optimizers[0]
        self.assertEqual(len(opt.param_groups), 1)


class TestDCPWithParamGroups(unittest.TestCase):
    def test_state_dict_round_trip(self):
        """Optimizer state_dict save/load works with multiple optimizers."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*\.bias$",
                    fused=False,
                    lr=1e-3,
                    weight_decay=0.0,
                ),
                AdamW.Config(
                    pattern=r"embed_tokens\.",
                    fused=False,
                    lr=1e-4,
                    weight_decay=0.1,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
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

    def test_cuda_graph_optimizer_state_round_trip(self):
        source_model = torch.nn.Linear(2, 2)
        config = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
        )
        source = config.build(model_parts=[source_model], enable_cuda_graph=True)
        source_model(torch.ones(1, 2)).sum().backward()
        source.step()
        old_group = source.optimizers[0].param_groups[0]
        old_group["lr"].fill_(5e-4)
        state_dict = source.state_dict()
        self.assertFalse(any(key.endswith(".capturable") for key in state_dict))
        for fqn in old_group["param_names"]:
            state_dict[f"param_groups.{fqn}.capturable"] = False
        source.load_state_dict(state_dict)
        captured_group = source.optimizers[0].param_groups[0]
        self.assertIsNot(captured_group, old_group)
        self.assertTrue(captured_group["capturable"])
        self.assertIsInstance(captured_group["lr"], torch.Tensor)

        with TemporaryDirectory() as checkpoint_dir:
            dcp.save({"optimizer": source}, checkpoint_id=checkpoint_dir, no_dist=True)
            target_model = torch.nn.Linear(2, 2)
            target = OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
            ).build(model_parts=[target_model])
            dcp.load({"optimizer": target}, checkpoint_id=checkpoint_dir, no_dist=True)
            captured_target = config.build(
                model_parts=[torch.nn.Linear(2, 2)],
                enable_cuda_graph=True,
            )
            dcp.load(
                {"optimizer": captured_target},
                checkpoint_id=checkpoint_dir,
                no_dist=True,
            )
            captured_target.step()

        target_group = target.optimizers[0].param_groups[0]
        self.assertIsInstance(target_group["lr"], float)
        self.assertAlmostEqual(target_group["lr"], 5e-4)
        self.assertFalse(target_group["capturable"])
        self.assertTrue(captured_target.optimizers[0].param_groups[0]["capturable"])
        self.assertIsInstance(
            captured_target.optimizers[0].param_groups[0]["lr"], torch.Tensor
        )

    def test_cuda_graph_setting_controls_capturable(self):
        eager = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
        ).build(model_parts=[torch.nn.Linear(2, 2)])
        captured = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
        ).build(
            model_parts=[torch.nn.Linear(2, 2)],
            enable_cuda_graph=True,
        )

        self.assertFalse(eager.optimizers[0].param_groups[0]["capturable"])
        self.assertTrue(captured.optimizers[0].param_groups[0]["capturable"])

    def test_optimizer_config_rejects_capturable(self):
        with self.assertRaises(TypeError):
            OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-3, capturable=True)]
            )


class TestMixedOptimizers(unittest.TestCase):
    def test_mixed_optimizer_types(self):
        """Different optimizer for a param group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                Adam.Config(
                    pattern=r"output\.",
                    fused=False,
                    lr=5e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
            ],
        )
        container = config.build(model_parts=[model])
        opt_types = {type(opt).__name__ for opt in container.optimizers}
        self.assertEqual(opt_types, {"AdamW", "Adam"})

        adam = next(opt for opt in container.optimizers if type(opt) is Adam)
        self.assertEqual(adam.param_groups[0]["lr"], 5e-4)
        self.assertEqual(adam.param_groups[0]["betas"], (0.9, 0.95))

    def test_pattern_not_leaked_to_state_dict(self):
        """Pattern is logging-only; it must not enter the optimizer or state dict."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r"output\.",
                    fused=False,
                    lr=1e-3,
                    weight_decay=0.0,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
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
            optimizers=[
                Adam.Config(
                    pattern=r"output\.",
                    fused=False,
                    lr=5e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
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
        config = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3, fused=False)]
        )
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
            optimizers=[
                AdamW.Config(
                    pattern=r"output\.",
                    fused=False,
                    lr=5e-4,
                    weight_decay=0.0,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
            ],
        )
        lr_config = LRSchedulersContainer.Config(warmup_steps=10, decay_type="linear")
        scheduler, _ = self._build_scheduler(config, lr_config, model)
        metrics = scheduler.get_metrics()
        # Repeated AdamW instances receive stable indexed metric keys.
        self.assertEqual(set(metrics), {"lr/AdamW/0", "lr/AdamW/1"})

    def test_mixed_optimizer_gets_separate_schedulers(self):
        """Mixed optimizers should each get their own scheduler."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            optimizers=[
                Adam.Config(
                    pattern=r"output\.",
                    fused=False,
                    lr=5e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
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
            optimizers=[
                Adam.Config(
                    pattern=r"output\.",
                    fused=False,
                    lr=5e-4,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                ),
                AdamW.Config(pattern=r".*", fused=False, lr=1e-3, weight_decay=0.1),
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
            if type(opt) is Adam:
                self.assertAlmostEqual(base_lr, 5e-4, places=6)
            else:
                self.assertAlmostEqual(base_lr, 1e-3, places=6)

    def test_first_capturable_step_uses_stable_tensor_lr(self):
        model = torch.nn.Linear(2, 2)
        container = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
        ).build(model_parts=[model], enable_cuda_graph=True)
        optimizer = container.optimizers[0]
        group = optimizer.param_groups[0]
        lr = group["lr"]
        self.assertIsInstance(lr, torch.Tensor)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 / (step + 1)
        )

        model(torch.ones(1, 2)).sum().backward()
        container.step()
        scheduler.step()

        self.assertTrue(group["capturable"])
        self.assertEqual(group["lr"].device, model.weight.device)
        self.assertAlmostEqual(group["lr"].item(), 5e-4)

        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()

        self.assertIs(group["lr"], lr)
        self.assertAlmostEqual(group["lr"].item(), 1e-3 / 3)

    def test_cuda_graph_lr_metrics_stay_on_host(self):
        model = torch.nn.Linear(2, 2)
        config = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=1e-3)]
        )
        optimizers = config.build(model_parts=[model], enable_cuda_graph=True)
        group = optimizers.optimizers[0].param_groups[0]
        lr = group["lr"]
        schedulers = LRSchedulersContainer(
            optimizers, lr_lambda=lambda step: 1.0 / (step + 1)
        )

        optimizers.step()
        schedulers.step()

        self.assertIsInstance(group["lr"], torch.Tensor)
        self.assertIs(group["lr"], lr)
        self.assertEqual(schedulers.schedulers[0].base_lrs, [1e-3])
        self.assertIsInstance(schedulers.get_host_lrs_per_scheduler()[0][0], float)
        self.assertAlmostEqual(schedulers.get_metrics()["lr/AdamW"], 5e-4)

    def test_cuda_graph_rejects_unsupported_optimizer(self):
        class SGD(torch.optim.SGD, BaseOptimizer):
            @dataclass(kw_only=True, slots=True)
            class Config(BaseOptimizer.Config):
                lr: float = 1e-3

            def __init__(self, config: Config, *, params) -> None:
                super().__init__(params, lr=config.lr)

        model = torch.nn.Linear(2, 2)
        config = OptimizersContainer.Config(
            optimizers=[SGD.Config(pattern=r".*")],
        )

        with self.assertRaisesRegex(ValueError, "SGD does not support"):
            config.build(model_parts=[model], enable_cuda_graph=True)


if __name__ == "__main__":
    unittest.main()
