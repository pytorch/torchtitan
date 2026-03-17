# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleDict, ModuleList, Sequential


class TestModuleInitWeights(unittest.TestCase):
    """Tests for Module.init_weights behavior.

    Module.init_weights provides a default no-op implementation so that
    subclasses without learnable parameters (or loaded from checkpoints)
    do not need to override it.
    """

    def test_default_init_weights_is_noop(self):
        """Subclass without init_weights gets the default no-op."""

        class SimpleModule(Module):
            def __init__(self):
                super().__init__()

        m = SimpleModule()
        m.init_weights()  # should not raise

    def test_init_weights_implemented(self):
        """Subclass with init_weights works normally."""

        class GoodModule(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear.Config(bias=True).build(
                    in_features=4, out_features=4
                )

            def init_weights(self, **kwargs):
                nn.init.zeros_(self.linear.weight)

        m = GoodModule()
        m.init_weights()
        self.assertTrue(torch.all(m.linear.weight == 0))


class TestDiamondInheritance(unittest.TestCase):
    """Tests for diamond inheritance: class Foo(nn.SomeModule, Module).

    The diamond pattern will be used if a Module component also inherits from
    an nn.Module (e.g., nn.Embedding) to reuse PyTorch's implementation while
    satisfying the Module protocol.
    These tests ensure the pattern works correctly and catches regressions.
    """

    class TestEmbedding(nn.Embedding, Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__(num_embeddings, embedding_dim)

        def init_weights(self, **kwargs):
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def test_module_has_no_init(self):
        """Module must not define __init__ in its own __dict__.

        If Module defined __init__, diamond inheritance (e.g. nn.Embedding +
        Module) may break the __init__ structure.
        """
        self.assertNotIn(
            "__init__",
            Module.__dict__,
            "Module must not define __init__. Adding __init__ to Module "
            "may break diamond inheritance (e.g. nn.Embedding + Module). "
            "Please verify all the use cases and ensure the change doesn't "
            "break them. After that, we can consider to remove this test.",
        )

    def test_forward(self):
        """Forward pass works through nn.Embedding's implementation."""
        emb = self.TestEmbedding(100, 32)
        out = emb(torch.tensor([0, 1, 2]))
        self.assertEqual(out.shape, torch.Size([3, 32]))

    def test_init_weights(self):
        """init_weights runs the subclass implementation."""
        emb = self.TestEmbedding(100, 32)
        emb.init_weights()
        # Weight should have been re-initialized; just check it's finite
        self.assertTrue(torch.all(torch.isfinite(emb.weight)))

    def test_isinstance_checks(self):
        """Diamond class is an instance of all parent types."""
        emb = self.TestEmbedding(100, 32)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertIsInstance(emb, nn.Module)
        self.assertIsInstance(emb, Module)

    def test_default_init_weights_noop_diamond(self):
        """Diamond class without init_weights gets the default no-op."""

        class SimpleEmbedding(nn.Embedding, Module):
            def __init__(self, num_embeddings, embedding_dim):
                super().__init__(num_embeddings, embedding_dim)

        emb = SimpleEmbedding(10, 4)
        emb.init_weights()  # should not raise

    def test_module_hierarchy_is_flat(self):
        """Diamond embedding adds no extra layer to the module tree."""

        class Model(Module):
            def __init__(self):
                super().__init__()
                self.embed = TestDiamondInheritance.TestEmbedding(100, 32)
                self.linear = Linear.Config(bias=True).build(
                    in_features=32, out_features=16
                )

        model = Model()
        param_names = {name for name, _ in model.named_parameters()}
        self.assertEqual(param_names, {"embed.weight", "linear.weight", "linear.bias"})

    def test_nn_module_init_called_once(self):
        """nn.Module.__init__ is called exactly once (no double init)."""
        call_count = 0
        orig_init = nn.Module.__init__

        def counting_init(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            orig_init(self, *args, **kwargs)

        nn.Module.__init__ = counting_init
        try:
            self.TestEmbedding(50, 16)
            self.assertEqual(call_count, 1)
        finally:
            nn.Module.__init__ = orig_init


class TestFromNnModule(unittest.TestCase):
    """Tests for Module.from_nn_module utility."""

    def test_is_subclass(self):
        """Created class is subclass of both original and Module."""
        Conv2d = Module.from_nn_module(nn.Conv2d)
        self.assertTrue(issubclass(Conv2d, nn.Conv2d))
        self.assertTrue(issubclass(Conv2d, Module))

    def test_isinstance(self):
        """Instance satisfies isinstance checks for both original and Module."""
        Conv2d = Module.from_nn_module(nn.Conv2d)
        m = Conv2d(3, 16, 3)
        self.assertIsInstance(m, nn.Conv2d)
        self.assertIsInstance(m, Module)

    def test_init_weights_calls_reset_parameters(self):
        """For classes with reset_parameters, init_weights delegates to it."""
        LayerNorm = Module.from_nn_module(nn.LayerNorm)
        m = LayerNorm(32)
        # Manually set weight to zeros, then init_weights should reset
        nn.init.zeros_(m.weight)
        m.init_weights()
        # After reset_parameters, weight should be ones for LayerNorm
        self.assertTrue(torch.allclose(m.weight, torch.ones(32)))

    def test_init_weights_noop_for_parameterless(self):
        """For classes without reset_parameters, init_weights is a no-op."""
        GELU = Module.from_nn_module(nn.GELU)
        m = GELU()
        m.init_weights()  # should not raise

    def test_cache(self):
        """Repeated calls return the same class object."""
        cls1 = Module.from_nn_module(nn.Conv2d)
        cls2 = Module.from_nn_module(nn.Conv2d)
        self.assertIs(cls1, cls2)

    def test_forward_unchanged(self):
        """Forward output is identical to original class."""
        LayerNorm = Module.from_nn_module(nn.LayerNorm)
        torch.manual_seed(42)
        orig = nn.LayerNorm(16)
        wrapped = LayerNorm(16)
        # Copy weights
        wrapped.load_state_dict(orig.state_dict())
        x = torch.randn(2, 16)
        torch.testing.assert_close(orig(x), wrapped(x))

    def test_state_dict_unchanged(self):
        """state_dict keys and values match the original class."""
        Conv2d = Module.from_nn_module(nn.Conv2d)
        orig = nn.Conv2d(3, 16, 3)
        wrapped = Conv2d(3, 16, 3)
        wrapped.load_state_dict(orig.state_dict())
        for key in orig.state_dict():
            self.assertIn(key, wrapped.state_dict())
            torch.testing.assert_close(
                orig.state_dict()[key], wrapped.state_dict()[key]
            )


class TestContainerInitWeights(unittest.TestCase):
    """Tests for ModuleList, ModuleDict, Sequential init_weights."""

    def test_module_list_init_weights(self):
        """ModuleList.init_weights calls init_weights on each child."""
        LayerNorm = Module.from_nn_module(nn.LayerNorm)
        norms = ModuleList([LayerNorm(8) for _ in range(3)])
        for n in norms:
            nn.init.zeros_(n.weight)
        norms.init_weights()
        for n in norms:
            self.assertTrue(torch.allclose(n.weight, torch.ones(8)))

    def test_module_dict_init_weights(self):
        """ModuleDict.init_weights calls init_weights on each child."""
        LayerNorm = Module.from_nn_module(nn.LayerNorm)
        norms = ModuleDict({"a": LayerNorm(8), "b": LayerNorm(8)})
        for n in norms.values():
            nn.init.zeros_(n.weight)
        norms.init_weights()
        for n in norms.values():
            self.assertTrue(torch.allclose(n.weight, torch.ones(8)))

    def test_sequential_init_weights(self):
        """Sequential.init_weights calls init_weights on each child."""
        linear = Linear.Config(bias=False).build(in_features=4, out_features=4)
        GELU = Module.from_nn_module(nn.GELU)
        seq = Sequential(linear, GELU())
        seq.init_weights()  # should not raise

    def test_containers_are_module(self):
        """Container instances satisfy Module protocol."""
        self.assertIsInstance(ModuleList(), Module)
        self.assertIsInstance(ModuleDict(), Module)
        self.assertIsInstance(Sequential(), Module)


class TestVerifyModuleProtocol(unittest.TestCase):
    """Tests for BaseModel.verify_module_protocol."""

    def test_passes_for_all_module(self):
        """No error when all submodules are Module instances."""
        from torchtitan.protocols.model import BaseModel

        class GoodModel(BaseModel):
            @dataclass(kw_only=True, slots=True)
            class Config(BaseModel.Config):
                def update_from_config(self, *, trainer_config, **kwargs):
                    pass

                def get_nparams_and_flops(self, model, seq_len):
                    return (0, 0)

            def __init__(self):
                super().__init__()
                self.linear = Linear.Config().build(in_features=4, out_features=4)

        model = GoodModel()
        model.verify_module_protocol()  # should not raise

    def test_default_raises_for_plain_nn_module(self):
        """Default verify_module_protocol raises when plain nn.Module child exists."""
        from torchtitan.protocols.model import BaseModel

        class BadModel(BaseModel):
            @dataclass(kw_only=True, slots=True)
            class Config(BaseModel.Config):
                def update_from_config(self, *, trainer_config, **kwargs):
                    pass

                def get_nparams_and_flops(self, model, seq_len):
                    return (0, 0)

            def __init__(self):
                super().__init__()
                self.plain = nn.Linear(4, 4)

        model = BadModel()
        with self.assertRaises(RuntimeError):
            model.verify_module_protocol()

    def test_override_skips_verification(self):
        """Subclass can override verify_module_protocol to skip verification."""
        from torchtitan.protocols.model import BaseModel

        class ThirdPartyModel(BaseModel):
            @dataclass(kw_only=True, slots=True)
            class Config(BaseModel.Config):
                def update_from_config(self, *, trainer_config, **kwargs):
                    pass

                def get_nparams_and_flops(self, model, seq_len):
                    return (0, 0)

            def __init__(self):
                super().__init__()
                self.plain = nn.Linear(4, 4)  # third-party module

            def verify_module_protocol(self) -> None:
                pass  # skip for third-party internals

        model = ThirdPartyModel()
        model.verify_module_protocol()  # should not raise


if __name__ == "__main__":
    unittest.main()
