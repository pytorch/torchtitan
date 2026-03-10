# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.protocols.module import Module
from torchtitan.protocols.state_initializer import StateInitializer


class SimpleStateInitializer(StateInitializer):
    @dataclass(kw_only=True, slots=True)
    class Config(StateInitializer.Config):
        pass

    def init_states(self, module: nn.Module, *, buffer_device=None) -> None:
        if hasattr(module, "linear"):
            nn.init.zeros_(module.linear.weight)


class TestModuleInitStates(unittest.TestCase):
    """Tests for Module.init_states enforcement.

    Module.init_states delegates to _state_initializer. If no state_initializer
    is provided, it uses the default NoOpStateInitializer.
    """

    def test_default_state_initializer_is_noop(self):
        """Subclass with default Config gets a NoOpStateInitializer (no error)."""

        class BasicModule(Module):
            @dataclass(kw_only=True, slots=True)
            class Config(Module.Config):
                pass

            def __init__(self, config: Config):
                super().__init__(config)

        config = BasicModule.Config()
        m = BasicModule(config)
        # Should not raise — NoOpStateInitializer.init_states is a no-op
        m.init_states()

    def test_init_states_with_initializer(self):
        """Subclass with state_initializer works normally."""

        class GoodModule(Module):
            @dataclass(kw_only=True, slots=True)
            class Config(Module.Config):
                state_initializer: StateInitializer.Config = field(
                    default_factory=SimpleStateInitializer.Config
                )

            def __init__(self, config: Config):
                super().__init__(config)
                self.linear = nn.Linear(4, 4)

        config = GoodModule.Config()
        m = GoodModule(config)
        m.init_states()
        self.assertTrue(torch.all(m.linear.weight == 0))


class TestDiamondInheritance(unittest.TestCase):
    """Tests for diamond inheritance: class Foo(Module, nn.SomeModule).

    The diamond pattern is used when a Module component also inherits from
    an nn.Module (e.g., nn.Embedding) to reuse PyTorch's implementation while
    satisfying the Module protocol.
    These tests ensure the pattern works correctly and catches regressions.
    """

    class EmbeddingInitializer(StateInitializer):
        @dataclass(kw_only=True, slots=True)
        class Config(StateInitializer.Config):
            pass

        def init_states(self, module: nn.Module, *, buffer_device=None) -> None:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    class TestEmbedding(Module, nn.Embedding):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            state_initializer: StateInitializer.Config = field(
                default_factory=lambda: TestDiamondInheritance.EmbeddingInitializer.Config()
            )

        def __init__(self, config, num_embeddings, embedding_dim):
            super().__init__(config, num_embeddings, embedding_dim)

    def test_forward(self):
        """Forward pass works through nn.Embedding's implementation."""
        config = self.TestEmbedding.Config()
        emb = self.TestEmbedding(config, 100, 32)
        out = emb(torch.tensor([0, 1, 2]))
        self.assertEqual(out.shape, torch.Size([3, 32]))

    def test_init_states(self):
        """init_states runs the state initializer implementation."""
        config = self.TestEmbedding.Config()
        emb = self.TestEmbedding(config, 100, 32)
        emb.init_states()
        # Weight should have been re-initialized; just check it's finite
        self.assertTrue(torch.all(torch.isfinite(emb.weight)))

    def test_isinstance_checks(self):
        """Diamond class is an instance of all parent types."""
        config = self.TestEmbedding.Config()
        emb = self.TestEmbedding(config, 100, 32)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertIsInstance(emb, nn.Module)
        self.assertIsInstance(emb, Module)

    def test_default_state_initializer(self):
        """Diamond class with default Config gets NoOpStateInitializer."""

        class SimpleEmbedding(Module, nn.Embedding):
            @dataclass(kw_only=True, slots=True)
            class Config(Module.Config):
                pass

            def __init__(self, config, num_embeddings, embedding_dim):
                super().__init__(config, num_embeddings, embedding_dim)

        config = SimpleEmbedding.Config()
        emb = SimpleEmbedding(config, 10, 4)
        # Should not raise — NoOpStateInitializer
        emb.init_states()

    def test_module_hierarchy_is_flat(self):
        """Diamond embedding adds no extra layer to the module tree."""

        config = TestDiamondInheritance.TestEmbedding.Config()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = TestDiamondInheritance.TestEmbedding(config, 100, 32)
                self.linear = nn.Linear(32, 16)

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
            config = TestDiamondInheritance.TestEmbedding.Config()
            TestDiamondInheritance.TestEmbedding(config, 50, 16)
            self.assertEqual(call_count, 1)
        finally:
            nn.Module.__init__ = orig_init


if __name__ == "__main__":
    unittest.main()
