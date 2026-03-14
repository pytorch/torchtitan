# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.protocols.module import Module


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
                self.linear = nn.Linear(4, 4)

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

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = TestDiamondInheritance.TestEmbedding(100, 32)
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
            self.TestEmbedding(50, 16)
            self.assertEqual(call_count, 1)
        finally:
            nn.Module.__init__ = orig_init


if __name__ == "__main__":
    unittest.main()
