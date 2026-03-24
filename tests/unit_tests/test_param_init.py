# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.protocols.module import (
    _SkipParamInitType,
    Module,
    set_param_init,
    SKIP_PARAM_INIT,
    validate_param_init,
)


class TestSetParamInit(unittest.TestCase):
    """Tests for set_param_init helper."""

    def test_sets_param_init(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        m = M()
        set_param_init(m, {"weight": nn.init.zeros_})
        self.assertIsNotNone(m._param_init)
        self.assertIn("weight", m._param_init)

    def test_raises_on_double_set(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        m = M()
        set_param_init(m, {"weight": nn.init.zeros_})
        with self.assertRaises(ValueError):
            set_param_init(m, {"weight": nn.init.ones_})


class TestValidateParamInit(unittest.TestCase):
    """Tests for validate_param_init."""

    def test_raises_for_missing_module(self):
        class Child(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

        m = Parent()
        with self.assertRaises(ValueError):
            validate_param_init(m)

    def test_raises_for_missing_param(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))
                self.bias = nn.Parameter(torch.empty(4))

        m = M()
        set_param_init(m, {"weight": nn.init.zeros_})  # missing bias
        with self.assertRaises(ValueError):
            validate_param_init(m)

    def test_passes_when_complete(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        m = M()
        set_param_init(m, {"weight": nn.init.zeros_})
        validate_param_init(m)  # should not raise

    def test_skip_param_init_passes(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4))

        m = M()
        object.__setattr__(m, "_param_init", SKIP_PARAM_INIT)
        validate_param_init(m)  # should not raise


class TestSkipParamInit(unittest.TestCase):
    """Tests for SKIP_PARAM_INIT sentinel."""

    def test_singleton(self):
        self.assertIs(SKIP_PARAM_INIT, _SkipParamInitType())

    def test_repr(self):
        self.assertEqual(repr(SKIP_PARAM_INIT), "SKIP_PARAM_INIT")

    def test_init_states_skips_init(self):
        """SKIP_PARAM_INIT prevents _init_self_parameters from raising."""

        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4))

        m = M()
        object.__setattr__(m, "_param_init", SKIP_PARAM_INIT)
        m.init_states()  # should not raise
        self.assertTrue(torch.all(m.weight == 1))  # weight unchanged


if __name__ == "__main__":
    unittest.main()
