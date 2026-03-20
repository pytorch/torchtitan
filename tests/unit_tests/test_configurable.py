# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field

from torchtitan.config.configurable import Configurable


class TestConfigurable(unittest.TestCase):
    class OldStyleComponent(Configurable):
        """__init__ takes extra kwargs (not config fields). To be deprecated."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config, *, dim: int):
            self.config = config
            self.dim = dim

    class NewStyleComponent(Configurable):
        """Config fields filled via build() using field(init=False)."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5
            dim: int = field(init=False)
            hidden: int = field(init=False)

        def __init__(self, config: Config):
            self.config = config

    class NoKwargsComponent(Configurable):
        """Takes only config, no extra kwargs."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config):
            self.config = config

    def test_valid_config(self):
        """Config with kw_only=True and slots=True should work."""

        class MyComponent(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                x: int = 5

            def __init__(self, config: Config):
                self.config = config

        cfg = MyComponent.Config(x=10)
        obj = cfg.build()
        self.assertIsInstance(obj, MyComponent)
        self.assertEqual(obj.config.x, 10)

    def test_missing_slots_raises(self):
        """Config without slots=True must be rejected."""
        with self.assertRaises(TypeError):

            class BadSlots(Configurable):
                @dataclass(kw_only=True)
                class Config(Configurable.Config):
                    x: int = 5

    def test_missing_kw_only_raises(self):
        """Config without kw_only=True must be rejected."""
        with self.assertRaises(TypeError):

            class BadKwOnly(Configurable):
                @dataclass(slots=True)
                class Config(Configurable.Config):
                    x: int = 5

    def test_missing_both_raises(self):
        """Config without kw_only=True or slots=True must be rejected."""
        with self.assertRaises(TypeError):

            class BadBoth(Configurable):
                @dataclass
                class Config(Configurable.Config):
                    x: int = 5

    def test_build_without_owner_raises(self):
        """Calling build() on the base Config should raise NotImplementedError."""
        cfg = Configurable.Config()
        with self.assertRaises(NotImplementedError):
            cfg.build()

    def test_old_style_forwarding(self):
        """kwargs not in config fields are forwarded to __init__."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        self.assertIsInstance(obj, self.OldStyleComponent)
        self.assertEqual(obj.config.x, 10)
        self.assertEqual(obj.dim, 64)

    def test_new_style_absorption(self):
        """kwargs matching config fields are absorbed into a cloned config."""
        cfg = self.NewStyleComponent.Config(x=10)
        obj = cfg.build(dim=64, hidden=128)
        self.assertIsInstance(obj, self.NewStyleComponent)
        self.assertEqual(obj.config.x, 10)
        self.assertEqual(obj.config.dim, 64)
        self.assertEqual(obj.config.hidden, 128)

    def test_mixed_kwargs_raises(self):
        """Mixing config fields and non-config kwargs raises TypeError."""
        cfg = self.NewStyleComponent.Config(x=10)
        with self.assertRaises(TypeError):
            cfg.build(dim=64, not_a_field=99)

    def test_clone_isolation_old_style(self):
        """Original config is not mutated in old-style path."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        obj.config.x = 999
        self.assertEqual(cfg.x, 10)

    def test_clone_isolation_new_style(self):
        """Original config is not mutated in new-style path."""
        cfg = self.NewStyleComponent.Config(x=10)
        obj = cfg.build(dim=64, hidden=128)
        obj.config.dim = 999
        self.assertFalse(hasattr(cfg, "dim"))

    def test_mismatch_raises(self):
        """Pre-specified field value != kwarg value raises ValueError."""
        cfg = self.NewStyleComponent.Config()
        cfg.dim = 32
        with self.assertRaises(ValueError):
            cfg.build(dim=64)

    def test_matching_pre_specified_value(self):
        """Pre-specified field value == kwarg value is accepted."""
        cfg = self.NewStyleComponent.Config()
        cfg.dim = 64
        obj = cfg.build(dim=64, hidden=128)
        self.assertEqual(obj.config.dim, 64)
        self.assertEqual(obj.config.hidden, 128)

    def test_no_kwargs(self):
        """build() with no kwargs clones config and constructs."""
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        self.assertIsInstance(obj, self.NoKwargsComponent)
        self.assertEqual(obj.config.x, 42)

    def test_no_kwargs_clone_isolation(self):
        """build() with no kwargs still clones the config."""
        cfg = self.NoKwargsComponent.Config(x=42)
        obj = cfg.build()
        obj.config.x = 999
        self.assertEqual(cfg.x, 42)

    def test_to_dict_two_layer(self):
        """to_dict serializes nested configs (two layers deep)."""

        class Inner(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                a: int = 1
                b: int = field(init=False)

            def __init__(self, config: Config):
                self.config = config

        class Outer(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                x: int = 10
                inner: Inner.Config = field(default_factory=Inner.Config)
                dim: int = field(init=False)

            def __init__(self, config: Config):
                self.config = config

        # Before build: unset field(init=False) slots are skipped
        cfg = Outer.Config(x=42)
        d = cfg.to_dict()
        self.assertEqual(d["x"], 42)
        self.assertNotIn("dim", d)
        # Inner config is serialised via its own to_dict
        self.assertIn("inner", d)
        self.assertEqual(d["inner"]["a"], 1)
        self.assertNotIn("b", d["inner"])

        # After build: all fields present
        obj = cfg.build(dim=128)
        obj.config.inner.b = 256
        d2 = obj.config.to_dict()
        self.assertEqual(d2["x"], 42)
        self.assertEqual(d2["dim"], 128)
        self.assertEqual(d2["inner"]["a"], 1)
        self.assertEqual(d2["inner"]["b"], 256)

    def test_repr_with_unset_init_false(self):
        """repr() must not crash when field(init=False) slots are unset."""
        cfg = self.NewStyleComponent.Config(x=10)
        # Before build: dim and hidden are unset
        r = repr(cfg)
        self.assertIn("x=10", r)
        self.assertIn("dim=<UNSET>", r)
        self.assertIn("hidden=<UNSET>", r)

        # After build: all fields set
        obj = cfg.build(dim=64, hidden=128)
        r2 = repr(obj.config)
        self.assertIn("x=10", r2)
        self.assertIn("dim=64", r2)
        self.assertIn("hidden=128", r2)
        self.assertNotIn("UNSET", r2)

    def test_repr_no_init_false_fields(self):
        """repr() works normally when there are no field(init=False) fields."""
        cfg = self.NoKwargsComponent.Config(x=42)
        r = repr(cfg)
        self.assertIn("x=42", r)
        self.assertNotIn("UNSET", r)

    def test_init_false_with_inheritance(self):
        """Child config can redeclare field with default."""

        class ChildComponent(self.NewStyleComponent):
            @dataclass(kw_only=True, slots=True)
            class Config(TestConfigurable.NewStyleComponent.Config):
                dim: int = 64
                hidden: int = 128

            def __init__(self, config: Config):
                self.config = config

        cfg = ChildComponent.Config()
        obj = cfg.build()
        self.assertEqual(obj.config.dim, 64)
        self.assertEqual(obj.config.hidden, 128)

        # Can also override via __init__
        cfg2 = ChildComponent.Config(dim=256, hidden=512)
        obj2 = cfg2.build()
        self.assertEqual(obj2.config.dim, 256)
        self.assertEqual(obj2.config.hidden, 512)


if __name__ == "__main__":
    unittest.main()
