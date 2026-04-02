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
        """__init__ takes extra runtime kwargs (not config fields)."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config, *, dim: int):
            self.config = config
            self.dim = dim

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

    def test_clone_isolation_old_style(self):
        """Original config is not mutated in old-style path."""
        cfg = self.OldStyleComponent.Config(x=10)
        obj = cfg.build(dim=64)
        obj.config.x = 999
        self.assertEqual(cfg.x, 10)

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
                b: int = 2

            def __init__(self, config: Config):
                self.config = config

        class Outer(Configurable):
            @dataclass(kw_only=True, slots=True)
            class Config(Configurable.Config):
                x: int = 10
                inner: Inner.Config = field(default_factory=Inner.Config)

            def __init__(self, config: Config):
                self.config = config

        cfg = Outer.Config(x=42)
        d = cfg.to_dict()
        self.assertEqual(d["x"], 42)
        # Inner config is serialised via its own to_dict
        self.assertIn("inner", d)
        self.assertEqual(d["inner"]["a"], 1)
        self.assertEqual(d["inner"]["b"], 2)

        # After build: all fields present
        obj = cfg.build()
        d2 = obj.config.to_dict()
        self.assertEqual(d2["x"], 42)
        self.assertEqual(d2["inner"]["a"], 1)
        self.assertEqual(d2["inner"]["b"], 2)

    def test_repr(self):
        """repr() works for configs."""
        cfg = self.NoKwargsComponent.Config(x=42)
        r = repr(cfg)
        self.assertIn("x=42", r)


if __name__ == "__main__":
    unittest.main()
