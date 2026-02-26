# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

from torchtitan.config.configurable import Configurable


class TestConfigurable(unittest.TestCase):
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


class TestBuildAutoDetection(unittest.TestCase):
    """Tests for the auto-detection logic in Config.build()."""

    class OldStyleComponent(Configurable):
        """Component whose __init__ takes extra kwargs (not config fields)."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config, *, dim: int):
            self.config = config
            self.dim = dim

    class NewStyleComponent(Configurable):
        """Component whose optional config fields are filled via build()."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5
            dim: int | None = None
            hidden: int | None = None

        def __init__(self, config: Config):
            self.config = config

    class NoKwargsComponent(Configurable):
        """Component that takes only config, no extra kwargs."""

        @dataclass(kw_only=True, slots=True)
        class Config(Configurable.Config):
            x: int = 5

        def __init__(self, config: Config):
            self.config = config

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
        # Mutating the built object's config doesn't affect original
        obj.config.x = 999
        self.assertEqual(cfg.x, 10)

    def test_clone_isolation_new_style(self):
        """Original config is not mutated in new-style path."""
        cfg = self.NewStyleComponent.Config(x=10)
        obj = cfg.build(dim=64, hidden=128)
        obj.config.dim = 999
        self.assertIsNone(cfg.dim)

    def test_mismatch_raises(self):
        """Pre-specified field value != kwarg value raises ValueError."""
        cfg = self.NewStyleComponent.Config(dim=32)
        with self.assertRaises(ValueError):
            cfg.build(dim=64)

    def test_matching_pre_specified_value(self):
        """Pre-specified field value == kwarg value is accepted."""
        cfg = self.NewStyleComponent.Config(dim=64)
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


if __name__ == "__main__":
    unittest.main()
