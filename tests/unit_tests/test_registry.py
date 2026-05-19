# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field

from torchtitan.config.configurable import Configurable
from torchtitan.registry import (
    _REGISTRY,
    apply_overrides,
    clear_registry,
    OverrideConfig,
    register,
)


class ComponentA(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 64

    def __init__(self, config: Config):
        self.config = config


class ComponentB(Configurable):
    """Replacement for ComponentA."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 64
        extra: int = 128

    def __init__(self, config: Config):
        self.config = config


class ParentComponent(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        child: ComponentA.Config = field(default_factory=ComponentA.Config)
        children: list[ComponentA.Config] = field(
            default_factory=lambda: [
                ComponentA.Config(dim=32),
                ComponentA.Config(dim=48),
            ]
        )

    def __init__(self, config: Config):
        self.config = config


class TestRegistry(unittest.TestCase):
    def setUp(self):
        clear_registry()

    def tearDown(self):
        clear_registry()

    def test_register_and_apply(self):
        @register("test_swap", target=ComponentA.Config, description="test swap")
        def swap_a_to_b(cfg: ComponentA.Config) -> ComponentB.Config:
            return ComponentB.Config(dim=cfg.dim, extra=256)

        self.assertIn("test_swap", _REGISTRY)

        parent_cfg = ParentComponent.Config()
        override_cfg = OverrideConfig()  # no modules to import
        replacements = apply_overrides(parent_cfg, override_cfg)

        # child and two list items should be replaced
        self.assertEqual(len(replacements), 3)
        self.assertIsInstance(parent_cfg.child, ComponentB.Config)
        self.assertEqual(parent_cfg.child.extra, 256)
        for child_cfg in parent_cfg.children:
            self.assertIsInstance(child_cfg, ComponentB.Config)
            self.assertEqual(child_cfg.extra, 256)

    def test_factory_returning_none_skips(self):
        @register("selective", target=ComponentA.Config)
        def selective_swap(cfg: ComponentA.Config):
            if cfg.dim == 32:
                return ComponentB.Config(dim=cfg.dim)
            return None  # skip

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(parent_cfg, OverrideConfig())

        # Only the child with dim=32 in the list should be replaced
        self.assertEqual(len(replacements), 1)
        self.assertIsInstance(parent_cfg.child, ComponentA.Config)  # dim=64, skipped
        self.assertIsInstance(parent_cfg.children[0], ComponentB.Config)  # dim=32
        self.assertIsInstance(
            parent_cfg.children[1], ComponentA.Config
        )  # dim=48, skipped

    def test_duplicate_name_raises(self):
        @register("dup", target=ComponentA.Config)
        def first(cfg):
            return cfg

        with self.assertRaises(ValueError, msg="already registered"):

            @register("dup", target=ComponentA.Config)
            def second(cfg):
                return cfg

    def test_conflict_detection(self):
        @register("swap1", target=ComponentA.Config)
        def swap1(cfg):
            return ComponentB.Config(dim=cfg.dim)

        # Manually register a second override for the same target
        # (bypassing the name check by using a different name)
        @register("swap2", target=ComponentA.Config)
        def swap2(cfg):
            return ComponentB.Config(dim=cfg.dim)

        parent_cfg = ParentComponent.Config()
        with self.assertRaises(ValueError, msg="Conflicting"):
            apply_overrides(parent_cfg, OverrideConfig())

    def test_clear_registry(self):
        @register("temp", target=ComponentA.Config)
        def temp(cfg):
            return cfg

        self.assertEqual(len(_REGISTRY), 1)
        clear_registry()
        self.assertEqual(len(_REGISTRY), 0)

    def test_no_overrides_is_noop(self):
        parent_cfg = ParentComponent.Config(child=ComponentA.Config(dim=100))
        replacements = apply_overrides(parent_cfg, OverrideConfig())
        self.assertEqual(len(replacements), 0)
        self.assertEqual(parent_cfg.child.dim, 100)

    def test_bad_module_import_raises(self):
        override_cfg = OverrideConfig(modules=["nonexistent.module.path"])
        parent_cfg = ParentComponent.Config()
        with self.assertRaises(ImportError):
            apply_overrides(parent_cfg, override_cfg)

    def test_logging_format(self):
        @register("fmt_test", target=ComponentA.Config, description="format test")
        def fmt_swap(cfg):
            return ComponentB.Config(dim=cfg.dim)

        parent_cfg = ParentComponent.Config()
        replacements = apply_overrides(parent_cfg, OverrideConfig())
        for line in replacements:
            self.assertIn("[Override]", line)
            self.assertIn("fmt_test", line)
            self.assertIn("->", line)


if __name__ == "__main__":
    unittest.main()
