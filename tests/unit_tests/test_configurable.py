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


if __name__ == "__main__":
    unittest.main()
