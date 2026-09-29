# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.components.validate import BaseValidator, Validator


class TestValidatorFrequency(unittest.TestCase):
    def test_frequency_must_be_positive(self):
        for config_cls in (BaseValidator.Config, Validator.Config):
            for freq in (0, -1):
                with self.subTest(config=config_cls.__qualname__, freq=freq):
                    with self.assertRaisesRegex(
                        ValueError, "validation frequency must be positive"
                    ):
                        config_cls(freq=freq)

    def test_steps_must_be_positive_or_neg1(self):
        for steps in (0, -2):
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(
                    ValueError, "validation steps must be positive or -1"
                ):
                    Validator.Config(steps=steps)

    def test_steps_accepts_positive_and_neg1(self):
        for steps in (-1, 1):
            with self.subTest(steps=steps):
                config = Validator.Config(steps=steps)
                self.assertEqual(config.steps, steps)

    def test_should_validate_at_configured_frequency(self):
        validator = BaseValidator(config=BaseValidator.Config(freq=3))

        self.assertTrue(validator.should_validate(1))
        self.assertFalse(validator.should_validate(2))
        self.assertTrue(validator.should_validate(3))
        self.assertFalse(validator.should_validate(4))


if __name__ == "__main__":
    unittest.main()
