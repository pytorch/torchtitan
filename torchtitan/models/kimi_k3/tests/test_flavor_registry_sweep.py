# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Constructs every registered flavor end-to-end on CPU.

Catches upstream config-API drift in flavors the unit tests never touch
(the pressure-test carriers, the 48B downscales, the fp8 variant). Pure
config construction -- no weights are materialized.
"""

import inspect
import unittest

import torchtitan.models.kimi_k3 as kimi_k3
from torchtitan.models.kimi_k3 import config_registry


class TestFlavorRegistrySweep(unittest.TestCase):
    def test_every_kimi_model_spec_builds(self):
        for flavor in config_registry.flavor_names():
            with self.subTest(flavor=flavor):
                spec = kimi_k3.model_registry(flavor)
                self.assertIsNotNone(spec.parallelize_fn)

    def test_every_trainer_config_builds(self):
        for name, fn in sorted(vars(config_registry).items()):
            if not (
                inspect.isfunction(fn)
                and fn.__module__ == config_registry.__name__
                and name.startswith("kimi_linear_")
            ):
                continue
            with self.subTest(flavor=name):
                try:
                    cfg = fn()
                except ValueError as e:
                    # Float8 swap requires SM89+; the fp8 flavor is
                    # hardware-gated, not a config error.
                    if "float8 is only supported" in str(e):
                        self.skipTest("float8 requires SM89+ hardware")
                    raise
                self.assertIsNotNone(cfg.model_spec)

    def test_unknown_flavor_raises_value_error(self):
        with self.assertRaises(ValueError):
            kimi_k3.model_registry("no_such_flavor")


class TestBlockSizeFitsTheModel(unittest.TestCase):
    """No flavor may declare an AttnRes block size larger than its layer count.

    A size above the layer count is not a partition, and it is what a flavor gets by
    inheriting one from a full-depth parent and then truncating the layers. Thirteen diag
    flavors were in that state and none of them could build; they are diagnostic
    flavors, so the matrix never touches them and nothing noticed.

    Written as a sweep rather than per flavor because the failure came from a builder
    that several flavors share, and the next one would too.
    """

    def test_every_zero_argument_flavor(self):
        import inspect

        from torchtitan.models.kimi_k3 import config_registry as cr

        checked = 0
        for name in dir(cr):
            if not (name.startswith("kimi_k3_") or name.startswith("kimi_linear")):
                continue
            fn = getattr(cr, name)
            if not callable(fn) or inspect.signature(fn).parameters:
                continue
            with self.subTest(flavor=name):
                cfg = fn()
                spec = cfg.model_spec.model
                size = getattr(spec, "attn_res_block_size", None)
                layers = spec.kimi_config.num_hidden_layers
                checked += 1
                if size is None:
                    continue
                self.assertLessEqual(
                    size,
                    layers,
                    f"{name} declares block size {size} over {layers} layers",
                )
        # A sweep that swept nothing would pass silently.
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
