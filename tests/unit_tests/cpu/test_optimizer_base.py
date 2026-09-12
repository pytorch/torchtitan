# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from typing import ClassVar

import torch

from torchtitan.components.optimizer.base import Adam, AdamW, Optimizer


class TestOptimizerConfigSplit(unittest.TestCase):
    def test_all_public_fields_are_param_group_kwargs_by_default(self):
        config = AdamW.Config(lr=1e-3)
        self.assertEqual(
            config.to_param_group_kwargs(),
            {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1},
        )
        self.assertEqual(config.to_factory_kwargs(), {})

    def test_factory_fields_are_split_out(self):
        class Fake(Optimizer):
            @dataclass(kw_only=True, slots=True)
            class Config(Optimizer.Config):
                _FACTORY_FIELDS: ClassVar[frozenset[str]] = frozenset({"widget"})

                lr: float = 1e-3
                widget: str = "w"

        config = Fake.Config()
        self.assertEqual(config.to_param_group_kwargs(), {"lr": 1e-3})
        self.assertEqual(config.to_factory_kwargs(), {"widget": "w"})

    def test_adamw_defaults_match_the_previous_optimizer_kwargs(self):
        # These defaults must equal what default_adamw used to write into
        # optimizer_kwargs, or saved parameter groups shift.
        config = AdamW.Config(lr=8e-4)
        self.assertEqual(config.betas, (0.9, 0.95))
        self.assertEqual(config.eps, 1e-8)
        self.assertEqual(config.weight_decay, 0.1)

    def test_from_torch_keeps_the_torch_class_name(self):
        # _log_optimizer reports type(optimizer).__name__.
        self.assertEqual(AdamW.__name__, "AdamW")
        self.assertEqual(Adam.__name__, "Adam")

    def test_build_constructs_a_real_torch_optimizer(self):
        param = torch.nn.Parameter(torch.zeros(2, 2))
        groups = [{"params": [param], "lr": 1e-3}]
        optimizer = AdamW.Config(lr=1e-3).build(params=groups)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(optimizer, Optimizer)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)


class TestDistMuonConfig(unittest.TestCase):
    """DistMuon is the case that motivates the two-channel Config."""

    def _config(self):
        from torchtitan.distributed.flex_shard import DistMuon

        return DistMuon.Config(lr=1e-3, compute_sharding_by_fqn={}, bucket_configs=())

    def test_factory_fields_are_not_param_group_kwargs(self):
        config = self._config()
        param_group_kwargs = config.to_param_group_kwargs()
        self.assertNotIn("compute_sharding_by_fqn", param_group_kwargs)
        self.assertNotIn("bucket_configs", param_group_kwargs)
        self.assertEqual(
            set(config.to_factory_kwargs()),
            {"compute_sharding_by_fqn", "bucket_configs"},
        )

    def test_defaults_match_the_previous_constructor_defaults(self):
        # to_param_group_kwargs emits every field, whereas the old signature
        # let unset keys fall through to constructor defaults. The values must
        # agree or saved parameter groups shift.
        config = self._config()
        self.assertEqual(config.weight_decay, 0.1)
        self.assertEqual(config.momentum, 0.95)
        self.assertTrue(config.nesterov)
        self.assertEqual(config.ns_coefficients, (3.4445, -4.7750, 2.0315))
        self.assertEqual(config.eps, 1e-7)
        self.assertEqual(config.ns_steps, 5)
        self.assertIsNone(config.adjust_lr_fn)

    def test_foreach_defaults_false_so_a_global_foreach_is_overridden(self):
        # _validate_groups rejects a truthy foreach, and Kimi K2.7 sets
        # implementation="foreach" for the container as a whole.
        self.assertIs(self._config().to_param_group_kwargs()["foreach"], False)


if __name__ == "__main__":
    unittest.main()
