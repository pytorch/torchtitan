# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.distributed.flex_shard import NoRedistribution
from torchtitan.models.kimi_k2_7.config_registry import (
    _flex_shard_muon_optimizer,
    model_registry,
)


class TestKimiK25FlexShardMuonConfig(unittest.TestCase):
    def test_declares_each_runtime_physical_bucket_alternative(self):
        optimizer_config = _flex_shard_muon_optimizer(
            model_registry("debugmodel"),
            lr=8e-4,
        )
        factory_kwargs = optimizer_config.optimizer_factory_kwargs_by_name["Muon"]
        compute_sharding_by_fqn = factory_kwargs["compute_sharding_by_fqn"]
        bucket_configs = factory_kwargs["bucket_configs"]

        for fqn in compute_sharding_by_fqn:
            matching_redistributions = {
                config.redistribution_mesh_axis_names
                for config in bucket_configs
                if fqn in config.patterns
            }
            expected_redistributions = {NoRedistribution(), ("dp_shard",)}
            if ".moe.routed_experts." in fqn:
                expected_redistributions.add(("efsdp",))
            self.assertEqual(
                matching_redistributions,
                expected_redistributions,
                fqn,
            )
            for redistribution in matching_redistributions:
                if isinstance(redistribution, NoRedistribution):
                    continue
                self.assertNotIn("ep", redistribution, fqn)


if __name__ == "__main__":
    unittest.main()
