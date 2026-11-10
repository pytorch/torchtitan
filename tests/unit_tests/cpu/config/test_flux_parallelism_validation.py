# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reject unsupported TP/EP degrees for FLUX (Issue #5854)."""

import unittest

try:
    from torchtitan.config import DebugConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.config.validation import validate_model_training_config
    from torchtitan.models.flux.flavors import _flux_debug

    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False


def _validate_flux(
    *,
    tensor_parallel_degree: int = 1,
    expert_parallel_degree: int = 1,
) -> None:
    seq_len = 256
    config = _flux_debug(seq_len=seq_len)
    validate_model_training_config(
        config,
        parallelism=ParallelismConfig(
            tensor_parallel_degree=tensor_parallel_degree,
            expert_parallel_degree=expert_parallel_degree,
        ),
        training=TrainingConfig(disable_accelerator_graphs=True),
        debug=DebugConfig(),
        activation_remat=None,
        max_num_documents=None,
    )


@unittest.skipUnless(_IMPORTS_OK, "torchtitan model imports not available")
class TestFluxParallelismValidation(unittest.TestCase):
    def test_flux_tp1_ep1_does_not_raise(self):
        _validate_flux()

    def test_flux_tensor_parallel_degree_gt1_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"FluxModel does not support tensor parallelism"
        ):
            _validate_flux(tensor_parallel_degree=2)

    def test_flux_expert_parallel_degree_gt1_raises(self):
        with self.assertRaisesRegex(ValueError, r"FluxModel has no MoE layers"):
            _validate_flux(expert_parallel_degree=2)


if __name__ == "__main__":
    unittest.main()
