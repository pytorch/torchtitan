# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for rejecting parallelism degrees a model does not implement.

Covers the silent-TP/EP failure mode: ``ParallelismConfig`` exposes
``tensor_parallel_degree`` / ``expert_parallel_degree`` for every model, but
TP/EP sharding is implemented per model. FLUX only implements FSDP; setting
``tp=4`` on a FLUX run was accepted by config validation, silently ignored
at parallelization time, and then failed as an OOM during model construction
(full unsharded weights on every rank) instead of a clear config error.

The fix adds early ValueErrors in ``validate_model_training_config``:
- models with ``supports_tensor_parallel = False`` reject ``tp > 1``
- dense models (no MoE layers) reject ``ep > 1``
"""

import unittest

try:
    from torchtitan.config import DebugConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.config.validation import validate_model_training_config
    from torchtitan.models.flux.flavors import build_model_config

    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False


def _validate(model_config, tp: int = 1, ep: int = 1) -> None:
    validate_model_training_config(
        model_config,
        parallelism=ParallelismConfig(
            tensor_parallel_degree=tp, expert_parallel_degree=ep
        ),
        training=TrainingConfig(disable_accelerator_graphs=True),
        debug=DebugConfig(),
        activation_remat=None,
        max_num_documents=None,
    )


@unittest.skipUnless(_IMPORTS_OK, "torchtitan model imports not available")
class TestUnsupportedParallelismValidation(unittest.TestCase):
    """Models that do not implement TP/EP must reject tp/ep > 1 up front."""

    @classmethod
    def setUpClass(cls):
        cls.flux_config = build_model_config(
            "flux-debug", parallelism=ParallelismConfig()
        )

    # ------------------------------------------------------------------
    # FLUX + tensor_parallel_degree > 1  →  should raise
    # ------------------------------------------------------------------

    def test_flux_tp_rejected(self):
        """FLUX does not implement TP; tp=4 must raise, not silently no-op."""
        with self.assertRaisesRegex(
            ValueError, "does not implement tensor parallelism"
        ):
            _validate(self.flux_config, tp=4)

    def test_flux_tp2_rejected(self):
        with self.assertRaisesRegex(ValueError, "tensor_parallel_degree = 2"):
            _validate(self.flux_config, tp=2)

    # ------------------------------------------------------------------
    # FLUX + expert_parallel_degree > 1  →  should raise (dense model)
    # ------------------------------------------------------------------

    def test_flux_ep_rejected(self):
        """FLUX is dense (no MoE); ep=4 must raise, not silently no-op."""
        with self.assertRaisesRegex(ValueError, "has no MoE"):
            _validate(self.flux_config, ep=4)

    # ------------------------------------------------------------------
    # FSDP-only configurations  →  should not raise
    # ------------------------------------------------------------------

    def test_flux_fsdp_only_does_not_raise(self):
        """tp=1/ep=1 with FSDP sharding is the supported FLUX path."""
        _validate(self.flux_config, tp=1, ep=1)

    def test_flux_capability_flags(self):
        """FLUX declares supports_pipeline/tensor_parallel = False."""
        from torchtitan.models.flux.model import FluxModel

        self.assertFalse(FluxModel.supports_tensor_parallel)
        self.assertFalse(FluxModel.supports_pipeline_parallel)


if __name__ == "__main__":
    unittest.main()
