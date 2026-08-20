# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.context_parallel import validate_cp_backend


class TestValidateCpBackend(unittest.TestCase):
    """``validate_cp_backend`` gates CP on the backend that implements it."""

    @staticmethod
    def _parallelism(*, spmd_backend: str, cp: int) -> ParallelismConfig:
        return ParallelismConfig(spmd_backend=spmd_backend, context_parallel_degree=cp)

    def test_rejects_cp_on_partial_dtensor(self):
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            validate_cp_backend(self._parallelism(spmd_backend="partial_dtensor", cp=2))

    def test_allows_cp_on_spmd_types(self):
        validate_cp_backend(self._parallelism(spmd_backend="spmd_types", cp=2))

    def test_allows_partial_dtensor_without_cp(self):
        # Only CP is gated; partial_dtensor stays valid for FSDP/TP/EP runs.
        validate_cp_backend(self._parallelism(spmd_backend="partial_dtensor", cp=1))


class TestDecoderConfigCpValidation(unittest.TestCase):
    """``Decoder.Config.update_from_config`` applies the CP gates at config time."""

    @staticmethod
    def _config(*, spmd_backend: str, cp: int, varlen: bool = False):
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel,
            llama3_debugmodel_varlen_attn,
        )

        config = (llama3_debugmodel_varlen_attn if varlen else llama3_debugmodel)()
        config.parallelism.spmd_backend = spmd_backend
        config.parallelism.context_parallel_degree = cp
        config.training.seq_len = 512
        return config

    def test_rejects_cp_on_partial_dtensor(self):
        config = self._config(spmd_backend="partial_dtensor", cp=2)
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            config.model_spec.model.update_from_config(config=config)

    def test_allows_partial_dtensor_without_cp(self):
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.model_spec.model.update_from_config(config=config)

    def test_allows_flex_cp_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2)
        config.model_spec.model.update_from_config(config=config)

    def test_rejects_varlen_cp_on_spmd_types(self):
        # Only FlexAttention's BlockMask represents global key positions for CP.
        config = self._config(spmd_backend="spmd_types", cp=2, varlen=True)
        with self.assertRaisesRegex(NotImplementedError, "VarlenAttention"):
            config.model_spec.model.update_from_config(config=config)


class TestFluxConfigCpValidation(unittest.TestCase):
    """Flux is not a ``Decoder`` but applies the same backend gate."""

    @staticmethod
    def _config(*, spmd_backend: str, cp: int):
        pytest.importorskip(
            "torchtitan.models.flux.config_registry",
            reason="Flux requires optional image dependencies",
        )
        from torchtitan.models.flux.config_registry import flux_debugmodel

        config = flux_debugmodel()
        config.parallelism.spmd_backend = spmd_backend
        config.parallelism.context_parallel_degree = cp
        return config

    def test_rejects_cp_on_partial_dtensor(self):
        config = self._config(spmd_backend="partial_dtensor", cp=2)
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            config.model_spec.model.update_from_config(config=config)

    def test_allows_partial_dtensor_without_cp(self):
        # Flux on partial_dtensor is FSDP-only but still a valid configuration.
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.model_spec.model.update_from_config(config=config)


if __name__ == "__main__":
    unittest.main()
