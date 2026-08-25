# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

import pytest

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.context_parallel import validate_cp_backend
from torchtitan.protocols.module import Module


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
    def _config(
        *, spmd_backend: str, cp: int, varlen: bool = False, cp_kernel: bool = False
    ):
        from torchtitan.models.common.cp_attention import (
            AllGatherCPFlexAttention,
            use_cp_kernel,
        )
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel,
            llama3_debugmodel_varlen_attn,
        )

        config = (llama3_debugmodel_varlen_attn if varlen else llama3_debugmodel)()
        if cp_kernel:
            use_cp_kernel(config, AllGatherCPFlexAttention)
        config.parallelism.spmd_backend = spmd_backend
        config.parallelism.context_parallel_degree = cp
        config.training.max_context_length = 512
        return config

    def test_rejects_cp_on_partial_dtensor(self):
        config = self._config(spmd_backend="partial_dtensor", cp=2)
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            config.model_spec.model.update_from_config(config=config)

    def test_allows_partial_dtensor_without_cp(self):
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.model_spec.model.update_from_config(config=config)

    def test_allows_cp_kernel_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2, cp_kernel=True)
        config.model_spec.model.update_from_config(config=config)

    def test_allows_plain_flex_without_cp(self):
        config = self._config(spmd_backend="spmd_types", cp=1)
        config.model_spec.model.update_from_config(config=config)

    def test_rejects_cp_kernel_without_cp(self):
        config = self._config(spmd_backend="spmd_types", cp=1, cp_kernel=True)
        with self.assertRaisesRegex(ValueError, "context parallel degree is 1"):
            config.model_spec.model.update_from_config(config=config)

    def test_rejects_plain_flex_cp_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2)
        with self.assertRaisesRegex(ValueError, "AllGatherCPFlexAttention"):
            config.model_spec.model.update_from_config(config=config)

    def test_rejects_varlen_cp_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2, varlen=True)
        with self.assertRaisesRegex(ValueError, "ContextParallelKernel"):
            config.model_spec.model.update_from_config(config=config)

    def test_rejects_an_unrecognized_kernel_cp_on_spmd_types(self):
        class LocalOnlyAttention(Module):
            @dataclass(kw_only=True, slots=True)
            class Config(Module.Config):
                pass

        config = self._config(spmd_backend="spmd_types", cp=2)
        for layer in config.model_spec.model.layers:
            layer.attention.inner_attention = LocalOnlyAttention.Config()
        with self.assertRaisesRegex(ValueError, "ContextParallelKernel"):
            config.model_spec.model.update_from_config(config=config)


class TestShippedCpRecipe(unittest.TestCase):
    def test_muse_glimmer_cp_recipe_passes_the_gate(self):
        from torchtitan_recipes.muse_glimmer import muse_glimmer_30b_allgather_cp8

        config = muse_glimmer_30b_allgather_cp8()
        self.assertGreater(config.parallelism.context_parallel_degree, 1)
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
