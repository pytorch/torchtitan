# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect
import unittest
from dataclasses import dataclass

import pytest

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.context_parallel import validate_context_parallel
from torchtitan.protocols.module import Module
from torchtitan.transforms import ContextParallelTransform


class _NoAttentionModel(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass


class TestBackendGate(unittest.TestCase):
    """CP requires the backend that implements it."""

    def _validate(self, *, spmd_backend: str, cp: int) -> None:
        validate_context_parallel(
            _NoAttentionModel.Config(),
            ParallelismConfig(spmd_backend=spmd_backend, context_parallel_degree=cp),
        )

    def test_rejects_cp_on_partial_dtensor(self):
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            self._validate(spmd_backend="partial_dtensor", cp=2)

    def test_allows_cp_on_spmd_types(self):
        self._validate(spmd_backend="spmd_types", cp=2)

    def test_allows_partial_dtensor_without_cp(self):
        # Only CP is gated; partial_dtensor stays valid for FSDP/TP/EP runs.
        self._validate(spmd_backend="partial_dtensor", cp=1)


class TestDecoderConfigCpValidation(unittest.TestCase):
    """``Trainer.Config.__post_init__`` applies the CP gates at config time."""

    @staticmethod
    def _config(
        *, spmd_backend: str, cp: int, varlen: bool = False, cp_kernel: bool = False
    ):
        from torchtitan.models.common.cp_attention import AllGatherCPFlexAttention
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel,
            llama3_debugmodel_varlen_attn,
        )

        config = (llama3_debugmodel_varlen_attn if varlen else llama3_debugmodel)()
        if cp_kernel:
            # Apply the transform without its final validation.
            ContextParallelTransform(kernel=AllGatherCPFlexAttention).transform(
                config.model_spec.model
            )
        config.parallelism.spmd_backend = spmd_backend
        config.parallelism.context_parallel_degree = cp
        config.training.max_context_length = 512
        return config

    def test_rejects_cp_on_partial_dtensor(self):
        config = self._config(spmd_backend="partial_dtensor", cp=2)
        with self.assertRaisesRegex(ValueError, "spmd_backend='spmd_types'"):
            config.__post_init__()

    def test_allows_partial_dtensor_without_cp(self):
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.__post_init__()

    def test_allows_cp_kernel_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2, cp_kernel=True)
        config.__post_init__()

    def test_allows_plain_flex_without_cp(self):
        config = self._config(spmd_backend="spmd_types", cp=1)
        config.__post_init__()

    def test_rejects_cp_kernel_without_cp(self):
        config = self._config(spmd_backend="spmd_types", cp=1, cp_kernel=True)
        with self.assertRaisesRegex(ValueError, "context parallel degree is 1"):
            config.__post_init__()

    def test_rejects_plain_flex_cp_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2)
        with self.assertRaisesRegex(ValueError, "AllGatherCPFlexAttention"):
            config.__post_init__()

    def test_rejects_varlen_cp_on_spmd_types(self):
        config = self._config(spmd_backend="spmd_types", cp=2, varlen=True)
        with self.assertRaisesRegex(ValueError, "ContextParallelKernel"):
            config.__post_init__()

    def test_rejects_an_unrecognized_kernel_cp_on_spmd_types(self):
        class LocalOnlyAttention(Module):
            @dataclass(kw_only=True, slots=True)
            class Config(Module.Config):
                pass

        config = self._config(spmd_backend="spmd_types", cp=2)
        for layer in config.model_spec.model.layers:
            layer.attention.inner_attention = LocalOnlyAttention.Config()
        with self.assertRaisesRegex(ValueError, "ContextParallelKernel"):
            config.__post_init__()


class TestShippedCpRecipes(unittest.TestCase):
    """Validate every shipped CP recipe after construction."""

    _MODULES = (
        "torchtitan_recipes.muse_glimmer",
        "torchtitan_recipes.tests.models",
        "torchtitan_recipes.tests.features",
        "torchtitan_recipes.tests.h100",
    )

    @classmethod
    def _recipes(cls):
        for name in cls._MODULES:
            module = importlib.import_module(name)
            for fn_name, fn in vars(module).items():
                if fn_name.startswith("_") or not inspect.isfunction(fn):
                    continue
                # Include local functions that take no arguments.
                if fn.__module__ != name or inspect.signature(fn).parameters:
                    continue
                yield f"{name}.{fn_name}", fn

    def test_every_cp_recipe_passes_the_gate(self):
        checked = 0
        for name, fn in self._recipes():
            config = fn()
            if config.parallelism.context_parallel_degree == 1:
                continue
            with self.subTest(recipe=name):
                config.__post_init__()
            checked += 1
        # Ensure recipe discovery found at least one CP recipe.
        self.assertGreater(checked, 0)


class TestFluxConfigCpValidation(unittest.TestCase):
    """Flux is not a ``Decoder`` and is covered by the same central gate."""

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
            config.__post_init__()

    def test_allows_partial_dtensor_without_cp(self):
        # Flux on partial_dtensor is FSDP-only but still a valid configuration.
        config = self._config(spmd_backend="partial_dtensor", cp=1)
        config.__post_init__()


if __name__ == "__main__":
    unittest.main()
