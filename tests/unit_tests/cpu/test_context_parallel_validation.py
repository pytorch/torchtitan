# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect
import unittest
from dataclasses import dataclass
from unittest import mock

from torchtitan.config.transform import ContextParallelTransform
from torchtitan.protocols.module import Module


class TestDecoderConfigCpValidation(unittest.TestCase):
    """``Trainer.Config.__post_init__`` applies the CP gates at config time."""

    @staticmethod
    def _config(*, cp: int, varlen: bool = False, cp_kernel: bool = False):
        from torchtitan.models.common.cp_attention import (
            KVAllGatherCPFlexInnerAttention,
        )
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel,
            llama3_debugmodel_varlen_attn,
        )

        config = (llama3_debugmodel_varlen_attn if varlen else llama3_debugmodel)()
        if cp_kernel:
            # Apply the transform without its final validation.
            ContextParallelTransform(
                inner_attention=KVAllGatherCPFlexInnerAttention
            ).transform(config.model_spec.model)
        config.parallelism.context_parallel_degree = cp
        config.training.max_context_length = 512
        return config

    def test_allows_cp_kernel(self):
        config = self._config(cp=2, cp_kernel=True)
        config.__post_init__()

    def test_allows_plain_flex_without_cp(self):
        config = self._config(cp=1)
        config.__post_init__()

    def test_rejects_cp_kernel_without_cp(self):
        config = self._config(cp=1, cp_kernel=True)
        with self.assertRaisesRegex(ValueError, "context parallel degree is 1"):
            config.__post_init__()

    def test_rejects_plain_flex_cp(self):
        config = self._config(cp=2)
        with self.assertRaisesRegex(ValueError, "KVAllGatherCPFlexInnerAttention"):
            config.__post_init__()

    def test_rejects_varlen_cp(self):
        config = self._config(cp=2, varlen=True)
        with self.assertRaisesRegex(ValueError, "CPInnerAttention"):
            config.__post_init__()

    def test_rejects_an_unrecognized_kernel_cp(self):
        class LocalOnlyAttention(Module):
            @dataclass(kw_only=True, slots=True)
            class Config(Module.Config):
                pass

        config = self._config(cp=2)
        for layer in config.model_spec.model.layers:
            layer.attention.inner_attention = LocalOnlyAttention.Config()
        with self.assertRaisesRegex(ValueError, "CPInnerAttention"):
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
            with mock.patch(
                "torchtitan.components.quantization.float8.has_cuda_capability",
                return_value=True,
            ):
                config = fn()
            if config.parallelism.context_parallel_degree == 1:
                continue
            with self.subTest(recipe=name):
                config.__post_init__()
            checked += 1
        # Ensure recipe discovery found at least one CP recipe.
        self.assertGreater(checked, 0)

    def test_allows_mtp_cp(self):
        from torchtitan.models.deepseek_v3.config_registry import (
            deepseek_v3_debugmodel_mtp,
        )

        config = deepseek_v3_debugmodel_mtp()
        config.parallelism.context_parallel_degree = 2
        config.model_spec.model.update_from_config(config=config)


if __name__ == "__main__":
    unittest.main()
