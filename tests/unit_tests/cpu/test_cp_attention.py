# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernel selection and mesh lookup."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.cp_attention import (
    AllGatherCPFlexAttention,
    ContextParallelKernel,
)


class TestKernelSelection(unittest.TestCase):
    def test_cp_kernel_is_a_flex_kernel(self):
        self.assertIsInstance(AllGatherCPFlexAttention.Config(), FlexAttention.Config)

    def test_cp_kernel_inherits_flex_fields(self):
        config = AllGatherCPFlexAttention.Config(block_size=256)
        self.assertEqual(config.block_size, 256)

    def test_cp_kernel_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("allgather_cp_flex")

    def test_plain_flex_is_not_a_cp_kernel(self):
        kernel = get_attention_config("flex")._owner
        assert kernel is not None
        self.assertFalse(issubclass(kernel, ContextParallelKernel))

    def test_swap_keeps_the_tuning_of_the_kernel_it_replaces(self):
        from torchtitan.models.common.cp_attention import use_cp_kernel
        from torchtitan.models.llama3 import model_registry
        from torchtitan.models.llama3.config_registry import llama3_debugmodel

        config = llama3_debugmodel()
        config.model_spec = model_registry("debugmodel", attn_backend="flex")
        tuned = config.model_spec.model.layers[0].attention.inner_attention
        tuned.block_size = (256, 128)
        tuned.kernel_options = {"BACKEND": "FLASH"}

        use_cp_kernel(config, AllGatherCPFlexAttention)

        swapped = config.model_spec.model.layers[0].attention.inner_attention
        self.assertIsInstance(swapped, AllGatherCPFlexAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_swap_rejects_a_kernel_that_is_not_context_parallel(self):
        from torchtitan.models.common.cp_attention import use_cp_kernel
        from torchtitan.models.llama3.config_registry import llama3_debugmodel

        with self.assertRaisesRegex(ValueError, "must inherit ContextParallelKernel"):
            use_cp_kernel(llama3_debugmodel(), FlexAttention)


class _FakeMesh:
    def __init__(self, cp_size: int | None):
        self.mesh_dim_names = ("dp", "tp") if cp_size is None else ("dp", "cp", "tp")
        self._cp_size = cp_size

    def get_group(self, axis):
        assert axis == "cp"
        return SimpleNamespace(size=lambda: self._cp_size)


def _in_mesh(cp_size):
    return mock.patch(
        "torchtitan.models.common.cp_attention.current_spmd_mesh",
        return_value=_FakeMesh(cp_size),
    )


class TestCpGroup(unittest.TestCase):
    """A CP kernel demands a CP group; it never degrades to plain attention."""

    @staticmethod
    def _kernel():
        return AllGatherCPFlexAttention(AllGatherCPFlexAttention.Config())

    def test_cp_axis_above_one_yields_its_group(self):
        with _in_mesh(8):
            self.assertEqual(self._kernel().cp_group.size(), 8)

    def test_no_mesh_context_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "requires an active SPMD mesh"):
            self._kernel().cp_group

    def test_degree_one_is_an_error(self):
        with _in_mesh(1), self.assertRaisesRegex(
            RuntimeError, "requires an active CP mesh"
        ):
            self._kernel().cp_group

    def test_mesh_without_a_cp_axis_is_an_error(self):
        with _in_mesh(None), self.assertRaisesRegex(
            RuntimeError, "requires an active CP mesh"
        ):
            self._kernel().cp_group

    def test_forward_without_a_cp_group_is_an_error(self):
        num_tokens, heads, head_dim = 8, 2, 16
        q, k, v = (torch.randn(num_tokens, heads, head_dim) for _ in range(3))
        with _in_mesh(1), self.assertRaisesRegex(
            RuntimeError, "requires an active CP mesh"
        ):
            self._kernel().forward(q, k, v)

    def test_the_kernel_holds_no_mesh_state(self):
        """Nothing is captured at parallelize time, so none of it can go stale."""
        self.assertNotIn("parallelize", ContextParallelKernel.__dict__)


if __name__ == "__main__":
    unittest.main()
