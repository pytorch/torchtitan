# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.full_dtensor import (
    _find_tied_parameters,
    _remove_sdpa_math_backend,
    _restore_tied_parameters,
    get_dp_mesh_dims,
    validate_config,
)
from torchtitan.models.common.attention import ScaledDotProductAttentionWrapper


def _make_parallel_dims(**kwargs) -> ParallelDims:
    """Create a ParallelDims with sensible defaults for testing."""
    defaults = dict(
        dp_replicate=1,
        dp_shard=-1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=4,
    )
    defaults.update(kwargs)
    return ParallelDims(**defaults)


class TestValidateConfig(unittest.TestCase):
    """Test validate_config rejects unsupported parallelism combinations."""

    def _model_config(self, attn_backend="sdpa"):
        return SimpleNamespace(
            layer=SimpleNamespace(attention=SimpleNamespace(attn_backend=attn_backend))
        )

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_rejects_pp(self):
        pd = _make_parallel_dims(pp=2, world_size=8)
        with self.assertRaises(NotImplementedError, msg="Pipeline Parallel"):
            validate_config(pd, self._model_config())

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_rejects_tp(self):
        pd = _make_parallel_dims(tp=2, world_size=8)
        with self.assertRaises(NotImplementedError, msg="Tensor Parallel"):
            validate_config(pd, self._model_config())

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_rejects_ep(self):
        pd = _make_parallel_dims(ep=2, world_size=8)
        with self.assertRaises(NotImplementedError, msg="Expert Parallel"):
            validate_config(pd, self._model_config())

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_rejects_cp(self):
        pd = _make_parallel_dims(cp=2, world_size=8)
        with self.assertRaises(NotImplementedError, msg="Context Parallel"):
            validate_config(pd, self._model_config())

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_rejects_flex_attention(self):
        pd = _make_parallel_dims()
        with self.assertRaises(NotImplementedError, msg="flex"):
            validate_config(pd, self._model_config(attn_backend="flex"))

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_rejects_varlen_attention(self):
        pd = _make_parallel_dims()
        with self.assertRaises(NotImplementedError, msg="varlen"):
            validate_config(pd, self._model_config(attn_backend="varlen"))

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_accepts_fsdp_only(self):
        pd = _make_parallel_dims()
        # Should not raise
        validate_config(pd, self._model_config())

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_accepts_hsdp(self):
        pd = _make_parallel_dims(dp_replicate=2, dp_shard=2)
        validate_config(pd, self._model_config())

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_accepts_model_config_without_layer(self):
        """Model configs without layer attr (e.g. flux) should pass."""
        pd = _make_parallel_dims()
        validate_config(pd, SimpleNamespace())


class TestFindTiedParameters(unittest.TestCase):
    """Test _find_tied_parameters detects shared parameter objects."""

    def test_no_tied_params(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        groups = _find_tied_parameters(model)
        self.assertEqual(len(groups), 0)

    def test_tied_params_detected(self):
        model = nn.Module()
        embed = nn.Embedding(10, 4)
        output = nn.Linear(4, 10, bias=False)
        output.weight = embed.weight  # tie weights
        model.embed = embed
        model.output = output

        groups = _find_tied_parameters(model)
        self.assertEqual(len(groups), 1)
        # The group should contain 2 locations
        self.assertEqual(len(groups[0]), 2)

    def test_three_way_tie(self):
        model = nn.Module()
        shared_param = nn.Parameter(torch.randn(4, 4))
        a = nn.Linear(4, 4, bias=False)
        b = nn.Linear(4, 4, bias=False)
        c = nn.Linear(4, 4, bias=False)
        a.weight = shared_param
        b.weight = shared_param
        c.weight = shared_param
        model.a = a
        model.b = b
        model.c = c

        groups = _find_tied_parameters(model)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)


class TestRestoreTiedParameters(unittest.TestCase):
    """Test _restore_tied_parameters re-ties broken parameter identity."""

    def test_restore_after_break(self):
        model = nn.Module()
        embed = nn.Embedding(10, 4)
        output = nn.Linear(4, 10, bias=False)
        output.weight = embed.weight
        model.embed = embed
        model.output = output

        # Record tied groups
        groups = _find_tied_parameters(model)
        self.assertTrue(model.embed.weight is model.output.weight)

        # Simulate distribute_module breaking the tie
        model.output._parameters["weight"] = nn.Parameter(
            model.output.weight.data.clone()
        )
        self.assertFalse(model.embed.weight is model.output.weight)

        # Restore
        _restore_tied_parameters(groups)
        self.assertTrue(model.embed.weight is model.output.weight)


class TestGetDpMeshDims(unittest.TestCase):
    """Test get_dp_mesh_dims returns correct DataParallelMeshDims."""

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_fsdp_only(self):
        pd = _make_parallel_dims(dp_replicate=1, dp_shard=4)
        dims = get_dp_mesh_dims(pd)
        self.assertEqual(dims.shard, "fsdp")
        self.assertIsNone(dims.replicate)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_hsdp(self):
        pd = _make_parallel_dims(dp_replicate=2, dp_shard=2)
        dims = get_dp_mesh_dims(pd)
        self.assertEqual(dims.shard, "fsdp")
        self.assertEqual(dims.replicate, "dp_replicate")

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_replicate_only(self):
        """When only dp_replicate is set, shard should be None."""
        pd = _make_parallel_dims(dp_replicate=4, dp_shard=1)
        dims = get_dp_mesh_dims(pd)
        self.assertIsNone(dims.shard)
        self.assertEqual(dims.replicate, "dp_replicate")


class TestRemoveSdpaMathBackend(unittest.TestCase):
    """Test _remove_sdpa_math_backend strips MATH from SDPA modules."""

    def test_removes_math_backend(self):
        model = nn.Module()
        attn = ScaledDotProductAttentionWrapper()
        attn.sdpa_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
        ]
        model.attn = attn

        _remove_sdpa_math_backend(model)

        self.assertNotIn(SDPBackend.MATH, attn.sdpa_backends)
        self.assertIn(SDPBackend.FLASH_ATTENTION, attn.sdpa_backends)
        self.assertIn(SDPBackend.EFFICIENT_ATTENTION, attn.sdpa_backends)

    def test_noop_when_no_math(self):
        model = nn.Module()
        attn = ScaledDotProductAttentionWrapper()
        attn.sdpa_backends = [SDPBackend.FLASH_ATTENTION]
        model.attn = attn

        _remove_sdpa_math_backend(model)

        self.assertEqual(attn.sdpa_backends, [SDPBackend.FLASH_ATTENTION])

    def test_noop_for_non_sdpa_modules(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        # Should not raise
        _remove_sdpa_math_backend(model)


if __name__ == "__main__":
    unittest.main()
