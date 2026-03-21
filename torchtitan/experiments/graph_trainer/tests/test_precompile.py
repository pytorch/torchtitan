# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import tempfile
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import torch

from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter


class TestDiskStorageAdapter(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            data = pickle.dumps({"hello": "world", "values": [1, 2, 3]})

            path = storage.save("test_key", data)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(storage.exists("test_key"))

            loaded = storage.load("test_key")
            self.assertEqual(data, loaded)

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            with self.assertRaises(FileNotFoundError):
                storage.load("nonexistent")

    def test_exists_false_for_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            self.assertFalse(storage.exists("missing"))

    def test_save_creates_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            storage = DiskStorageAdapter(nested)
            data = b"test"
            path = storage.save("key", data)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(storage.load("key"), data)


class TestPrecompiledArtifact(unittest.TestCase):
    def test_artifact_pickle_roundtrip(self):
        from torchtitan.experiments.graph_trainer.precompile import PrecompiledArtifact

        artifact = PrecompiledArtifact(
            serialized_fn=b"fake_serialized_data",
            params_spec=["layer.weight", "layer.bias"],
            buffers_spec=["running_mean"],
            in_spec=None,
            out_spec=None,
            metadata={"world_size": 8, "model_name": "test"},
        )

        data = pickle.dumps(artifact)
        loaded = pickle.loads(data)

        self.assertEqual(loaded.serialized_fn, artifact.serialized_fn)
        self.assertEqual(loaded.params_spec, artifact.params_spec)
        self.assertEqual(loaded.buffers_spec, artifact.buffers_spec)
        self.assertEqual(loaded.metadata, artifact.metadata)


class TestGraphTrainerCompileConfig(unittest.TestCase):
    def test_precompile_config_defaults(self):
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        config = GraphTrainerCompileConfig()
        self.assertFalse(config.precompile)
        self.assertEqual(config.precompile_artifact_dir, "/tmp/precompile_artifacts")

    def test_precompile_config_custom(self):
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        config = GraphTrainerCompileConfig(
            enable=True,
            precompile=True,
            precompile_artifact_dir="/tmp/test_artifacts",
        )
        self.assertTrue(config.precompile)
        self.assertEqual(config.precompile_artifact_dir, "/tmp/test_artifacts")


@dataclass
class _StubCompileConfig:
    mode: str = "aot"
    backend: str = "aot_eager"
    passes: list = field(default_factory=list)
    joint_passes: list = field(default_factory=list)


@dataclass
class _StubParallelDims:
    world_size: int = 8
    dp_replicate: int = 1
    dp_shard: int = 2
    cp: int = 1
    tp: int = 2
    pp: int = 2
    ep: int = 1
    etp: int = 1


def _make_stub_model(params=None, buffers=None):
    """
    Build a mock model with controlled named_parameters() and
    named_buffers() for deterministic fingerprint testing.
    """
    if params is None:
        params = [
            ("layer.weight", torch.zeros(4, 4)),
            ("layer.bias", torch.zeros(4)),
        ]
    if buffers is None:
        buffers = [("running_mean", torch.zeros(4))]

    model = MagicMock()
    model.named_parameters.return_value = iter(params)
    model.named_buffers.return_value = iter(buffers)
    return model


class TestConfigFingerprint(unittest.TestCase):
    def test_deterministic(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        cfg = _StubCompileConfig()
        dims = _StubParallelDims()

        fp1 = compute_config_fingerprint(_make_stub_model(), cfg, dims)
        fp2 = compute_config_fingerprint(_make_stub_model(), cfg, dims)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 16)

    def test_model_shape_sensitivity(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        cfg = _StubCompileConfig()
        dims = _StubParallelDims()

        model_a = _make_stub_model(params=[("w", torch.zeros(4, 4))], buffers=[])
        model_b = _make_stub_model(params=[("w", torch.zeros(8, 8))], buffers=[])
        fp_a = compute_config_fingerprint(model_a, cfg, dims)
        fp_b = compute_config_fingerprint(model_b, cfg, dims)
        self.assertNotEqual(fp_a, fp_b)

    def test_parallelism_sensitivity(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        cfg = _StubCompileConfig()
        model = _make_stub_model()

        dims_tp2 = _StubParallelDims(tp=2)
        dims_tp4 = _StubParallelDims(tp=4)
        fp_tp2 = compute_config_fingerprint(model, cfg, dims_tp2)
        fp_tp4 = compute_config_fingerprint(_make_stub_model(), cfg, dims_tp4)
        self.assertNotEqual(fp_tp2, fp_tp4)

    def test_compile_config_sensitivity(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        dims = _StubParallelDims()

        cfg_a = _StubCompileConfig(passes=["pass_a"])
        cfg_b = _StubCompileConfig(passes=["pass_a", "pass_b"])
        fp_a = compute_config_fingerprint(_make_stub_model(), cfg_a, dims)
        fp_b = compute_config_fingerprint(_make_stub_model(), cfg_b, dims)
        self.assertNotEqual(fp_a, fp_b)

    def test_pass_order_insensitive(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        dims = _StubParallelDims()

        cfg_ab = _StubCompileConfig(passes=["a", "b"])
        cfg_ba = _StubCompileConfig(passes=["b", "a"])
        fp_ab = compute_config_fingerprint(_make_stub_model(), cfg_ab, dims)
        fp_ba = compute_config_fingerprint(_make_stub_model(), cfg_ba, dims)
        self.assertEqual(fp_ab, fp_ba)


if __name__ == "__main__":
    unittest.main()
