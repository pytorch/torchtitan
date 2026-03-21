# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import tempfile
import unittest

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

    def test_fake_tensors_config_default(self):
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        config = GraphTrainerCompileConfig()
        self.assertFalse(config.fake_tensors)

    def test_fake_tensors_config_custom(self):
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        config = GraphTrainerCompileConfig(
            enable=True,
            fake_tensors=True,
        )
        self.assertTrue(config.fake_tensors)


if __name__ == "__main__":
    unittest.main()
