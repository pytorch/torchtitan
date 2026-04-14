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
from unittest.mock import MagicMock, patch

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

    def test_delete_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("key", b"data")
            self.assertTrue(storage.exists("key"))
            storage.delete("key")
            self.assertFalse(storage.exists("key"))

    def test_delete_nonexistent_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.delete("nonexistent")

    def test_save_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("key", b"original")
            storage.save("key", b"updated")
            self.assertEqual(storage.load("key"), b"updated")

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            with self.assertRaises(ValueError):
                storage.save("../../escape", b"data")


class TestPrecompiledArtifact(unittest.TestCase):
    def test_artifact_pickle_roundtrip(self):
        from torchtitan.experiments.graph_trainer.precompile import PrecompiledArtifact

        artifact = PrecompiledArtifact(
            serialized_fn=b"fake_serialized_data",
            params_spec=("layer.weight", "layer.bias"),
            buffers_spec=("running_mean",),
            out_spec=None,
            metadata={"world_size": 8, "model_name": "test"},
        )

        data = pickle.dumps(artifact)
        loaded = pickle.loads(data)

        self.assertEqual(loaded.serialized_fn, artifact.serialized_fn)
        self.assertEqual(loaded.params_spec, artifact.params_spec)
        self.assertEqual(loaded.buffers_spec, artifact.buffers_spec)
        self.assertEqual(loaded.metadata, artifact.metadata)


class TestApplyCompileValidation(unittest.TestCase):
    """Test that apply_compile raises on invalid precompile configurations."""

    def _make_args(self, **compile_overrides):
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed import ParallelDims
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        compile_config = GraphTrainerCompileConfig(
            enable=True, mode="aot", **compile_overrides
        )
        parallelism = ParallelismConfig()
        parallel_dims = ParallelDims(
            dp_shard=2, dp_replicate=1, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=2
        )
        return dict(
            model=torch.nn.Linear(4, 4),
            compile_config=compile_config,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
            dump_folder="/tmp/test_dump",
        )

    def test_precompile_missing_artifact_raises(self):
        from torchtitan.experiments.graph_trainer.compile import apply_compile

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(
                precompile_artifact_dir=tmpdir,
                passes=["full_inductor_compilation"],
            )
            with self.assertRaisesRegex(ValueError, "not found"):
                apply_compile(**args)

    def test_precompile_without_serializable_pass_raises(self):
        from torchtitan.experiments.graph_trainer.compile import apply_compile

        args = self._make_args(
            precompile_artifact_dir="/tmp/test",
            passes=["auto_bucketing"],
        )
        with self.assertRaisesRegex(ValueError, "serializable output"):
            apply_compile(**args)


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
    # Use side_effect (not return_value) so each call produces a
    # fresh iterator — just like real nn.Module methods. A single
    # return_value=iter(...) would be exhausted after the first call.
    model.named_parameters.side_effect = lambda: iter(params)
    model.named_buffers.side_effect = lambda: iter(buffers)
    return model


class TestPrecompileSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        from torch._dynamo.aot_compile_types import (
            BundledAOTAutogradSerializableCallable,
        )

        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_load,
            precompile_save,
        )

        model = torch.nn.Linear(4, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            compiled_fn = MagicMock(spec=BundledAOTAutogradSerializableCallable)
            with patch.object(
                BundledAOTAutogradSerializableCallable,
                "serialize_compile_artifacts",
                return_value=b"fake_serialized",
            ):
                precompile_save(
                    model,
                    compiled_fn,
                    storage,
                    out_spec=None,
                    config_fingerprint="abc123",
                )

            self.assertTrue(storage.exists("default"))

            # Load should succeed with matching model
            wrapper = precompile_load(model, storage, expected_fingerprint="abc123")
            self.assertTrue(callable(wrapper))

    def test_load_param_mismatch(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_load,
            PrecompiledArtifact,
        )

        artifact = PrecompiledArtifact(
            serialized_fn=b"fake",
            params_spec=("layer.weight", "layer.bias"),
            buffers_spec=(),
            out_spec=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("default", pickle.dumps(artifact))

            # Model with different params should fail
            model = torch.nn.Linear(8, 8)
            model.extra = torch.nn.Parameter(torch.zeros(1))
            with self.assertRaisesRegex(ValueError, "Parameter mismatch"):
                precompile_load(model, storage, expected_fingerprint="")

    def test_load_buffer_mismatch(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_load,
            PrecompiledArtifact,
        )

        artifact = PrecompiledArtifact(
            serialized_fn=b"fake",
            params_spec=("weight", "bias"),
            buffers_spec=("running_mean",),
            out_spec=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("default", pickle.dumps(artifact))

            # nn.Linear has no buffers, so buffers_spec won't match
            model = torch.nn.Linear(4, 4)
            with self.assertRaisesRegex(ValueError, "Buffer mismatch"):
                precompile_load(model, storage, expected_fingerprint="")

    def test_load_fingerprint_mismatch(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_load,
            PrecompiledArtifact,
        )

        model = torch.nn.Linear(4, 4)
        artifact = PrecompiledArtifact(
            serialized_fn=b"fake",
            params_spec=tuple(n for n, _ in model.named_parameters()),
            buffers_spec=tuple(n for n, _ in model.named_buffers()),
            out_spec=None,
            config_fingerprint="old_fingerprint",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("default", pickle.dumps(artifact))

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                precompile_load(model, storage, expected_fingerprint="new_fingerprint")

    def test_load_with_out_spec_unflattens(self):
        from torch._dynamo.aot_compile_types import (
            BundledAOTAutogradSerializableCallable,
        )

        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_load,
            PrecompiledArtifact,
        )

        model = torch.nn.Linear(4, 4)
        # Build an out_spec from a dict so we can verify unflattening
        example_output = {"loss": torch.tensor(1.0), "logits": torch.tensor(2.0)}
        flat_values, out_spec = torch.utils._pytree.tree_flatten(example_output)

        artifact = PrecompiledArtifact(
            serialized_fn=b"fake",
            params_spec=tuple(n for n, _ in model.named_parameters()),
            buffers_spec=tuple(n for n, _ in model.named_buffers()),
            out_spec=out_spec,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("default", pickle.dumps(artifact))

            # Mock deserialize to return a fn that returns flat outputs
            def fake_compiled_fn(*args, **kwargs):
                return flat_values

            with patch.object(
                BundledAOTAutogradSerializableCallable,
                "deserialize_compile_artifacts",
                return_value=fake_compiled_fn,
            ):
                wrapper = precompile_load(model, storage, expected_fingerprint="")
                result = wrapper((), {})

            self.assertIsInstance(result, dict)
            self.assertIn("loss", result)
            self.assertIn("logits", result)
            self.assertEqual(result["loss"].item(), 1.0)
            self.assertEqual(result["logits"].item(), 2.0)

    def test_load_legacy_artifact_warns(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            precompile_load,
            PrecompiledArtifact,
        )

        model = torch.nn.Linear(4, 4)
        artifact = PrecompiledArtifact(
            serialized_fn=b"fake",
            params_spec=tuple(n for n, _ in model.named_parameters()),
            buffers_spec=tuple(n for n, _ in model.named_buffers()),
            out_spec=None,
            config_fingerprint="",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save("default", pickle.dumps(artifact))

            with self.assertLogs(level="WARNING") as cm:
                precompile_load(model, storage, expected_fingerprint="some_fp")
            self.assertTrue(any("legacy artifact" in msg for msg in cm.output))


class TestPrecompileSaveValidation(unittest.TestCase):
    def test_non_serializable_compiled_fn_raises(self):
        from torchtitan.experiments.graph_trainer.precompile import precompile_save

        model = torch.nn.Linear(4, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)

            def not_serializable(*args):
                return None

            with self.assertRaisesRegex(
                TypeError, "BundledAOTAutogradSerializableCallable"
            ):
                precompile_save(
                    model,
                    not_serializable,
                    storage,
                    out_spec=None,
                )

    def test_unwrap_from_wrapped_attribute(self):
        """Test that precompile_save can unwrap a plain function whose
        __wrapped__ attribute is a BundledAOTAutogradSerializableCallable,
        matching PyTorch's aot_compile_joint_with_descriptors behavior."""
        from functools import wraps

        from torch._dynamo.aot_compile_types import (
            BundledAOTAutogradSerializableCallable,
        )

        from torchtitan.experiments.graph_trainer.precompile import _unwrap_serializable

        inner = MagicMock(spec=BundledAOTAutogradSerializableCallable)

        @wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        result = _unwrap_serializable(wrapper)
        self.assertIs(result, inner)


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

    def test_pass_order_sensitive(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        dims = _StubParallelDims()

        cfg_ab = _StubCompileConfig(passes=["a", "b"])
        cfg_ba = _StubCompileConfig(passes=["b", "a"])
        fp_ab = compute_config_fingerprint(_make_stub_model(), cfg_ab, dims)
        fp_ba = compute_config_fingerprint(_make_stub_model(), cfg_ba, dims)
        self.assertNotEqual(fp_ab, fp_ba)


if __name__ == "__main__":
    unittest.main()
