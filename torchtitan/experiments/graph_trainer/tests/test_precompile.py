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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from torchtitan.experiments.graph_trainer.configs import EpOverlapConfig
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


@dataclass
class _StubCompileConfig:
    mode: str = "aot_fx_trace"
    backend: str = "aot_eager"
    passes: list = field(default_factory=list)
    ep_overlap: EpOverlapConfig = field(default_factory=EpOverlapConfig)


@dataclass
class _StubParallelDims:
    world_size: int = 8
    dp_replicate: int = 1
    dp_shard: int = 2
    cp: int = 1
    tp: int = 2
    pp: int = 2
    ep: int = 1


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

        cfg_graph_batch = _StubCompileConfig(ep_overlap=EpOverlapConfig(enabled=True))
        cfg_graph_seq = _StubCompileConfig(
            ep_overlap=EpOverlapConfig(
                enabled=True,
                chunk_dim="seq",
                module_fqn="layers.*.moe",
            ),
        )
        fp_graph_batch = compute_config_fingerprint(
            _make_stub_model(), cfg_graph_batch, dims
        )
        fp_graph_seq = compute_config_fingerprint(
            _make_stub_model(), cfg_graph_seq, dims
        )
        self.assertNotEqual(fp_graph_batch, fp_graph_seq)

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


class TestPrecompileLossSetup(unittest.TestCase):
    def test_chunked_loss_setup_matches_trainer_boundary(self):
        from torchtitan.experiments.graph_trainer.chunked_loss import (
            ChunkedCELossWithParamGrads,
        )
        from torchtitan.experiments.graph_trainer.precompile_main import (
            _prepare_loss_for_precompile,
        )

        lm_head = torch.nn.Linear(2, 3)
        model = SimpleNamespace(lm_head=lm_head, _skip_lm_head=False)
        loss_fn = ChunkedCELossWithParamGrads.Config().build()

        _prepare_loss_for_precompile(model, loss_fn)

        self.assertIs(loss_fn.lm_head, lm_head)
        self.assertTrue(model._skip_lm_head)


class TestPrecompiledFxTraceArtifact(unittest.TestCase):
    def test_artifact_pickle_roundtrip(self):
        from torchtitan.experiments.graph_trainer.make_fx_tracer import SubclassLayout
        from torchtitan.experiments.graph_trainer.precompile import (
            PrecompiledFxTraceArtifact,
        )

        flat_vals, spec = torch.utils._pytree.tree_flatten({"a": torch.zeros(2)})
        artifact = PrecompiledFxTraceArtifact(
            serialized_gm=b"fake_serialized_data",
            state_fqns=["w1", "w2"],
            num_flat_inputs=4,
            input_subclass_layouts={
                0: SubclassLayout(1, None),
                1: SubclassLayout(1, None),
            },
            num_flat_outputs=2,
            output_subclass_layouts={0: SubclassLayout(1, None)},
            output_spec=spec,
            tensor_input_indices=[0, 1, 2, 3],
            config_fingerprint="test_fp_123",
        )

        data = pickle.dumps(artifact)
        loaded = pickle.loads(data)

        self.assertEqual(loaded.serialized_gm, artifact.serialized_gm)
        self.assertEqual(len(loaded.state_fqns), 2)
        self.assertEqual(loaded.num_flat_inputs, 4)
        self.assertEqual(len(loaded.input_subclass_layouts), 2)
        self.assertEqual(loaded.num_flat_outputs, 2)
        self.assertEqual(loaded.config_fingerprint, "test_fp_123")

    def test_artifact_pickle_with_blockmask_treespec(self):
        """Verify artifact pickles when user_inputs_spec contains BlockMask.

        BlockMask's pytree context stores a _MaskModWrapper holding the
        mask_mod closure, which is not picklable. The artifact must not
        serialize user_inputs_spec (the mask_mod is already compiled into
        standalone Inductor HOPs baked into serialized_gm).
        """
        from torch.nn.attention.flex_attention import create_block_mask

        from torchtitan.experiments.graph_trainer.common_utils import (
            maybe_register_blockmask_pytree_node,
        )
        from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
        from torchtitan.experiments.graph_trainer.precompile import (
            PrecompiledFxTraceArtifact,
        )
        from torchtitan.models.common.attention import get_causal_mask_mod

        maybe_register_blockmask_pytree_node()

        mask_mod = get_causal_mask_mod()
        block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=128, KV_LEN=128)

        # Build a user_inputs_spec that includes BlockMask — this is what
        # minimal_fx_tracer produces when FlexAttention is configured.
        _, blockmask_spec = torch.utils._pytree.tree_flatten(
            ((torch.zeros(2),), {"attention_masks": block_mask})
        )

        # Sanity: the raw TreeSpec itself is NOT picklable (the bug).
        with self.assertRaises((TypeError, AttributeError)):
            pickle.dumps(blockmask_spec)

        # Build a TracedResult with the unpicklable spec, then create
        # the artifact via from_traced_result — this must succeed because
        # user_inputs_spec is excluded from serialization.
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        dummy_spec = torch.utils._pytree.tree_flatten(((), {}))[1]
        traced_result = TracedResult(
            gm=gm,
            example_inputs=(),
            num_flat_inputs=0,
            input_subclass_layouts={},
            user_inputs_spec=blockmask_spec,
            tensor_input_indices=[],
            num_flat_outputs=0,
            output_subclass_layouts={},
            output_spec=dummy_spec,
            state_fqns=[],
        )

        artifact = PrecompiledFxTraceArtifact.from_traced_result(traced_result)
        data = pickle.dumps(artifact)
        loaded = pickle.loads(data)
        self.assertEqual(loaded.serialized_gm, artifact.serialized_gm)

    def test_fx_trace_save_load_fingerprint_mismatch(self):
        from torchtitan.experiments.graph_trainer.precompile import (
            _FX_TRACE_ARTIFACT_KEY,
            precompile_fx_trace_load,
            PrecompiledFxTraceArtifact,
        )

        flat_vals, spec = torch.utils._pytree.tree_flatten({"a": torch.zeros(2)})
        artifact = PrecompiledFxTraceArtifact(
            serialized_gm=b"fake",
            state_fqns=["w"],
            num_flat_inputs=2,
            input_subclass_layouts={},
            num_flat_outputs=1,
            output_subclass_layouts={},
            output_spec=spec,
            tensor_input_indices=[0, 1],
            config_fingerprint="old_fp",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            storage.save(_FX_TRACE_ARTIFACT_KEY, pickle.dumps(artifact))

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                precompile_fx_trace_load(
                    storage,
                    expected_fingerprint="new_fp",
                )


class TestCudagraphPass(unittest.TestCase):
    """Test cudagraph_pass behavior."""

    def test_non_graphmodule_raises(self):
        """cudagraph_pass rejects non-GraphModule callables (e.g.
        OutputCode from full_inductor_compilation)."""
        from torchtitan.experiments.graph_trainer.passes import cudagraph_pass

        def plain_fn(*args):
            return args

        with self.assertRaisesRegex(TypeError, "requires a GraphModule"):
            cudagraph_pass(plain_fn, (torch.zeros(4),), static_input_indices=[0])

    def test_graphmodule_wraps_forward(self):
        """cudagraph_pass wraps gm.forward with CUDAGraphWrapper."""
        from torchtitan.experiments.graph_trainer.passes import cudagraph_pass

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        example_inputs = (torch.zeros(4),)

        with patch(
            "torchtitan.experiments.graph_trainer.cudagraph.CUDAGraphWrapper"
        ) as MockWrapper:
            mock_instance = MagicMock()
            MockWrapper.return_value = mock_instance
            result = cudagraph_pass(gm, example_inputs, static_input_indices=[0])
            self.assertIs(result, gm)
            self.assertIs(gm.forward, mock_instance)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_minimal_async_ep_custom_ops_are_wrapped_by_cudagraph_pass(self):
        """MinimalAsyncEP custom ops should not force cudagraph_pass fallback."""
        import torchtitan.distributed.minimal_async_ep  # noqa: F401
        from torchtitan.experiments.graph_trainer.passes import cudagraph_pass

        def cuda_i64(*shape):
            return torch.empty(*shape, device="cuda", dtype=torch.int64)

        def cuda_f32(*shape):
            return torch.empty(*shape, device="cuda", dtype=torch.float32)

        graph = torch.fx.Graph()
        op_outputs = [
            (
                torch.ops.minimal_async_ep.dispatch.default,
                (
                    cuda_f32(16, 8),
                    cuda_i64(12),
                    cuda_i64(12),
                    cuda_i64(16),
                    cuda_i64(16),
                    cuda_i64(1),
                    cuda_i64(12),
                    cuda_i64(12),
                    cuda_i64(4),
                ),
            ),
            (
                torch.ops.minimal_async_ep.combine.default,
                (cuda_f32(4, 8), cuda_f32(12, 8)),
            ),
            (
                torch.ops.minimal_async_ep.dispatch_backward.default,
                cuda_f32(4, 8),
            ),
            (
                torch.ops.minimal_async_ep.combine_backward.default,
                (cuda_f32(16, 8), cuda_f32(12)),
            ),
        ]
        nodes = []
        for target, meta_val in op_outputs:
            node = graph.call_function(target, args=())
            node.meta["val"] = meta_val
            nodes.append(node)
        graph.output(tuple(nodes))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with patch(
            "torchtitan.experiments.graph_trainer.cudagraph.CUDAGraphWrapper"
        ) as MockWrapper:
            mock_instance = MagicMock()
            MockWrapper.return_value = mock_instance
            result = cudagraph_pass(gm, (), static_input_indices=[])

            self.assertIs(result, gm)
            self.assertIs(gm.forward, mock_instance)
            MockWrapper.assert_called_once()
            _, example_inputs, static_input_indices = MockWrapper.call_args.args
            self.assertEqual(example_inputs, ())
            self.assertEqual(static_input_indices, [])


class TestCudagraphFingerprintConsistency(unittest.TestCase):
    """Test that save and load paths produce the same fingerprint.

    Both paths compute the fingerprint from the original (unmodified)
    compile_config — cudagraph stripping in precompile_main happens
    AFTER fingerprint computation, so no manual filtering is needed.
    """

    def test_cudagraph_included_in_fingerprint(self):
        """Cudagraph in passes should produce a different fingerprint
        than without cudagraph — no filtering is applied."""
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        dims = _StubParallelDims()

        cfg_with = _StubCompileConfig(passes=["full_inductor_compilation", "cudagraph"])
        cfg_without = _StubCompileConfig(passes=["full_inductor_compilation"])

        fp_with = compute_config_fingerprint(_make_stub_model(), cfg_with, dims)
        fp_without = compute_config_fingerprint(_make_stub_model(), cfg_without, dims)

        self.assertNotEqual(fp_with, fp_without)

    def test_same_config_produces_same_fingerprint(self):
        """Both save and load paths use the same unmodified config,
        so the fingerprint is identical."""
        from torchtitan.experiments.graph_trainer.precompile import (
            compute_config_fingerprint,
        )

        dims = _StubParallelDims()

        cfg = _StubCompileConfig(passes=["full_inductor_compilation", "cudagraph"])

        fp1 = compute_config_fingerprint(_make_stub_model(), cfg, dims)
        fp2 = compute_config_fingerprint(_make_stub_model(), cfg, dims)

        self.assertEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
