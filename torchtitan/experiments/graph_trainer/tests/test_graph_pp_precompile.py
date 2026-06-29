# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.graph_trainer.graph_pp.precompile import (
    _deserialize_gm,
    _serialize_gm,
    build_graph_pp_stage_loader,
    compute_graph_pp_fingerprint,
    ensure_schedule_precompilable,
    graph_pp_rank_artifact_key,
    SerializedRankBundle,
    SerializedStageBundle,
)


@dataclass
class _StubCompileConfig:
    mode: str = "aot_fx_trace"
    backend: str = "inductor"
    passes: list = field(default_factory=list)
    disable_passes: list = field(default_factory=list)
    inductor_compilation: str = "regional"
    enable: bool = True
    enable_passes: bool = True
    numerics_changing_optim: bool = False
    memory_policy: str = "default"
    cpu_offload_prefetch_n_layers: int = 1
    cpu_offload_defer_n_layers: int = 1
    cpu_offload_budget_gb: float = 100.0


@dataclass
class _StubParallelDims:
    world_size: int = 8
    dp_replicate: int = 1
    dp_shard: int = 2
    cp: int = 1
    tp: int = 1
    pp: int = 2
    ep: int = 1


@dataclass
class _StubParallelism:
    enable_async_tensor_parallel: bool = False
    fsdp_reshard_after_forward: str = "default"
    pipeline_parallel_microbatch_size: int = 1


@dataclass
class _StubTraining:
    seq_len: int = 2048
    local_batch_size: int = 8
    global_batch_size: int = 16
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"


@dataclass
class _StubLoss:
    num_chunks: int = 8


@dataclass
class _StubLossSameFields:
    # Same field(s) as _StubLoss but a distinct type, to isolate the loss:type
    # qualname hash from the per-field hash loop.
    num_chunks: int = 8


@dataclass
class _StubOtherLoss:
    global_vocab_size: int = 1000


@dataclass
class _StubDebug:
    deterministic: bool = False
    deterministic_warn_only: bool = False


def _make_fx_gm(scale: float = 2.0):
    """A tiny fake-traced GraphModule with placeholder meta['val'] populated."""

    def fn(x):
        return x * scale + x.sum()

    return make_fx(fn, tracing_mode="fake")(torch.randn(4, 4))


def _make_fake_stage(*, stage_index, is_first, is_last, compiled=True):
    from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphCallables

    callables = GraphCallables(fw=_make_fx_gm(2.0), full_bw=_make_fx_gm(3.0))
    return SimpleNamespace(
        stage_index=stage_index,
        is_first=is_first,
        is_last=is_last,
        graph_callables=callables,
        graph_meta=SimpleNamespace(num_user_outputs=1, partition="stub"),
        trace_spec=SimpleNamespace(output_spec=None, output_grad_spec=None),
        _graph_pp_callables_compiled=compiled,
    )


class TestArtifactKey(unittest.TestCase):
    def test_per_rank_keys_are_distinct_and_flat(self):
        keys = {graph_pp_rank_artifact_key(r) for r in range(4)}
        self.assertEqual(len(keys), 4)
        for key in keys:
            self.assertNotIn("/", key)
            self.assertTrue(key.startswith("graph_pp_pp"))


class TestScheduleGuard(unittest.TestCase):
    def test_overlap_schedule_rejected(self):
        with self.assertRaises(NotImplementedError):
            ensure_schedule_precompilable("DualPipeV")

    def test_supported_schedules_allowed(self):
        ensure_schedule_precompilable("Interleaved1F1B")
        ensure_schedule_precompilable("ZBVZeroBubble")


class TestGraphModuleRoundTrip(unittest.TestCase):
    def test_serialize_deserialize_preserves_computation(self):
        gm = _make_fx_gm(2.0)
        data = _serialize_gm(gm)
        self.assertIsInstance(data, bytes)

        loaded = _deserialize_gm(data)

        # The deserialized graph must reproduce the original computation. The
        # GraphPickler distributed metadata filter strips meta["val"]; that is
        # fine because GraphPP precompile serializes compiled graphs that are
        # executed directly at load (no Inductor reinvocation), matching the
        # non-PP fx_trace artifact.
        x = torch.randn(4, 4)
        torch.testing.assert_close(gm(x), loaded(x))


class TestSerializedBundleRoundTrip(unittest.TestCase):
    def test_stage_bundle_roundtrip_reconstructs_callables(self):
        stage = _make_fake_stage(stage_index=0, is_first=True, is_last=False)
        serialized = SerializedStageBundle.from_stage(stage)

        # Optional callables that were absent stay absent.
        self.assertEqual(set(serialized.serialized_callables), {"fw", "full_bw"})

        callables = serialized.to_graph_callables()
        self.assertIsNotNone(callables.fw)
        self.assertIsNotNone(callables.full_bw)
        self.assertIsNone(callables.bw_di)

        x = torch.randn(4, 4)
        torch.testing.assert_close(callables.fw(x), stage.graph_callables.fw(x))
        torch.testing.assert_close(
            callables.full_bw(x), stage.graph_callables.full_bw(x)
        )

    def test_compiled_flag_round_trips(self):
        for compiled in (True, False):
            stage = _make_fake_stage(
                stage_index=0, is_first=True, is_last=False, compiled=compiled
            )
            serialized = SerializedStageBundle.from_stage(stage)
            self.assertEqual(serialized.compiled, compiled)

    def test_rank_bundle_pickles_and_loader_installs(self):
        stages = [
            _make_fake_stage(stage_index=0, is_first=True, is_last=False),
            _make_fake_stage(stage_index=1, is_first=False, is_last=True),
        ]
        bundle = SerializedRankBundle.from_stages(
            stages, pp_rank=0, schedule_name="Interleaved1F1B"
        )
        # The whole bundle (GraphPickler bytes nested in a picklable dataclass)
        # must survive plain pickle for disk storage.
        restored: SerializedRankBundle = pickle.loads(pickle.dumps(bundle))
        self.assertEqual([s.stage_index for s in restored.stages], [0, 1])

        loader = build_graph_pp_stage_loader(restored)
        target = _make_fake_stage(stage_index=1, is_first=False, is_last=True)
        target.graph_callables = None
        target.graph_meta = None
        target._graph_pp_callables_compiled = False
        loader(target)
        self.assertIsNotNone(target.graph_callables)
        self.assertIsNotNone(target.graph_meta)
        # The loader restores the saved compiled state (True here), so the
        # downstream short-circuit skips recompilation.
        self.assertTrue(target._graph_pp_callables_compiled)

    def test_loader_rejects_unknown_stage(self):
        stages = [_make_fake_stage(stage_index=0, is_first=True, is_last=True)]
        bundle = SerializedRankBundle.from_stages(
            stages, pp_rank=0, schedule_name="Interleaved1F1B"
        )
        loader = build_graph_pp_stage_loader(bundle)
        missing = _make_fake_stage(stage_index=7, is_first=False, is_last=False)
        with self.assertRaises(ValueError):
            loader(missing)


class TestGraphPpFingerprint(unittest.TestCase):
    def _parts(self):
        return [nn.Linear(4, 4), nn.Linear(4, 2)]

    def _fp(
        self,
        parts=None,
        cc=None,
        pd=None,
        schedule_name="Interleaved1F1B",
        parallelism=None,
        training=None,
        loss_config=None,
        debug_config=None,
    ):
        return compute_graph_pp_fingerprint(
            parts if parts is not None else self._parts(),
            cc if cc is not None else _StubCompileConfig(),
            pd if pd is not None else _StubParallelDims(),
            schedule_name=schedule_name,
            parallelism=parallelism if parallelism is not None else _StubParallelism(),
            training=training if training is not None else _StubTraining(),
            loss_config=loss_config if loss_config is not None else _StubLoss(),
            debug_config=debug_config if debug_config is not None else _StubDebug(),
        )

    def test_deterministic(self):
        a = self._fp()
        b = self._fp()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_schedule_name_sensitivity(self):
        self.assertNotEqual(
            self._fp(schedule_name="Interleaved1F1B"),
            self._fp(schedule_name="ZBVZeroBubble"),
        )

    def test_parallelism_sensitivity(self):
        self.assertNotEqual(
            self._fp(pd=_StubParallelDims(pp=2)),
            self._fp(pd=_StubParallelDims(pp=4)),
        )

    def test_inductor_compilation_sensitivity(self):
        self.assertNotEqual(
            self._fp(cc=_StubCompileConfig(inductor_compilation="regional")),
            self._fp(cc=_StubCompileConfig(inductor_compilation="full")),
        )

    def test_model_shape_sensitivity(self):
        self.assertNotEqual(
            self._fp(parts=[nn.Linear(4, 4)]),
            self._fp(parts=[nn.Linear(4, 8)]),
        )

    def test_graph_changing_compile_flags_sensitivity(self):
        # These all change the saved/compiled graph and must invalidate stale
        # artifacts (the bug the adversarial review found).
        base = self._fp()
        self.assertNotEqual(
            base, self._fp(cc=_StubCompileConfig(numerics_changing_optim=True))
        )
        self.assertNotEqual(
            base, self._fp(cc=_StubCompileConfig(memory_policy="sac_and_offload"))
        )
        self.assertNotEqual(base, self._fp(cc=_StubCompileConfig(enable=False)))
        self.assertNotEqual(
            base, self._fp(cc=_StubCompileConfig(cpu_offload_defer_n_layers=3))
        )
        self.assertNotEqual(
            base, self._fp(cc=_StubCompileConfig(disable_passes=["cudagraph_pass"]))
        )

    def test_parallelism_graph_flags_sensitivity(self):
        base = self._fp()
        self.assertNotEqual(
            base,
            self._fp(parallelism=_StubParallelism(enable_async_tensor_parallel=True)),
        )
        self.assertNotEqual(
            base,
            self._fp(parallelism=_StubParallelism(fsdp_reshard_after_forward="never")),
        )

    def test_microbatch_shape_sensitivity(self):
        # The microbatch shape is baked into the traced graphs as static shapes,
        # so changing the per-microbatch size or sequence length must invalidate
        # a saved artifact.
        base = self._fp()
        self.assertNotEqual(
            base,
            self._fp(parallelism=_StubParallelism(pipeline_parallel_microbatch_size=2)),
        )
        self.assertNotEqual(base, self._fp(training=_StubTraining(seq_len=4096)))

    def test_batch_size_sensitivity(self):
        # The resolved global batch size feeds the global_valid_tokens loss
        # divisor baked into the last stage's loss graph; a mismatch would
        # silently rescale the loss.
        base = self._fp()
        self.assertNotEqual(
            base, self._fp(training=_StubTraining(global_batch_size=32))
        )
        self.assertNotEqual(base, self._fp(training=_StubTraining(local_batch_size=4)))

    def test_mixed_precision_sensitivity(self):
        # mixed_precision_param/reduce set the FSDP all-gather/reduce-scatter
        # dtypes baked into the unshard/reduce_grad graphs but do not change the
        # stored param dtype, so the param loop misses them: a mismatch would
        # load graphs that cast to the wrong dtype with no fingerprint change.
        base = self._fp()
        self.assertNotEqual(
            base, self._fp(training=_StubTraining(mixed_precision_param="float32"))
        )
        self.assertNotEqual(
            base, self._fp(training=_StubTraining(mixed_precision_reduce="bfloat16"))
        )

    def test_loss_config_sensitivity(self):
        # num_chunks is baked into the last stage's chunked-loss graph (static
        # chunk shapes + unrolled chunk count); the loss type changes the whole
        # last-stage loss structure. Both must invalidate a stale artifact.
        base = self._fp()
        self.assertNotEqual(base, self._fp(loss_config=_StubLoss(num_chunks=16)))
        self.assertNotEqual(base, self._fp(loss_config=_StubOtherLoss()))
        # Same fields, different type: isolates the loss-type qualname hash from
        # the per-field loop (two loss families with identical fields must not
        # collide into the same artifact).
        self.assertNotEqual(base, self._fp(loss_config=_StubLossSameFields()))

    def test_deterministic_mode_sensitivity(self):
        # The save process sets use_deterministic_algorithms from
        # debug.deterministic before tracing and the compiled backward captures
        # it; a save/load mismatch must invalidate the artifact instead of
        # tripping the runtime determinism assert.
        base = self._fp()
        self.assertNotEqual(base, self._fp(debug_config=_StubDebug(deterministic=True)))
        self.assertNotEqual(
            base, self._fp(debug_config=_StubDebug(deterministic_warn_only=True))
        )


if __name__ == "__main__":
    unittest.main()
