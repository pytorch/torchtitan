# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Kimi Linear's PP adapter plumbing.

Focused on the parts that are Kimi-specific (FQN name remapping +
AttnRes-presence detection via ``num_blocks`` attr). The heavy lift —
``CrossStageCacheAdapter`` / ``RankLocalCache`` / the hook+detach
bridge — is tested in ``torchtitan/models/kimi_k3/tests/`` and
reused verbatim.
"""

from __future__ import annotations

import unittest

from torchtitan.models.kimi_k3.pipeline_adapter import (
    _KIMI_ATTN_RES_LAST_STAGE_FQNS,
    _kimi_llm_fqns,
)


class TestKimiFQNRemapping(unittest.TestCase):
    def test_embed_tokens_and_lm_head_replacements(self):
        """``tok_embeddings`` → ``embed_tokens``, ``output`` → ``lm_head``."""
        # 2 stages, 4 layers, default weights.
        fqns = _kimi_llm_fqns(num_stages=2, num_layers=4)
        # Stage 0 should start with embed_tokens, stage 1 ends with lm_head.
        self.assertEqual(fqns[0][0], "embed_tokens")
        self.assertIn("lm_head", fqns[-1])
        self.assertNotIn("tok_embeddings", fqns[0])
        self.assertNotIn("output", fqns[-1])

    def test_layers_preserved(self):
        """Layer FQNs (``layers.N``) pass through untouched."""
        fqns = _kimi_llm_fqns(num_stages=2, num_layers=4)
        flat = [name for stage in fqns for name in stage]
        for i in range(4):
            self.assertIn(f"layers.{i}", flat)

    def test_stage_count(self):
        """Requested stage count matches output length."""
        for n in (1, 2, 4, 8):
            fqns = _kimi_llm_fqns(
                num_stages=n,
                num_layers=max(n, 4),
            )
            self.assertEqual(len(fqns), n)

    def test_attn_res_extra_fqns_constant(self):
        """Last-stage AttnRes extras are exactly the two final modules."""
        self.assertEqual(
            _KIMI_ATTN_RES_LAST_STAGE_FQNS,
            ("output_res_proj", "output_res_norm"),
        )


class TestPipeliningFnInModelSpec(unittest.TestCase):
    def test_all_flavors_wire_pipelining_fn(self):
        """Every registered flavor's ModelSpec points at
        ``pipeline_kimi_k3_with_cache_adapter``. Runtime detection
        (baseline vs AttnRes) happens inside that function via
        ``num_blocks`` attr check, not at registration time.
        """
        from torchtitan.models.kimi_k3 import flavor_names, model_registry
        from torchtitan.models.kimi_k3.pipeline_adapter import (
            pipeline_kimi_k3_with_cache_adapter,
        )

        for flavor in flavor_names():
            spec = model_registry(flavor)
            self.assertEqual(
                spec.pipelining_fn,
                pipeline_kimi_k3_with_cache_adapter,
                f"{flavor}: pipelining_fn not wired",
            )


class TestContiguousSplitGuard(unittest.TestCase):
    """The layer->stage discovery verifies the layout it cannot replace.

    ``stages`` is the local rank's stages, so the discovery can never see every
    layer and the map it builds is always partial. What it can do is check the
    contiguous default against the layers this rank actually holds.
    """

    @staticmethod
    def _stage(stage_index: int, layer_ids):
        from torch import nn

        submod = nn.Module()
        if layer_ids is not None:
            submod.layers = nn.ModuleDict({str(i): nn.Identity() for i in layer_ids})
        stage = nn.Module()
        stage.submod = submod
        stage.stage_index = stage_index
        return stage

    def _infer(self, stages):
        from torchtitan.models.kimi_k3.layout import (
            _infer_block_layout_tables_from_stages,
        )

        # 8 layers over 2 stages -> 4 per stage; blocks of 4 -> 2 blocks.
        return _infer_block_layout_tables_from_stages(
            stages, pp_size=2, num_blocks=2, n_layers=8, layers_per_block=4
        )

    def test_a_contiguous_rank_is_accepted(self):
        tables = self._infer([self._stage(1, [4, 5, 6, 7])])
        self.assertEqual(tables.num_blocks, 2)

    def test_a_non_contiguous_split_raises_instead_of_mislaying_blocks(self):
        # Stage 1 holding the first four layers contradicts the default, which
        # would route block deltas to the wrong stage.
        with self.assertRaises(ValueError) as ctx:
            self._infer([self._stage(1, [0, 1, 2, 3])])
        self.assertIn("non-contiguous", str(ctx.exception))

    def test_stages_without_layers_leave_nothing_to_verify(self):
        tables = self._infer([self._stage(0, None)])
        self.assertEqual(tables.num_blocks, 2)


class TestStepEndSweep(unittest.TestCase):
    """What the step-end sweep evicts.

    Only backward marks a microbatch as seen, so a sweep keyed on the seen-set
    alone cannot reach anything a forward-only pass cached.
    """

    @staticmethod
    def _adapter():
        from torch import nn

        from torchtitan.models.kimi_k3.pipeline_adapter import CrossStageCacheAdapter

        return CrossStageCacheAdapter(nn.Identity(), stage_id=0, num_stages=1)

    def test_a_forward_only_microbatch_is_evicted(self):
        import torch

        adapter = self._adapter()
        adapter._cache.append(0, torch.zeros(2), (0, 0, 0))
        # Evaluation reaches exactly this state: cached blocks, nothing marked.
        self.assertEqual(adapter._cache._seen_mbs, set())
        adapter._drop_all_cached_and_clear()
        self.assertEqual(adapter._cache.get_blocks(0), [])

    def test_a_backward_marked_microbatch_is_still_evicted(self):
        import torch

        adapter = self._adapter()
        adapter._cache.append(1, torch.zeros(2), (0, 0, 0))
        adapter.on_microbatch_end(1)
        adapter._drop_all_cached_and_clear()
        self.assertEqual(adapter._cache.get_blocks(1), [])
        self.assertEqual(adapter._cache._seen_mbs, set())


class TestMultiCommitProducers(unittest.TestCase):
    """A stage whose layer span is wider than one AttnRes block.

    ``layers_per_stage > layers_per_block`` puts several block boundaries on one
    stage, so that stage commits several blocks. The layout used to refuse this
    with a NotImplementedError naming ``_RecvBlockGradsFromConsumers``, a class
    deleted when the custom grad P2P was replaced by the rank-local capture and
    augment hooks; the table that restriction protected
    (``consumer_stages_of``) had no readers left. Every remaining table is keyed
    by the commit's index within its producer stage, which is what these tests
    pin down -- the runtime keys its cache and its hooks the same way.
    """

    @staticmethod
    def _tables(*, P, V, n_layers, layers_per_block):
        from torchtitan.models.kimi_k3.layout import BlockLayoutTables

        return BlockLayoutTables(
            pp_size=P,
            virtual_stages_per_rank=V,
            num_blocks=-(-n_layers // layers_per_block),
            n_layers=n_layers,
            layers_per_block=layers_per_block,
        )

    def test_two_boundaries_on_one_stage_build_a_table(self):
        # K3's 12-layer blocks over 96 layers at pp=2, V=2: 24 layers a stage,
        # so two commits each.
        tables = self._tables(P=2, V=2, n_layers=96, layers_per_block=12)
        self.assertEqual(tables.commits_at(0), [0, 1])
        self.assertEqual(tables.commits_at(3), [6, 7])

    def test_every_block_has_exactly_one_producer(self):
        tables = self._tables(P=2, V=2, n_layers=96, layers_per_block=12)
        owned = [b for s in range(4) for b in tables.commits_at(s)]
        self.assertEqual(sorted(owned), list(range(8)))
        for b in range(8):
            self.assertEqual(
                tables.producer_stage_of_block(b),
                next(s for s in range(4) if b in tables.commits_at(s)),
            )

    def test_captures_are_counted_per_commit_not_per_stage(self):
        # Both of stage 0's commits are read by stage 2 (same rank, later
        # virtual stage), so each commit expects its own single deposit. A
        # per-stage count would say 2 for one slot and 0 for the other.
        tables = self._tables(P=2, V=2, n_layers=96, layers_per_block=12)
        self.assertEqual(tables.expected_same_rank_captures(0, 0), 1)
        self.assertEqual(tables.expected_same_rank_captures(0, 1), 1)
        # Out-of-range commit index stays 0 rather than raising.
        self.assertEqual(tables.expected_same_rank_captures(0, 2), 0)

    def test_a_cache_consumer_is_always_a_later_stage(self):
        # The grad bridge assumes it: a consumer deposits during ITS backward,
        # which under Interleaved1F1B precedes the producer's own.
        for P, V, n, bs in ((2, 2, 96, 12), (2, 2, 16, 2), (1, 2, 16, 2)):
            tables = self._tables(P=P, V=V, n_layers=n, layers_per_block=bs)
            for b in range(tables.num_blocks):
                producer = tables.producer_stage_of_block(b)
                for consumer in tables.cache_consumers_of_block(b):
                    self.assertGreater(consumer, producer, f"P={P} block={b}")


class TestCaptureCountMismatchRaises(unittest.TestCase):
    """A capture-count mismatch means a gradient was dropped, so it raises.

    It used to warn, which left the run to take the step with an incomplete
    gradient for that block and nothing but a log line to say so.
    """

    def test_a_missing_consumer_deposit_raises_during_backward(self):
        import torch

        from torchtitan.models.kimi_k3.pipeline_adapter import (
            _install_augment_hook,
            RankLocalCache,
        )

        cache = RankLocalCache()
        block = torch.zeros(2, requires_grad=True)
        # Layout says one same-rank consumer will deposit; none does.
        _install_augment_hook(block, (0, 0, 0), cache, expected_captures=1)
        with self.assertRaises(RuntimeError) as ctx:
            (block * 2).sum().backward()
        self.assertIn("capture-count mismatch", str(ctx.exception))

    def test_the_expected_deposit_passes_and_is_summed_in(self):
        import torch

        from torchtitan.models.kimi_k3.pipeline_adapter import (
            _install_augment_hook,
            RankLocalCache,
        )

        cache = RankLocalCache()
        block = torch.zeros(2, requires_grad=True)
        _install_augment_hook(block, (0, 0, 0), cache, expected_captures=1)
        cache.capture_grad((0, 0, 0), torch.ones(2) * 3.0)
        (block * 2).sum().backward()
        # 2 from the local graph plus the consumer's 3.
        self.assertTrue(torch.equal(block.grad, torch.full((2,), 5.0)))


class TestStepEndSlotSweep(unittest.TestCase):
    """The step-end sweep clears captured-grad slots outright.

    The mb-keyed drop only reaches slots whose micro-batch still had cached
    blocks. A step that dies inside one micro-batch's backward -- OOM being the
    ordinary cause -- leaves a slot holding a grad tensor that nothing else
    frees, and the sweep runs from the step patch's ``finally``.
    """

    def test_a_slot_with_no_cached_blocks_is_still_cleared(self):
        import torch
        from torch import nn

        from torchtitan.models.kimi_k3.pipeline_adapter import CrossStageCacheAdapter

        adapter = CrossStageCacheAdapter(
            nn.Identity(), stage_id=0, num_stages=1, pp_rank=91
        )
        adapter._cache.capture_grad((7, 0, 0), torch.ones(2))
        self.assertEqual(adapter._cache.get_blocks(7), [])
        adapter._drop_all_cached_and_clear()
        self.assertEqual(adapter._cache.pop_grad((7, 0, 0)), (None, 0))

    def test_the_sweep_reports_how_many_it_cleared(self):
        import torch

        from torchtitan.models.kimi_k3.pipeline_adapter import RankLocalCache

        cache = RankLocalCache()
        cache.capture_grad((0, 0, 0), torch.ones(2))
        cache.capture_grad((1, 0, 0), torch.ones(2))
        self.assertEqual(cache.clear_capture_slots(), 2)
        self.assertEqual(cache.clear_capture_slots(), 0)


if __name__ == "__main__":
    unittest.main()


class TestShapeInferencePlaceholder(unittest.TestCase):
    """The delta placeholder must have the shape the runtime actually sends.

    Pipelining sizes the next stage's recv buffer from what shape inference returns, so a
    placeholder of the wrong rank is not a cosmetic mismatch -- the consumer then receives
    a carrier it does not recognise. The runtime sends ``torch.stack(pieces, dim=1)`` over
    ``[T, D]`` pieces, so the shape is ``[T, K, D]`` with T the flattened batch-sequence.

    Two earlier forms were wrong and both needed ``expected_K != N`` to show it, which no
    16-layer pp2 x vp2 run produces: an empty commit used ``partial_out.shape`` whole and
    returned a four-dimensional ``[K, B, L, D]``; a non-empty one used
    ``new_blocks_out.shape[1:]`` and put the block axis first. The four-dimensional case
    is what broke 32 layers at pp8 x vp2, as "got multiple values for argument 'blocks'" --
    the consumer's ``_has_blocks_signature`` tests ``dim() == 3``, so a rank-4 carrier fell
    through to the positional slot that ``blocks`` occupies.
    """

    def _adapter(self, *, stage_id, num_stages, layout, pp_rank):
        from torch import nn

        from torchtitan.models.kimi_k3.pipeline_adapter import CrossStageCacheAdapter

        wrapped = nn.Module()
        wrapped._return_only_new_blocks = True
        return CrossStageCacheAdapter(
            wrapped,
            stage_id=stage_id,
            num_stages=num_stages,
            layout_tables=layout,
            pp_rank=pp_rank,
        )

    def _layout(self):
        from torchtitan.models.kimi_k3.layout import BlockLayoutTables

        # 16 stages over 32 layers with blocks of 4: stages alternate between committing
        # one block and committing none, which is the empty-commit case.
        return BlockLayoutTables(
            pp_size=8,
            virtual_stages_per_rank=2,
            num_blocks=8,
            n_layers=32,
            layers_per_block=4,
        )

    def _placeholder(self, adapter, *, n_new):
        import torch

        partial = torch.zeros(1, 512, 1280, requires_grad=True)
        blocks = torch.zeros(512, n_new, 1280)
        adapter._call_wrapped_naive = lambda args, kwargs: (partial, blocks)
        return adapter._forward_shape_inference(partial)

    def test_an_empty_commit_still_yields_a_rank_three_carrier(self):
        layout = self._layout()
        # A stage whose delta differs from its commit count, so the placeholder path runs.
        stage = next(
            s
            for s in range(16)
            if len(layout.delta_to_send(s)) != len(layout.commits_at(s))
        )
        adapter = self._adapter(
            stage_id=stage, num_stages=16, layout=layout, pp_rank=stage % 8
        )
        _, carrier = self._placeholder(adapter, n_new=0)
        self.assertEqual(carrier.dim(), 3, f"rank must be 3, got {carrier.shape}")
        self.assertEqual(carrier.shape[0], 512, "T comes first, as stack(dim=1) emits")
        self.assertEqual(carrier.shape[1], len(layout.delta_to_send(stage)))
        self.assertEqual(carrier.shape[2], 1280)

    def test_the_carrier_keeps_requires_grad(self):
        """A requires_grad=False placeholder makes the consumer drop the backward edge."""
        layout = self._layout()
        stage = next(
            s
            for s in range(16)
            if len(layout.delta_to_send(s)) != len(layout.commits_at(s))
        )
        adapter = self._adapter(
            stage_id=stage, num_stages=16, layout=layout, pp_rank=stage % 8
        )
        _, carrier = self._placeholder(adapter, n_new=0)
        self.assertTrue(carrier.requires_grad)

    def test_a_consumer_recognises_the_placeholder_as_a_block_carrier(self):
        """The end-to-end property: dim() == 3 is what _has_blocks_signature tests."""
        import torch

        from torchtitan.models.kimi_k3.pipeline_adapter import CrossStageCacheAdapter

        layout = self._layout()
        stage = next(
            s
            for s in range(16)
            if len(layout.delta_to_send(s)) != len(layout.commits_at(s))
        )
        adapter = self._adapter(
            stage_id=stage, num_stages=16, layout=layout, pp_rank=stage % 8
        )
        _, carrier = self._placeholder(adapter, n_new=0)
        self.assertTrue(
            CrossStageCacheAdapter._has_blocks_signature(
                (torch.zeros(1, 512, 1280), carrier)
            ),
            "the consumer would pass this positionally into 'blocks' instead",
        )
