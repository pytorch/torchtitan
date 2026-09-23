# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The block routing tables, on CPU: uneven splits, the store, the deltas."""

import unittest

from torchtitan.models.kimi_k3.pipeline_parallel import _require_loop_style
from torchtitan.models.kimi_k3.pipeline_parallel.layout import (
    BlockLayoutTables,
    infer_block_layout_tables,
    layer_to_stage_from_split,
)


def _uneven_map() -> dict[int, int]:
    # 24 layers over 4 stages as 5 / 7 / 6 / 6.
    ranges = [(0, 5), (5, 12), (12, 18), (18, 24)]
    return {
        layer: stage for stage, (lo, hi) in enumerate(ranges) for layer in range(lo, hi)
    }


# Two ranks, two stages each, interleaved: stage s runs on rank s % 2.
_STAGE_TO_RANK = {0: 0, 1: 1, 2: 0, 3: 1}

# The split the B200 pp4 x vp4 cell spells out, repeated here so the test needs
# no recipe module: 16 stages, one layer per stage from layer 5 on.
_PP4_VP4_SPLIT = [
    ["vision_encoder", "tok_embeddings", "layers.0"],
    ["layers.1", "layers.2"],
    ["layers.3", "layers.4"],
    *[[f"layers.{i}"] for i in range(5, 17)],
    ["norm", "lm_head", "output_res_proj", "output_res_norm"],
]


def _tables(cache: bool = True) -> BlockLayoutTables:
    return BlockLayoutTables(
        stage_to_rank=_STAGE_TO_RANK,
        n_layers=24,
        layers_per_block=12,
        layer_to_stage=_uneven_map(),
        cache=cache,
    )


_EIGHT_STAGES = [
    ["vision_encoder", "tok_embeddings", "layers.0", "layers.1"],
    ["layers.2", "layers.3", "layers.4"],
    ["layers.5", "layers.6", "layers.7"],
    ["layers.8", "layers.9"],
    ["layers.10", "layers.11"],
    ["layers.12", "layers.13"],
    ["layers.14", "layers.15"],
    ["layers.16", "norm", "lm_head", "output_res_proj", "output_res_norm"],
]


class TestRouting(unittest.TestCase):
    def test_tables_follow_the_map_not_an_equal_split(self):
        tables = _tables()
        self.assertEqual(tables.producer_stage_of_block(0), 0)
        self.assertEqual(tables.producer_stage_of_block(1), 2)
        self.assertEqual(tables.commits_at(1), [])
        self.assertEqual(tables.delta_to_send(0), [0])
        self.assertEqual(tables.delta_to_send(1), [])
        self.assertEqual(tables.delta_to_send(2), [1])
        self.assertEqual(tables.delta_to_send(3), [])
        self.assertEqual(tables.cache_at_entry(2), frozenset({0}))
        self.assertEqual(tables.cache_at_entry(3), frozenset({0}))
        self.assertEqual(tables.cache_readers_of_block(0), [2, 3])
        self.assertEqual(tables.deposits_expected(0, 0), 1)
        self.assertEqual(tables.deposits_expected(0, 1), 1)
        self.assertEqual(tables.deposits_expected(1, 2), 0)

    def test_without_the_cache_every_hop_carries_everything(self):
        tables = _tables(cache=False)
        self.assertEqual(tables.delta_to_send(0), [0])
        self.assertEqual(tables.delta_to_send(1), [0])
        self.assertEqual(tables.delta_to_send(2), [0, 1])
        for stage in range(4):
            self.assertEqual(tables.cache_at_entry(stage), frozenset())
        self.assertEqual(tables.deposits_expected(0, 0), 0)

    def test_infer_accepts_the_map_and_rejects_a_broken_one(self):
        common = dict(
            stage_to_rank=_STAGE_TO_RANK,
            n_layers=24,
            layers_per_block=12,
        )
        tables = infer_block_layout_tables(layer_to_stage=_uneven_map(), **common)
        self.assertEqual(tables.producer_stage_of_block(1), 2)
        incomplete = _uneven_map()
        del incomplete[7]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            infer_block_layout_tables(layer_to_stage=incomplete, **common)
        scrambled = _uneven_map()
        scrambled[7], scrambled[20] = scrambled[20], scrambled[7]
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            infer_block_layout_tables(layer_to_stage=scrambled, **common)

    def test_the_map_is_read_off_the_split(self):
        split = [
            ["tok_embeddings", "vision_encoder", "layers.0", "layers.1"],
            ["layers.2"],
            ["layers.3", "norm", "lm_head", "output_res_proj"],
        ]
        self.assertEqual(layer_to_stage_from_split(split), {0: 0, 1: 0, 2: 1, 3: 2})


class TestSplit(unittest.TestCase):
    def _tables(self, split, pp):
        layer_to_stage = layer_to_stage_from_split(split)
        self.assertEqual(sorted(layer_to_stage), list(range(17)))
        return BlockLayoutTables(
            stage_to_rank={s: s % pp for s in range(len(split))},
            n_layers=17,
            layers_per_block=4,
            layer_to_stage=layer_to_stage,
            cache=True,
        )

    def test_the_pp2_vp2_cell_has_every_transport_path(self):
        from torchtitan.distributed.pipeline_parallel import (
            _generate_llm_fqn_per_model_part,
        )

        split = _generate_llm_fqn_per_model_part(4, 17)
        self.assertEqual([len(stage) for stage in split], [5, 5, 5, 5])
        tables = self._tables(split, pp=2)
        self.assertEqual(
            [tables.producer_stage_of_block(b) for b in range(5)], [0, 1, 1, 2, 3]
        )
        self.assertEqual(
            [tables.delta_to_send(s) for s in range(3)], [[0], [1, 2], [3]]
        )
        self.assertEqual(tables.cache_at_entry(2), frozenset({0}))
        self.assertEqual(tables.cache_at_entry(3), frozenset({0, 1, 2}))

    def test_an_eight_stage_split_collects_three_deposits_per_block(self):
        split = _EIGHT_STAGES
        self.assertEqual(len(split), 8)
        self.assertEqual(split[0][:2], ["vision_encoder", "tok_embeddings"])
        self.assertEqual(split[-1][-2:], ["output_res_proj", "output_res_norm"])
        tables = self._tables(split, pp=2)
        self.assertEqual(
            [tables.producer_stage_of_block(b) for b in range(5)], [0, 1, 3, 5, 7]
        )
        self.assertEqual(
            [tables.delta_to_send(s) for s in range(7)],
            [[0], [1], [], [2], [], [3], []],
        )
        self.assertEqual(tables.cache_at_entry(7), frozenset({0, 1, 2, 3}))
        self.assertEqual(tables.deposits_expected(0, 0), 3)
        self.assertEqual(tables.deposits_expected(0, 1), 3)

    def test_the_pp4_vp4_cell_runs_one_layer_per_stage(self):
        split = _PP4_VP4_SPLIT
        self.assertEqual(len(split), 16)
        self.assertEqual(sum(n.startswith("layers.") for s in split for n in s), 17)
        self.assertEqual(
            split[-1], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )
        tables = self._tables(split, pp=4)
        self.assertEqual(
            [tables.producer_stage_of_block(b) for b in range(5)], [0, 2, 6, 10, 14]
        )
        self.assertEqual(tables.delta_to_send(2), [0, 1])
        self.assertEqual(tables.delta_to_send(5), [])
        self.assertEqual(tables.delta_to_send(14), [4])
        self.assertEqual(tables.cache_at_entry(15), frozenset({0, 1, 2, 3}))
        self.assertEqual(tables.deposits_expected(0, 0), 3)


class TestLoopStyleGuard(unittest.TestCase):
    def test_a_loop_style_assignment_passes(self):
        _require_loop_style(object(), {0: 0, 1: 1, 2: 0, 3: 1}, pp=2)

    def test_any_other_assignment_is_refused(self):
        # Stages 0..3 on ranks 0, 1, 1, 0, the v shape: stage 2 revisits rank 1
        # out of index order, so the rank store would be asked for blocks it
        # never saw. The map is what is checked, whatever schedule produced it.
        with self.assertRaisesRegex(ValueError, "unsupported with attn_res_cache"):
            _require_loop_style(object(), {0: 0, 1: 1, 2: 1, 3: 0}, pp=2)


if __name__ == "__main__":
    unittest.main()
