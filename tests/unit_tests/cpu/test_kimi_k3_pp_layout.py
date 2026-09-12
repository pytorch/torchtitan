# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The block routing tables, on CPU: uneven splits, the store, the deltas."""

import unittest
from types import SimpleNamespace

from torchtitan.models.kimi_k3.layout import (
    BlockLayoutTables,
    infer_block_layout_tables_from_stages,
    layer_to_stage_from_split,
)


def _uneven_map() -> dict[int, int]:
    # 24 layers over 4 stages as 5 / 7 / 6 / 6: the first stage carries the
    # embedding and the last the head, so neither holds a full share.
    ranges = [(0, 5), (5, 12), (12, 18), (18, 24)]
    return {
        layer: stage for stage, (lo, hi) in enumerate(ranges) for layer in range(lo, hi)
    }


# Two ranks, two stages each, interleaved: stage s runs on rank s % 2.
_STAGE_TO_RANK = {0: 0, 1: 1, 2: 0, 3: 1}


def _tables(cache: bool = True) -> BlockLayoutTables:
    return BlockLayoutTables(
        stage_to_rank=_STAGE_TO_RANK,
        num_blocks=2,
        n_layers=24,
        layers_per_block=12,
        layer_to_stage=_uneven_map(),
        cache=cache,
    )


class TestRouting(unittest.TestCase):
    def test_tables_follow_the_map_not_an_equal_split(self):
        tables = _tables()
        # Block 0 opens at layer 0 (stage 0); block 1 at layer 12, which the
        # uneven split puts on stage 2, not stage 1 as an equal split would.
        self.assertEqual(tables.producer_stage_of_block(0), 0)
        self.assertEqual(tables.producer_stage_of_block(1), 2)
        self.assertEqual(tables.commits_at(1), [])
        # Stage 2 runs on rank 0, which committed block 0 at stage 0, so the
        # hop from stage 1 carries nothing; stage 3 runs on rank 1, which kept
        # block 0 when stage 1 received it, so the hop from stage 2 carries
        # only the new block.
        self.assertEqual(tables.delta_to_send(0), [0])
        self.assertEqual(tables.delta_to_send(1), [])
        self.assertEqual(tables.delta_to_send(2), [1])
        self.assertEqual(tables.delta_to_send(3), [])
        self.assertEqual(tables.cache_at_entry(2), frozenset({0}))
        self.assertEqual(tables.cache_at_entry(3), frozenset({0}))
        # Block 0 is read from the store by stages 2 and 3: one deposit for
        # its producer (stage 0, rank 0) and one for the stage that received
        # it on rank 1 (stage 1).
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
        stages = [SimpleNamespace(stage_index=0), SimpleNamespace(stage_index=2)]
        common = dict(
            stage_to_rank=_STAGE_TO_RANK,
            num_blocks=2,
            n_layers=24,
            layers_per_block=12,
        )
        tables = infer_block_layout_tables_from_stages(
            stages, layer_to_stage=_uneven_map(), **common
        )
        self.assertEqual(tables.producer_stage_of_block(1), 2)
        incomplete = _uneven_map()
        del incomplete[7]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            infer_block_layout_tables_from_stages(
                stages, layer_to_stage=incomplete, **common
            )
        scrambled = _uneven_map()
        scrambled[7], scrambled[20] = scrambled[20], scrambled[7]
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            infer_block_layout_tables_from_stages(
                stages, layer_to_stage=scrambled, **common
            )

    def test_the_map_is_read_off_the_split(self):
        """Every rank computes the split, so the layer-to-stage map needs no
        collective; only ``layers.<n>`` names count, pinned modules do not."""
        split = [
            ["tok_embeddings", "vision_encoder", "layers.0", "layers.1"],
            ["layers.2"],
            ["layers.3", "norm", "lm_head", "output_res_proj"],
        ]
        self.assertEqual(layer_to_stage_from_split(split), {0: 0, 1: 0, 2: 1, 3: 2})


if __name__ == "__main__":
    unittest.main()
