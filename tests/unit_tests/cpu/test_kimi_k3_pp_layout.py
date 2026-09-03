# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The block layout tables over an uneven pipeline split, on CPU."""

import unittest
from types import SimpleNamespace

import torch.nn as nn

from torchtitan.models.kimi_k3.layout import (
    BlockLayoutTables,
    infer_block_layout_tables_from_stages,
    local_layer_to_stage,
)


def _uneven_map() -> dict[int, int]:
    # 24 layers over 4 stages as 5 / 7 / 6 / 6: the first stage carries the
    # embedding and the last the head, so neither holds a full share.
    ranges = [(0, 5), (5, 12), (12, 18), (18, 24)]
    return {
        layer: stage for stage, (lo, hi) in enumerate(ranges) for layer in range(lo, hi)
    }


class TestUnevenSplit(unittest.TestCase):
    def test_tables_follow_the_map_not_an_equal_split(self):
        tables = BlockLayoutTables(
            pp_size=2,
            virtual_stages_per_rank=2,
            num_blocks=2,
            n_layers=24,
            layers_per_block=12,
            layer_to_stage=_uneven_map(),
        )
        # Block 0 opens at layer 0 (stage 0); block 1 at layer 12, which the
        # uneven split puts on stage 2, not stage 1 as an equal split would.
        self.assertEqual(tables.producer_stage_of_block(0), 0)
        self.assertEqual(tables.producer_stage_of_block(1), 2)
        self.assertEqual(tables.commits_at(1), [])
        # Stage 2 runs on rank 0, which committed block 0 at stage 0, so the
        # hop from stage 1 carries nothing; stage 3 runs on rank 1, which
        # cached block 0 when stage 1 received it, so the hop from stage 2
        # carries only the new block.
        self.assertEqual(tables.delta_to_send(0), [0])
        self.assertEqual(tables.delta_to_send(1), [])
        self.assertEqual(tables.delta_to_send(2), [1])
        self.assertEqual(tables.delta_to_send(3), [])
        self.assertEqual(tables.rank_cache_at_entry(0, 1), frozenset({0}))
        self.assertEqual(tables.rank_cache_at_entry(1, 1), frozenset({0}))

    def test_infer_accepts_the_map_and_rejects_a_broken_one(self):
        stages = [SimpleNamespace(stage_index=0), SimpleNamespace(stage_index=2)]
        tables = infer_block_layout_tables_from_stages(
            stages,
            pp_size=2,
            num_blocks=2,
            n_layers=24,
            layers_per_block=12,
            layer_to_stage=_uneven_map(),
        )
        self.assertEqual(tables.producer_stage_of_block(1), 2)
        incomplete = _uneven_map()
        del incomplete[7]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            infer_block_layout_tables_from_stages(
                stages,
                pp_size=2,
                num_blocks=2,
                n_layers=24,
                layers_per_block=12,
                layer_to_stage=incomplete,
            )
        scrambled = _uneven_map()
        scrambled[7], scrambled[20] = scrambled[20], scrambled[7]
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            infer_block_layout_tables_from_stages(
                stages,
                pp_size=2,
                num_blocks=2,
                n_layers=24,
                layers_per_block=12,
                layer_to_stage=scrambled,
            )

    def test_local_map_reads_the_stage_modules(self):
        class _Part(nn.Module):
            def __init__(self, ids):
                super().__init__()
                self.layers = nn.ModuleDict({str(i): nn.Linear(2, 2) for i in ids})

        stages = [
            SimpleNamespace(stage_index=0, submod=_Part(range(0, 5))),
            SimpleNamespace(stage_index=2, submod=_Part(range(12, 18))),
        ]
        local = local_layer_to_stage(stages)
        self.assertEqual({local[i] for i in range(0, 5)}, {0})
        self.assertEqual({local[i] for i in range(12, 18)}, {2})
        self.assertEqual(len(local), 11)


if __name__ == "__main__":
    unittest.main()
